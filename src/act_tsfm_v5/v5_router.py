"""ACT-TSFM V5: Prediction-Assisted Anytime-Valid Paired Racing with Cost-Aware Sampling."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Sequence, Any

from act_tsfm.evaluator.engine import RevealOnlyEvaluator
from act_tsfm.model_descriptors import CELL_LIST

from .sampling_design import FinitePopulationSamplingDesign
from .paired_estimators import AIPWPairedEstimator
from .confidence_sequences import AnytimeValidConfidenceSequence
from .residual_predictor import ResidualScalePredictor
from .v4_context_adapter import V4ContextAdapter, PRIMARY_MODELS
from .bundle_selection import generate_candidate_bundles
from .kg_star import KGStarScorer
from .thompson_fallback import ThompsonFallbackPolicy
from .racing_elimination import RacingEliminationManager
from .certification import CertificationManager


class V5Policy:
    """ACT-TSFM V5 Policy.

    Combines:
    1. V4 In-Context Transformer auxiliary paired-difference predictions g_t(w).
    2. Paired model evaluations on identical financial test windows.
    3. Finite-population AIPW estimation.
    4. Anytime-valid confidence sequences for elimination and certification.
    5. Cost-aware Neyman sampling allocation pi_t*(w) proportional to s_t(w) / sqrt(C(w)).
    6. Racing-style elimination and per-cell certification.
    7. Never-idle fallback policy.
    8. Separate reporting of provisional recommendations and certification status.
    """
    def __init__(
        self,
        seed: int = 42,
        models: Sequence[str] | None = None,
        use_transformer: bool = True,
        use_neyman_allocation: bool = True,
        use_auxiliary_losses: bool = True,
        device: str = "cuda"
    ):
        self.name = "v5"
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.models = list(models) if models is not None else PRIMARY_MODELS
        
        self.use_transformer = use_transformer
        self.use_neyman_allocation = use_neyman_allocation
        self.use_auxiliary_losses = use_auxiliary_losses
        self.device = device
        
        # Sub-components
        self.adapter = V4ContextAdapter(device=device, model_list=self.models) if use_transformer else None
        self.residual_predictor = ResidualScalePredictor()
        self.kg_scorer = KGStarScorer(max_depth=2)
        self.thompson_fallback = ThompsonFallbackPolicy(seed=seed)
        self.racing = RacingEliminationManager(CELL_LIST, self.models, delta=0.05)
        self.certifier = CertificationManager(CELL_LIST, epsilon_cell=0.05)
        
        # Per-cell sampling designs and per-(pair, cell) AIPW estimators
        self.sampling_designs: dict[str, FinitePopulationSamplingDesign] = {}
        self.aipw_estimators: dict[tuple[str, str, str], AIPWPairedEstimator] = {}
        self.confidence_seqs: dict[tuple[str, str, str], AnytimeValidConfidenceSequence] = {}
        
        # Precomputed auxiliary loss cache (for HAR & GARCH)
        self.auxiliary_loss_df: pd.DataFrame | None = None
        self._initialized = False
        self._last_ledger_len = 0
        self._last_pred_ledger_len = -1
        self._cached_predictions: dict[tuple[str, str, str], float] | None = None
        self.w_to_cell: dict[str, str] = {}
        self.cell_to_windows: dict[str, list[str]] = {}

    def _init_structures(self, available_windows: pd.DataFrame):
        if self._initialized:
            return
            
        self.w_to_cell = dict(zip(available_windows["window_id"], available_windows["cell_id"]))
        for c in CELL_LIST:
            w_list = available_windows[available_windows["cell_id"] == c]["window_id"].tolist()
            self.cell_to_windows[c] = w_list
            self.sampling_designs[c] = FinitePopulationSamplingDesign(w_list, seed=self.seed)

        if self.adapter:
            self.adapter.init_descriptors(available_windows)

        # Initialize estimators for all unique pairs (m1, m2) and cells
        for i, m1 in enumerate(self.models):
            for m2 in self.models[i+1:]:
                for c in CELL_LIST:
                    w_list = self.cell_to_windows[c]
                    key = (m1, m2, c)
                    self.aipw_estimators[key] = AIPWPairedEstimator(w_list, clipping_bound=3.0)
                    self.confidence_seqs[key] = AnytimeValidConfidenceSequence(
                        population_size=len(w_list),
                        delta=self.racing.alpha_pair,
                        bound=3.0
                    )
        self._initialized = True

    def set_auxiliary_losses(self, aux_df: pd.DataFrame):
        """Sets precomputed cheap model auxiliary losses (HAR/GARCH) if enabled."""
        if self.use_auxiliary_losses:
            self.auxiliary_loss_df = aux_df

    def _sync_ledger(self, evaluator: RevealOnlyEvaluator):
        """Processes new observations revealed in evaluator's ledger."""
        new_entries = evaluator.ledger[self._last_ledger_len:]
        if not new_entries:
            return

        for entry in new_entries:
            m = entry["model_id"]
            w = entry["window_id"]
            c = entry["cell_id"]
            loss = float(entry["qlike_loss"])
            
            # Check if paired counterpart is also revealed on this window
            for other_m in self.models:
                if other_m == m:
                    continue
                other_key = (other_m, w)
                if other_key in evaluator.revealed_observations:
                    # Retrieve other model's loss from ledger
                    other_entry = next(e for e in evaluator.ledger if e["model_id"] == other_m and e["window_id"] == w)
                    other_loss = float(other_entry["qlike_loss"])
                    
                    pair_key = (m, other_m, c) if (m, other_m, c) in self.aipw_estimators else (other_m, m, c)
                    diff = (loss - other_loss) if pair_key[0] == m else (other_loss - loss)
                    
                    if w not in self.aipw_estimators[pair_key].observed_D:
                        # Determine sampling probability from design
                        des = self.sampling_designs[c]
                        q_t = des.compute_sampling_probabilities().get(w, 1.0 / max(1, des.n_unobserved + 1))
                        y_inc = self.aipw_estimators[pair_key].update(w, diff, q_t)
                        self.confidence_seqs[pair_key].update(y_inc)
                        des.mark_observed(w)
                        self.residual_predictor.update(c, diff - self.aipw_estimators[pair_key].predictions.get(w, 0.0))

        self._last_ledger_len = len(evaluator.ledger)

    def select_action(
        self,
        evaluator: RevealOnlyEvaluator,
        available_windows: pd.DataFrame
    ) -> tuple[list[str], str] | None:
        """Selects the next paired bundle a = (S, w) to evaluate."""
        self._init_structures(available_windows)
        self._sync_ledger(evaluator)
        
        # Stop if all cells are fully certified
        if self.certifier.check_global_certification():
            return None

        # Predict paired differences with V4 transformer prior (cached periodically)
        predictions = {}
        if self.adapter:
            cur_l = len(evaluator.ledger)
            if self._cached_predictions is None or (cur_l - self._last_pred_ledger_len) >= 8:
                self._cached_predictions = self.adapter.predict_paired_differences(
                    evaluator.ledger,
                    auxiliary_losses=self.auxiliary_loss_df
                )
                self._last_pred_ledger_len = cur_l
                for (m1, m2, c), val in self._cached_predictions.items():
                    if (m1, m2, c) in self.aipw_estimators:
                        w_dict = {w: val for w in self.cell_to_windows[c]}
                        self.aipw_estimators[(m1, m2, c)].set_predictions(w_dict)
            predictions = self._cached_predictions

        # Update eliminations and certifications across cells
        for c in CELL_LIST:
            leader = self._get_cell_leader(c)
            active = list(self.racing.active_arms[c])
            
            # 1. Elimination check: LCB(Delta_{cand, leader}) > 0
            def get_lcb(cand, lead, cell):
                key = (cand, lead, cell) if (cand, lead, cell) in self.confidence_seqs else (lead, cand, cell)
                lcb, ucb, _ = self.confidence_seqs[key].get_bounds()
                return lcb if key[0] == cand else -ucb

            self.racing.check_eliminations(c, leader, get_lcb)
            
            # 2. Certification check: UCB(Delta_{leader, ch}) <= epsilon
            def get_ucb(lead, ch, cell):
                key = (lead, ch, cell) if (lead, ch, cell) in self.confidence_seqs else (ch, lead, cell)
                lcb, ucb, _ = self.confidence_seqs[key].get_bounds()
                return ucb if key[0] == lead else -lcb

            self.certifier.evaluate_cell_certification(c, leader, active, get_ucb)

        # Prioritize uncertified cells
        uncertified_cells = [c for c in CELL_LIST if c not in self.certifier.certified_cells]
        target_cells = uncertified_cells if uncertified_cells else CELL_LIST
        
        best_bundle = None
        best_score = -1.0

        for c in target_cells:
            leader = self._get_cell_leader(c)
            active = list(self.racing.active_arms[c])
            des = self.sampling_designs[c]
            
            if des.n_unobserved == 0:
                continue

            # Candidate windows drawn according to sampling design
            res_scales = {}
            if self.use_neyman_allocation:
                for w in des.unobserved_windows:
                    res_scales[w] = self.residual_predictor.predict_scale(c, w)

            candidate_w_info = des.sample_next_window(residuals_std=res_scales)
            if candidate_w_info is None:
                continue
            w_id, _ = candidate_w_info

            bundles = generate_candidate_bundles(c, w_id, leader, active, evaluator)
            for models_sub, _, cost in bundles:
                ch = models_sub[1] if len(models_sub) > 1 else models_sub[0]
                pred_gap = predictions.get((leader, ch, c), 0.10)
                unc = res_scales.get(w_id, 0.20)
                score = self.kg_scorer.score_bundle(models_sub, w_id, cost, pred_gap, unc)

                if score > best_score:
                    best_score = score
                    best_bundle = (models_sub, w_id)

        if best_bundle is not None and best_score > 1e-4:
            return best_bundle

        # Never-Idle Fallback: Top-two Thompson Sampling
        for c in target_cells:
            leader = self._get_cell_leader(c)
            active = list(self.racing.active_arms[c])
            means = {m: self._get_model_mean_loss(m, c) for m in active}
            variances = {m: 0.15 for m in active}
            
            fb = self.thompson_fallback.select_fallback_bundle(
                active, means, variances, self.sampling_designs[c], evaluator
            )
            if fb is not None:
                return fb

        return None

    def _get_cell_leader(self, cell_id: str) -> str:
        """Returns the provisional leader (model with lowest estimated mean loss) in cell."""
        losses = {m: self._get_model_mean_loss(m, cell_id) for m in self.models}
        return min(losses.keys(), key=lambda m: losses[m])

    def _get_model_mean_loss(self, model_id: str, cell_id: str) -> float:
        """Computes current AIPW or ledger mean loss for model_id in cell_id."""
        diffs = []
        for other_m in self.models:
            if other_m == model_id:
                continue
            key = (model_id, other_m, cell_id) if (model_id, other_m, cell_id) in self.aipw_estimators else (other_m, model_id, cell_id)
            if key in self.aipw_estimators:
                d = self.aipw_estimators[key].get_estimate()
                diffs.append(d if key[0] == model_id else -d)
        if diffs:
            # Anchored loss: relative to average of others
            return 0.35 + float(np.mean(diffs))
        return 0.35

    def make_recommendations(
        self,
        evaluator: RevealOnlyEvaluator
    ) -> tuple[str, dict[str, str]]:
        """Produces provisional model recommendations based on current loss estimates:

            m_hat = argmin_m L_m
            pi_hat(c) = argmin_m L_{m,c}
        Certification status is reported independently.
        """
        self._sync_ledger(evaluator)
        
        routing = {}
        model_overall_losses = {m: 0.0 for m in self.models}
        
        for c in CELL_LIST:
            best_m = self._get_cell_leader(c)
            routing[c] = best_m
            for m in self.models:
                model_overall_losses[m] += self._get_model_mean_loss(m, c)

        global_rec = min(self.models, key=lambda m: model_overall_losses[m])
        return global_rec, routing

