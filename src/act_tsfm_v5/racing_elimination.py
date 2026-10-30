"""Racing-style pairwise elimination and active candidate arm management."""

from __future__ import annotations
import numpy as np


class RacingEliminationManager:
    """Maintains active candidate arms per cell and eliminates statistically

    inferior candidates based on anytime-valid confidence sequence lower bounds:
        LCB(Delta_{A, B, c}) > 0  => Eliminate A.
    """
    def __init__(
        self,
        cell_list: list[str],
        all_models: list[str],
        delta: float = 0.05
    ):
        self.cells = list(cell_list)
        self.all_models = list(all_models)
        self.delta = delta
        
        # Family-wise error allocation per cell and pair
        # alpha_c = delta / (N_cells * N_pairs)
        n_pairs = len(all_models) * (len(all_models) - 1) // 2
        self.alpha_pair = delta / max(1, len(self.cells) * max(1, n_pairs))
        
        # Active candidate set per cell
        self.active_arms: dict[str, set[str]] = {
            c: set(all_models) for c in self.cells
        }
        self.elimination_history: list[dict] = []

    def check_eliminations(
        self,
        cell_id: str,
        provisional_leader: str,
        lcb_func
    ) -> list[str]:
        """Checks if any active candidate A in cell_id can be eliminated by the leader.

        lcb_func(m_cand, m_lead, cell_id) -> float
        """
        active = list(self.active_arms[cell_id])
        eliminated = []
        
        for cand in active:
            if cand == provisional_leader:
                continue
            # Difference Delta_{cand, leader} = Loss(cand) - Loss(leader)
            lcb = lcb_func(cand, provisional_leader, cell_id)
            if lcb > 0.0:
                self.active_arms[cell_id].remove(cand)
                eliminated.append(cand)
                self.elimination_history.append({
                    "cell_id": cell_id,
                    "eliminated_model": cand,
                    "dominating_leader": provisional_leader,
                    "lcb": float(lcb)
                })
        return eliminated

