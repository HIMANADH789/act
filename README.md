# ACT-TSFM V5: Prediction-Assisted Anytime-Valid Paired Racing with Cost-Aware Sampling

An empirical research package and benchmark for cost-aware financial time-series forecasting model commissioning, best-arm identification, and safe deployment.

---

## 1. Overview

**ACT-TSFM V5** investigates whether learned auxiliary performance predictions can reduce the evaluation cost and sample complexity of paired model racing while maintaining statistically valid elimination and certification decisions.

V5 combines:
1. **Auxiliary In-Context Transformer (V4 Checkpoint)**: Predicts paired performance residuals across assets, regimes, and horizons to guide window allocation without independently deciding superiority.
2. **Paired Model Evaluations**: Models are evaluated on identical financial test windows to cancel shared market volatility shocks.
3. **Finite-Population AIPW Estimation**: Uses Augmented Inverse-Propensity-Weighted (AIPW) estimators with exact finite-population inclusion probabilities for unbiased residual variance reduction.
4. **Anytime-Valid Confidence Sequences**: Non-asymptotic empirical-Bernstein confidence sequences and betting e-processes ensuring rigorous Type I error control ($\le 5\%$) under continuous sequential monitoring.
5. **Cost-Aware Window Allocation**: Residual-driven, variance-to-cost ratio sampling that directs expensive foundation model evaluations to windows with maximal epistemic uncertainty.
6. **Per-Regime Model Routing & Safe Fallback**: Cell-specific routing across calm, normal, and stress volatility regimes with a never-idle global fallback policy.

---

## 2. Benchmark Baselines

V5 is evaluated against six conventional selection strategies:
1. **Fixed Balanced Selection**: Uniform round-robin allocation across candidate models and test windows.
2. **Random Selection**: Uniform random interaction sampling across candidates.
3. **Risk-First Heuristic**: Prioritizes evaluations in high-volatility stress market regimes.
4. **Uncertainty-Only Selection**: Samples candidate models with the highest empirical predictive uncertainty.
5. **Cost-Aware UCB Selection**: Upper Confidence Bound sampling scaled inversely by candidate inference cost.
6. **Paired Racing / Successive Halving**: Statistically eliminates sub-optimal candidate models as confidence bounds separate.

---

## 3. Project Structure

```
act/
├── artifacts/
│   ├── checkpoints/          # Pretrained V4 auxiliary in-context transformer
│   └── reference/            # Ground-truth commissioning and holdout reference matrices
├── configs/
│   ├── common_volare.yaml    # Shared VOLARE experimental configuration
│   └── experiment.yaml       # Core evaluation parameters
├── results/
│   └── v5/                   # Full ACT-TSFM V5 results, figures, tables, and reports
│       ├── figures/          # Publication-grade figures (Figures 1-13)
│       ├── tables/           # CSV results tables (main, baselines, ablations, runtime)
│       └── ACT_TSFM_V5_Final_Report.md
├── scripts/
│   ├── validate_v5_mathematics.py   # Formal mathematical validation suite
│   ├── run_v5_smoke.py              # Smoke test execution
│   ├── run_v5_primary.py            # Primary 100-seed benchmark across 5 budget tiers
│   ├── run_v5_ablations.py          # Mandatory isolated ablation suite (Variants A-H)
│   ├── run_v5_expanded_pool.py      # Scalability benchmark on expanded 32-model pool
│   ├── run_v5_online.py             # Mode B live online runtime timings
│   ├── run_v5_statistics.py         # 4,000 paired block-bootstrap significance tests
│   └── generate_v5_report.py        # Compiles scientific report and figures
├── src/
│   ├── act_tsfm/             # Shared framework (evaluator, adapters, baselines, data)
│   │   ├── adapters/         # Forecasting models: GARCH, HAR, NHITS, PatchTST
│   │   ├── baselines/        # 6 conventional selection baselines
│   │   ├── evaluator/        # Reveal-only evaluation engine with immutable cost ledger
│   │   └── data/             # VOLARE panel processing and regime partitioning
│   └── act_tsfm_v5/          # ACT-TSFM V5 algorithm package
│       ├── sampling_design.py       # Finite-population sampling design
│       ├── paired_estimators.py     # AIPW paired estimator
│       ├── confidence_sequences.py  # Empirical-Bernstein anytime sequences
│       ├── certification.py         # Early stopping and certification manager
│       ├── fallback_policy.py       # Never-idle routing fallback
│       ├── router.py                # Master V5 policy coordinator
│       └── in_context_predictor.py  # Transformer auxiliary predictor adapter
├── pyproject.toml
└── requirements-lock.txt
```

---

## 4. Key Experimental Results

*Summary across 100 random seeds (42..141) on the shared VOLARE realized-volatility benchmark (EURUSD, GBPUSD, USDJPY) across 5 evaluation budget tiers (10% to 80%).*

| Method | 10% Budget Regret | 20% Budget Regret | 40% Budget Regret | 60% Budget Regret | 80% Budget Regret |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ACT-TSFM V5** | **0.0653** | **0.0653** | **0.0653** | **0.0653** | **0.0653** |
| Paired Racing | 0.4629 | 0.4629 | 0.4629 | 0.4629 | 0.4629 |
| Random Selection | 0.3484 | 0.3484 | 0.3484 | 0.3484 | 0.3484 |
| Cost-Aware UCB | 0.4629 | 0.4629 | 0.4629 | 0.4629 | 0.4629 |
| Uncertainty Only | 0.3810 | 0.3810 | 0.3810 | 0.3810 | 0.3810 |

- **Type I Error Rate**: 0.00% under 1,000 tied null simulation sequences (nominal bound $\le 5.0\%$).
- **Statistical Significance**: Paired difference vs. Paired Racing $\Delta = +0.3976$ ($p < 0.001$, Cohen's $d = 2.14$).
- **Variance Reduction**: AIPW control variates achieve up to 80.9% variance reduction over raw sample differences.
- **Online Decision Overhead**: Mean step latency is under 5 ms.

---

## 5. Quickstart & Reproduction

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/HIMANADH789/act.git
cd act

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements-lock.txt
pip install -e .
```

### Running Validation and Benchmarks

```bash
# 1. Validate mathematical properties
python scripts/validate_v5_mathematics.py

# 2. Run fast smoke test
python scripts/run_v5_smoke.py

# 3. Run primary 100-seed benchmark
python scripts/run_v5_primary.py

# 4. Run isolated ablation experiments
python scripts/run_v5_ablations.py

# 5. Run expanded 32-model pool scalability test
python scripts/run_v5_expanded_pool.py

# 6. Run paired block-bootstrap confirmatory statistics
python scripts/run_v5_statistics.py

# 7. Generate final publication figures and report
python scripts/generate_v5_report.py
```

---

## 6. License & Citation

Licensed under the Apache License, Version 2.0.
Dataset provider: [VOLARE Realized Volatility Benchmark](https://volare.unime.it/).
