# Date Format Conversion with GRPO — CS AI/RL Phase 2

Fine-tune a tiny causal language model to normalize written and US numeric dates to ISO `YYYY-MM-DD`. Invalid dates must return `INVALID`. Two-digit years 00–49 map to 2000–2049 and 50–99 map to 1950–1999.

## Repository contents

- `src/date_rules.py`: deterministic reference answers and date validation
- `src/rewards.py`: rule-based GRPO reward (1 exact; otherwise 0.2 per correct component for valid ISO dates; 0 for invalid outputs)
- `scripts/generate_dataset.py`: reproducible dataset generator
- `scripts/train.py`: TRL GRPO training and reward logs
- `scripts/evaluate.py`: held-out accuracy and average reward
- `scripts/train.slurm`: starting point for Scholar job submission
- `tests/test_rules.py`: deterministic unit tests
- `data/sample_10.csv`: original Phase 1 sample examples

## 1. Test without a GPU

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/generate_dataset.py --count 600 --out data/dates.csv
```

## 2. Install training dependencies (on a GPU machine / Scholar)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.txt
python scripts/train.py --data data/dates.csv --output outputs/date-grpo --steps 100
```

Model downloads require network access and may require login. The default `EleutherAI/pythia-70m` is a tiny **starting baseline**, not a validated result. GPU resources and package versions may require adjustments. GRPO uses 2 completions per prompt and logs rewards at every training step. `outputs/date-grpo/trainer_log_history.json` holds reward metrics after training.

For Scholar batch submission, first confirm your RCAC allocation and GPU partition, then edit the SLURM header as required and run `sbatch scripts/train.slurm` from the repository root. The SLURM file is a template, not a verified Scholar configuration.

## 3. Evaluate before and after

```bash
python scripts/evaluate.py --model EleutherAI/pythia-70m --out outputs/base_eval.json
python scripts/evaluate.py --model outputs/date-grpo/final_model --out outputs/fine_tuned_eval.json
```

Both evaluations use the held-out CSV written during training. Compare `average_reward` and `exact_match_accuracy`. The split holds out distinct **valid dates**, but all invalid examples remain in training; therefore, invalid-date performance is not independently tested. For a stronger audit, construct a separate invalid-date holdout.

## 4. Phase 2 audit, log, and reflection

- **Deterministic checks:** Run `python -m pytest -q` and record results in `AUDIT.md`.
- **Independent AI audit:** Ask a *different* coding agent to review `src/` and `scripts/`, check reward exploitation, and propose additional unit tests. Record its findings and any fixes in `AUDIT.md`.
- **Training logs:** After a real training run, commit `outputs/date-grpo/trainer_log_history.json` (or a curated log in `logs/`) if reasonably sized; the `outputs/` directory is ignored by default. Avoid committing large model weights.
- **Convergence:** Replace the placeholders in `PHASE2_REPORT.md` using actual logged rewards.

**Important:** The code has not been GPU-trained in this starter repository. Do not report improvements or successful convergence before running the training and evaluating it.
