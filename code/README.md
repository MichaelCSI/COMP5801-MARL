# When Messages Fail: Robust Peer SoC Communication in CityLearn

COMP 5801 MARL project. Six buildings in CityLearn (Challenge 2023 Phase 3.3)
each control a battery with their own MAPPO agent. Before acting, each agent
sends its battery state of charge (SoC) to its peers over a lossy channel. We
measure how much of the messages' benefit survives when they drop, and whether
training under random loss (M1-RND) makes the policies robust.

| Arm | Description |
|-----|-------------|
| M0 | MAPPO, no messages (primary baseline) |
| M1 | MAPPO with peer SoC + availability mask, 100% delivery in training |
| M1-RND | Same inputs, delivery probability drawn per episode from {0, 0.5, 0.75, 1} |
| I0 | IPPO, no messages (does the centralized critic help?) |
| M-FULL | Every peer's true SoC and net load, never dropped (upper bound) |
| M1-HOLD | Evaluation of M1 only: a dropped message reuses the last SoC received |
| M1-RND-NM | Optional: M1-RND without the mask |

Message arms are evaluated at delivery levels {100%, 90%, 50%, 25%, 0%}; arms without
messages are evaluated once.

The learner is the reference MAPPO implementation of Yu et al. (NeurIPS 2022),
[marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy), vendored
unmodified in `third_party/onpolicy/`.

## Repository layout

```
marl/
  comms/        lossy peer-SoC message channel and delivery schedules
  envs/         project CityLearn env (reward, outages, KPIs, frozen constants), env contract, toy envs
  baselines.py  R0 and R1 reference controllers
  mappo/        glue around on-policy (messages in actor inputs, map arms to on-policy...)
scripts/        command-line entry points (training, speed check, baselines, alignment check, freeze, export check)
tests/          unit and integration tests
third_party/    vendored on-policy MAPPO code (MIT, pinned commit; see VENDORED.md)
WorkPlan.md     work plan per team member
requirements.txt, test_marl_env.py   dependencies and installation check
```

## Setup

Use Python 3.10, 3.11 or 3.12.

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install PyTorch first, so CityLearn doesn't pull in a different build:

   ```bash
   # NVIDIA GPU (CUDA 12.8)
   pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

   # macOS or CPU only
   pip install torch==2.10.0 torchvision==0.25.0
   ```

3. Install the rest (CityLearn, numpy, scipy, matplotlib, pytest):

   ```bash
   pip install -r requirements.txt
   ```

4. Check the installation:

   ```bash
   python test_marl_env.py
   ```

   It prints the PyTorch version and whether a GPU is available, then runs a
   24-step multi-agent CityLearn episode with random actions and reports
   each agent's observation size, action space and rewards. It uses the
   `citylearn_challenge_2020_climate_zone_1` dataset, not the project's 2023
   Phase 3.3 dataset. CityLearn downloads a dataset the first time it is used,
   so the first run needs internet access.

Run every command below from the repository root.

## Running

### Train one run: `scripts/train.py`

Trains one (arm, seed) pair. It writes `runs/<arm>_s<seed>.jsonl` (config +
one line per update) and `runs/<arm>_s<seed>.pt` (checkpoint).

```bash
python -m scripts.train --arm M1-RND --seed 0 --steps 200000 --env citylearn
python -m scripts.train --arm M0 --seed 0 --steps 20000 --env toy   # fast check on the toy env
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--arm` | `M0` | `M0`, `M1`, `M1-RND`, `I0`, `M-FULL`, `M1-RND-NM` |
| `--seed` | `0` | Run seed (the same seed gives the same outages in every arm) |
| `--steps` | `20000` | Total environment steps |
| `--env` | `toy` | `toy`, `citylearn` (project env: district reward, outage seeding, KPIs) or `citylearn-temp` (old adapter) |
| `--episode-steps` | `168` | Training episode length in hours |
| `--rollout-steps` | `1000` | Steps collected per PPO update |
| `--out` | `runs` | Output directory |

### Speed check: `scripts/speed_check.py`

Measures raw environment steps/s and MAPPO steps/s, and estimates hours per
training run (the proposal's target is 2 hours or less).

```bash
python -m scripts.speed_check                        # CityLearn
python -m scripts.speed_check --env toy --steps 1000
```

Options: `--env` (`citylearn`), `--steps` (`720`, steps to time),
`--train-steps` (`200000`, planned steps per run for the estimate).

## Testing

Run all tests:

```bash
python -m pytest tests -v
python -m unittest discover -s tests -t . -v   # same tests, without pytest
```

Tests that need torch or CityLearn skip themselves when those packages are
missing.

| File | What it checks | Needs | Run alone |
|------|----------------|-------|-----------|
| `tests/test_channel.py` | Message format, delivery rate and independence, seeding, forced-mask and hold-last modes, M1-RND schedule | numpy | `python -m pytest tests/test_channel.py -v` |
| `tests/test_message_env.py` | Actor/critic inputs (critic never sees drops), raw messages, M-FULL inputs, paired seeds across arms, env contract, evaluation, every arm on the project env | numpy | `python -m pytest tests/test_message_env.py -v` |
| `tests/test_onpolicy_args.py` | Arms map to identical on-policy settings; evaluation conditions per arm; vendored on-policy files match upstream | numpy | `python -m pytest tests/test_onpolicy_args.py -v` |
| `tests/test_citylearn_adapter.py` | Temporary CityLearn adapter against a fake CityLearn: split windows, truncation at episode end, battery energy, seed range | numpy | `python -m pytest tests/test_citylearn_adapter.py -v` |
| `tests/test_district_env.py` | Project env against a fake CityLearn: reward formula (exports unpenalized) and R0 normalization, outage calendar seeding, KPIs, R0/R1, per-building load in `info`, peer-SoC feature data | numpy | `python -m pytest tests/test_district_env.py -v` |
| `tests/test_onpolicy_runner.py` | Training with on-policy MAPPO for every arm, reproducibility, checkpoints, learning-from-messages check (about a minute) | torch | `python -m pytest tests/test_onpolicy_runner.py -v -s` |
| `tests/test_citylearn_smoke.py` | CityLearn adapter on the real 2023 Phase 3.3 dataset (split windows, truncation); one training update per arm and steps/s | citylearn, torch | `python -m pytest tests/test_citylearn_smoke.py -v -s` |

Without pytest, use `python -m unittest tests.test_channel -v` (and so on).
`-s` shows the printed learning and speed results.



## Part C — Training and Experimental Evaluation

Part C covers the training, validation, held-out testing, and statistical analysis of three MAPPO controllers: M0, M1, and M1-RND.

The experiments use five independent training seeds per controller, with 200,000 environment steps per seed. Performance is evaluated under different peer SoC message delivery probabilities.

### 1. Environment Setup

Follow the installation instructions in the main **Setup** section before running Part C.

The experiments were executed using Python 3.10, PyTorch 2.10.0, and CPU training.

Run all commands from the repository root.

**Activate the virtual environment (Windows):**

```cmd
venv_marl\Scripts\activate
```

If the environment was created using the main Setup instructions, activate `.venv` instead.

**Verify dependencies:**

```cmd
python -m pip check
```

**Run Part C unit tests:**

```cmd
python -m unittest discover -s tests -p "test_part_c.py" -v
```

### 2. Training

Three controllers are trained independently:

- **M0:** MAPPO without peer SoC communication.
- **M1:** MAPPO trained with perfect peer SoC communication.
- **M1-RND:** MAPPO trained with randomized message delivery.

**Training configuration:**

| Parameter | Value |
|---|---|
| Training seeds | 0, 1, 2, 3, 4 |
| Training steps per seed | 200,000 |
| Rollout length | 1,000 steps |
| Total trained models | 15 |
| Total training steps | 3,000,000 |
| Training device | CPU |

**Train M0:**

```cmd
python -m scripts.part_c_train --arms M0 --seeds 0 1 2 3 4 --steps 200000 --rollout-steps 1000 --out runs
```

**Train M1:**

```cmd
python -m scripts.part_c_train --arms M1 --seeds 0 1 2 3 4 --steps 200000 --rollout-steps 1000 --out runs
```

**Train M1-RND:**

```cmd
python -m scripts.part_c_train --arms M1-RND --seeds 0 1 2 3 4 --steps 200000 --rollout-steps 1000 --out runs
```

Each training run generates:

- `runs/<arm>_s<seed>.pt` — trained model checkpoint.
- `runs/<arm>_s<seed>.jsonl` — training configuration and progress logs.

### 3. Validation Evaluation

The trained controllers are evaluated on the validation split using different message delivery probabilities.

**Evaluation configuration:**

| Parameter | Value |
|---|---|
| Training seeds | 5 |
| Evaluation replicates | 10 per condition per seed |
| Delivery probabilities | 1.0, 0.9, 0.5, 0.25, 0.0 |
| Evaluation transitions per run | 359 |
| M0 conditions | 1 |
| M1 conditions | 9 |
| M1-RND conditions | 9 |
| Total validation records | 950 |

M1 includes the HOLD extension, which reuses the last delivered peer SoC value when a message is dropped. M1-RND includes forced-mask evaluations.

**Evaluate M0:**

```cmd
python -m scripts.part_c_experiments --split validation --arms M0 --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results\part_c\validation_m0
```

**Evaluate M1:**

```cmd
python -m scripts.part_c_experiments --split validation --arms M1 --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results\part_c\validation_m1
```

**Evaluate M1-RND:**

```cmd
python -m scripts.part_c_experiments --split validation --arms M1-RND --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results\part_c\validation_m1rnd
```

Each evaluation generates a `replicates.csv`, `manifest.json`, and trajectory files in the specified output directory.

**Combine validation results:**

```cmd
python -c "from pathlib import Path; import pandas as pd; b=Path('results/part_c'); paths=[b/'validation_m0/replicates.csv',b/'validation_m1/replicates.csv',b/'validation_m1rnd/replicates.csv']; assert all(p.is_file() for p in paths), 'Missing validation CSV'; d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True); assert (d.split=='validation').all(), 'Non-validation records'; assert not d.duplicated(['split','arm','condition','seed','replicate']).any(), 'Duplicate evaluations'; assert len(d)==950, 'Unexpected record count'; out=b/'validation_combined'; out.mkdir(parents=True,exist_ok=True); d.to_csv(out/'replicates.csv',index=False); print(d.groupby('arm').size()); print('Total records:',len(d))"
```

**Generate the validation statistical report:**

```cmd
python -m scripts.part_c_report --split validation --input results\part_c\validation_combined\replicates.csv --out results\part_c\validation_combined\analysis --delta 5
```

The `--delta 5` argument specifies a minimum meaningful improvement threshold of 5% for the primary KPI.

### 4. Held-Out Test Evaluation

After validation, the existing trained checkpoints are evaluated on the held-out test split without further model tuning.

The same five training seeds and ten evaluation replicates per condition are used.

**Evaluate M0:**

```cmd
python -m scripts.part_c_experiments --split test --final-test --arms M0 --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results\part_c\test_m0
```

**Evaluate M1:**

```cmd
python -m scripts.part_c_experiments --split test --final-test --arms M1 --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results\part_c\test_m1
```

**Evaluate M1-RND:**

```cmd
python -m scripts.part_c_experiments --split test --final-test --arms M1-RND --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results\part_c\test_m1rnd
```

**Combine test results:**

```cmd
python -c "from pathlib import Path; import pandas as pd; b=Path('results/part_c'); paths=[b/'test_m0/replicates.csv',b/'test_m1/replicates.csv',b/'test_m1rnd/replicates.csv']; assert all(p.is_file() for p in paths), 'Missing test CSV'; d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True); assert (d.split=='test').all(), 'Non-test records'; assert not d.duplicated(['split','arm','condition','seed','replicate']).any(), 'Duplicate evaluations'; assert len(d)==950, 'Unexpected record count'; out=b/'test_combined'; out.mkdir(parents=True,exist_ok=True); d.to_csv(out/'replicates.csv',index=False); print(d.groupby('arm').size()); print('Total records:',len(d))"
```

**Generate the final statistical report:**

```cmd
python -m scripts.part_c_report --split test --input results\part_c\test_combined\replicates.csv --out results\part_c\test_combined\analysis --delta 5
```

The statistical analysis uses seed-paired 95% confidence intervals across the five independent training seeds. Evaluation replicates are averaged within each seed to avoid pseudo-replication.

**Generated analysis files:**

```text
results/part_c/
├── validation_combined/
│   ├── replicates.csv
│   └── analysis/
│       ├── per_seed.csv
│       ├── paired_vs_M0.csv
│       ├── robustness.csv
│       ├── robustness_arm_difference.csv
│       ├── hypotheses.json
│       ├── delivery_sweep.png
│       ├── robustness.png
│       ├── synchrony.png
│       └── message_extensions.png
└── test_combined/
    ├── replicates.csv
    └── analysis/
        ├── per_seed.csv
        ├── paired_vs_M0.csv
        ├── robustness.csv
        ├── robustness_arm_difference.csv
        ├── hypotheses.json
        ├── delivery_sweep.png
        ├── robustness.png
        ├── synchrony.png
        └── message_extensions.png
```

The generated reports cover communication benefit (H1), robustness under message loss (H2), and battery synchrony (H3).

The held-out test results are used for final reporting rather than hyperparameter tuning.
