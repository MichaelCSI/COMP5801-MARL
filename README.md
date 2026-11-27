# When Messages Fail: Robust Peer SoC Communication in CityLearn

COMP 5801 / CSI 5388 MARL project. Six buildings in CityLearn (Challenge 2023 Phase 3.3) each control a battery using an independent actor. Before acting, communication-enabled agents send their battery state of charge (SoC) to their peers over a lossy channel.

We investigate whether peer SoC communication improves district-level battery coordination, how performance changes under message loss, and whether training under randomized message delivery produces more robust policies.

Supplementary experiments investigate whether a centralized critic improves performance and whether providing actors with richer peer information improves coordination.

| Arm | Description | Status |
|-----|-------------|--------|
| R0 | No-battery reference controller | Evaluated |
| R1 | Rule-based battery controller | Evaluated |
| M0 | MAPPO, no messages (primary learned baseline) | Trained and evaluated |
| M1 | MAPPO with peer SoC and availability mask, 100% delivery during training | Trained and evaluated |
| M1-RND | Same inputs as M1, delivery probability drawn per episode from {0, 0.5, 0.75, 1} | Trained and evaluated |
| I0 | Independent PPO (IPPO), no messages, local critic | Trained and evaluated |
| M-FULL | MAPPO with access to every peer's true SoC and net load, without message loss | Trained and evaluated |
| M1-HOLD | Evaluation of M1 only: a dropped message reuses the last SoC received | Evaluated |
| M1-RND forced mask | Evaluation of M1-RND with availability masks forced to one while message loss remains active | Evaluated |
| M1-RND-NM | Optional: M1-RND without the availability mask | Not evaluated |

Message-enabled arms are evaluated at delivery probabilities {100%, 90%, 50%, 25%, 0%}. Arms without a lossy communication channel have one evaluation condition.

M-FULL serves as an informational upper-bound baseline rather than a guaranteed upper bound on learned policy performance.

The learner is based on the reference MAPPO implementation of Yu et al. (NeurIPS 2022), [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy), vendored unmodified in `third_party/onpolicy/`.

## Repository layout

All paths below are relative to the project's `code/` directory.

```text
marl/
  comms/        Lossy peer-SoC message channel and delivery schedules
  envs/         CityLearn environment, reward, outages, KPIs, frozen constants,
                environment contract and toy environments
  baselines.py  R0 and R1 reference controllers
  mappo/        MAPPO/IPPO integration, actor and critic inputs, arm configuration
  part_c/       Experimental metrics, statistical analysis and diagnostics

scripts/        Command-line entry points for training, evaluation, statistical
                reporting, alignment checks and visualization

tests/          Unit and integration tests

third_party/    Vendored on-policy MAPPO code
                (MIT, pinned commit; see VENDORED.md)

results/        Experimental results, statistical summaries and figures

runs/           Locally generated training checkpoints and JSONL logs

WorkPlan.md     Work plan per team member

requirements.txt
test_marl_env.py
```

## Setup

Use Python 3.10, 3.11 or 3.12.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```cmd
.venv\Scripts\activate
```

### 2. Install PyTorch first

Install PyTorch before the remaining dependencies so CityLearn does not pull in a different build.

NVIDIA GPU (CUDA 12.8):

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

macOS or CPU only:

```bash
pip install torch==2.10.0 torchvision==0.25.0
```

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

Part C additionally requires pandas, SciPy, matplotlib and scikit-learn for statistical analysis, visualization and peer-SoC predictability analysis.

### 4. Check the installation

```bash
python -m pip check
python test_marl_env.py
```

The installation check prints the PyTorch version and whether a GPU is available, then runs a 24-step multi-agent CityLearn episode with random actions and reports each agent's observation size, action space and rewards.

It uses the `citylearn_challenge_2020_climate_zone_1` dataset, not the project's 2023 Phase 3.3 dataset.

CityLearn downloads a dataset the first time it is used, so the first run requires internet access.

Run all subsequent commands from the project's `code/` directory, which contains the `scripts/`, `marl/` and `tests/` directories.

## Running

### Train one run: `scripts/train.py`

Trains one (arm, seed) pair. It writes `runs/<arm>_s<seed>.jsonl` (configuration and one record per update) and `runs/<arm>_s<seed>.pt` (checkpoint).

```bash
python -m scripts.train --arm M1-RND --seed 0 --steps 200000 --env citylearn
python -m scripts.train --arm M0 --seed 0 --steps 20000 --env toy
```

The toy environment provides a fast functional check without executing the full CityLearn experiment.

| Option | Default | Meaning |
|--------|---------|---------|
| `--arm` | `M0` | `M0`, `M1`, `M1-RND`, `I0`, `M-FULL`, `M1-RND-NM` |
| `--seed` | `0` | Run seed; corresponding environment reset seeds are paired across arms |
| `--steps` | `20000` | Total environment steps |
| `--env` | `toy` | `toy`, `citylearn` (project environment) or `citylearn-temp` (old adapter) |
| `--episode-steps` | `168` | Training episode length in hours |
| `--rollout-steps` | `1000` | Steps collected per PPO update |
| `--out` | `runs` | Output directory |

### Speed check: `scripts/speed_check.py`

Measures raw environment throughput and MAPPO rollout throughput, and estimates computational requirements relative to the proposal's target of two hours or less per training run.

```bash
python -m scripts.speed_check
python -m scripts.speed_check --env toy --steps 1000
```

Options:

| Option | Default | Meaning |
|--------|---------|---------|
| `--env` | `citylearn` | Environment used for the speed check |
| `--steps` | `720` | Number of steps to time |
| `--train-steps` | `200000` | Planned training steps per run |

Rollout throughput estimates should not be interpreted as complete training wall-clock durations because PPO optimization and other overhead may be excluded.

## Testing

Run all tests:

```bash
python -m pytest tests -v
```

Alternatively, without pytest:

```bash
python -m unittest discover -s tests -t . -v
```

Tests that require PyTorch or CityLearn skip themselves when those packages are missing.

| File | What it checks | Dependencies | Run individually |
|------|----------------|--------------|------------------|
| `tests/test_channel.py` | Message format, delivery rate and independence, seeding, forced-mask and hold-last modes, M1-RND schedule | NumPy | `python -m pytest tests/test_channel.py -v` |
| `tests/test_message_env.py` | Actor/critic inputs, raw messages, M-FULL peer inputs, paired seeds, normalization, environment contract and evaluation | NumPy | `python -m pytest tests/test_message_env.py -v` |
| `tests/test_onpolicy_args.py` | Arm configurations, IPPO local critic, on-policy settings, evaluation conditions and vendored file integrity | NumPy | `python -m pytest tests/test_onpolicy_args.py -v` |
| `tests/test_citylearn_adapter.py` | CityLearn adapter against a fake environment, split windows, truncation, battery energy and seed range | NumPy | `python -m pytest tests/test_citylearn_adapter.py -v` |
| `tests/test_district_env.py` | Reward formula, R0 normalization, outage seeding, KPIs, R0/R1 and peer-SoC feature data | NumPy | `python -m pytest tests/test_district_env.py -v` |
| `tests/test_onpolicy_runner.py` | Training for every arm, reproducibility, checkpoint handling, evaluation and learning-from-messages checks | PyTorch | `python -m pytest tests/test_onpolicy_runner.py -v -s` |
| `tests/test_citylearn_smoke.py` | Real CityLearn 2023 Phase 3.3 environment, split windows, truncation, M-FULL net-load inputs, training updates and project reward | CityLearn, PyTorch | `python -m pytest tests/test_citylearn_smoke.py -v -s` |
| `tests/test_part_c.py` | Part C metrics, statistical analysis and experimental utilities | Part C dependencies | `python -m pytest tests/test_part_c.py -v` |

The `-s` argument displays printed learning and speed diagnostics.

The learning-signal test in `test_onpolicy_runner.py` may take longer than the remaining unit tests because it performs policy training.

# Part C — Training and Experimental Evaluation

Part C covers experimental execution, validation, test evaluation, statistical analysis, communication robustness, battery-energy synchrony, peer-SoC predictability and training diagnostics.

The original experiments investigate three MAPPO controllers: M0, M1 and M1-RND.

Two supplementary controllers, I0 and M-FULL, were subsequently trained and evaluated in response to the proposal feedback.

All five learned controllers use five independent training seeds and 200,000 environment steps per seed.

## 1. Environment setup

Follow the installation instructions in the main Setup section before running Part C.

The reported experiments were executed using:

| Component | Configuration |
|-----------|---------------|
| Operating system | Windows |
| Python | 3.10.10 |
| PyTorch | 2.10.0+cu128 |
| Processor | Intel Core Ultra 7 255HX |
| CPU cores / logical processors | 20 / 20 |
| Physical memory | Approximately 32 GB |
| Training device | CPU |
| Dataset | CityLearn Challenge 2023 Phase 3.3 |
| Number of buildings | 6 |
| Controllable assets | Batteries |

Although CUDA was available, the reported training runs used the CPU.

### Activate the virtual environment (Windows)

```cmd
venv_marl\Scripts\activate
```

If the environment was created using the main Setup instructions, activate `.venv` instead.

### Verify dependencies

```cmd
python -m pip check
```

### Run Part C unit tests

```cmd
python -m unittest discover -s tests -p "test_part_c.py" -v
```

### Verify controller and environment implementations

```cmd
python -m pytest tests/test_message_env.py tests/test_onpolicy_args.py tests/test_onpolicy_runner.py -v
python -m pytest tests/test_citylearn_smoke.py -v -s
```

The relevant IPPO, M-FULL, actor/critic input, checkpoint and evaluation tests passed. All 13 real CityLearn integration tests passed.

In the recorded combined 61-test execution, 60 tests passed before the final learning-signal test. Completion of that final test was not confirmed in the recorded output.

## 2. Experimental configuration

### Dataset partitioning

The 2,208-hour dataset is partitioned chronologically into training, validation and test blocks.

| Split | Days | Hour indices |
|-------|------|--------------|
| Training | 1–62 | 0–1487 |
| Validation | 63–77 | 1488–1847 |
| Test | 78–92 | 1848–2207 |

Training uses 168-hour episode windows within the training block.

Validation and test evaluations each contain 359 measured environment transitions.

The validation environment uses outage seed 7, while the test environment uses outage seed 0.

### Training configuration

| Parameter | Value |
|-----------|-------|
| Training seeds | 0, 1, 2, 3, 4 |
| Training steps per seed | 200,000 |
| Rollout length | 1,000 steps |
| Training episode length | 168 hours |
| PPO update iterations per run | 200 |
| Actor learning rate | 0.0005 |
| Critic learning rate | 0.0005 |
| PPO epochs | 15 |
| Mini-batches | 1 |
| PPO clipping parameter | 0.2 |
| Discount factor | 0.99 |
| GAE parameter | 0.95 |
| Entropy coefficient | 0.01 |
| Hidden layer size | 64 |
| Observation clipping | 10 |
| Reward weights | alpha = 1, beta = 1 |
| Training device | CPU |

### Experimental scale

| Component | Original experiments | Supplementary experiments | Total |
|-----------|---------------------:|--------------------------:|------:|
| Learned controllers | 3 | 2 | 5 |
| Training seeds per controller | 5 | 5 | 5 |
| Trained models | 15 | 10 | 25 |
| Environment steps | 3,000,000 | 2,000,000 | 5,000,000 |
| PPO update iterations | 3,000 | 2,000 | 5,000 |
| Validation records | 950 | 100 | 1,050 |
| Test records | 950 | 100 | 1,050 |
| Total evaluation records | 1,900 | 200 | 2,100 |

These totals exclude smoke runs, standalone baseline evaluations, reward-alignment checks and the peer-SoC predictability experiment.

## 3. Original controller training

The three original controllers are trained separately:

- **M0:** MAPPO without peer SoC communication.
- **M1:** MAPPO trained with perfect peer SoC communication.
- **M1-RND:** MAPPO trained with randomized message delivery probabilities.

### Train M0

```cmd
python -m scripts.part_c_train --arms M0 --seeds 0 1 2 3 4 --steps 200000 --rollout-steps 1000 --episode-steps 168 --out runs
```

### Train M1

```cmd
python -m scripts.part_c_train --arms M1 --seeds 0 1 2 3 4 --steps 200000 --rollout-steps 1000 --episode-steps 168 --out runs
```

### Train M1-RND

```cmd
python -m scripts.part_c_train --arms M1-RND --seeds 0 1 2 3 4 --steps 200000 --rollout-steps 1000 --episode-steps 168 --out runs
```

Each training run generates:

- `runs/<arm>_s<seed>.pt` — trained model checkpoint.
- `runs/<arm>_s<seed>.jsonl` — training configuration and progress logs.

The training wrapper delegates to `scripts.train.run()` and skips existing checkpoints.

For a fresh reproduction, use an empty output directory or verify that existing checkpoints match the intended configuration.

The training wrapper does not automatically enforce the proposal's G1 or D1 experimental decision points.

## 4. Original validation evaluation

The original trained controllers are evaluated on the validation block using different message delivery probabilities.

### Evaluation configuration

| Parameter | Value |
|-----------|-------|
| Training seeds | 5 |
| Evaluation replicates | 10 per condition per seed |
| Delivery probabilities | 1.0, 0.9, 0.5, 0.25, 0.0 |
| Evaluation transitions per run | 359 |
| Validation environment seed | 7 |
| M0 conditions | 1 |
| M1 conditions | 9 |
| M1-RND conditions | 9 |
| Total original validation records | 950 |

M1 includes the HOLD extension, which reuses the last delivered peer SoC value when a message is dropped.

M1-RND includes forced-mask evaluations, in which the availability mask is forced to one while the underlying message-loss process remains active.

### Evaluate M0

```cmd
python -m scripts.part_c_experiments --split validation --arms M0 --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results/part_c/validation_m0
```

### Evaluate M1

```cmd
python -m scripts.part_c_experiments --split validation --arms M1 --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results/part_c/validation_m1
```

### Evaluate M1-RND

```cmd
python -m scripts.part_c_experiments --split validation --arms M1-RND --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results/part_c/validation_m1rnd
```

Each evaluation generates a `replicates.csv`, `manifest.json` and trajectory archives in the specified output directory.

### Combine validation results

```cmd
python -c "from pathlib import Path; import pandas as pd; b=Path('results/part_c'); paths=[b/'validation_m0/replicates.csv',b/'validation_m1/replicates.csv',b/'validation_m1rnd/replicates.csv']; assert all(p.is_file() for p in paths), 'Missing validation CSV'; d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True); assert (d.split=='validation').all(), 'Non-validation records'; assert not d.duplicated(['split','arm','condition','seed','replicate']).any(), 'Duplicate evaluations'; assert len(d)==950, 'Unexpected record count'; out=b/'validation_combined'; out.mkdir(parents=True,exist_ok=True); d.to_csv(out/'replicates.csv',index=False); print(d.groupby('arm').size()); print('Total records:',len(d))"
```

### Generate the validation statistical report

```cmd
python -m scripts.part_c_report --split validation --input results/part_c/validation_combined/replicates.csv --out results/part_c/validation_combined/analysis --delta 5
```

The `--delta 5` argument specifies the original minimum meaningful relative improvement threshold of 5% for the primary KPI.

## 5. Original held-out test evaluation

The original trained checkpoints are evaluated on the test block without further model tuning.

The same five training seeds and ten evaluation replicates per condition are used. The test environment uses outage seed 0.

### Evaluate M0

```cmd
python -m scripts.part_c_experiments --split test --final-test --arms M0 --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results/part_c/test_m0
```

### Evaluate M1

```cmd
python -m scripts.part_c_experiments --split test --final-test --arms M1 --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results/part_c/test_m1
```

### Evaluate M1-RND

```cmd
python -m scripts.part_c_experiments --split test --final-test --arms M1-RND --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs --out results/part_c/test_m1rnd
```

### Combine test results

```cmd
python -c "from pathlib import Path; import pandas as pd; b=Path('results/part_c'); paths=[b/'test_m0/replicates.csv',b/'test_m1/replicates.csv',b/'test_m1rnd/replicates.csv']; assert all(p.is_file() for p in paths), 'Missing test CSV'; d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True); assert (d.split=='test').all(), 'Non-test records'; assert not d.duplicated(['split','arm','condition','seed','replicate']).any(), 'Duplicate evaluations'; assert len(d)==950, 'Unexpected record count'; out=b/'test_combined'; out.mkdir(parents=True,exist_ok=True); d.to_csv(out/'replicates.csv',index=False); print(d.groupby('arm').size()); print('Total records:',len(d))"
```

### Generate the original test statistical report

```cmd
python -m scripts.part_c_report --split test --input results/part_c/test_combined/replicates.csv --out results/part_c/test_combined/analysis --delta 5
```

The generated reports cover:

- H1: Communication benefit under perfect delivery.
- H2: Robustness under message loss.
- H3: Battery-energy synchrony.

Evaluation replicates are averaged within each training seed before calculating seed-paired 95% confidence intervals.

### Generated original analysis files

Both `validation_combined/` and `test_combined/` contain the following structure:

```text
replicates.csv
analysis/
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

The original evaluation results and statistical reports are preserved separately from the supplementary experiments.

## 6. Supplementary experiments: IPPO and full peer observability

Two additional controllers were trained and evaluated in response to the professor's proposal feedback.

### I0 — Independent PPO without messages

I0 uses the same local actor observations as M0, but replaces the centralized critic with a local critic.

This comparison investigates whether centralized critic information provides a performance advantage in the evaluated CityLearn environment.

### M-FULL — Full peer SoC and net-load observability

M-FULL provides each actor with the true SoC and normalized net load of every other building without a lossy communication channel.

This comparison investigates whether richer peer information improves district-level battery coordination.

M-FULL provides an informational upper-bound comparison for the selected peer variables, not a guaranteed performance upper bound.

| Controller | Actor input dimensions | Critic input dimensions | Peer information |
|------------|----------------------:|------------------------:|------------------|
| M0 | 30 | 85 | None |
| M1 | 40 | 85 | Peer SoC and availability masks |
| M1-RND | 40 | 85 | Peer SoC and availability masks |
| I0 | 30 | 30 | None |
| M-FULL | 40 | 85 | True peer SoC and normalized net load |

All controllers use the shared district reward.

### 6.1. Supplementary smoke training

Before full training, both additional controllers were verified using a short functional training run.

```cmd
python -m scripts.part_c_train --arms I0 M-FULL --seeds 0 --steps 2000 --rollout-steps 1000 --out runs/smoke_extended
```

### 6.2. Supplementary smoke evaluation

```cmd
python -m scripts.part_c_experiments --split validation --arms I0 M-FULL --seeds 0 --replicates 1 --checkpoints runs/smoke_extended --out results/part_c/smoke_extended
```

Both controllers successfully completed smoke training, checkpoint loading, evaluation and KPI generation.

Smoke results are functional checks only and are excluded from the full experimental analysis.

### 6.3. Train I0

```cmd
python -m scripts.part_c_train --arms I0 --seeds 0 1 2 3 4 --steps 200000 --rollout-steps 1000 --episode-steps 168 --out runs/extended
```

### 6.4. Train M-FULL

```cmd
python -m scripts.part_c_train --arms M-FULL --seeds 0 1 2 3 4 --steps 200000 --rollout-steps 1000 --episode-steps 168 --out runs/extended
```

The supplementary controllers use the same training budget, seed set, rollout length and episode length as the original controllers.

The ten supplementary checkpoints and JSONL logs are stored in `runs/extended/`, preserving the original 15 training runs.

### 6.5. Supplementary validation evaluation

```cmd
python -m scripts.part_c_experiments --split validation --arms I0 M-FULL --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs/extended --out results/part_c/extended/validation
```

### 6.6. Supplementary test evaluation

```cmd
python -m scripts.part_c_experiments --split test --final-test --arms I0 M-FULL --seeds 0 1 2 3 4 --replicates 10 --checkpoints runs/extended --out results/part_c/extended/test
```

Each supplementary evaluation block contains:

| Controller | Training seeds | Conditions | Replicates | Records |
|------------|---------------:|-----------:|-----------:|--------:|
| I0 | 5 | 1 | 10 | 50 |
| M-FULL | 5 | 1 | 10 | 50 |
| Total | | | | 100 |

Neither I0 nor M-FULL uses a stochastic message-delivery channel. Therefore, repeated deterministic evaluations of the same checkpoint under the fixed environment seed produce identical results.

The five training seeds remain the independent statistical units.

### 6.7. Supplementary statistical reporting

Generate the supplementary validation report:

```cmd
python -m scripts.part_c_extended_report --split validation
```

Generate the supplementary test report:

```cmd
python -m scripts.part_c_extended_report --split test
```

The script requires the following input files:

```text
results/part_c/validation_combined/replicates.csv
results/part_c/test_combined/replicates.csv
results/part_c/extended/validation/replicates.csv
results/part_c/extended/test/replicates.csv
```

Each report compares five learned controllers. For M1 and M1-RND, the perfect-delivery condition (p = 1) is selected.

The supplementary paired comparisons are:

- I0 versus M0.
- M-FULL versus M0.
- M-FULL versus M1 at perfect delivery.

These comparisons use unadjusted, exploratory 95% paired Student-t confidence intervals across five training seeds.

### Generated supplementary analysis files

Both `extended/validation/analysis/` and `extended/test/analysis/` contain:

```text
per_seed.csv
summary_by_arm.csv
paired_comparisons.csv
01_controller_comparison.png
02_paired_comparisons.png
03_peak_cost_tradeoff.png
04_battery_correlation.png
```

## 7. Reward–KPI alignment verification

The reward penalizes normalized electricity import cost and squared positive district net load.

Negative district net load (electricity export) is not penalized by the squared peak term.

The reward weights are fixed at alpha = 1 and beta = 1.

A retrospective alignment check evaluates whether the reward ranking agrees with the primary KPI ranking for R0, R1 and the five trained M0 checkpoints on the validation block.

### Execute the alignment check

```cmd
python -m scripts.alignment_check --m0 runs/M0_s0.pt --m0 runs/M0_s1.pt --m0 runs/M0_s2.pt --m0 runs/M0_s3.pt --m0 runs/M0_s4.pt --alpha 1 --beta 1 > results/alignment_check.txt 2>&1
```

All five M0 checkpoints satisfy the expected ranking:

- Higher episode return: M0 > R1 > R0.
- Lower primary KPI: M0 < R1 < R0.

The result is saved in:

```text
results/alignment_check.txt
```

This verifies reward–KPI ranking consistency for the evaluated controllers. It does not establish global alignment between the training reward and primary KPI for every possible policy.

The alignment check was completed retrospectively and should not be interpreted as evidence that the proposed prospective G1 gate was executed before the original experiments.

## 8. Peer-SoC predictability analysis

An additional diagnostic investigates whether peer battery SoC can be predicted using information already available in the receiving agent's observations.

The purpose is to determine whether peer SoC may be partially redundant with existing local and shared observations.

The analysis uses trajectories generated by the five trained M0 policies on the validation block.

### Regression configuration

| Parameter | Value |
|-----------|-------|
| Controller | M0 |
| Training seeds | 5 |
| Receiver–peer pairs per seed | 30 |
| Total regressions | 150 |
| Regression model | Ridge regression |
| Ridge regularization parameter | 1.0 |
| Input features | 30 receiving-agent observation features |
| Observations per trajectory | 359 |
| Chronological fitting subset | 251 observations |
| Chronological evaluation subset | 108 observations |
| Feature standardization | Fitted on the fitting subset only |
| Evaluation metrics | R-squared and MAE |

Separate regression models are fitted for each directed receiver–peer pair.

The regression fitting and evaluation subsets are both taken from the project validation block. The project's held-out test block is not used.

### Execute the predictability experiment

Windows CMD:

```cmd
for %s in (0 1 2 3 4) do python -m scripts.part_c_predictability --checkpoint runs/M0_s%s.pt --out results/part_c/predictability/M0_s%s.csv --alpha 1.0
```

When executing from a Windows batch file, replace `%s` with `%%s`.

On Windows, PyTorch is imported before scikit-learn in the predictability script to avoid the observed DLL import-order conflict.

### Generate predictability plots

```cmd
python -m scripts.part_c_predictability_plots
```

### Generated predictability files

```text
results/part_c/predictability/
├── M0_s0.csv
├── M0_s1.csv
├── M0_s2.csv
├── M0_s3.csv
├── M0_s4.csv
├── predictability_summary_by_seed.csv
├── predictability_pair_mean_r2.csv
├── 01_peer_soc_r2_heatmap.png
└── 02_peer_soc_r2_distribution.png
```

### Predictability results

| Metric | Result |
|--------|-------:|
| Mean R-squared | 0.1603 |
| Median R-squared | 0.1691 |
| Mean absolute error | 0.2332 |
| Regressions with negative R-squared | 46 / 150 |
| Regressions with R-squared >= 0.5 | 32 / 150 |

The results indicate that peer SoC is only partially predictable from the existing receiving-agent observations using the evaluated linear regression model.

These findings do not strongly support the explanation that peer SoC is redundant with existing observations.

The diagnostic does not rule out nonlinear or temporal predictability and does not establish the causal reason for the absence of a communication benefit.

The 150 regression results are descriptive and are not treated as independent policy training runs.

## 9. Training learning curves and optimization diagnostics

### Original three-controller training analysis

```cmd
python -m scripts.part_c_learning_curves
```

The original script reads the 15 training logs for M0, M1 and M1-RND.

Its outputs are stored in:

```text
results/part_c/training_analysis/
```

### Extended five-controller training analysis

```cmd
python -m scripts.part_c_extended_learning_curves
```

The extended script reads:

- 15 original training logs from `runs/`.
- 10 supplementary training logs from `runs/extended/`.

It verifies that each run contains 200 PPO update records and reaches 200,000 environment steps.

The expected total is 5,000 training update records.

### Training diagnostics methodology

Training-return curves use a trailing ten-update rolling mean calculated separately for each training seed.

The controller-level mean and pointwise 95% Student-t confidence intervals are then calculated across the five training seeds.

Individual seed learning curves are also generated to examine between-seed variability.

The optimization diagnostics include:

- Value loss.
- Policy loss.
- Policy entropy.
- Actor gradient norm.
- Critic gradient norm.
- PPO probability ratio.

These metrics provide descriptive training and optimization diagnostics. They do not independently establish policy convergence or optimality.

### Generated extended training files

```text
results/part_c/extended/training_analysis/
├── 01_training_learning_curves.png
├── 02_individual_seed_learning_curves.png
├── 03_optimization_diagnostics.png
├── training_configurations.csv
├── training_summary_per_seed.csv
└── training_summary_by_arm.csv
```

### Training-return summary

Initial and final returns are calculated over the first and last ten PPO updates, respectively.

| Controller | Initial mean return | Final mean return | Final return SD |
|------------|--------------------:|------------------:|----------------:|
| M0 | -235.4210 | -345.0108 | 2.1307 |
| M1 | -235.0272 | -343.4972 | 1.0628 |
| M1-RND | -234.9910 | -346.0745 | 2.1685 |
| I0 | -235.2659 | -346.5972 | 3.3926 |
| M-FULL | -235.0457 | -350.2405 | 4.2537 |

The five controllers exhibit similar, non-monotonic training-return trajectories.

Variation in training episode conditions may influence the observed returns, but this explanation has not been independently established from the realized episode-window sequences.

The learning curves should therefore be interpreted as training diagnostics rather than evidence of monotonic improvement or convergence.

## 10. Computational cost

All 25 full training runs were completed using CPU training.

Training-wrapper wall-clock durations are obtained from the `done in ...` messages in the training console output.

Rollout collection durations are estimated separately from the logged throughput for each PPO update.

| Controller | Mean wall-clock duration per run (min) | Total wall-clock duration (h) | Mean rollout collection duration per run (min) |
|------------|--------------------------------------:|-----------------------------:|----------------------------------------------:|
| M0 | 25.61 | 2.135 | 23.00 |
| M1 | 26.43 | 2.202 | 23.65 |
| M1-RND | 26.15 | 2.179 | 23.32 |
| I0 | 26.08 | 2.173 | 23.35 |
| M-FULL | 24.59 | 2.049 | 22.12 |
| Total | | 10.738 | |

The total estimated rollout collection time across all 25 full training runs is approximately 9.619 hours.

Rollout collection time excludes PPO optimization and other training overhead. It should not be confused with the complete wall-clock duration of a training run.

The total wall-clock and rollout collection durations are separate measurements and should not be added together.

All full training runs completed within the proposal's two-hour computational target per run.

## 11. Statistical methodology

### Primary KPI

The primary KPI is the average daily district peak relative to the no-battery baseline R0.

Lower values indicate better performance.

```text
Primary KPI = controller average daily district peak / R0 average daily district peak
```

Peak reduction relative to R0 is calculated as:

```text
Peak reduction (%) = 100 * (1 - Primary KPI)
```

### Secondary KPI

Electricity import cost relative to R0:

```text
Relative cost = controller electricity import cost / R0 electricity import cost
```

Lower values indicate better performance.

### Seed-paired relative improvement

For a candidate controller and a reference controller:

```text
Relative improvement (%) =
    100 * (reference KPI - candidate KPI) / reference KPI
```

Positive values indicate an improvement over the reference controller.

Negative values indicate worse performance.

Relative improvement is calculated separately for each corresponding pair of training seeds before computing the overall mean and confidence interval.

### Battery-energy synchrony

Battery-energy synchrony is measured using pairwise Pearson correlations between realized battery-energy trajectories.

For six buildings, there are 15 unique building pairs.

Two complementary statistics are reported:

- Mean signed pairwise correlation.
- Mean absolute pairwise correlation.

The signed correlation measures the direction of linear coupling, while the absolute correlation measures its magnitude.

Low correlation does not independently establish effective coordination or the absence of all forms of synchronized behaviour.

### Confidence intervals

The statistical analysis uses seed-level Student-t confidence intervals with five training seeds.

Evaluation replicates are averaged within each seed and condition before statistical comparisons.

The evaluation replicates are not treated as independent training runs, avoiding pseudo-replication.

With five paired training seeds, paired confidence intervals use four degrees of freedom.

The supplementary comparisons are exploratory and their confidence intervals are unadjusted for multiple comparisons.

## 12. Experimental results

### Five-controller comparison

M1 and M1-RND are evaluated at perfect message delivery (p = 1) in the following comparison.

| Controller | Validation primary KPI | Test primary KPI | Test peak reduction vs R0 |
|------------|-----------------------:|-----------------:|--------------------------:|
| M0 | 0.93496 | 0.92959 | 7.04% |
| M1 | 0.94595 | 0.94942 | 5.06% |
| M1-RND | 0.96844 | 0.98550 | 1.45% |
| I0 | 0.95323 | 0.94539 | 5.46% |
| M-FULL | 0.98093 | 0.98626 | 1.37% |

M0 achieves the lowest mean primary KPI among the five learned controllers at the displayed conditions on both validation and test blocks.

### Test secondary metrics

| Controller | Relative electricity cost | Mean signed correlation | Mean absolute correlation |
|------------|--------------------------:|------------------------:|--------------------------:|
| M0 | 0.98611 | 0.07380 | 0.10099 |
| M1 | 0.97813 | 0.09739 | 0.10786 |
| M1-RND | 0.98252 | 0.07232 | 0.09889 |
| I0 | 0.98486 | 0.09017 | 0.13240 |
| M-FULL | 0.98903 | 0.02131 | 0.09354 |

M1 achieves the lowest mean electricity cost among the five displayed controllers, whereas M0 achieves the strongest mean peak reduction.

M-FULL exhibits the lowest mean signed and absolute battery-energy correlations while achieving the weakest mean peak reduction among these conditions.

This indicates that lower battery-energy coupling does not necessarily correspond to better district peak reduction.

### Supplementary paired comparisons

Positive relative improvement favours the supplementary controller.

| Split | Comparison | Mean improvement | 95% confidence interval |
|-------|------------|-----------------:|-------------------------|
| Validation | I0 vs M0 | -2.01% | [-6.61%, +2.59%] |
| Validation | M-FULL vs M0 | -4.93% | [-9.16%, -0.70%] |
| Validation | M-FULL vs M1 | -3.71% | [-7.75%, +0.33%] |
| Test | I0 vs M0 | -1.76% | [-6.94%, +3.42%] |
| Test | M-FULL vs M0 | -6.16% | [-12.49%, +0.18%] |
| Test | M-FULL vs M1 | -3.94% | [-9.64%, +1.75%] |

### Interpretation

The original communication experiments do not demonstrate the proposed 5% primary-KPI improvement from perfect peer-SoC communication over M0.

The supplementary experiments provide additional context:

- I0 does not demonstrate a performance advantage over M0.
- The point estimates favour MAPPO over IPPO without messages, but the paired differences remain statistically inconclusive.
- M-FULL does not demonstrate improved peak reduction despite providing richer peer information.
- M-FULL performs worse than M0 on the validation block, with an unadjusted paired confidence interval excluding zero.
- All three supplementary test confidence intervals include zero.
- Lower battery-energy correlations do not necessarily correspond to stronger district peak reduction.
- Peer SoC is only partially predictable from existing observations under the evaluated ridge regression model.

Because I0 does not clearly outperform M0, the suggested conditional extension to IPPO with peer communication was not pursued.

The findings are restricted to the evaluated dataset, environment, reward, observation configuration, learning architectures and training budget.

They do not establish that communication is universally ineffective or that centralized critics are generally superior.

## 13. Experimental protocol deviations and limitations

### Decision point D1

The original proposal specified that a validation null bound for H1 would trigger the negative-result investigation and omit M1-RND training.

The validation results satisfied the null-bound criterion. However, M1-RND was trained and evaluated rather than omitted from the experimental study.

This constitutes a departure from the proposed conditional experimental protocol.

The M1-RND results are retained and reported transparently. The training wrapper does not automatically enforce D1.

The precise timing and rationale for the decision should be documented in the accompanying report based on the team's experimental records.

### Gate G1

The reward–KPI alignment check was performed retrospectively using the trained M0 checkpoints.

Although all five checkpoints passed the evaluated ranking check, this does not establish that the proposed prospective G1 gate was followed before training.

### Supplementary experiments

The I0 and M-FULL experiments were conducted after the original test evaluation in response to the professor's proposal feedback.

They retain the same training configuration and evaluation protocol as the original controllers.

However, because the original test block had already been examined, the supplementary results on that block are treated as exploratory rather than as a new, untouched confirmatory evaluation.

### Additional limitations

- Only five independent training seeds are used per learned controller.
- The environment contains six buildings from a single CityLearn dataset configuration.
- The fixed test outage seed produces only one outage hour in the test block.
- The supplementary confidence intervals are not adjusted for multiple comparisons.
- The training learning curves do not establish convergence or optimality.
- The peer-SoC predictability analysis examines linear predictability only.
- The full-observability baseline provides additional information but does not guarantee that MAPPO learns to exploit it.
- Low battery-energy correlation is not sufficient evidence of effective coordination.

## 14. Reproducibility and output locations

### Training outputs

```text
runs/
├── M0_s0.pt
├── M0_s0.jsonl
├── ...
├── M1_s0.pt
├── M1_s0.jsonl
├── ...
├── M1-RND_s0.pt
├── M1-RND_s0.jsonl
├── ...
├── smoke_extended/
│   ├── I0_s0.pt
│   ├── I0_s0.jsonl
│   ├── M-FULL_s0.pt
│   └── M-FULL_s0.jsonl
└── extended/
    ├── I0_s0.pt
    ├── I0_s0.jsonl
    ├── ...
    ├── M-FULL_s0.pt
    ├── M-FULL_s0.jsonl
    └── ...
```

Each full controller has checkpoints and training logs for seeds 0–4.

### Experimental result directories

```text
results/
├── alignment_check.txt
├── baselines.json
└── part_c/
    ├── validation_m0/
    ├── validation_m1/
    ├── validation_m1rnd/
    ├── validation_combined/
    │   ├── replicates.csv
    │   └── analysis/
    ├── test_m0/
    ├── test_m1/
    ├── test_m1rnd/
    ├── test_combined/
    │   ├── replicates.csv
    │   └── analysis/
    ├── smoke_extended/
    ├── predictability/
    ├── training_analysis/
    └── extended/
        ├── validation/
        │   ├── replicates.csv
        │   ├── manifest.json
        │   ├── trajectories/
        │   └── analysis/
        ├── test/
        │   ├── replicates.csv
        │   ├── manifest.json
        │   ├── trajectories/
        │   └── analysis/
        └── training_analysis/
```

### Part C script reference

| Script | Purpose |
|--------|---------|
| `scripts/part_c_train.py` | Multi-controller, multi-seed training scheduler |
| `scripts/part_c_experiments.py` | Validation and test evaluation, delivery sweep, HOLD and forced-mask conditions |
| `scripts/part_c_report.py` | Original H1–H3 statistical analysis and plots |
| `scripts/part_c_extended_report.py` | Supplementary five-controller comparison and paired statistical analysis |
| `scripts/alignment_check.py` | Reward–KPI ranking verification |
| `scripts/part_c_predictability.py` | Peer-SoC ridge regression diagnostic |
| `scripts/part_c_predictability_plots.py` | Predictability summaries, heatmap and distribution plots |
| `scripts/part_c_learning_curves.py` | Original three-controller training diagnostics |
| `scripts/part_c_extended_learning_curves.py` | Extended five-controller training diagnostics |

Every evaluation record contains the SHA-256 hash of the checkpoint used for that evaluation.

The original and supplementary experimental results are preserved separately to maintain provenance and prevent accidental modification of the original hypothesis analysis.

Statistical reports and figures can be regenerated from the corresponding evaluation CSV files and training JSONL logs without retraining the controllers.

Trained checkpoints are required to reproduce fresh policy evaluations.

Generated checkpoints, trajectory archives, virtual environments and other large local artifacts should be managed according to the repository's `.gitignore` and storage policy.

## 15. References

The MAPPO implementation is based on:

Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games," Advances in Neural Information Processing Systems (NeurIPS), 2022.

Reference implementation:

https://github.com/marlbenchmark/on-policy

CityLearn:

https://github.com/intelligent-environments-lab/CityLearn

The accompanying research report contains the complete literature review, experimental methodology, hypothesis definitions, results, limitations and references.