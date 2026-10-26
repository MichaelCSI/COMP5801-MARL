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
| I0, M1-RND-NM | Optional: IPPO without messages; M1-RND without the mask |

The learner is the reference MAPPO implementation of Yu et al. (NeurIPS 2022),
[marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy), vendored
unmodified in `third_party/onpolicy/`.

## Repository layout

```
marl/
  comms/        lossy peer-SoC message channel and delivery schedules
  envs/         env contract for the CityLearn wrapper, toy envs, temporary CityLearn adapter
  mappo/        glue around on-policy (messages in actor inputs, map arms to on-policy...)
scripts/        command-line entry points (training, speed check)
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

   Don't name it `MARL`: macOS and Windows file names ignore case, so a
   `MARL` folder is the same folder as the `marl/` package and the venv ends
   up inside the code.

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
| `--arm` | `M0` | `M0`, `M1`, `M1-RND`, `I0`, `M1-RND-NM` |
| `--seed` | `0` | Run seed (the same seed gives the same outages in every arm) |
| `--steps` | `20000` | Total environment steps |
| `--env` | `toy` | `toy` or `citylearn` (temporary adapter until the project wrapper lands) |
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
| `tests/test_channel.py` | Message format, delivery rate and independence, seeding, forced-mask mode, M1-RND schedule | numpy | `python -m pytest tests/test_channel.py -v` |
| `tests/test_message_env.py` | Actor/critic inputs (critic never sees drops), raw messages, paired seeds across arms, env contract, evaluation | numpy | `python -m pytest tests/test_message_env.py -v` |
| `tests/test_onpolicy_args.py` | Arms map to identical on-policy settings; vendored on-policy files match upstream | numpy | `python -m pytest tests/test_onpolicy_args.py -v` |
| `tests/test_citylearn_adapter.py` | Temporary CityLearn adapter against a fake CityLearn: split windows, truncation at episode end, battery energy, seed range | numpy | `python -m pytest tests/test_citylearn_adapter.py -v` |
| `tests/test_onpolicy_runner.py` | Training with on-policy MAPPO for every arm, reproducibility, checkpoints, learning-from-messages check (about a minute) | torch | `python -m pytest tests/test_onpolicy_runner.py -v -s` |
| `tests/test_citylearn_smoke.py` | CityLearn adapter on the real 2023 Phase 3.3 dataset (split windows, truncation); one training update per arm and steps/s | citylearn, torch | `python -m pytest tests/test_citylearn_smoke.py -v -s` |

Without pytest, use `python -m unittest tests.test_channel -v` (and so on).
`-s` shows the printed learning and speed results.
