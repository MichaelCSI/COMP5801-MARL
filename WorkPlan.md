# Part A: CityLearn wrapper, reward, KPIs, references, split, outages

CityLearn wrapper, reward, KPIs, references R0 and R1, split and outage seeding. Report: environment and reward sections; optional mask arm (M1-RND-NM).

## 1. What was built

`marl.envs.district_env.DistrictBatteryEnv` is the project environment. It
extends Michael's `CityLearnAdapter`, so the split windows, window seeds,
truncation at the window end, `battery_energy`, `soc()` and `global_state()`
are unchanged and still covered by `test_citylearn_adapter.py`. It passes
`check_env_contract` on every split, and `scripts/train.py --env citylearn` now
builds it (`--env citylearn-temp` keeps the old adapter).

| Piece | Proposal | Where |
|-------|----------|-------|
| Shared reward r = -(α ĉ + β ĥ), c = price · max(0, L), h = max(0, L)², normalized by R0 on the training block | section 2 | `marl/envs/reward.py` |
| Export check: how often L < 0 under R0/R1 | feedback | `scripts/export_check.py` → `results/export_check.json` |
| Peer-SoC informativeness data: own observation X and peers' SoC Y per agent | feedback | `marl/envs/features.py` |
| R0 norms, frozen observation set, evaluation seeds, building table | sections 1, 2 and 6 | `scripts/freeze_env.py` → `marl/envs/phase33_frozen.json` |
| Outage seeding on the 92-day calendar, per building, from the reset seed | section 1 | `marl/envs/outages.py` |
| KPIs: average daily district peak and cost, raw and relative to R0 on the same window and seed | section 6 | `marl/envs/kpis.py`, `env.kpis()` |
| R0 (zero action) and R1 (CityLearn `BasicBatteryRBC`) | section 4 | `marl/baselines.py`, `scripts/run_baselines.py` |
| Alignment check: R0, R1 and M0 rank the same by return and by the primary outcome on validation | section 2, gate G1 | `scripts/alignment_check.py` |

**Outages.** The schema's outage model (SAIFI 1.436/yr, CAIDI 331.2 min) is
kept, but its seed is replaced: for reset seed `s`, building `b` draws its
signal once over steps 0–2207 with a seed derived from `(s, b)`, and each episode
takes its window's slice. This goes through CityLearn's own hook,
`Building.reset_power_outage_signal`, so the `power_outage` observation and
the physics both see it. The same seed gives the same outages in every arm and
on any window, and the six buildings fail independently.

**Frozen constants** (`phase33_frozen.json`, written by `python -m scripts.freeze_env`):
c̄_R0 = 0.2470 and h̄_R0 = 67.145 (R0 over steps 0–1487, outage seed 0), so the
mean R0 reward on the training block is -(α + β). `VAL_SEED = 7` and
`TEST_SEED = 0` are the smallest seeds with at least one outage hour in their
block. The construction fails if the active observation set (30 per building,
85 in the global state) stops matching. Battery efficiency is the schema's
nominal 0.95/0.96; CityLearn applies it through a power-efficiency curve
(0.88–0.96 depending on the charge rate), which the report should mention.

**info per step:** `district_load` (L_t, kWh), `price`, `cost`,
`dataset_step`, `outage` (n,), `building_net_load` (n,, kWh; for a
full-observability arm or load messages), plus the adapter's
`battery_energy`.

## 2. Results so far

| Block (seed) | Arm | Primary (peak / R0) | Cost / R0 | Return |
|--------------|-----|---------------------|-----------|--------|
| validation (7) | R0 | 1.000 | 1.000 | -874.8 |
| validation (7) | R1 | 0.968 | 0.998 | -865.1 |
| test (0) | R0 | 1.000 | 1.000 | -999.0 |
| test (0) | R1 | 0.946 | 0.998 | -983.8 |

Speed check (M1-RND, 720-step windows): about 104 steps/s including updates,
about **0.53 h per 200k-step run** (target ≤ 2 h). The wrapper costs about 6%
over the temporary adapter.

## 2b. Changes after the proposal feedback

**Reward.** The peak term is now h = max(0, L)² instead of L². The old term
penalized midday solar export like an import peak, while the cost term and
the primary KPI (daily import peak) ignore export. `scripts/export_check.py`:

| Block (seed) | Arm | Export steps | Lowest L (kWh) | Share of old Σ L² from export |
|--------------|-----|--------------|----------------|-------------------------------|
| train (0) | R0 | 16 / 1487 (1.1%) | -1.32 | 0.01% |
| train (0) | R1 | 6 / 1487 (0.4%) | -1.65 | 0.01% |
| validation (7) | R0, R1 | 0 / 359 | 0.49, 1.15 | 0 |
| test (0) | R0, R1 | 0 / 359 | 1.79, 3.41 | 0 |

So for the references the change is nearly free: h̄_R0 moved from 67.149 to
67.145, R0/R1 KPIs are identical, and returns moved by at most 0.03 (test R0: -998.93 → -998.95).
It matters for learned policies, which could otherwise be pushed away from
discharging into a midday surplus. Re-frozen before any M0 run; the
alignment check still has to run once M0 exists. The frozen norms now carry
`h_formula = "max(0,L)^2"`, and the env refuses norms frozen for another
definition, so a stale `phase33_frozen.json` fails loudly instead of
mis-scaling the reward. The KPI field is now `mean_import_sq` (mean h_t).

**Report text, Section 1 (message channel).** "The messages are fixed
peer-state sharing over a lossy channel: every agent always sends its current
SoC to its five peers, and nothing about what to send is learned. The study
measures the value of this fixed signal and how much of it survives message
loss; learned communication is outside its scope."

**Report text, Section 2 (reward).** "h_t = max(0, L_t)². Squaring only the
district import keeps both terms consistent with each other and with the
primary outcome, the average daily import peak; exported energy (L_t < 0) is
neither costed nor penalized. Under no-battery control the district exports
in 1.1% of training hours and never in the validation or test blocks, so the
normalization constant h̄_R0 is practically unchanged."

**For Michael (M-FULL / load messages).** `info["building_net_load"]` gives
each building's net load in the step just taken; local observations already
contain `net_electricity_consumption` and `non_shiftable_load`. The schema has
no predicted-load observation, so a "predicted load" message would need a
forecaster of our own.

**For Vishal (informativeness check, after G1).**

```python
from marl.envs.features import collect, peer_soc_dataset
obs_seq, soc_seq = collect(val_env, controller, seed=VAL_SEED)   # controller: obs_list -> actions
data = peer_soc_dataset(obs_seq, soc_seq, val_env.observation_names)
data[i]["X"], data[i]["Y"]       # regress peers' SoC (T, 5) on agent i's observation (T, 30)
```

Peers are in the channel's order. `drop_own_soc=True` removes the agent's own
SoC from X. For M0, wrap the loaded policy as a raw-observation controller.

## 3. Running

```bash
python -m scripts.freeze_env           # only if the reward, schema or CityLearn changes
python -m scripts.export_check         # how often L < 0 -> results/export_check.json
python -m scripts.run_baselines        # R0/R1 KPIs -> results/baselines.json
python -m scripts.alignment_check --m0 runs/M0_s0.pt [--m0 runs/M0_s1.pt]
python -m pytest tests/test_district_env.py -v
```

**For Vishal:** evaluate with the frozen seeds and the KPI hook:

```python
from marl.envs.district_env import DistrictBatteryEnv, TEST_SEED
test_env = DistrictBatteryEnv(split="test")
recs = r.evaluate(test_env, p=0.5, n_replicates=10, env_seed=TEST_SEED,
                  on_episode_end=lambda e: e.kpis())
recs[0]["kpis"]["primary"]       # average daily peak / R0's (lower is better)
```

**Dataset download.** CityLearn fetches the dataset through the GitHub API,
which is rate limited (60 requests/h without auth). If you see a 403, clone
`citylearn-project/CityLearn` at tag `v3.0.2` and copy
`data/datasets/citylearn_challenge_2023_phase_3_3` into
`~/Library/Caches/citylearn/v3.0.2/datasets/` (Linux: `~/.cache/citylearn/...`).
The wrapper passes the cached `schema.json` path, so after that it never
contacts GitHub.

# Part B: MAPPO, message channel, random-loss training

MAPPO training code, message channel and mask, random-loss training (M1-RND), optional IPPO (I0). Report: method and baselines.

## 1. What we reuse and what we wrote

**Learner = reference MAPPO, not ours.** We use the official implementation
from Yu et al. [10], [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)
at commit `de66d7a` in `third_party/onpolicy/`,
under its MIT license; see `VENDORED.md` there. 

* `R_MAPPOPolicy`: actor, critic and Adam optimizers. There is one per
  building, because on-policy's *separated* mode (`share_policy=False`) means
  no parameter sharing.
* `R_MAPPO`: the PPO-clip update, clipped Huber value loss and ValueNorm.
* `SeparatedReplayBuffer`: per-agent rollout storage and GAE.
* `onpolicy.config.get_config()`: the reference hyper-parameter defaults.

**Our `marl/mappo/`:** on-policy provides the learner (networks,
PPO update, rollout buffer) but nothing about our experiment, and its own
training loop is tied to wandb, tensorboardX and its bundled environments, so
it isn't vendored. `marl/mappo/` is the glue between the two: it puts the lossy
peer-SoC messages into each actor's input and the true state into the critic's
(`message_env.py`), maps each arm onto on-policy's arguments (`config.py`), runs
on-policy's classes in a training loop (`runner.py`) and holds the evaluation
protocol (`evaluate.py`). Nothing can train without it.

**Our code (the contribution and the glue):**

| Piece | Proposal | Where |
|-------|----------|-------|
| Peer-SoC channel: 30 directed links, independent Bernoulli(p) per link per step, `[SoC, 1]` / `[0, 0]`, own RNG | section 1 | `marl/comms/channel.py` |
| Delivery schedule: fixed p (M1) or p ~ U{0, .5, .75, 1} per episode (M1-RND) | section 5 | `DeliverySchedule` |
| Mask check: evaluate with mask forced to 1 | section 6 | `PeerSoCChannel(force_mask=True)` |
| `MessageEnv`: actor input õᵢ = [norm(oᵢ), m₋ᵢ], critic input = true global state, paired reset seeds, per-episode p | section 1, section 5 | `marl/mappo/message_env.py` |
| Arms M0 / M1 / M1-RND (+ I0, M1-RND-NM) → on-policy arguments | section 4–6 | `marl/mappo/config.py` |
| Training loop on on-policy's separated runner (no wandb/tensorboardX), `bad_masks` for window truncation, checkpoints | section 5 | `marl/mappo/runner.py` |
| Evaluation at p ∈ {1, .9, .5, 0}, 10 replicates, identical outages, independent drops | section 6 | `marl/mappo/evaluate.py` |
| Env contract for Andrea's wrapper, toy envs, temporary CityLearn adapter | — | `marl/envs/` |
| CLI + speed check | section 5 checkpoint 1 | `scripts/train.py`, `scripts/speed_check.py` |

**Settings fixed by the design** (`FIXED_ONPOLICY_ARGS`; the same in every arm):
* `share_policy=False`: one actor per building.
* `algorithm_name="mappo"` with recurrence off: MLP policy.
* `use_centralized_V`: True for MAPPO, False for IPPO.
* `use_proper_time_limits=True`: training windows are time-limit cuts, not
  terminals.
* `use_feature_normalization=False`: we normalize each local feature with
  running statistics instead, and leave the `[SoC, mask]` columns raw. A
  LayerNorm over the whole input would mix raw-scale CityLearn features with
  the messages.

Everything else is on-policy's default (lr 5e-4, 15 PPO epochs, 1 minibatch,
clip 0.2, entropy 0.01, γ 0.99, λ 0.95, Huber value loss, ValueNorm,
64-unit ReLU MLP). Tuning goes through `MAPPOConfig(onpolicy_overrides={...})`
with the same budget for every arm.

Each building has its own critic, and with `use_centralized_V=True` all of them see the same true global state and the same shared reward. So they estimate the same V(s) with independent parameters. This is on-policy's standard non-shared MAPPO.

## 2. Tests

| Test | Guards against | File | Needs |
|------|----------------|------|-------|
| Message shape, peer order, no self-link, p=1 / p=0, SoC=0 vs drop, rate ≈ p, link independence, seeding, `force_mask`, schedule | channel bugs (section 1, section 5, section 6) | `test_channel.py` | numpy |
| Spaces per arm are `Box` with the right dims (on-policy picks heads by class name) | wrong network inputs | `test_message_env.py` | numpy |
| **Critic input identical at p=1 and p=0** | critic seeing dropped messages | `test_message_env.py` | numpy |
| Messages are raw SoC of the current step; local features normalized | stale or scaled messages | `test_message_env.py` | numpy |
| Shared reward replicated to every agent; auto-reset + truncation flags | wrong rewards / masks into on-policy | `test_message_env.py` | numpy |
| **Reset seeds identical across M0/M1/M1-RND/I0** | unpaired runs | `test_message_env.py` | numpy |
| M1-RND draws p per episode, realized rate tracks it; no-mask arm holds mask at 1 | random-loss training | `test_message_env.py` | numpy |
| Oracle on a messages-only task scores ≈ −(1−p)/3; replicates share outages, not drops; eval freezes normalizers | end-to-end message path, Vishal's sweep | `test_message_env.py` | numpy |
| Env contract passes on toy envs, rejects per-building reward | Andrea's wrapper breaking the trainer | `test_message_env.py` | numpy |
| Arms differ only in message fields; **on-policy args identical across MAPPO arms**; all other args = on-policy defaults; design-fixed args can't be overridden | confounded comparison, accidental tuning | `test_onpolicy_args.py` | numpy |
| **Vendored on-policy matches upstream hashes**; license + commit present | silent edits to the reference code | `test_onpolicy_args.py` | numpy |
| Runner uses on-policy's classes; 6 actors share no parameters; dims per arm | wrong wiring | `test_onpolicy_runner.py` | torch |
| Every arm trains with finite stats; same seed → same weights; truncation → `bad_masks` | crashes, irreproducible seeds | `test_onpolicy_runner.py` | torch |
| Checkpoint save/load (incl. ValueNorm) → identical eval actions | evaluation reusing trained policies | `test_onpolicy_runner.py` | torch |
| **Learning signal**: on a messages-only task M1 beats M0's ceiling (−1/3) and degrades at p=0 | MAPPO actually learns from messages | `test_onpolicy_runner.py` | torch |
| Adapter vs a fake CityLearn with 3.0.2's window, terminal-clock and seed behaviour: train windows random inside steps 0–1487 and paired across arms, test = whole block, episode end → `truncated`, battery energy from the right step, large seeds don't crash | test-data leakage, wrong value targets, wrong synchrony signal | `test_citylearn_adapter.py` | numpy |
| Evaluation records actual battery energy, which differs from the action once a battery is full | synchrony computed from commands | `test_message_env.py` | numpy |
| CityLearn adapter contract, split windows, truncation; one update per arm and steps/s | real-env integration, ≤ 2 h target | `test_citylearn_smoke.py` | citylearn (+ torch) |

**Untested until part C done:** KPI aggregation, paired t-intervals,
synchrony (Vishal). Part A's reward, R0 normalization, outage seeding and
KPIs are covered by `test_district_env.py` and `test_citylearn_smoke.py`.

## 3. Running

```bash
python -m pytest tests -v                       # in the project venv, .venv (torch + citylearn)
python -m unittest discover -s tests -t . -v    # same tests, no pytest needed
python -m scripts.speed_check                   # CityLearn steps/s and hours per run
python -m scripts.train --arm M1-RND --seed 0 --steps 20000 --env citylearn
```

No new dependencies: the vendored on-policy code needs only torch and numpy.

## 4. Citation (for the report)

> We use the reference MAPPO implementation of Yu et al. [10]
> (github.com/marlbenchmark/on-policy, commit de66d7a, MIT license) in its
> non-parameter-sharing ("separated") configuration. Our code adds the
> peer-SoC message channel, the random-loss training schedule and the
> evaluation protocol, and replaces the upstream runner's logging.

```bibtex
@inproceedings{yu2022surprising,
  title     = {The Surprising Effectiveness of {PPO} in Cooperative Multi-Agent Games},
  author    = {Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and
               Wang, Yu and Bayen, Alexandre and Wu, Yi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Datasets and Benchmarks Track},
  year      = {2022}
}
```

# Part C: ...

@Vishal the runner can call the part B code with:

```python
from marl.mappo import MAPPOConfig, OnPolicyRunner
r = OnPolicyRunner(env, MAPPOConfig.for_arm("M1-RND", seed=s, total_steps=N), logger=fn)
r.run(); r.save(path)                          # or scripts.train.run(arm, seed, ...)
r = OnPolicyRunner.load(path, test_env)
recs = r.evaluate(test_env, p=0.5, n_replicates=10, env_seed=TEST_SEED)
recs = r.evaluate(test_env, p=0.0, force_mask=True)    # mask check
```

Each record has `actions (T, n)` (commanded actions), `battery_energy (T, n)`
(what the batteries actually did; use this for synchrony), `soc (T, n)`, `delivered (T, n, n)`, `rewards (T,)`, `infos`,
`return` and `kpis`. The logger gets one dict per update with on-policy's
`value_loss`, `policy_loss`, `dist_entropy`, `ratio` and grad norms (averaged
over agents), plus `steps_per_sec`, `ep_return_mean` and `train_p_mean`.
