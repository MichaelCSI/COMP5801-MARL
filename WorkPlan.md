# Part A: ...

@Andrea, based on the part B setup, part A would connect with it via:

`marl.envs.check_env_contract(env)`:

```
env.n_agents, env.obs_dims, env.action_dims, env.state_dim
env.reset(seed=None, options=None) -> (obs_list, info)
env.step(actions_list) -> (obs_list, reward: float, terminated, truncated, info)
env.soc() -> (n,) true SoC        # message payload
env.global_state() -> (state_dim,) # critic input, never contains drops
```

* The reward is **one shared scalar** (section 2).
* `reset(seed=s)` must fix the outage realization and the training window for
  that seed. Reset seeds come from a stream that depends only on the run
  seed, never on the arm, so runs are paired.
* Report a window end as `truncated=True`. on-policy's `bad_masks` then stop
  the cut-off step from being treated as a real terminal. CityLearn 3.0.2
  never sets `truncated` (it reports every episode end as `terminated`), so
  the wrapper has to convert it, as the temporary adapter does.
* Put each battery's actual electricity use in the step just taken (kWh,
  + charging) in `info["battery_energy"]`. Evaluation records it, and it is the
  signal for the synchrony diagnostic. In CityLearn it is
  `building.electrical_storage.electricity_consumption[env.time_step - 1]`.
  Use the env clock: on the terminal step the buildings' clocks don't advance.
* **Known issue, outages:** in the Phase 3.3 schema every building's
  stochastic outage model has a fixed seed (57841) that `reset(seed)` doesn't
  reach. Outages are drawn per episode from the episode start, so with this
  seed there are none in 7-, 15- or 30-day episodes, and in a full 92-day run
  they fall on days 43, 70 and 89 in all six buildings at once. The wrapper
  must set the outage seed from the reset seed and place outages on the
  92-day calendar so the test block gets some.
* CityLearn picks random windows with `np.random.RandomState(seed * (episode +
  1))`, which fails once that reaches 2**32. Map reset seeds into a small range
  first (the adapter uses `1 + seed % 100000`).
* KPIs: pass `on_episode_end=lambda env: env.kpis()` (or similar) to
  `evaluate`, and the result lands in `record["kpis"]`.
* Until the wrapper is done, `marl/envs/citylearn_adapter.py` gives a
  battery-only, decentralized CityLearn 2023 Phase 3.3 env with a
  **placeholder** reward. It follows the split: `split="train"` gives random
  contiguous windows inside steps 0–1487; `"validation"` and `"test"` are one
  episode over their whole block. It does **not** fix the outage seeding.

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

**Untested until part A and C done:** the real reward and its R0
normalization, alignment check, split and outage seeding (Andrea); KPI
aggregation, paired t-intervals, synchrony (Vishal).

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
