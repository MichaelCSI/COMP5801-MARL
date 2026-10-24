# Member B (Michael): MAPPO, message channel, random-loss training

Scope from proposal §7: **MAPPO training code, message channel and mask,
random-loss training (M1-RND)**, optional IPPO (I0). Report: method and
baselines.

## 1. Plan

| # | Piece | Proposal | Where |
|---|-------|----------|-------|
| 1 | Peer-SoC channel: 30 directed links, independent Bernoulli(p) per link per step, `[SoC, 1]` / `[0, 0]`, own RNG | §1 | `marl/comms/channel.py` |
| 2 | Delivery schedule: fixed p (M1) or p ~ U{0, .5, .75, 1} per episode (M1-RND) | §5 | `DeliverySchedule` |
| 3 | Mask check: evaluate with mask forced to 1 | §6 | `PeerSoCChannel(force_mask=True)` |
| 4 | Actor input õᵢ = [oᵢ, m₋ᵢ]; critic input = true global state | §1, §5 | `marl/mappo/rollout.py` |
| 5 | MAPPO: one actor per building (no sharing), centralized critic, GAE, PPO-clip, value norm | §5 | `marl/mappo/{networks,trainer}.py` |
| 6 | Arms M0 / M1 / M1-RND (+ optional I0, M1-RND-NM) differ only in message fields | §4–6 | `marl/mappo/config.py` |
| 7 | Evaluation at p ∈ {1, .9, .5, 0}, 10 replicates, identical outages, independent drops | §6 | `evaluate()` |
| 8 | Env contract that Andrea's wrapper implements, toy envs, temporary CityLearn adapter | — | `marl/envs/` |
| 9 | CLI + speed check | §5 checkpoint 1 | `scripts/train.py`, `scripts/speed_check.py` |

Order: channel → buffer/GAE → rollout loop → networks/PPO → arms → eval →
CityLearn smoke → speed check. Weeks 1–2 (with Andrea): get M0 training on her
wrapper for gate G1.

## 2. Interfaces with the team

**Andrea's wrapper** only has to satisfy `marl.envs.check_env_contract(env)`:

```
env.n_agents, env.obs_dims, env.action_dims, env.state_dim
env.reset(seed=None, options=None) -> (obs_list, info)
env.step(actions_list) -> (obs_list, reward: float, terminated, truncated, info)
env.soc() -> (n,) true SoC        # message payload
env.global_state() -> (state_dim,) # critic input, never contains drops
```

* The reward is **one shared scalar** (§2). A per-building list fails the check.
* `reset(seed=s)` must fix the outage realization and the training window for
  that seed. The trainer draws reset seeds from a stream that depends only on
  the run seed, never on the arm. That is what makes runs paired and outages
  identical across arms.
* KPIs: pass `on_episode_end=lambda env: env.kpis()` (or similar) to
  `evaluate`, and the result lands in `record["kpis"]`.
* Until the wrapper lands, `marl/envs/citylearn_adapter.py` gives a
  battery-only, decentralized CityLearn 2023 Phase 3.3 env with a
  **placeholder** reward (sum of CityLearn's building rewards).

**Vishal's runner** calls:

```python
from marl.mappo import MAPPOConfig, MAPPOTrainer
tr = MAPPOTrainer(env, MAPPOConfig.for_arm("M1-RND", seed=s, total_steps=N), logger=fn)
tr.train(); tr.save(path)                       # or scripts.train.run(arm, seed, ...)
tr = MAPPOTrainer.load(path, test_env)
recs = tr.evaluate(test_env, p=0.5, n_replicates=10, env_seed=TEST_SEED)
recs = tr.evaluate(test_env, p=0.0, force_mask=True)   # mask check
```

Each record has `actions (T, n)` (battery actions, for the synchrony
diagnostic), `soc (T, n)`, `delivered (T, n, n)`, `rewards (T,)`, `infos`,
`return` and `kpis`. The logger gets a dict per update: losses, KL, clip
fraction, explained variance, `steps_per_sec`, `ep_return_mean`,
`train_p_mean`.

## 3. What I can test now, so it holds when the others plug in

| Test | Guards against | File | Needs |
|------|----------------|------|-------|
| Message shape 2(n−1), peer order, no self-link, 30 links | wrong input layout | `test_channel.py` | numpy |
| p=1 carries every peer's SoC; p=0 is all zeros | channel wiring | `test_channel.py` | numpy |
| Delivered SoC=0 → `[0,1]`, dropped → `[0,0]` | the reason the mask exists | `test_channel.py` | numpy |
| Empirical rate ≈ p; links independent across links and time | §1 loss model | `test_channel.py` | numpy |
| Seeded, own RNG, doesn't touch the global RNG | outages changing with the arm | `test_channel.py` | numpy |
| `force_mask` turns drops into SoC=0 | mask check (§6) | `test_channel.py` | numpy |
| Schedule draws {0, .5, .75, 1} uniformly; 0.9 never in training | M1-RND (§5) | `test_channel.py` | numpy |
| GAE: hand-computed values, λ=1 = discounted return, no cross-episode leak, truncation bootstrap | wrong learning targets | `test_rollout.py` | numpy |
| Running normalizer = numpy mean/var; frozen in eval | train/eval mismatch | `test_rollout.py` | numpy |
| Env contract passes on toy envs; rejects per-building reward | Andrea's wrapper breaking the trainer | `test_rollout.py` | numpy |
| Actor input dims per arm (M0 = 7, M1 = 17 on toy) | arms not comparable | `test_rollout.py` | numpy |
| **Critic input identical at p=1 and p=0** | critic seeing dropped messages | `test_rollout.py` | numpy |
| Message entries are raw SoC (not normalized) at the step the agent acts | stale / scaled messages | `test_rollout.py` | numpy |
| **Env reset seeds identical across M0/M1/M1-RND** | unpaired runs | `test_rollout.py` | numpy |
| M1-RND: per-episode p varies and the realized rate tracks it | random-loss training | `test_rollout.py` | numpy |
| Oracle policy on a messages-only task scores ≈ −(1−p)/3 | end-to-end message path | `test_rollout.py` | numpy |
| Eval replicates: same outages, different drops; record format | Vishal's sweep and statistics | `test_rollout.py` | numpy |
| Arms differ only in message fields | confounded comparison | `test_rollout.py` | numpy |
| Actors share no parameters | §5 "no parameter sharing" | `test_mappo_torch.py` | torch |
| Every arm trains, stats finite; same seed → same weights | runner crashes, irreproducible seeds | `test_mappo_torch.py` | torch |
| Checkpoint save/load → identical eval actions | evaluation reusing trained policies | `test_mappo_torch.py` | torch |
| **Learning signal**: on a task solvable only with messages, M1 beats M0's ceiling (−1/3); M1 then degrades at p=0 | MAPPO + channel actually learn from messages | `test_mappo_torch.py` | torch |
| CityLearn adapter: 6 buildings, battery-only, contract, SoC moves | real-env integration | `test_citylearn_smoke.py` | citylearn |
| One update per arm on CityLearn; steps/s → hours per run | speed check, ≤ 2 h target | `test_citylearn_smoke.py` | citylearn + torch |

**Can't test until the others deliver:** the real reward and its R0
normalization, alignment check (R0/R1/M0 ranking), train/val/test windows and
outage seeding (Andrea); KPI aggregation, paired t-intervals, synchrony
(Vishal). The contract check plus the paired-seed test cover the seams.

## 4. Running the tests

```bash
python -m pytest tests -v                       # in the MARL venv (torch + citylearn)
python -m unittest discover -s tests -t . -v    # same tests, no pytest needed
python -m scripts.speed_check                   # CityLearn steps/s and hours per run
python -m scripts.train --arm M1-RND --seed 0 --steps 20000 --env citylearn
```

Tests that need torch or citylearn skip themselves when those packages are
missing.

## 5. Defaults to revisit at the pilot

PPO: γ = 0.99, λ = 0.95, clip 0.2, 10 epochs, 4 minibatches, Adam (3e-4
actor, 1e-3 critic), grad-norm 0.5, entropy 0, actor log-std init −0.5,
64×64 tanh MLPs, observation and value normalization, bootstrap at window
truncation. Same values for every arm (equal tuning budget).
