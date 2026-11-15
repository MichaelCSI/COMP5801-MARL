# Vendored: marlbenchmark/on-policy (MAPPO reference implementation)

* **Upstream:** https://github.com/marlbenchmark/on-policy
* **Commit:** `de66d7a4b23fac2513f56f96f73b3f5cb96695ac` (2024-07-18)
* **License:** MIT, copyright (c) 2021 MAPPO contributors. See `LICENSE`.
* **Paper:** C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, Y. Wu,
  "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,"
  NeurIPS 2022 Datasets and Benchmarks. (Proposal reference [10].)

## Why vendored instead of `pip install`

on-policy is not on PyPI. Its `setup.py` uses `find_packages()`, which skips
`onpolicy/algorithms/utils/` and `onpolicy/algorithms/r_mappo/algorithm/`
(no `__init__.py`). A regular `pip install git+...` therefore produces a
package that cannot import `R_MAPPO`. Upstream's own instructions use
`pip install -e .` from a clone. Vendoring the files we use avoids that,
pins the exact code, and works offline.

## What is used

| File | Role here |
|------|-----------|
| `onpolicy/algorithms/r_mappo/r_mappo.py` | `R_MAPPO`: PPO-clip update, clipped Huber value loss, ValueNorm |
| `onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py` | `R_MAPPOPolicy`: actor + critic + Adam optimizers |
| `onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py` | `R_Actor`, `R_Critic` (MLP; RNN off) |
| `onpolicy/algorithms/utils/*.py` | MLP base, diagonal-Gaussian action head, init helpers |
| `onpolicy/utils/separated_buffer.py` | `SeparatedReplayBuffer`: per-agent rollout storage and GAE |
| `onpolicy/utils/valuenorm.py`, `util.py` | value normalization, losses, shape helpers |
| `onpolicy/config.py` | the reference hyper-parameter defaults (`get_config()`) |

We use on-policy's **separated** mode (`share_policy=False`): one actor per
building with no parameter sharing. With
`use_centralized_V=True` each agent's critic sees the true global state (MAPPO),
and with `False` it sees the agent's own input (IPPO). The training loop in
`marl/mappo/runner.py` follows `onpolicy/runner/separated/base_runner.py` and
`mpe_runner.py`. Those runner files aren't vendored, because they hard-import
wandb and tensorboardX and target the MPE env API.

## Modifications

* `onpolicy/__init__.py` is replaced by a one-line package file (upstream's
  imports every environment).
* **No other file is modified.** `MANIFEST.sha256` holds the upstream hashes,
  and `tests/test_vendored_onpolicy.py` fails if any vendored file changes.
