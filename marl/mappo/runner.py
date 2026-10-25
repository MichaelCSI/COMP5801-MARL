"""Training loop around the reference MAPPO implementation (needs torch).

Learner: marlbenchmark/on-policy @ de66d7a (MIT), vendored in
third_party/onpolicy. Yu et al., "The Surprising Effectiveness of PPO in
Cooperative Multi-Agent Games", NeurIPS 2022.

On-policy components used UNCHANGED:
    R_MAPPOPolicy         actor + critic + optimizers, one per building
    R_MAPPO               PPO-clip update, clipped Huber value loss, ValueNorm
    SeparatedReplayBuffer per-agent rollout storage and GAE

This loop follows on-policy's separated runner
(onpolicy/runner/separated/base_runner.py and mpe_runner.py: warmup ->
collect -> insert -> compute -> train) for one rollout thread. It drops the
wandb/tensorboardX logging, takes inputs from ``MessageEnv`` (peer-SoC
channel + true global state), and passes ``bad_masks`` for time-limit
truncation.

    cfg = MAPPOConfig.for_arm("M1-RND", seed=0, total_steps=200_000)
    runner = OnPolicyRunner(env, cfg, logger=print)
    runner.run()
    runner.save("runs/M1-RND_s0.pt")
    recs = runner.evaluate(test_env, p=0.5, n_replicates=10, env_seed=TEST_SEED)
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from .config import STREAM_EVAL, STREAM_TORCH, MAPPOConfig, stream
from .evaluate import evaluate as _evaluate
from .message_env import make_message_env
from .onpolicy_compat import ONPOLICY_COMMIT, ensure_on_path

ensure_on_path()
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy  # noqa: E402
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO  # noqa: E402
from onpolicy.utils.separated_buffer import SeparatedReplayBuffer  # noqa: E402
from onpolicy.utils.valuenorm import ValueNorm  # noqa: E402


def _t2n(x):
    return x.detach().cpu().numpy()


def _scalar(v) -> float:
    return float(v.item()) if hasattr(v, "item") else float(v)


class DecentralizedActors:
    """Execution-time policy: actor i sees only its own input (CTDE)."""

    def __init__(self, policies, args, device):
        self.policies, self.args, self.device = policies, args, device

    @torch.no_grad()
    def act(self, actor_inputs, deterministic: bool = True):
        out = []
        for pol, x in zip(self.policies, actor_inputs):
            pol.actor.eval()
            rnn = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), np.float32)
            masks = np.ones((1, 1), np.float32)
            a, _ = pol.act(np.asarray(x, np.float32)[None], rnn, masks,
                           deterministic=deterministic)
            out.append(_t2n(a)[0])
        width = max(o.shape[0] for o in out)
        acts = np.zeros((len(out), width), np.float32)
        for i, o in enumerate(out):
            acts[i, : o.shape[0]] = o
        return acts, np.zeros(len(out), np.float32)


class OnPolicyRunner:
    def __init__(self, env, cfg: MAPPOConfig, logger: Optional[Callable[[dict], None]] = None):
        self.cfg, self.logger = cfg, logger
        self.args = args = cfg.to_onpolicy_args()
        torch.manual_seed(int(stream(cfg.seed, STREAM_TORCH).generate_state(1)[0]))
        torch.set_num_threads(int(args.n_training_threads))
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device(cfg.device if use_cuda else "cpu")

        self.env = env
        self.menv = make_message_env(env, cfg)
        self.n_agents = env.n_agents
        self.use_centralized_V = args.use_centralized_V

        self.policy, self.trainer, self.buffer = [], [], []
        for i in range(self.n_agents):
            obs_sp = self.menv.observation_space[i]
            share_sp = self.menv.share_observation_space[i]
            act_sp = self.menv.action_space[i]
            pol = R_MAPPOPolicy(args, obs_sp, share_sp, act_sp, device=self.device)
            self.policy.append(pol)
            self.trainer.append(R_MAPPO(args, pol, device=self.device))
            self.buffer.append(SeparatedReplayBuffer(args, obs_sp, share_sp, act_sp))
        self.total_steps = 0
        self.updates = 0
        self.history: list = []

    # -- on-policy separated-runner steps -------------------------------------
    def warmup(self) -> None:
        obs, share = self.menv.reset()
        for i, b in enumerate(self.buffer):
            b.share_obs[0] = share[i][None].copy()
            b.obs[0] = obs[i][None].copy()

    @torch.no_grad()
    def collect(self, step: int):
        vals, acts, logps, rnns, rnns_c = [], [], [], [], []
        for i in range(self.n_agents):
            self.trainer[i].prep_rollout()
            b = self.buffer[i]
            v, a, lp, rs, rsc = self.policy[i].get_actions(
                b.share_obs[step], b.obs[step], b.rnn_states[step],
                b.rnn_states_critic[step], b.masks[step])
            vals.append(_t2n(v))
            acts.append(_t2n(a))
            logps.append(_t2n(lp))
            rnns.append(_t2n(rs))
            rnns_c.append(_t2n(rsc))
        return vals, acts, logps, rnns, rnns_c

    def insert(self, obs, share, rewards, done, truncated, vals, acts, logps, rnns, rnns_c):
        masks = np.zeros((1, 1), np.float32) if done else np.ones((1, 1), np.float32)
        bad_masks = np.zeros((1, 1), np.float32) if truncated else np.ones((1, 1), np.float32)
        for i, b in enumerate(self.buffer):
            rs, rsc = rnns[i], rnns_c[i]
            if done:
                rs, rsc = np.zeros_like(rs), np.zeros_like(rsc)
            b.insert(share[i][None], obs[i][None], rs, rsc, acts[i], logps[i], vals[i],
                     rewards[i][None], masks, bad_masks=bad_masks)

    @torch.no_grad()
    def compute(self) -> None:
        for i, b in enumerate(self.buffer):
            self.trainer[i].prep_rollout()
            nv = self.policy[i].get_values(b.share_obs[-1], b.rnn_states_critic[-1], b.masks[-1])
            b.compute_returns(_t2n(nv), self.trainer[i].value_normalizer)

    def train(self) -> list:
        infos = []
        for i, b in enumerate(self.buffer):
            self.trainer[i].prep_training()
            info = self.trainer[i].train(b)
            infos.append({k: _scalar(v) for k, v in info.items()})
            b.after_update()
        return infos

    # -- main loop ------------------------------------------------------------
    def run(self, total_steps: Optional[int] = None) -> list:
        T = int(self.args.episode_length)
        total = int(total_steps or self.args.num_env_steps)
        n_updates = max(1, total // T)
        self.warmup()
        for u in range(n_updates):
            if self.args.use_linear_lr_decay:
                for pol in self.policy:
                    pol.lr_decay(u, n_updates)
            n_eps_before = len(self.menv.episodes)
            t0 = time.perf_counter()
            for step in range(T):
                vals, acts, logps, rnns, rnns_c = self.collect(step)
                obs, share, rewards, done, truncated, _ = self.menv.step(
                    [a[0] for a in acts])
                self.insert(obs, share, rewards, done, truncated, vals, acts, logps,
                            rnns, rnns_c)
            t_collect = time.perf_counter() - t0
            self.total_steps += T
            self.compute()
            infos = self.train()
            self.updates += 1
            eps = self.menv.episodes[n_eps_before:]
            stats = {k: float(np.mean([inf[k] for inf in infos])) for k in infos[0]}
            stats.update({
                "arm": self.cfg.arm, "seed": self.cfg.seed, "update": self.updates,
                "env_steps": self.total_steps,
                "steps_per_sec": T / max(t_collect, 1e-9),
                "episodes": len(eps),
                "ep_return_mean": float(np.mean([e["return"] for e in eps])) if eps else None,
                "train_p_mean": (float(np.mean([e["p"] for e in eps]))
                                 if eps and self.cfg.use_messages else None),
            })
            self.history.append(stats)
            if self.logger:
                self.logger(stats)
        return self.history

    # -- evaluation -----------------------------------------------------------
    def actors(self) -> DecentralizedActors:
        return DecentralizedActors(self.policy, self.args, self.device)

    def evaluate(self, env=None, p: float = 1.0, n_replicates: int = 10, env_seed: int = 0,
                 channel_seed: Optional[int] = None, force_mask: bool = False,
                 deterministic: bool = True, on_episode_end=None, max_steps=None) -> list:
        """Frozen-policy evaluation at delivery level ``p`` (see marl.mappo.evaluate)."""
        if channel_seed is None:
            channel_seed = int(stream(self.cfg.seed, STREAM_EVAL).generate_state(1)[0])
        return _evaluate(env or self.env, self.actors(), self.menv.pre, self.cfg.use_messages,
                         p, n_replicates, env_seed, channel_seed, force_mask, deterministic,
                         self.cfg.critic, on_episode_end, max_steps)

    # -- checkpoints ----------------------------------------------------------
    def _valuenorm_state(self, tr):
        vn = tr.value_normalizer
        if isinstance(vn, ValueNorm):
            # stored explicitly: on GPU ValueNorm's tensors are not registered
            # in its state_dict (it calls .to() on nn.Parameters)
            return {k: getattr(vn, k).detach().cpu().clone()
                    for k in ("running_mean", "running_mean_sq", "debiasing_term")}
        return None

    def state_dict(self) -> dict:
        return {
            "config": self.cfg.to_dict(),
            "onpolicy_commit": ONPOLICY_COMMIT,
            "actor_dims": self.menv.actor_dims, "critic_dims": self.menv.critic_dims,
            "actors": [p.actor.state_dict() for p in self.policy],
            "critics": [p.critic.state_dict() for p in self.policy],
            "value_norms": [self._valuenorm_state(t) for t in self.trainer],
            "preprocessor": self.menv.pre.state_dict(),
            "total_steps": self.total_steps, "updates": self.updates,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, env, logger=None) -> "OnPolicyRunner":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        d = dict(ck["config"])
        d["rnd_p_choices"] = tuple(d["rnd_p_choices"])
        runner = cls(env, MAPPOConfig(**d), logger)
        if (runner.menv.actor_dims != ck["actor_dims"]
                or runner.menv.critic_dims != ck["critic_dims"]):
            raise ValueError("checkpoint was trained on an env with different dimensions")
        for pol, a, c in zip(runner.policy, ck["actors"], ck["critics"]):
            pol.actor.load_state_dict(a)
            pol.critic.load_state_dict(c)
        for tr, vn in zip(runner.trainer, ck["value_norms"]):
            if vn is not None:
                with torch.no_grad():
                    for k, v in vn.items():
                        getattr(tr.value_normalizer, k).copy_(v.to(runner.device))
        runner.menv.pre.load_state_dict(ck["preprocessor"])
        runner.total_steps, runner.updates = ck["total_steps"], ck["updates"]
        return runner
