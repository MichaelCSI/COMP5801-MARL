"""MAPPO / IPPO trainer with the peer-SoC message channel (torch).

    from marl.mappo import MAPPOConfig, MAPPOTrainer
    cfg = MAPPOConfig.for_arm("M1-RND", seed=0, total_steps=100_000)
    trainer = MAPPOTrainer(env, cfg, logger=print)
    trainer.train()
    trainer.save("runs/m1rnd_s0.pt")
    records = trainer.evaluate(test_env, p=0.5, n_replicates=10)

``env`` is anything that passes ``marl.envs.check_env_contract``.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from ..comms.channel import DeliverySchedule, PeerSoCChannel, message_dim
from .buffer import RolloutBuffer
from .config import (STREAM_CHANNEL, STREAM_ENV, STREAM_SCHEDULE, STREAM_TORCH,
                     MAPPOConfig, stream)
from .networks import MAPPOPolicy
from .normalize import Preprocessor
from .rollout import RolloutWorker, evaluate


def explained_variance(y_pred, y_true) -> float:
    var = np.var(y_true)
    return float("nan") if var == 0 else float(1 - np.var(y_true - y_pred) / var)


class MAPPOTrainer:
    def __init__(self, env, cfg: MAPPOConfig, logger: Optional[Callable[[dict], None]] = None):
        cfg.validate()
        self.env, self.cfg, self.logger = env, cfg, logger
        n = env.n_agents
        self.n_agents = n

        torch_seed = int(stream(cfg.seed, STREAM_TORCH).generate_state(1)[0])
        torch.manual_seed(torch_seed)
        self.rng = np.random.default_rng(stream(cfg.seed, STREAM_TORCH))

        msg = message_dim(n) if cfg.use_messages else 0
        self.actor_dims = [d + msg for d in env.obs_dims]
        self.critic_dims = [env.state_dim] if cfg.critic == "central" else list(self.actor_dims)

        self.pre = Preprocessor(env.obs_dims, env.state_dim, cfg.normalize_obs, cfg.obs_clip)
        self.policy = MAPPOPolicy(self.actor_dims, env.action_dims, self.critic_dims,
                                  cfg.hidden_sizes, cfg.init_log_std, cfg.normalize_value,
                                  cfg.device)
        # one optimizer per actor: independent learners, independent grad clipping
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.actor_lr, eps=1e-5)
                           for a in self.policy.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.critic_lr, eps=1e-5)
                            for c in self.policy.critics]

        channel = schedule = None
        if cfg.use_messages:
            channel = PeerSoCChannel(n, 1.0, seed=stream(cfg.seed, STREAM_CHANNEL),
                                     force_mask=cfg.force_mask_train)
            if cfg.train_schedule == "fixed":
                schedule = DeliverySchedule("fixed", p=cfg.train_p)
            else:
                schedule = DeliverySchedule("random", choices=tuple(cfg.rnd_p_choices),
                                            seed=stream(cfg.seed, STREAM_SCHEDULE))
        self.worker = RolloutWorker(env, self.policy, self.pre, cfg.use_messages, channel,
                                    schedule, cfg.critic, stream(cfg.seed, STREAM_ENV),
                                    cfg.bootstrap_on_truncation)
        self.buffer = RolloutBuffer(cfg.rollout_steps, self.actor_dims, env.action_dims,
                                    self.critic_dims)
        self.updates = 0
        self.history: list = []

    # -- training -------------------------------------------------------------
    def train(self, total_steps: Optional[int] = None) -> list:
        total = total_steps or self.cfg.total_steps
        n_updates = max(1, total // self.cfg.rollout_steps)
        for u in range(n_updates):
            if self.cfg.lr_anneal:
                frac = 1.0 - u / n_updates
                for o in self.actor_opts:
                    o.param_groups[0]["lr"] = frac * self.cfg.actor_lr
                for o in self.critic_opts:
                    o.param_groups[0]["lr"] = frac * self.cfg.critic_lr
            t0 = time.perf_counter()
            roll = self.worker.collect(self.buffer)
            t_collect = time.perf_counter() - t0
            stats = self.update()
            eps = roll["episodes"]
            stats.update({
                "arm": self.cfg.arm, "seed": self.cfg.seed, "update": self.updates,
                "env_steps": self.worker.total_steps,
                "steps_per_sec": self.cfg.rollout_steps / max(t_collect, 1e-9),
                "episodes": len(eps),
                "ep_return_mean": float(np.mean([e["return"] for e in eps])) if eps else None,
                "train_p_mean": (float(np.mean([e["p"] for e in eps]))
                                 if eps and self.cfg.use_messages else None),
            })
            self.history.append(stats)
            if self.logger:
                self.logger(stats)
        return self.history

    def update(self) -> dict:
        cfg, buf = self.cfg, self.buffer
        adv, ret = buf.compute(cfg.gamma, cfg.gae_lambda)       # (T, K)
        T = buf.ptr
        # central critic: one advantage shared by all agents (shared reward);
        # local critics: agent i uses its own critic's advantage
        if cfg.critic == "central":
            adv_agent = np.repeat(adv[:, :1], self.n_agents, axis=1)
        else:
            adv_agent = adv
        adv_agent = (adv_agent - adv_agent.mean(0)) / (adv_agent.std(0) + 1e-8)

        if cfg.normalize_value:
            for k, rms in enumerate(self.policy.value_rms):
                rms.update(ret[:, k])
        old_v = buf.values[:T]
        if cfg.normalize_value:
            ret_n = np.stack([self.policy.value_rms[k].normalize(ret[:, k])
                              for k in range(buf.K)], 1)
            old_v_n = np.stack([self.policy.value_rms[k].normalize(old_v[:, k])
                                for k in range(buf.K)], 1)
        else:
            ret_n, old_v_n = ret, old_v

        t = self.policy._t
        A_in = [t(x[:T]) for x in buf.actor_inputs]
        A_act = [t(x[:T]) for x in buf.actions]
        C_in = [t(x[:T]) for x in buf.critic_inputs]
        old_logp, ADV = t(buf.logp[:T]), t(adv_agent)
        RET, OLDV = t(ret_n), t(old_v_n)

        logs = {"pi_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
        stop = False
        for _ in range(cfg.ppo_epochs):
            for mb in buf.minibatches(cfg.num_minibatches, self.rng):
                mb = torch.as_tensor(mb, device=self.policy.device)
                kls = []
                for i, (actor, opt) in enumerate(zip(self.policy.actors, self.actor_opts)):
                    d = actor.dist(A_in[i][mb])
                    logp = d.log_prob(A_act[i][mb]).sum(-1)
                    ratio = (logp - old_logp[mb, i]).exp()
                    a = ADV[mb, i]
                    pg = -torch.min(ratio * a,
                                    ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a).mean()
                    ent = d.entropy().sum(-1).mean()
                    loss = pg - cfg.entropy_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        lr = logp - old_logp[mb, i]
                        kl = float(((lr.exp() - 1) - lr).mean())
                        kls.append(kl)
                        logs["pi_loss"].append(float(pg))
                        logs["entropy"].append(float(ent))
                        logs["approx_kl"].append(kl)
                        logs["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip_eps)
                                                      .float().mean()))
                for k, (critic, opt) in enumerate(zip(self.policy.critics, self.critic_opts)):
                    v = critic(C_in[k][mb])
                    v_clip = OLDV[mb, k] + (v - OLDV[mb, k]).clamp(-cfg.value_clip_eps,
                                                                   cfg.value_clip_eps)
                    vl = torch.max((v - RET[mb, k]) ** 2, (v_clip - RET[mb, k]) ** 2).mean()
                    opt.zero_grad()
                    (cfg.value_coef * vl).backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                    opt.step()
                    logs["v_loss"].append(float(vl))
                if cfg.target_kl is not None and np.mean(kls) > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        self.updates += 1
        out = {k: float(np.mean(v)) for k, v in logs.items()}
        out["explained_var"] = explained_variance(old_v.ravel(), ret.ravel())
        out["log_std"] = float(np.mean([a.log_std.detach().cpu().numpy().mean()
                                        for a in self.policy.actors]))
        return out

    # -- evaluation -----------------------------------------------------------
    def evaluate(self, env=None, p: float = 1.0, n_replicates: int = 10, env_seed: int = 0,
                 channel_seed: Optional[int] = None, force_mask: bool = False,
                 deterministic: bool = True, on_episode_end=None, max_steps=None) -> list:
        """Evaluate the trained (frozen) policy at delivery level ``p``.

        ``channel_seed`` defaults to a value derived from the training seed,
        independent of the arm and of ``p``, so paired comparisons line up.
        """
        if channel_seed is None:
            channel_seed = int(np.random.SeedSequence([self.cfg.seed, 4]).generate_state(1)[0])
        return evaluate(env or self.env, self.policy, self.pre, self.cfg.use_messages, p,
                        n_replicates, env_seed, channel_seed, force_mask, deterministic,
                        on_episode_end, max_steps)

    # -- checkpoints ----------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "config": self.cfg.to_dict(),
            "actor_dims": self.actor_dims, "critic_dims": self.critic_dims,
            "policy": self.policy.state_dict(),
            "policy_extra": self.policy.extra_state(),
            "preprocessor": self.pre.state_dict(),
            "updates": self.updates,
            "env_steps": self.worker.total_steps,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, env, logger=None) -> "MAPPOTrainer":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg_d = dict(ck["config"])
        for key in ("hidden_sizes", "rnd_p_choices"):
            cfg_d[key] = tuple(cfg_d[key])
        tr = cls(env, MAPPOConfig(**cfg_d), logger)
        if tr.actor_dims != ck["actor_dims"] or tr.critic_dims != ck["critic_dims"]:
            raise ValueError("checkpoint was trained on an env with different dimensions")
        tr.policy.load_state_dict(ck["policy"])
        tr.policy.load_extra_state(ck["policy_extra"])
        tr.pre.load_state_dict(ck["preprocessor"])
        tr.updates = ck["updates"]
        return tr
