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

    config = MAPPOConfig.for_arm("M1-RND", seed=0, total_steps=200_000)
    runner = OnPolicyRunner(env, config, logger=print)
    runner.run()
    runner.save("runs/M1-RND_s0.pt")
    recs = runner.evaluate(test_env, p=0.5, n_replicates=10, env_seed=TEST_SEED)
"""
from __future__ import annotations

import time
from typing import Callable, NamedTuple, Optional

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


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _to_float(value) -> float:
    """on-policy's train() returns a mix of tensors and floats."""
    return float(value.item()) if hasattr(value, "item") else float(value)


class StepOutputs(NamedTuple):
    """What every agent's policy produced for one step (one list entry per agent)."""
    values: list
    actions: list
    action_log_probs: list
    actor_rnn_states: list    # unused by the MLP policy, but the buffer stores them
    critic_rnn_states: list


class DecentralizedActors:
    """Execution-time policy: actor i sees only its own input (CTDE)."""

    def __init__(self, policies, args, device):
        self.policies, self.args, self.device = policies, args, device

    @torch.no_grad()
    def act(self, actor_inputs, deterministic: bool = True):
        per_agent_actions = []
        for policy, agent_input in zip(self.policies, actor_inputs):
            policy.actor.eval()
            # on-policy's act() takes RNN state and masks even for MLP policies
            rnn_state = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), np.float32)
            masks = np.ones((1, 1), np.float32)
            action, _ = policy.act(np.asarray(agent_input, np.float32)[None], rnn_state, masks,
                                   deterministic=deterministic)
            per_agent_actions.append(_to_numpy(action)[0])

        # pad to a rectangular (n_agents, max action dim) array
        width = max(action.shape[0] for action in per_agent_actions)
        actions = np.zeros((len(per_agent_actions), width), np.float32)
        for agent, action in enumerate(per_agent_actions):
            actions[agent, : action.shape[0]] = action
        return actions, np.zeros(len(per_agent_actions), np.float32)


class OnPolicyRunner:
    def __init__(self, env, config: MAPPOConfig,
                 logger: Optional[Callable[[dict], None]] = None):
        self.config, self.logger = config, logger
        self.args = args = config.to_onpolicy_args()

        torch.manual_seed(int(stream(config.seed, STREAM_TORCH).generate_state(1)[0]))
        torch.set_num_threads(int(args.n_training_threads))
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device(config.device if use_cuda else "cpu")

        self.env = env
        self.message_env = make_message_env(env, config)
        self.n_agents = env.n_agents
        self.use_centralized_V = args.use_centralized_V

        # One on-policy policy (actor + critic), trainer and buffer per building
        self.policies, self.trainers, self.buffers = [], [], []
        for agent in range(self.n_agents):
            obs_space = self.message_env.observation_space[agent]
            critic_space = self.message_env.share_observation_space[agent]
            action_space = self.message_env.action_space[agent]

            policy = R_MAPPOPolicy(args, obs_space, critic_space, action_space,
                                   device=self.device)
            self.policies.append(policy)
            self.trainers.append(R_MAPPO(args, policy, device=self.device))
            self.buffers.append(SeparatedReplayBuffer(args, obs_space, critic_space,
                                                      action_space))

        self.total_steps = 0
        self.updates = 0
        self.history: list = []

    # -- on-policy separated-runner steps -------------------------------------
    def warmup(self) -> None:
        """Put the first episode's inputs in slot 0 of every buffer."""
        actor_inputs, critic_inputs = self.message_env.reset()
        for agent, buffer in enumerate(self.buffers):
            buffer.share_obs[0] = critic_inputs[agent][None].copy()
            buffer.obs[0] = actor_inputs[agent][None].copy()

    @torch.no_grad()
    def collect(self, step: int) -> StepOutputs:
        """Each agent's action, log-prob and value for buffer slot ``step``."""
        outputs = StepOutputs([], [], [], [], [])
        for agent in range(self.n_agents):
            self.trainers[agent].prep_rollout()
            buffer = self.buffers[agent]
            value, action, log_prob, actor_rnn, critic_rnn = self.policies[agent].get_actions(
                buffer.share_obs[step], buffer.obs[step], buffer.rnn_states[step],
                buffer.rnn_states_critic[step], buffer.masks[step])

            outputs.values.append(_to_numpy(value))
            outputs.actions.append(_to_numpy(action))
            outputs.action_log_probs.append(_to_numpy(log_prob))
            outputs.actor_rnn_states.append(_to_numpy(actor_rnn))
            outputs.critic_rnn_states.append(_to_numpy(critic_rnn))
        return outputs

    def insert(self, actor_inputs, critic_inputs, rewards, done: bool, time_limit: bool,
               outputs: StepOutputs) -> None:
        """Store one step in every agent's buffer."""
        # masks = 0: episode ended here; bad_masks = 0: it ended by time limit
        masks = np.zeros((1, 1), np.float32) if done else np.ones((1, 1), np.float32)
        bad_masks = np.zeros((1, 1), np.float32) if time_limit else np.ones((1, 1), np.float32)

        for agent, buffer in enumerate(self.buffers):
            actor_rnn = outputs.actor_rnn_states[agent]
            critic_rnn = outputs.critic_rnn_states[agent]
            if done:
                actor_rnn, critic_rnn = np.zeros_like(actor_rnn), np.zeros_like(critic_rnn)

            buffer.insert(critic_inputs[agent][None], actor_inputs[agent][None],
                          actor_rnn, critic_rnn, outputs.actions[agent],
                          outputs.action_log_probs[agent], outputs.values[agent],
                          rewards[agent][None], masks, bad_masks=bad_masks)

    @torch.no_grad()
    def compute(self) -> None:
        """GAE returns in every buffer, bootstrapped from the state after the rollout."""
        for agent, buffer in enumerate(self.buffers):
            self.trainers[agent].prep_rollout()
            next_value = self.policies[agent].get_values(
                buffer.share_obs[-1], buffer.rnn_states_critic[-1], buffer.masks[-1])
            buffer.compute_returns(_to_numpy(next_value), self.trainers[agent].value_normalizer)

    def train(self) -> list:
        """One PPO update per agent; returns each agent's on-policy training stats."""
        per_agent_stats = []
        for agent, buffer in enumerate(self.buffers):
            self.trainers[agent].prep_training()
            train_info = self.trainers[agent].train(buffer)
            per_agent_stats.append({name: _to_float(value) for name, value in train_info.items()})
            buffer.after_update()
        return per_agent_stats

    # -- main loop ------------------------------------------------------------
    def run(self, total_steps: Optional[int] = None) -> list:
        rollout_length = int(self.args.episode_length)   # on-policy's name for it
        total = int(total_steps or self.args.num_env_steps)
        n_updates = max(1, total // rollout_length)
        self.warmup()

        for update in range(n_updates):
            if self.args.use_linear_lr_decay:
                for policy in self.policies:
                    policy.lr_decay(update, n_updates)

            # collect one rollout
            episodes_before = len(self.message_env.episodes)
            collect_start = time.perf_counter()
            for step in range(rollout_length):
                outputs = self.collect(step)
                actor_inputs, critic_inputs, rewards, done, time_limit, _ = self.message_env.step(
                    [action[0] for action in outputs.actions])
                self.insert(actor_inputs, critic_inputs, rewards, done, time_limit, outputs)
            collect_seconds = time.perf_counter() - collect_start
            self.total_steps += rollout_length

            # learn from it
            self.compute()
            per_agent_stats = self.train()
            self.updates += 1

            # log: on-policy's stats averaged over agents, plus episode stats
            new_episodes = self.message_env.episodes[episodes_before:]
            stats = {name: float(np.mean([agent_stats[name] for agent_stats in per_agent_stats]))
                     for name in per_agent_stats[0]}
            stats.update({
                "arm": self.config.arm, "seed": self.config.seed, "update": self.updates,
                "env_steps": self.total_steps,
                "steps_per_sec": rollout_length / max(collect_seconds, 1e-9),
                "episodes": len(new_episodes),
                "ep_return_mean": (float(np.mean([ep["return"] for ep in new_episodes]))
                                   if new_episodes else None),
                "train_p_mean": (float(np.mean([ep["p"] for ep in new_episodes]))
                                 if new_episodes and self.config.use_messages else None),
            })
            self.history.append(stats)
            if self.logger:
                self.logger(stats)
        return self.history

    # -- evaluation -----------------------------------------------------------
    def actors(self) -> DecentralizedActors:
        return DecentralizedActors(self.policies, self.args, self.device)

    def evaluate(self, env=None, p: float = 1.0, n_replicates: int = 10, env_seed: int = 0,
                 channel_seed: Optional[int] = None, force_mask: bool = False,
                 deterministic: bool = True, on_episode_end=None, max_steps=None) -> list:
        """Frozen-policy evaluation at delivery level ``p`` (see marl.mappo.evaluate)."""
        if channel_seed is None:
            channel_seed = int(stream(self.config.seed, STREAM_EVAL).generate_state(1)[0])
        return _evaluate(env or self.env, self.actors(), self.message_env.preprocessor,
                         self.config.use_messages, p, n_replicates, env_seed, channel_seed,
                         force_mask, deterministic, self.config.critic, on_episode_end,
                         max_steps)

    # -- checkpoints ----------------------------------------------------------
    VALUENORM_TENSORS = ("running_mean", "running_mean_sq", "debiasing_term")

    def _valuenorm_state(self, trainer):
        value_norm = trainer.value_normalizer
        if not isinstance(value_norm, ValueNorm):
            return None
        # stored explicitly: on GPU ValueNorm's tensors are not registered
        # in its state_dict (it calls .to() on nn.Parameters)
        return {name: getattr(value_norm, name).detach().cpu().clone()
                for name in self.VALUENORM_TENSORS}

    def state_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "onpolicy_commit": ONPOLICY_COMMIT,
            "actor_dims": self.message_env.actor_dims,
            "critic_dims": self.message_env.critic_dims,
            "actors": [policy.actor.state_dict() for policy in self.policies],
            "critics": [policy.critic.state_dict() for policy in self.policies],
            "value_norms": [self._valuenorm_state(trainer) for trainer in self.trainers],
            "preprocessor": self.message_env.preprocessor.state_dict(),
            "total_steps": self.total_steps,
            "updates": self.updates,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, env, logger=None) -> "OnPolicyRunner":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config_fields = dict(checkpoint["config"])
        config_fields["rnd_p_choices"] = tuple(config_fields["rnd_p_choices"])  # saved as list
        runner = cls(env, MAPPOConfig(**config_fields), logger)

        if (runner.message_env.actor_dims != checkpoint["actor_dims"]
                or runner.message_env.critic_dims != checkpoint["critic_dims"]):
            raise ValueError("checkpoint was trained on an env with different dimensions")

        for policy, actor_weights, critic_weights in zip(
                runner.policies, checkpoint["actors"], checkpoint["critics"]):
            policy.actor.load_state_dict(actor_weights)
            policy.critic.load_state_dict(critic_weights)

        for trainer, saved_norm in zip(runner.trainers, checkpoint["value_norms"]):
            if saved_norm is None:
                continue
            with torch.no_grad():
                for name, tensor in saved_norm.items():
                    getattr(trainer.value_normalizer, name).copy_(tensor.to(runner.device))

        runner.message_env.preprocessor.load_state_dict(checkpoint["preprocessor"])
        runner.total_steps, runner.updates = checkpoint["total_steps"], checkpoint["updates"]
        return runner
