import re
import time
from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...buffers import NStepReplay, ReplayBuffer
from ..actorcritic_base import ActorCriticBase
from .noise import add_mixed_normal_noise, add_normal_noise
from .schedule_util import ExponentialSchedule, LinearSchedule
from .utils import RewardShaper, RunningMeanStd


def create_simple_mlp(in_dim, out_dim, hidden_layers, act_type="ELU", act_kwargs=dict(inplace=True)):
    layer_nums = [in_dim, *hidden_layers, out_dim]
    model = []
    for idx, (in_f, out_f) in enumerate(zip(layer_nums[:-1], layer_nums[1:])):
        model.append(nn.Linear(in_f, out_f))
        if idx < len(layer_nums) - 2:
            module = torch.nn.modules.activation
            Cls = getattr(module, act_type)
            act = Cls(**act_kwargs)
            model.append(act)
    return nn.Sequential(*model)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None):
        super().__init__()
        if isinstance(in_dim, Sequence):
            in_dim = in_dim[0]
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        self.net = create_simple_mlp(in_dim=in_dim, out_dim=out_dim, hidden_layers=hidden_layers)

    def forward(self, x):
        return self.net(x)


class TanhMLPPolicy(MLP):
    def forward(self, state):
        return super().forward(state).tanh()


class DoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        if isinstance(state_dim, Sequence):
            state_dim = state_dim[0]
        self.net_q1 = MLP(in_dim=state_dim + act_dim, out_dim=1)
        self.net_q2 = MLP(in_dim=state_dim + act_dim, out_dim=1)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x), self.net_q2(input_x)  # two Q values

    def get_q1(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        input_x = torch.cat((state, action), dim=1)
        return self.net_q1(input_x)


class DDPG(ActorCriticBase):
    def __init__(self, env, output_dir, full_cfg):
        self.network_config = full_cfg.agent.network
        self.ddpg_config = full_cfg.agent.ddpg
        self.num_actors = self.ddpg_config.num_actors
        super().__init__(env, output_dir, full_cfg)

        obs_dim = self.obs_space['obs']
        self.obs_dim = obs_dim
        self.actor = TanhMLPPolicy(obs_dim, self.action_dim)
        self.critic = DoubleQ(obs_dim, self.action_dim).to(self.device)

        print(self.actor)
        print(self.critic, '\n')

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.ddpg_config.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.ddpg_config.critic_lr)

        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor) if not self.ddpg_config.no_tgt_actor else self.actor

        if self.normalize_input:
            self.obs_rms = {
                k: RunningMeanStd(v, device=self.device) if re.match(self.input_keys_normalize, k) else nn.Identity()
                for k, v in self.obs_space.items()
            }
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        if self.ddpg_config.noise.decay == 'linear':
            self.noise_scheduler = LinearSchedule(
                start_val=self.ddpg_config.noise.std_max,
                end_val=self.ddpg_config.noise.std_min,
                total_iters=self.ddpg_config.noise.lin_decay_iters,
            )
        elif self.ddpg_config.noise.decay == 'exp':
            self.noise_scheduler = ExponentialSchedule(
                start_val=self.ddpg_config.noise.std_max,
                gamma=self.ddpg_config.exp_decay_rate,
                end_val=self.ddpg_config.noise.std_min,
            )
        else:
            self.noise_scheduler = None

        self.memory = ReplayBuffer(
            self.obs_space, self.action_dim, capacity=int(self.ddpg_config.memory_size), device=self.device
        )
        self.n_step_buffer = NStepReplay(
            self.obs_space, self.action_dim, self.num_actors, self.ddpg_config.nstep, device=self.device
        )

        self.reward_shaper = RewardShaper(**self.ddpg_config['reward_shaper'])

        self.epoch_num = -1
        self.global_steps = 0
        self.max_agent_steps = int(self.ddpg_config['max_agent_steps'])

    def get_noise_std(self):
        if self.noise_scheduler is None:
            return self.ddpg_config.noise.std_max
        else:
            return self.noise_scheduler.val()

    def update_noise(self):
        if self.noise_scheduler is not None:
            self.noise_scheduler.step()

    def get_actions(self, obs, sample=True):
        if self.normalize_input:
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        obs = obs['obs']
        actions = self.actor(obs)
        if sample:
            if self.ddpg_config.noise.type == 'fixed':
                actions = add_normal_noise(actions, std=self.get_noise_std(), out_bounds=[-1.0, 1.0])
            elif self.ddpg_config.noise.type == 'mixed':
                actions = add_mixed_normal_noise(
                    actions,
                    std_min=self.ddpg_config.noise.std_min,
                    std_max=self.ddpg_config.noise.std_max,
                    out_bounds=[-1.0, 1.0],
                )
            else:
                raise NotImplementedError
        return actions

    @torch.no_grad()
    def get_tgt_policy_actions(self, obs, sample=True):
        actions = self.actor_target(obs)
        if sample:
            actions = add_normal_noise(
                actions,
                std=self.ddpg_config.noise.tgt_pol_std,
                noise_bounds=[-self.ddpg_config.noise.tgt_pol_noise_bound, self.ddpg_config.noise.tgt_pol_noise_bound],
                out_bounds=[-1.0, 1.0],
            )
        return actions

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False) -> list:
        traj_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_actions = torch.empty((self.num_actors, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.num_actors, timesteps), device=self.device)
        traj_next_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_dones = torch.empty((self.num_actors, timesteps), device=self.device)

        obs = self.obs
        for i in range(timesteps):
            if self.normalize_input:
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
            if random:
                action = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                action = self.get_actions(obs, sample=True)

            next_obs, reward, done, info = env.step(action)
            next_obs = self._convert_obs(next_obs)

            done_indices = torch.where(done)[0].tolist()
            self.update_tracker(reward, done_indices, info)
            if self.ddpg_config.handle_timeout:
                done = handle_timeout(done, info)

            for k, v in obs.items():
                traj_obs[k][:, i] = v
            traj_actions[:, i] = action
            traj_dones[:, i] = done
            traj_rewards[:, i] = reward
            for k, v in next_obs.items():
                traj_next_obs[k][:, i] = v.clone()
            obs = next_obs
        self.obs = obs

        traj_rewards = self.reward_shaper(traj_rewards.reshape(self.num_actors, timesteps, 1))
        traj_dones = traj_dones.reshape(self.num_actors, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)

        return data, timesteps * self.num_actors

    def update_net(self, memory):
        critic_loss_list = list()
        actor_loss_list = list()
        for i in range(self.ddpg_config.mini_epochs):
            obs, action, reward, next_obs, done = memory.sample_batch(self.ddpg_config.batch_size)
            if self.normalize_input:
                obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
                next_obs = {k: self.obs_rms[k].normalize(v) for k, v in next_obs.items()}
            obs, next_obs = obs['obs'], next_obs['obs']
            critic_loss, critic_grad_norm = self.update_critic(obs, action, reward, next_obs, done)
            critic_loss_list.append(critic_loss)

            actor_loss, actor_grad_norm = self.update_actor(obs)
            actor_loss_list.append(actor_loss)

            soft_update(self.critic_target, self.critic, self.ddpg_config.tau)
            if not self.ddpg_config.no_tgt_actor:
                soft_update(self.actor_target, self.actor, self.ddpg_config.tau)

        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "metrics/episode_rewards": self.episode_rewards.mean(),
            "metrics/episode_lengths": self.episode_lengths.mean(),
        }
        # self.add_info_tracker_log(log_info)
        return log_info

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_actions = self.get_tgt_policy_actions(next_obs)
            target_Q = self.critic_target.get_q_min(next_obs, next_actions)
            target_Q = reward + (1 - done) * (self.ddpg_config.gamma**self.ddpg_config.nstep) * target_Q

        current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        grad_norm = self.optimizer_update(self.critic_optimizer, critic_loss)

        return critic_loss.item(), grad_norm

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        action = self.actor(obs)
        Q = self.critic.get_q_min(obs, action)
        actor_loss = -Q.mean()
        grad_norm = self.optimizer_update(self.actor_optimizer, actor_loss)
        self.critic.requires_grad_(True)
        return actor_loss.item(), grad_norm

    def train(self):
        _t = time.perf_counter()
        _last_t = _t

        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.ones((self.num_actors,), dtype=torch.bool, device=self.device)

        self.set_eval()
        trajectory, steps = self.explore_env(self.env, self.ddpg_config.warm_up, random=True)
        self.memory.add_to_buffer(trajectory)
        self.global_steps += steps

        while self.global_steps < self.max_agent_steps:
            self.epoch_num += 1
            self.set_eval()
            trajectory, steps = self.explore_env(self.env, self.ddpg_config.horizon_len, random=False)
            self.global_steps += steps
            self.memory.add_to_buffer(trajectory)

            self.set_train()
            metrics = self.update_net(self.memory)
            self.write_metrics(self.global_steps, metrics)

    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if self.ddpg_config.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                parameters=optimizer.param_groups[0]["params"],
                max_norm=self.ddpg_config.max_grad_norm,
            )
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

        self.obs_rms.eval()

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

        self.obs_rms.eval()

    def restore_train(self, f):
        if not f:
            return


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


def handle_timeout(dones, info):
    timeout_key = 'TimeLimit.truncated'
    timeout_envs = None
    if timeout_key in info:
        timeout_envs = info[timeout_key]
    if timeout_envs is not None:
        dones = dones * (~timeout_envs)
    return dones
