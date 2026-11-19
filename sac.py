"""
Soft Actor-Critic (Haarnoja et al. 2018, v2 with twin critics and automatic
entropy tuning), implemented from scratch in PyTorch.

  - squashed-Gaussian actor (tanh), reparameterised sampling
  - twin Q critics + Polyak-averaged target critics
  - learned temperature alpha with target entropy = -|A|
  - only true termination (not time-limit truncation) cuts the bootstrap
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(inp, out, hidden=256, layers=2):
    dims, mods = [inp] + [hidden] * layers, []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.ReLU()]
    mods.append(nn.Linear(dims[-1], out))
    return nn.Sequential(*mods)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=1_000_000, device="cpu"):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.d = np.zeros((size, 1), np.float32)
        self.size, self.ptr, self.n, self.device = size, 0, 0, device

    def add(self, o, a, r, o2, terminated):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, float(terminated)
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, batch)
        f = lambda x: torch.as_tensor(x[idx], device=self.device)
        return f(self.o), f(self.a), f(self.r), f(self.o2), f(self.d)


class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def forward(self, obs, deterministic=False, with_logp=True):
        mu, log_std = self.net(obs).chunk(2, -1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = None
        if with_logp:
            logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
            # tanh change of variables (numerically stable form)
            logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x), self.q2(x)


class SAC:
    def __init__(self, obs_dim, act_dim, device="cpu", hidden=256, gamma=0.99,
                 tau=0.005, lr=3e-4, init_alpha=0.2, target_entropy=None):
        self.device, self.gamma, self.tau = torch.device(device), gamma, tau
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = TwinQ(obs_dim, act_dim, hidden).to(self.device)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = -float(act_dim) if target_entropy is None else target_entropy
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic, with_logp=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch):
        o, a, r, o2, d = batch
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            tq1, tq2 = self.critic_targ(o2, a2)
            y = r + self.gamma * (1 - d) * (torch.min(tq1, tq2) - alpha * logp2.unsqueeze(-1))
        q1, q2 = self.critic(o, a)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        a_new, logp = self.actor(o)
        q1n, q2n = self.critic(o, a_new)
        pi_loss = (alpha * logp - torch.min(q1n, q2n).squeeze(-1)).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return dict(q_loss=q_loss.item(), pi_loss=pi_loss.item(),
                    alpha=alpha.item(), entropy=-logp.mean().item(), q=q1.mean().item())

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "log_alpha": self.log_alpha.detach().cpu()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha.data.copy_(ck["log_alpha"].to(self.device))
