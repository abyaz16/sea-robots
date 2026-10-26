"""Actor-critic network and observation normaliser for the fast loop.

One network serves every ROV. The observation is egocentric and carries a
role one-hot, so a single set of weights controls lifters and the connector
vehicle, and the swarm size is not baked into the architecture. That is both
cheaper to train and the right abstraction: a vehicle's control problem is
local, and what differs between roles is the task, not the physics.

Deliberately small. The policy has to run at 50 Hz on a vehicle with a power
budget, and a 2x256 MLP is roughly 70k parameters -- microseconds of inference
on any embedded accelerator. A large network here would undercut the whole
argument that this can live onboard.
"""

from typing import Tuple

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Welford running mean/variance normaliser, frozen at evaluation time.

    Observation scales here span metres, m/s, radians and normalised
    tensions. Without this, PPO spends its first million steps learning the
    scale of its own inputs instead of the task.
    """

    def __init__(self, dim: int, clip: float = 8.0) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        n = torch.tensor(float(x.shape[0]), device=x.device)
        mean = x.mean(0)
        var = x.var(0, unbiased=False)
        delta = mean - self.mean
        tot = self.count + n
        new_mean = self.mean + delta * n / tot
        m_a = self.var * self.count
        m_b = var * n
        new_var = (m_a + m_b + delta.pow(2) * self.count * n / tot) / tot
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / (self.var + 1e-8).sqrt()
        return z.clamp(-self.clip, self.clip)


def mlp(sizes, act=nn.Tanh, out_act=None) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian policy with a state-independent log-std, plus a value head.

    tanh activations rather than ReLU: the control output needs to be smooth
    in the observation, because a policy that is piecewise-linear with sharp
    kinks produces thruster chatter, and thruster chatter both wastes energy
    and excites the cable.
    """

    def __init__(self, obs_dim: int, act_dim: int = 4, hidden: int = 256,
                 log_std_init: float = -0.7) -> None:
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.pi = mlp([obs_dim, hidden, hidden, act_dim])
        self.vf = mlp([obs_dim, hidden, hidden, 1])
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        # small final-layer gain: start near zero action, so the first
        # rollouts explore around "do nothing" rather than slamming thrusters
        for m in (self.pi, self.vf):
            last = [l for l in m if isinstance(l, nn.Linear)][-1]
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        z = self.norm(obs)
        mean = torch.tanh(self.pi(z))
        return torch.distributions.Normal(mean, self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.vf(self.norm(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        logp = d.log_prob(a).sum(-1)
        return a.clamp(-1.0, 1.0), logp, self.value(obs)

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        logp = d.log_prob(act).sum(-1)
        ent = d.entropy().sum(-1)
        return logp, ent, self.value(obs)
