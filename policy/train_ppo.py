"""PPO training entrypoint. Runs locally on CPU and on Nebius AI Cloud GPUs.

Usage
-----
    python -m policy.train_ppo --steps 2000000 --envs 256
    python -m policy.train_ppo --steps 10000000 --envs 4096 --device cuda

The only difference between the two is `--envs` and `--device`; the physics,
the policy and the curriculum are identical. That is the point of batching
the simulator in PyTorch rather than wrapping a C++ engine: the same code
path scales from a laptop to a GPU node.

Each ROV contributes a sample, so a run of N environment steps yields 4N
policy samples. Step counts in the logs are environment steps unless stated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from policy.curriculum import Stage, default_curriculum, describe
from policy.nets import ActorCritic
from sim.env import RepairEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Leviathan fast loop")
    p.add_argument("--steps", type=int, default=2_000_000,
                   help="total environment steps")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--rollout", type=int, default=32,
                   help="control steps per PPO update")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=3e-3)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/runs/ppo")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    return p.parse_args()


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if args.threads:
            torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)

        self.stages = default_curriculum()
        self.stage_i = 0
        self.stage_steps = 0
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)

        st = self.stages[0]
        self.env = RepairEnv(batch=args.envs, profile=st.profile,
                             device=self.device, seed=args.seed,
                             task=st.task(), use_sonar=st.use_sonar)
        self.net = ActorCritic(self.env.obs_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, eps=1e-5)

        self.obs = self.env.reset()
        self.global_step = 0
        self.history: list[dict] = []
        # rolling episode bookkeeping
        self.ep_success: list[float] = []
        self.ep_return = torch.zeros(args.envs, device=self.device)
        self.ep_returns: list[float] = []

    # ------------------------------------------------------------- curriculum

    def maybe_advance_stage(self) -> bool:
        st = self.stages[self.stage_i]
        recent = self.ep_success[-200:]
        rate = sum(recent) / max(len(recent), 1)
        ready = (len(recent) >= 60 and rate >= st.advance_success) \
            or self.stage_steps >= st.max_steps
        if not ready or self.stage_i + 1 >= len(self.stages):
            return False
        self.stage_i += 1
        self.stage_steps = 0
        nxt = self.stages[self.stage_i]
        print("\n>>> stage %d -> %d  (%s)  success=%.2f  %s" % (
            self.stage_i - 1, self.stage_i, nxt.name, rate, nxt.note))
        # rebuild the env with the new task spec, carrying the policy over
        self.env = RepairEnv(batch=self.args.envs, profile=nxt.profile,
                             device=self.device,
                             seed=self.args.seed + 100 * self.stage_i,
                             task=nxt.task(), use_sonar=nxt.use_sonar)
        self.obs = self.env.reset()
        self.ep_success.clear()
        return True

    # ---------------------------------------------------------------- rollout

    def collect(self) -> dict:
        a = self.args
        B, K = a.envs, self.env.n_rov
        T = a.rollout
        dev = self.device

        obs_buf = torch.zeros(T, B, K, self.env.obs_dim, device=dev)
        act_buf = torch.zeros(T, B, K, 4, device=dev)
        logp_buf = torch.zeros(T, B, K, device=dev)
        val_buf = torch.zeros(T, B, K, device=dev)
        rew_buf = torch.zeros(T, B, K, device=dev)
        done_buf = torch.zeros(T, B, device=dev)

        for t in range(T):
            with torch.no_grad():
                act, logp, val = self.net.act(self.obs)
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = \
                self.obs, act, logp, val

            nxt, rew, done, info = self.env.step(act)
            rew_buf[t] = rew
            done_buf[t] = done.to(rew.dtype)
            self.ep_return += rew.mean(-1)

            if bool(done.any()):
                for s in info["success"][done].tolist():
                    self.ep_success.append(float(s))
                self.ep_returns.extend(self.ep_return[done].tolist())
                self.ep_return = torch.where(done, torch.zeros_like(self.ep_return),
                                             self.ep_return)
                nxt = self.env.reset(done)
            self.obs = nxt

        with torch.no_grad():
            last_val = self.net.value(self.obs)

        # GAE-lambda. The done flag is per environment, so it broadcasts
        # across the ROVs of that environment -- they share an episode.
        adv = torch.zeros_like(rew_buf)
        gae = torch.zeros(B, K, device=dev)
        for t in reversed(range(T)):
            nonterm = (1.0 - done_buf[t]).unsqueeze(-1)
            nv = last_val if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + a.gamma * nv * nonterm - val_buf[t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val_buf

        self.global_step += T * B
        self.stage_steps += T * B
        self.net.norm.update(obs_buf)
        flat = lambda x, d: x.reshape(T * B * K, *d)
        return dict(obs=flat(obs_buf, (self.env.obs_dim,)),
                    act=flat(act_buf, (4,)),
                    logp=flat(logp_buf, ()), adv=flat(adv, ()),
                    ret=flat(ret, ()), val=flat(val_buf, ()))

    # ----------------------------------------------------------------- update

    def update(self, batch: dict) -> dict:
        a = self.args
        n = batch["obs"].shape[0]
        mb = max(1, n // a.minibatches)
        adv = batch["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        stats = dict(pi_loss=0.0, vf_loss=0.0, entropy=0.0, kl=0.0, clipfrac=0.0)
        count = 0
        for _ in range(a.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                logp, ent, val = self.net.evaluate(batch["obs"][idx],
                                                   batch["act"][idx])
                ratio = (logp - batch["logp"][idx]).exp()
                a_i = adv[idx]
                p1 = ratio * a_i
                p2 = ratio.clamp(1 - a.clip, 1 + a.clip) * a_i
                pi_loss = -torch.min(p1, p2).mean()
                vf_loss = (val - batch["ret"][idx]).pow(2).mean()
                loss = pi_loss + a.vf_coef * vf_loss - a.ent_coef * ent.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), a.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["pi_loss"] += float(pi_loss)
                    stats["vf_loss"] += float(vf_loss)
                    stats["entropy"] += float(ent.mean())
                    stats["kl"] += float((batch["logp"][idx] - logp).mean())
                    stats["clipfrac"] += float(
                        ((ratio - 1).abs() > a.clip).float().mean())
                count += 1
        return {k: v / max(count, 1) for k, v in stats.items()}

    # ------------------------------------------------------------------- loop

    def run(self) -> None:
        a = self.args
        print("device: %s | obs_dim %d | %d envs x %d ROVs | %d threads" % (
            self.device, self.env.obs_dim, a.envs, self.env.n_rov,
            torch.get_num_threads()))
        print(describe(self.stages))
        print("\n%8s %7s %9s %8s %8s %7s %7s %7s" % (
            "step", "stage", "return", "success", "entropy", "kl", "vf", "sps"))

        t0 = time.perf_counter()
        it = 0
        while self.global_step < a.steps:
            batch = self.collect()
            st = self.update(batch)
            self.maybe_advance_stage()
            it += 1

            if it % a.log_every == 0:
                recent = self.ep_success[-200:]
                rate = sum(recent) / max(len(recent), 1)
                ret = (sum(self.ep_returns[-100:])
                       / max(len(self.ep_returns[-100:]), 1))
                sps = self.global_step / (time.perf_counter() - t0)
                print("%8d %7d %9.1f %8.2f %8.3f %7.4f %7.1f %7.0f" % (
                    self.global_step, self.stage_i, ret, rate, st["entropy"],
                    st["kl"], st["vf_loss"], sps))
                self.history.append(dict(
                    step=self.global_step, stage=self.stage_i,
                    stage_name=self.stages[self.stage_i].name,
                    ep_return=ret, success=rate, **st))
                self.save()

        self.save()
        print("\ndone: %d env steps (%d policy samples) in %.1f min" % (
            self.global_step, self.global_step * self.env.n_rov,
            (time.perf_counter() - t0) / 60.0))

    def save(self) -> None:
        torch.save(dict(model=self.net.state_dict(),
                        obs_dim=self.env.obs_dim,
                        step=self.global_step,
                        stage=self.stage_i), self.out / "policy.pt")
        (self.out / "history.json").write_text(json.dumps(self.history, indent=1))


if __name__ == "__main__":
    Trainer(parse_args()).run()
