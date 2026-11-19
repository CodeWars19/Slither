"""Train SAC on the snake goal-reaching task.

    python train.py                       # CPG action space, 200k steps
    python train.py --mode direct         # raw joint-angle targets (harder)
    python train.py --steps 50000 --device cuda
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from sac import SAC, ReplayBuffer
from snake_env import SnakeEnv


def evaluate(agent, env, episodes, seed):
    rets, succ, steps = [], [], []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        done, R, t = False, 0.0, 0
        while not done:
            o, r, term, trunc, info = env.step(agent.act(o, deterministic=True))
            R, t, done = R + r, t + 1, term or trunc
        rets.append(R)
        succ.append(info["success"])
        steps.append(t)
    return float(np.mean(rets)), float(np.mean(succ)), float(np.mean(steps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="cpg", choices=["cpg", "direct"])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--start-steps", type=int, default=5_000, help="uniform-random warm-up")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--utd", type=int, default=1, help="gradient updates per env step")
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/snake_sac")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = SnakeEnv(action_mode=args.mode)
    eval_env = SnakeEnv(action_mode=args.mode)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SAC(obs_dim, act_dim, args.device, args.hidden, args.gamma, lr=args.lr)
    buf = ReplayBuffer(obs_dim, act_dim, size=min(args.steps, 1_000_000), device=args.device)

    log = open(os.path.join(args.out, "log.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "eval_return", "eval_success", "eval_len", "train_return",
                     "alpha", "entropy", "q_loss", "q"])

    o, _ = env.reset(seed=args.seed)
    ep_ret, recent, best, stats, t0 = 0.0, [], -np.inf, {}, time.time()
    for step in range(1, args.steps + 1):
        a = env.action_space.sample() if step <= args.start_steps else agent.act(o)
        o2, r, term, trunc, _ = env.step(a)
        buf.add(o, a, r, o2, term)          # truncation still bootstraps
        o, ep_ret = o2, ep_ret + r
        if term or trunc:
            recent.append(ep_ret)
            o, _ = env.reset()
            ep_ret = 0.0

        if step > args.start_steps:
            for _ in range(args.utd):
                stats = agent.update(buf.sample(args.batch))

        if step % args.eval_every == 0:
            er, es, el = evaluate(agent, eval_env, args.eval_episodes, seed=10_000)
            tr = float(np.mean(recent[-20:])) if recent else float("nan")
            writer.writerow([step, er, es, el, tr, stats.get("alpha"), stats.get("entropy"),
                             stats.get("q_loss"), stats.get("q")])
            log.flush()
            print(f"step {step:7d} | eval return {er:7.2f} success {es:4.0%} len {el:5.1f} "
                  f"| train return {tr:7.2f} | alpha {stats.get('alpha', 0):.3f} "
                  f"| {time.time() - t0:5.0f}s", flush=True)
            agent.save(os.path.join(args.out, "last.pt"))
            if er > best:
                best = er
                agent.save(os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
