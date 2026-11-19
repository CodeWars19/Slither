"""Roll out a trained policy, report success rate, and save a GIF.

    python evaluate.py --ckpt runs/snake_sac/best.pt --mode cpg --episodes 5
"""
import argparse

import numpy as np
from PIL import Image

from sac import SAC
from snake_env import SnakeEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", default="cpg", choices=["cpg", "direct"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--gif", default="snake_rollout.gif")
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    env = SnakeEnv(action_mode=args.mode, render_mode=None if args.no_render else "rgb_array")
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0], "cpu", args.hidden)
    agent.load(args.ckpt)

    frames, wins = [], 0
    for ep in range(args.episodes):
        o, _ = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            if not args.no_render and env.t % 2 == 0:
                frames.append(env.render())
            o, r, term, trunc, info = env.step(agent.act(o, deterministic=True))
            done = term or trunc
        wins += info["success"]
        print(f"episode {ep}: success={info['success']} steps={env.t} final_dist={info['distance']:.2f} m")
        if not args.no_render:
            frames.extend([env.render()] * 5)
    print(f"success rate: {wins / args.episodes:.0%}")

    if frames:
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=80, loop=0)
        print("saved", args.gif)


if __name__ == "__main__":
    main()
