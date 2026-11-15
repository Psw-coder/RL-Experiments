"""Render one episode of a trained agent to a GIF."""
import runtime
from runtime import local_path
import argparse

import ale_py
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from atari import FireReset
from rpo import Agent

gym.register_envs(ale_py)

p = argparse.ArgumentParser()
p.add_argument('checkpoint', nargs='?', default='runs/breakout_rpo_s0/latest.pt')
p.add_argument('--seed', type=int, default=10000)
p.add_argument('--scan', type=int, default=20,
               help='try this many seeds and render the best episode')
p.add_argument('--stride', type=int, default=2, help='keep every Nth env step')
p.add_argument('--max-frames', type=int, default=900)
p.add_argument('--out', default='breakout.gif')
args = p.parse_args()
torch.set_num_threads(2)

saved = torch.load(local_path(args.checkpoint), map_location='cpu', weights_only=True)
agent = Agent()
agent.load_state_dict(saved['agent'])
agent.eval()


def run(seed, collect=False):
    env = gym.make('ALE/Breakout-v5', frameskip=1,
                   repeat_action_probability=0.25, full_action_space=False,
                   render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4,
        screen_size=84, terminal_on_life_loss=False, grayscale_obs=True,
        scale_obs=False)
    env = FireReset(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    torch.manual_seed(seed)  # actions reproduce exactly between scan and render
    obs, _ = env.reset(seed=seed)
    frames, score, steps, done = [], 0.0, 0, False
    while not done:
        with torch.no_grad():
            dist, _ = agent(torch.as_tensor(np.asarray(obs)[None]))
            action = dist.sample()
        obs, reward, terminated, truncated, info = env.step(action.item())
        score += reward
        steps += 1
        done = terminated or truncated
        if collect and steps % args.stride == 0:
            frames.append(Image.fromarray(env.render()))
        if collect and len(frames) >= args.max_frames:
            break
        if not collect and steps >= 4000:
            break
    env.close()
    return score, steps, frames, done


best_seed, best_score = None, -1.0
for seed in range(args.seed, args.seed + args.scan):
    score, steps, _, _ = run(seed)
    print(f'seed {seed}: score={int(score)} steps={steps}', flush=True)
    if score > best_score:
        best_seed, best_score = seed, score

print(f'rendering best seed {best_seed} (score {int(best_score)})', flush=True)
score, steps, frames, done = run(best_seed, collect=True)
out = local_path(args.out)
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=66 * args.stride, loop=0, optimize=True)
print(f'{out} | frames={len(frames)} steps={steps} score={int(score)} '
      f'size={out.stat().st_size / 1e6:.1f}MB finished={done}')
