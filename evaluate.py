import runtime
from runtime import local_path
import argparse
import json
import numpy as np
import torch
from atari import make_env
from rpo import Agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', help='Project-relative checkpoint path')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if args.episodes <= 0:
        p.error('--episodes must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    checkpoint = local_path(args.checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    agent = Agent().to(args.device)
    agent.load_state_dict(saved['agent'])
    agent.eval()
    env = make_env(training=False)
    scores = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            score = 0.0
            while True:
                with torch.no_grad():
                    dist, _ = agent(torch.as_tensor(np.asarray(obs)[None], device=args.device))
                    action = dist.probs.argmax(-1) if args.deterministic else dist.sample()
                obs, reward, terminated, truncated, info = env.step(action.item())
                score += reward
                if terminated or truncated:
                    # Includes any rewards during FIRE reset, if present.
                    score = float(info.get('episode', {}).get('r', score))
                    break
            scores.append(score)
            print(f'Episode {ep + 1}: {score}', flush=True)
    finally:
        env.close()
    result = {'checkpoint': str(checkpoint), 'seed': args.seed,
              'deterministic': args.deterministic, 'scores': scores,
              'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
    out = checkpoint.parent / 'evaluation.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
