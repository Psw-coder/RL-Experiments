import runtime
from runtime import local_path
import argparse
from datetime import datetime
from functools import partial
import json
import math
import random
import time
import numpy as np
import torch
import gymnasium as gym
from atari import make_env
from rpo import Agent, targets, pair_indices, policy_loss
import ppo as ppo_algorithm
from monitor import TrainingMonitor


def main():
    parser = argparse.ArgumentParser(description='RPO / PPO experiments for Breakout')
    parser.add_argument('--algorithm', choices=['rpo', 'ppo'], help='Override config algorithm (default: rpo)')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--total-timesteps', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--rollout-length', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--beta', type=float, help='Reflective objective weight (RPO only)')
    parser.add_argument('--critic-target', choices=['gae', 'discounted'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--plot-every', type=int, help='Redraw curves every N updates')
    args = parser.parse_args()
    cfg = json.loads(local_path(args.config).read_text())
    cfg.setdefault('plot_every', 10)
    cfg.setdefault('algorithm', 'rpo')
    cfg.setdefault('critic_target', 'discounted')
    for key in ('algorithm', 'total_timesteps', 'num_envs', 'rollout_length', 'seed', 'device', 'plot_every', 'beta', 'critic_target'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    algorithm = cfg['algorithm']
    if algorithm not in ('rpo', 'ppo'):
        raise ValueError('algorithm must be rpo or ppo')
    if cfg['critic_target'] not in ('gae', 'discounted'):
        raise ValueError('critic_target must be gae or discounted')
    target_fn = ppo_algorithm.targets if cfg['critic_target'] == 'gae' else targets
    if algorithm == 'ppo':
        cfg.pop('beta', None)
        cfg.pop('next_clip_coef', None)
    n, horizon = cfg['num_envs'], cfg['rollout_length']
    batch_size = n * horizon
    if min(n, horizon, cfg['total_timesteps'], cfg['epochs'], cfg['num_minibatches'], cfg['save_every'], cfg['plot_every']) <= 0:
        raise ValueError('Step counts and batch parameters must be positive')
    if batch_size % cfg['num_minibatches'] or batch_size < 2:
        raise ValueError('Rollout batch must divide evenly into minibatches and contain >=2 samples')
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg['torch_threads'])
    torch.backends.cudnn.deterministic = True
    device = cfg['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = args.run_name or algorithm + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run = local_path('runs/' + name)
    run.mkdir(parents=True, exist_ok=False)
    updates = math.ceil(cfg['total_timesteps'] / batch_size)
    resolved = cfg | {'resolved_device': device, 'batch_size': batch_size,
        'minibatch_size': batch_size // cfg['num_minibatches'], 'total_updates': updates,
        'actual_total_timesteps': updates * batch_size, 'run_dir': str(run),
        'torch_version': str(torch.__version__),
        'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None}
    config_text = json.dumps(resolved, indent=2, ensure_ascii=False)
    print('Training configuration:\n' + config_text, flush=True)
    (run / 'config.json').write_text(config_text)
    print(f"Live dashboard: {run / 'dashboard.html'}", flush=True)
    agent = Agent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg['learning_rate'])
    # Pipe observations avoid allocating persistent files outside the project.
    envs = gym.vector.AsyncVectorEnv([partial(make_env, training=True) for _ in range(n)],
        shared_memory=False, context='spawn', autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    start = time.monotonic()
    steps = 0
    monitor = TrainingMonitor(run, resolved, updates * batch_size)
    status = 'Interrupted or failed'
    def tensor(x):
        return torch.as_tensor(x, device=device)
    def save(update):
        path = run / 'latest.pt'
        torch.save({'agent': agent.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': cfg, 'steps': steps, 'update': update}, run / 'latest.tmp')
        (run / 'latest.tmp').replace(path)
    try:
        obs, _ = envs.reset(seed=seed)
        with (run / 'metrics.jsonl').open('w', buffering=1) as log:
            for update in range(1, updates + 1):
                states = np.empty((horizon, n, 4, 84, 84), dtype=np.uint8)
                actions = np.empty((horizon, n), dtype=np.int64)
                old_logp = np.empty((horizon, n), dtype=np.float32)
                values, rewards, next_values = [np.empty((horizon, n), dtype=np.float32) for _ in range(3)]
                terminated, truncated = [np.empty((horizon, n), dtype=bool) for _ in range(2)]
                episode_returns = []
                for t in range(horizon):
                    states[t] = obs
                    with torch.no_grad():
                        dist, value = agent(tensor(obs))
                        action = dist.sample()
                    actions[t] = action.cpu().numpy()
                    old_logp[t] = dist.log_prob(action).cpu().numpy()
                    values[t] = value.cpu().numpy()
                    obs, rewards[t], terminated[t], truncated[t], info = envs.step(actions[t])
                    bootstrap_obs = obs.copy()
                    for i in np.flatnonzero(truncated[t] & ~terminated[t]):
                        bootstrap_obs[i] = info['final_obs'][i]
                    with torch.no_grad():
                        next_values[t] = agent(tensor(bootstrap_obs))[1].cpu().numpy()
                    final_info = info.get('final_info', {})
                    ep = final_info.get('episode', {})
                    if 'r' in ep:
                        mask = final_info.get('_episode', np.ones(n, dtype=bool))
                        episode_returns.extend(np.asarray(ep['r'])[mask].tolist())
                    steps += n
                adv, ret = target_fn(rewards, values, next_values, terminated, truncated,
                                   cfg['gamma'], cfg['gae_lambda'])
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                flat_states = states.reshape(-1, 4, 84, 84)
                a, lp, advantages, returns = [tensor(x.reshape(-1)) for x in (actions, old_logp, adv, ret)]
                done = (terminated | truncated).reshape(-1)
                metrics = []
                for epoch in range(cfg['epochs']):
                    for idx in np.split(np.random.permutation(batch_size), cfg['num_minibatches']):
                        dist, value = agent(tensor(flat_states[idx]))
                        new_logp = dist.log_prob(a[idx])
                        if algorithm == 'ppo':
                            loss_pi = ppo_algorithm.policy_loss(
                                new_logp, lp[idx], advantages[idx], cfg['clip_coef'])
                            ppo, reflection = loss_pi, loss_pi.new_zeros(())
                        else:
                            nxt, valid = pair_indices(idx, n, horizon, done)
                            next_dist, _ = agent(tensor(flat_states[nxt]))
                            loss_pi, ppo, reflection = policy_loss(new_logp, lp[idx], advantages[idx],
                                next_dist.log_prob(a[nxt]), lp[nxt], advantages[nxt], tensor(valid),
                                cfg['beta'], cfg['clip_coef'], cfg['next_clip_coef'])
                        value_loss = 0.5 * (value - returns[idx]).square().mean()
                        entropy = dist.entropy().mean()
                        loss = loss_pi + value_loss - cfg['entropy_coef'] * entropy
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite training loss')
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg['max_grad_norm'])
                        if not torch.isfinite(grad):
                            raise FloatingPointError('Nonfinite gradient')
                        optimizer.step()
                        with torch.no_grad():
                            log_ratio = new_logp - lp[idx]
                            ratio = log_ratio.exp()
                            approx_kl = (ratio - 1 - log_ratio).mean()
                            clip_fraction = ((ratio - 1).abs() > cfg['clip_coef']).float().mean()
                        metrics.append([x.detach().item() for x in
                            (loss, ppo, reflection, value_loss, entropy, approx_kl, clip_fraction)])
                row = dict(zip(('loss', 'ppo_loss', 'reflective_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction'),
                               np.mean(metrics, axis=0).tolist()))
                return_variance = float(np.var(ret))
                row['explained_variance'] = float(1 - np.var(ret - values) / return_variance) if return_variance > 1e-8 else None
                row['elapsed_seconds'] = time.monotonic() - start
                row.update(update=update, steps=steps, fps=int(steps / (time.monotonic() - start)),
                           episode_returns=episode_returns, valid_pair_fraction=float((~done[:-n]).mean()) if horizon > 1 else 0.)
                monitor.record(row)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                if update == 1 or update % cfg['plot_every'] == 0:
                    monitor.draw()
                if update % cfg['save_every'] == 0 or update == updates:
                    save(update)
        status = 'Completed'
    finally:
        envs.close()
        monitor.close(status)
    print(f'Checkpoint: {run / "latest.pt"}')


if __name__ == '__main__':
    main()
