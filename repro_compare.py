"""Paper-protocol PPO vs RPO comparison across seeds.

Runs train.py with the paper/author Atari settings (discounted critic, sticky
actions p=0.25, author network initialization) for PPO and RPO, then reports
each run's final training return over its last 100 full games.  This matches
the metric behind Table 2 of arXiv:2406.03678.
"""
import runtime
from runtime import ROOT, local_path
import argparse
from datetime import datetime
import json
import shutil
import subprocess
import sys
import traceback
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(run):
    return [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='repro_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--algorithms', nargs='+', choices=['ppo', 'rpo'], default=['ppo', 'rpo'])
    p.add_argument('--beta', type=float, default=3.0, help='Reflective weight used by RPO')
    p.add_argument('--critic-target', choices=['gae', 'discounted'], default='discounted')
    p.add_argument('--total-timesteps', type=int, default=50_000_000)
    p.add_argument('--eval-episodes', type=int, default=0,
                   help='Optional separate evaluation per run (0 disables)')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--rollout-length', type=int, default=256)
    args = p.parse_args()
    if min(args.total_timesteps, args.num_envs, args.rollout_length) <= 0:
        p.error('counts must be positive')
    if args.eval_episodes < 0:
        p.error('--eval-episodes must be >= 0')
    if not args.seeds or not args.algorithms:
        p.error('--seeds and --algorithms must not be empty')

    out = local_path('runs/' + args.name)
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    source.mkdir()
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, source / path.name)
    frozen_runtime = source / 'runtime.py'
    frozen_runtime.write_text(frozen_runtime.read_text().replace(
        'ROOT = Path(__file__).resolve().parent', 'ROOT = Path(' + repr(str(ROOT)) + ')'))
    cfg = json.loads((ROOT / 'config.json').read_text())
    cfg.update(critic_target=args.critic_target, total_timesteps=args.total_timesteps,
               device=args.device, num_envs=args.num_envs, rollout_length=args.rollout_length)
    config = out / 'base_config.json'
    config.write_text(json.dumps(cfg, indent=2))
    status = {'state': 'running', 'settings': vars(args), 'runs': {}}

    def save_status():
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(out / 'status.json')

    def execute(command, log):
        with log.open('w') as f:
            subprocess.run([sys.executable, '-B', *command], cwd=ROOT,
                           stdout=f, stderr=subprocess.STDOUT, check=True)

    try:
        save_status()
        for seed in args.seeds:
            for algorithm in args.algorithms:
                label = f'{algorithm}_seed{seed}'
                name = f'{args.name}/{label}'
                run = local_path('runs/' + name)
                status['phase'] = f'train_{label}'
                save_status()
                print(f'=== training {label} ===', flush=True)
                command = [str(source / 'train.py'), '--config', str(config),
                           '--run-name', name, '--algorithm', algorithm,
                           '--critic-target', args.critic_target, '--seed', str(seed)]
                if algorithm == 'rpo':
                    command += ['--beta', str(args.beta)]
                execute(command, out / f'{label}_train.log')
                if args.eval_episodes:
                    status['phase'] = f'evaluate_{label}'
                    save_status()
                    execute([str(source / 'evaluate.py'), str(run / 'latest.pt'),
                             '--episodes', str(args.eval_episodes),
                             '--seed', '10000', '--device', args.device],
                            out / f'{label}_eval.log')
                rows = read_rows(run)
                final = rows[-1]
                record = {'steps': final['steps'],
                          'final_training_mean100': final.get('return_mean_100'),
                          'episodes': final.get('episodes')}
                if args.eval_episodes:
                    record['evaluation'] = json.loads(
                        (run / 'evaluation.json').read_text())
                status['runs'][label] = record
                save_status()
                print(f'{label}: steps={final["steps"]:,} '
                      f'return_mean_100={final.get("return_mean_100")}', flush=True)

        # Aggregate the paper metric across seeds.
        summary = {}
        for algorithm in args.algorithms:
            finals = [status['runs'][f'{algorithm}_seed{seed}']['final_training_mean100']
                      for seed in args.seeds]
            finals = [float(value) for value in finals if value is not None]
            summary[algorithm] = {
                'seeds': list(args.seeds),
                'final_training_mean100': finals,
                'mean': float(np.mean(finals)) if finals else None,
                'std': float(np.std(finals)) if finals else None,
            }
            evaluations = [status['runs'][f'{algorithm}_seed{seed}'].get('evaluation')
                           for seed in args.seeds]
            evaluations = [value for value in evaluations if value]
            if evaluations:
                means = [value['mean'] for value in evaluations]
                summary[algorithm]['evaluation_mean'] = float(np.mean(means))
                summary[algorithm]['evaluation_std_across_seeds'] = float(np.std(means))

        # Plot each seed faintly and the across-seed mean.
        figure, ax = plt.subplots(figsize=(9, 5))
        colors = {'ppo': 'tab:blue', 'rpo': 'tab:orange'}
        for algorithm in args.algorithms:
            steps = None
            curves = []
            for seed in args.seeds:
                rows = read_rows(local_path(f'runs/{args.name}/{algorithm}_seed{seed}'))
                x = [row['steps'] for row in rows]
                y = [np.nan if row.get('return_mean_100') is None
                     else row['return_mean_100'] for row in rows]
                ax.plot(x, y, color=colors[algorithm], alpha=0.25, linewidth=1)
                if steps is None:
                    steps = x
                curves.append(y)
            if steps and curves and all(len(curve) == len(steps) for curve in curves):
                array = np.asarray(curves, dtype=float)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    mean = np.nanmean(array, axis=0)
                    std = np.nanstd(array, axis=0)
                ax.plot(steps, mean, color=colors[algorithm], linewidth=2,
                        label=f'{algorithm.upper()} mean')
                ax.fill_between(steps, mean - std, mean + std,
                                color=colors[algorithm], alpha=0.15)
        ax.set(xlabel='Environment steps', ylabel='Training return (last 100 episodes)',
               title=f'PPO vs RPO, {len(args.seeds)} seeds, {args.critic_target} critic')
        ax.legend()
        figure.tight_layout()
        figure.savefig(out / 'comparison.png', dpi=160)
        plt.close(figure)

        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        lines = ['# PPO vs RPO (paper protocol)', '',
                 f'Seeds: {args.seeds}; steps: {args.total_timesteps:,}; '
                 f'critic target: {args.critic_target}; metric: final training '
                 'return over the last 100 full games.', '',
                 '| algorithm | seeds | mean | std | per-seed final mean |',
                 '|---|---:|---:|---:|---|']
        for algorithm in args.algorithms:
            result = summary[algorithm]
            per_seed = ', '.join(f'{value:.1f}' for value in result['final_training_mean100'])
            mean = 'n/a' if result['mean'] is None else f"{result['mean']:.1f}"
            std = 'n/a' if result['std'] is None else f"{result['std']:.1f}"
            lines.append(f'| {algorithm.upper()} | {len(result["final_training_mean100"])} '
                         f'| {mean} | {std} | {per_seed} |')
        if args.eval_episodes:
            lines += ['', f'Evaluation: {args.eval_episodes} full games per run, '
                          'sampled actions, seed=10000.']
        lines += ['', 'Paper Table 2 reference for Breakout: PPO 233, RPO 306 '
                      '(3 seeds, last 100 episodes).', '',
                  '![Training curves](comparison.png)']
        (out / 'report.md').write_text('\n'.join(lines) + '\n')
        status.update(state='completed', phase='report_written')
        print(f'Report: {out / "report.md"}', flush=True)
    except BaseException:
        status.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        save_status()


if __name__ == '__main__':
    main()
