"""Run a matched beta ablation and write a report after evaluation."""
import runtime
from runtime import ROOT, local_path
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='unified_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--total-timesteps', type=int, default=50_000_000)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--rollout-length', type=int, default=256)
    p.add_argument('--critic-target', choices=['gae', 'discounted'], default=None,
                   help='Critic target; default uses config.json (discounted, as in the paper)')
    args = p.parse_args()
    if min(args.total_timesteps, args.episodes, args.num_envs, args.rollout_length) <= 0:
        p.error('counts must be positive')
    out = local_path('runs/' + args.name)
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    source.mkdir()
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, source / path.name)
    # Execute frozen sources while keeping inputs/outputs rooted in the project.
    frozen_runtime = source / 'runtime.py'
    frozen_runtime.write_text(frozen_runtime.read_text().replace(
        'ROOT = Path(__file__).resolve().parent', 'ROOT = Path(' + repr(str(ROOT)) + ')'))
    cfg = json.loads((ROOT / 'config.json').read_text())
    cfg.setdefault('critic_target', 'discounted')
    cfg.update(algorithm='rpo', seed=args.seed,
               total_timesteps=args.total_timesteps, device=args.device,
               num_envs=args.num_envs, rollout_length=args.rollout_length)
    if args.critic_target:
        cfg['critic_target'] = args.critic_target
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
        for beta in (0, 3):
            name = args.name + f'/beta{beta}'
            run = local_path('runs/' + name)
            status['phase'] = f'train_beta{beta}'
            save_status()
            execute([str(source / 'train.py'), '--config', str(config), '--run-name', name,
                     '--beta', str(beta)], out / f'beta{beta}_train.log')
            status['phase'] = f'evaluate_beta{beta}'
            save_status()
            execute([str(source / 'evaluate.py'), str(run / 'latest.pt'), '--episodes', str(args.episodes),
                     '--seed', '10000', '--device', args.device], out / f'beta{beta}_eval.log')
            status['runs'][str(beta)] = str(run)
            save_status()
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        summary = {}
        fig, ax = plt.subplots(figsize=(9, 5))
        for beta, run in status['runs'].items():
            run = Path(run)
            rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            evaluation = json.loads((run / 'evaluation.json').read_text())
            last = rows[-1]
            scores = [score for row in rows if row['steps'] > .9 * last['steps'] for score in row['episode_returns']]
            summary[beta] = {'steps': last['steps'], 'final_training_mean100': last['return_mean_100'],
                             'last_10_percent_episode_mean': float(np.mean(scores)) if scores else None,
                             'evaluation': evaluation}
            points = [r for r in rows if r['return_mean_100'] is not None]
            ax.plot([r['steps'] for r in points], [r['return_mean_100'] for r in points], label=f'beta={beta}')
        ax.set(xlabel='Environment steps', ylabel='Training return (last 100 games)',
               title=f'{cfg["critic_target"]} critic, seed={args.seed}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'comparison.png', dpi=160)
        plt.close(fig)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        lines = ['# RPO beta ablation', '', f'Both arms use the {cfg["critic_target"]} critic, the same RPO code path, and identical settings except beta.', '',
                 '| beta | steps | training mean (last 100 games) | evaluation mean | evaluation std |',
                 '|---|---:|---:|---:|---:|']
        for beta, result in summary.items():
            ev = result['evaluation']
            lines.append(f"| {beta} | {result['steps']} | {result['final_training_mean100']} | {ev['mean']:.2f} | {ev['std']:.2f} |")
        lines += ['', f'Evaluation: {args.episodes} full games per arm, sampled actions, seed=10000.',
                  'This is one training seed; it does not establish performance across seeds.', '', '![Training curves](comparison.png)']
        (out / 'report.md').write_text('\n'.join(lines) + '\n')
        status.update(state='completed', phase='report_written')
    except BaseException:
        status.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        save_status()


if __name__ == '__main__':
    main()
