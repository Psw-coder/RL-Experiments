"""Local training curves and an automatically refreshing browser view."""
import runtime
import html
import json
from collections import deque
from datetime import timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def duration(seconds):
    return str(timedelta(seconds=int(max(seconds, 0))))


class TrainingMonitor:
    def __init__(self, run, config, total_steps):
        self.run = run
        self.config = config
        self.algorithm = config.get('algorithm', 'rpo').upper()
        self.total_steps = total_steps
        self.history = []
        self.scores = deque(maxlen=100)
        self.episodes = 0
        self.status = 'Running'
        self.figure, self.axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        self.write_page()

    def record(self, row):
        self.scores.extend(row['episode_returns'])
        self.episodes += len(row['episode_returns'])
        row['return_mean_100'] = float(np.mean(self.scores)) if self.scores else None
        row['episodes'] = self.episodes
        row['progress_percent'] = 100 * row['steps'] / self.total_steps
        row['eta_seconds'] = max(0, (self.total_steps - row['steps']) / max(row['fps'], 1))
        self.history.append(row.copy())
        score = f"{row['return_mean_100']:.2f}" if self.scores else 'waiting for full game'
        print(f"[{row['progress_percent']:6.2f}%] update={row['update']} "
              f"steps={row['steps']:,}/{self.total_steps:,} "
              f"return100={score} episodes={self.episodes} "
              f"loss={row['loss']:.4f} entropy={row['entropy']:.3f} "
              f"KL={row['approx_kl']:.5f} FPS={row['fps']} "
              f"elapsed={duration(row['elapsed_seconds'])} ETA={duration(row['eta_seconds'])}", flush=True)

    def draw(self):
        if not self.history:
            return
        x = [r['steps'] for r in self.history]
        panels = [
            ('Full-game score (last 100 games)', [('return_mean_100', 'Mean raw score')]),
            ('Losses', [('loss', 'Total'), ('ppo_loss', 'PPO'),
                        ('reflective_loss', 'Reflection (before beta)'), ('value_loss', 'Value')]),
            ('Policy entropy', [('entropy', 'Entropy')]),
            ('Policy update size', [('approx_kl', 'Approximate KL'), ('clip_fraction', 'Clip fraction')]),
            ('Value fit', [('explained_variance', 'Explained variance')]),
            ('Training throughput', [('fps', 'Environment steps / second')]),
        ]
        for ax, (title, series) in zip(self.axes.flat, panels):
            ax.clear()
            for key, label in series:
                if key == 'reflective_loss' and self.algorithm == 'PPO':
                    continue
                y = [r.get(key) if r.get(key) is not None else np.nan for r in self.history]
                ax.plot(x, y, label=label, linewidth=1.5, marker='.' if len(x) < 3 else None)
            ax.set_title(title)
            ax.set_xlabel('Environment steps')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        if not self.scores:
            self.axes[0, 0].text(.5, .5, 'Waiting for a completed game',
                                  ha='center', transform=self.axes[0, 0].transAxes)
        row = self.history[-1]
        self.figure.suptitle(f"Breakout {self.algorithm} | {self.status} | {row['progress_percent']:.2f}% | "
                            f"{row['steps']:,} steps | ETA {duration(row['eta_seconds'])}", fontsize=15)
        self.figure.savefig(self.run / 'curves.tmp.png', dpi=120)
        (self.run / 'curves.tmp.png').replace(self.run / 'curves.png')
        self.write_page()

    def write_page(self):
        step = self.history[-1]['steps'] if self.history else 0
        content = f'''<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>Breakout {html.escape(self.algorithm)} training</title>
<style>body{{font:16px system-ui;margin:24px;background:#f4f6f8;color:#152035}}
img{{width:100%;max-width:1600px}}pre{{background:white;padding:16px}}</style></head>
<body><h1>Breakout {html.escape(self.algorithm)} — {html.escape(self.status)}</h1>
<p>{step:,} / {self.total_steps:,} environment steps. Refreshes every 10 seconds.
Curves update every {self.config['plot_every']} training updates.</p>
<img src="curves.png?v={step}" alt="Curves appear after the first training update">
<details><summary>Training configuration</summary><pre>{html.escape(json.dumps(self.config, indent=2))}</pre></details>
</body></html>'''
        (self.run / 'dashboard.tmp.html').write_text(content, encoding='utf-8')
        (self.run / 'dashboard.tmp.html').replace(self.run / 'dashboard.html')

    def close(self, status):
        self.status = status
        try:
            self.draw()
            self.write_page()
        finally:
            plt.close(self.figure)
