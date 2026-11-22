import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', default='outputs/date-grpo/trainer_log_history.json')
    p.add_argument('--out', default='figures/training_curve.png')
    p.add_argument('--window', type=int, default=10, help='moving-average window')
    args = p.parse_args()

    # Keep only the per-step training rows (the last entry is an end-of-run summary).
    df = pd.read_json(args.log).dropna(subset=['reward'])
    std = df.get('reward_std', 0)
    smooth = df['reward'].rolling(args.window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(df['step'], df['reward'] - std, df['reward'] + std,
                    alpha=0.15, label='±1 std (within batch)')
    ax.plot(df['step'], df['reward'], alpha=0.5, linewidth=1, label='avg reward per step')
    ax.plot(df['step'], smooth, linewidth=2, label=f'{args.window}-step moving average')
    ax.set(xlabel='Training step', ylabel='Average reward (0 to 1)', ylim=(-0.05, 1.05),
           title='GRPO training curve: date-format conversion')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    # Numbers to quote in the write-up.
    k = max(1, len(df) // 10)
    print(f"steps logged: {len(df)}")
    print(f"first {k} steps avg reward: {df['reward'].head(k).mean():.3f}")
    print(f"last  {k} steps avg reward: {df['reward'].tail(k).mean():.3f}")
    print(f"max: {df['reward'].max():.3f}  min: {df['reward'].min():.3f}")
    for col in ['frac_reward_zero_std', 'kl', 'loss']:
        if col in df:
            print(f"{col}: mean {df[col].mean():.4f}, last {df[col].iloc[-1]:.4f}")
    print('saved', args.out)


if __name__ == '__main__':
    main()
