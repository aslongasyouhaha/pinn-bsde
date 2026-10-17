"""Load experiment artifacts and export comparison figures without retraining."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(output_dir):
    return pd.read_csv(Path(output_dir) / 'comparison.csv')


def plot_results(output_dir):
    root = Path(output_dir)
    frame = load_results(root)
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    saved = []
    for product, group in frame.groupby('product', sort=False):
        labels = [f'{row.method}\nseed {row.seed}' for row in group.itertuples()]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, group.prediction, label='Network price')
        ax.axhline(group.continuous_analytic.iloc[0], color='black', label='Continuous analytic')
        ax.axhline(group.discrete_mc.iloc[0], color='tab:orange', linestyle='--', label='Discrete MC')
        if 'bridge_mc' in group and group.bridge_mc.notna().any():
            ax.axhline(group.bridge_mc.iloc[0], color='tab:green', linestyle=':', label='Bridge MC')
        ax.set(title=product + ': initial price', ylabel='Price')
        ax.legend()
        fig.tight_layout()
        path = figures / f'{product}_prices.png'
        fig.savefig(path, dpi=160)
        saved.append(path)
        plt.close(fig)
        methods = list(group.method.unique())
        fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), squeeze=False)
        for ax, method in zip(axes[0], methods):
            for row in group[group.method == method].itertuples():
                history = pd.read_csv(root / row.run_dir / 'history.csv')
                for column in history.columns:
                    if column == 'mc_anchor' and row.mc_anchor_weight == 0:
                        continue
                    if column != 'epoch' and (history[column] > 0).any():
                        ax.semilogy(history.epoch, history[column].clip(lower=1e-16), label=f'{column} (seed {row.seed})')
            ax.set(title=method, xlabel='Epoch', ylabel='Loss (different objectives)')
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = figures / f'{product}_losses.png'
        fig.savefig(path, dpi=160)
        saved.append(path)
        plt.close(fig)
    return frame, saved


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    args = parser.parse_args()
    _, paths = plot_results(args.output_dir)
    for path in paths:
        print(path)
