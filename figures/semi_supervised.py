# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""Figures of the semi-supervised learning experiment: grand_comparison.pdf, mech_mnist.pdf
and budget_mnist.pdf."""

import os
from collections import defaultdict

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import style
from style import COLOR, LABEL, LINESTYLE, MARKER, plt
from amogmm import data
from amogmm.results import RESULTS, load

ORDER = ['amo_ca', 'amo', 'bare_ca', 'bare', 'nn']
WIDTH = {'amo': 3.2, 'bare': 2.0, 'nn': 2.0, 'bare_ca': 2.2, 'amo_ca': 3.0}
MSIZE = {'amo': 6.8, 'bare': 4.5, 'nn': 5.0, 'bare_ca': 5.0, 'amo_ca': 6.8}
ZORDER = {'amo': 2, 'bare': 2, 'nn': 5, 'bare_ca': 4, 'amo_ca': 6}
DIGIT_COLORS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3',
                '#8c8c8c', '#ccb974', '#64b5cd']


def accuracy_table(rows):
    """{(dataset, model, labels): (mean, sem)} over seeds."""
    per = defaultdict(dict)
    for r in rows:
        per[(r['dataset'], r['model'], r['labels'])][r['seed']] = r['test_acc']
    out = {}
    for k, d in per.items():
        v = np.array([d[s] for s in sorted(d)])
        out[k] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
    return out


def draw(ax, xs, ys, es, m):
    ax.errorbar(xs, ys, yerr=es, color=COLOR[m], marker=MARKER[m], markersize=MSIZE[m],
                linewidth=WIDTH[m], label=LABEL[m], linestyle=LINESTYLE[m], capsize=2,
                elinewidth=0.9, markeredgecolor='white' if m != 'nn' else COLOR[m],
                markeredgewidth=0.7, zorder=ZORDER[m])


def grand_comparison(rows):
    stats = accuracy_table(rows)
    style.setup(0.80)
    fig, axes = plt.subplots(2, 2, squeeze=False)
    accs = [v[0] for k, v in stats.items() if k[1] in ORDER]
    deltas = []
    for j, ds in enumerate(style.DATASETS):
        levels = sorted({k[2] for k in stats if k[0] == ds}, key=lambda lv: style.count(lv, ds))
        nn = {lv: stats[(ds, 'nn', lv)][0] for lv in levels}
        for m in ORDER:
            pts = [(style.count(lv, ds), *stats[(ds, m, lv)]) for lv in levels]
            xs, ys, es = zip(*pts)
            draw(axes[0][j], xs, ys, es, m)
            if m != 'nn':
                d = [y - nn[lv] for y, lv in zip(ys, levels)]
                deltas += d
                draw(axes[1][j], xs, d, es, m)
        axes[1][j].axhline(0.0, color='#52514e', linewidth=1.4, linestyle='--', zorder=1)
        axes[0][j].set_title(style.TITLE[ds])
        for ax in (axes[0][j], axes[1][j]):
            xs = [style.count(lv, ds) for lv in levels]
            style.log_label_axis(ax, ds, levels, keep={xs[0], xs[-1], 1, 10, 100, 1000})
            style.decorate(ax, spine_color='#333333')
            ax.yaxis.set_major_formatter(style.pct_formatter(100))
        axes[1][j].set_xlabel('labeled examples per class')
        axes[0][j].tick_params(labelbottom=False)

    pad = 0.05 * (max(accs) - min(accs))
    dpad = 0.05 * (max(deltas) - min(deltas))
    for j in range(2):
        axes[0][j].set_ylim(min(accs) - pad, max(accs) + pad)
        axes[1][j].set_ylim(min(deltas) - dpad, max(deltas) + dpad)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    axes[0][0].set_ylabel('test accuracy')
    axes[1][0].set_ylabel('accuracy $-$ NN baseline')
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.6, columnspacing=1.6)
    style.save(fig, 'grand_comparison.pdf')


def neighbor_agreement(means, cls, k=5):
    """Fraction of the k nearest neighbors of a component that share its class."""
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(means).kneighbors(means)
    return float((cls[idx[:, 1:]] == cls[:, None]).mean())


def mechanism():
    """Component means in a t-SNE projection. The edge color is the hardened class
    argmax_k p(k|c), the fill runs from filled at max_k p(k|c) = 1 to unfilled at 1/K, and
    decided components are drawn on top."""
    d = np.load(os.path.join(RESULTS, 'examples', 'mechanism.npz'))
    both = np.vstack([d['amo_means'], d['amo_ca_means']])
    Z = TSNE(n_components=2, perplexity=20, init='pca', random_state=0).fit_transform(both)
    Z = {'amo': Z[:len(d['amo_means'])], 'amo_ca': Z[len(d['amo_means']):]}
    style.setup(0.42)
    fig, axes = plt.subplots(1, 2)
    for ax, m, title in ((axes[0], 'amo', 'tabular class-assignments'),
                         (axes[1], 'amo_ca', 'amortized class-assignments')):
        pkc = d[f'{m}_pkc']
        cls = pkc.argmax(0)
        decided = np.clip((pkc.max(0) - 1 / len(pkc)) / (1 - 1 / len(pkc)), 0, 1)
        edge = np.array([mcolors.to_rgb(DIGIT_COLORS[k]) for k in cls])
        order = np.argsort(decided)
        ax.scatter(Z[m][order, 0], Z[m][order, 1], s=24, facecolors=np.c_[edge, decided][order],
                   edgecolors=edge[order], linewidths=0.9)
        agree = neighbor_agreement(d[f'{m}_means'], cls)
        ax.set_title(f'{title}\nnearest-neighbor class agreement: {100 * agree:.0f}{style.pct()}')
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color('#c3cad3')
    handles = [Line2D([], [], ls='', marker='o', ms=5, mfc=DIGIT_COLORS[k], mec=DIGIT_COLORS[k],
                      mew=0.9, label=str(k)) for k in range(10)]
    handles.append(Line2D([], [], ls='', marker='o', ms=5, mfc='none', mec='#333333', mew=0.9,
                          label='undecided'))
    fig.legend(handles=handles, loc='lower center', ncol=11, frameon=False,
               bbox_to_anchor=(0.5, -0.13), columnspacing=0.9, handletextpad=0.2)
    fig.tight_layout()
    style.save(fig, 'mech_mnist.pdf')


def budget(digit=3, n=12):
    p = np.load(os.path.join(RESULTS, 'examples', 'probabilities.npz'))
    X, _ = data.load('mnist')
    pick = p['test_index'][np.where(p['y'] == digit)[0][:n]]
    style.setup(0.13)
    fig, axes = plt.subplots(1, n)
    for j, (ax, i) in enumerate(zip(axes, pick)):
        ax.imshow(X[i].reshape(28, 28), cmap='gray_r')
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        if j == 0:
            ax.add_patch(Rectangle((0, 0), 27, 27, fill=False, edgecolor='#1b7f5a', linewidth=2.2))
            ax.set_xlabel(f'$y={digit}$', color='#1b7f5a')
        else:
            ax.set_xlabel('?', color='#b9bec6')
    fig.tight_layout()
    style.save(fig, 'budget_mnist.pdf')


if __name__ == '__main__':
    grand_comparison(load('semi_supervised'))
    mechanism()
    budget()
