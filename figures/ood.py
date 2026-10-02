# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""Figures of the out-of-distribution experiment: ood_auroc.pdf, ood_roc.pdf,
ood_perclass.pdf and oodhist_mnist.pdf. The baselines are scored by their maximum softmax
probability, the generative models by log p(x)."""

import os
from collections import defaultdict

import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

import style
from style import COLOR, GRAY, LABEL, LINESTYLE, MARKER, plt
from amogmm.data import CIFAR_CLASSES
from amogmm.results import RESULTS, load

ORDER = ['nn', 'nn_matched', 'bare', 'amo', 'amo_ca']
WIDTH = {'nn': 2.0, 'nn_matched': 1.4, 'bare': 2.0, 'amo': 2.2, 'amo_ca': 2.8}
CLASS_NAMES = {'mnist': [str(i) for i in range(10)], 'cifar10': CIFAR_CLASSES}


def score(r):
    return r['msp'] if r['model'].startswith('nn') else r['log_px']


def seed_stats(rows, metric):
    """{(dataset, model, labels): (mean, sem)}, with the folds averaged within a seed."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per[(r['dataset'], r['model'], r['labels'])][r['seed']].append(score(r)[metric])
    out = {}
    for k, d in per.items():
        v = np.array([np.mean(x) for x in d.values()])
        out[k] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
    return out


def better_arrow(ax, up, x=0.07, lo=0.06, hi=0.26):
    y0, y1 = (lo, hi) if up else (hi, lo)
    ax.annotate('', xy=(x, y1), xytext=(x, y0), xycoords='axes fraction',
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=GRAY, linewidth=0.9))
    ax.text(x + 0.035, (lo + hi) / 2, 'better', transform=ax.transAxes, ha='left',
            va='center', color=GRAY)


def auroc_fpr(rows):
    stats = {m: seed_stats(rows, m) for m in ('auroc', 'fpr95')}
    style.setup(0.80)
    fig, axes = plt.subplots(2, 2, squeeze=False, sharex='col', sharey='row')
    for j, ds in enumerate(style.DATASETS):
        for i, (metric, ylabel) in enumerate((('auroc', 'AUROC (held-out class)'),
                                              ('fpr95', f'FPR at 95{style.pct()} TPR'))):
            ax = axes[i][j]
            st = stats[metric]
            levels = sorted({k[2] for k in st if k[0] == ds}, key=lambda lv: style.count(lv, ds))
            xs = [style.count(lv, ds) for lv in levels]
            for m in ORDER:
                v = [st[(ds, m, lv)] for lv in levels]
                ax.errorbar(xs, [x[0] for x in v], yerr=[x[1] for x in v], label=LABEL[m],
                            color=COLOR[m], marker=MARKER[m], linestyle=LINESTYLE[m],
                            linewidth=WIDTH[m], markersize=5.5, capsize=2.5, elinewidth=0.9)
            style.log_label_axis(ax, ds, levels)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            if metric == 'auroc':
                ax.axhline(0.5, color='#b9bec6', linewidth=0.9, linestyle=':')
            else:
                ax.yaxis.set_major_formatter(style.pct_formatter(100))
            style.decorate(ax)
            if i == 0:
                ax.set_title(style.TITLE[ds])
            else:
                ax.set_xlabel('labeled examples per class')
            if j == 0:
                ax.set_ylabel(ylabel)
                better_arrow(ax, up=metric == 'auroc')
    for row in axes:
        lo, hi = row[0].get_ylim()
        row[0].set_ylim(lo, np.ceil(hi * 10 - 1e-9) / 10)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    style.save(fig, 'ood_auroc.pdf')


def level_title(ds, labels):
    return f'{style.TITLE[ds]}, ' + ('all labels' if labels == 'all' else f'{labels} label/class')


def roc(rows, levels=('1', 'all')):
    """Mean ROC curves, averaged horizontally (mean FPR at every TPR), so that every curve
    passes through the mean FPR@95TPR."""
    grid = np.linspace(0, 1, 101)
    tgrid = np.linspace(0, 1, 201)
    style.setup(0.95)
    fig, axes = plt.subplots(2, 2, squeeze=False, sharex=True, sharey=True)
    for i, labels in enumerate(levels):
        for j, ds in enumerate(style.DATASETS):
            ax = axes[i][j]
            ax.set_box_aspect(1)
            for m in ORDER:
                runs = [score(r) for r in rows
                        if (r['dataset'], r['model'], r['labels']) == (ds, m, labels)]
                fpr = np.mean([np.interp(tgrid, o['roc_tpr'], grid) for o in runs], axis=0)
                ax.plot(fpr, tgrid, label=LABEL[m], color=COLOR[m], linestyle=LINESTYLE[m],
                        linewidth=WIDTH[m])
                f95 = float(np.mean([o['fpr95'] for o in runs]))
                ax.plot([f95, f95], [0, 0.95], color=COLOR[m], linestyle=':', linewidth=1.1,
                        alpha=0.85)
                ax.plot([f95], [0.95], marker='o', color=COLOR[m], markersize=4.5,
                        markeredgecolor='white', markeredgewidth=0.6, zorder=5)
            ax.axhline(0.95, color=GRAY, linestyle=':', linewidth=1.0)
            ax.plot([0, 1], [0, 1], color='#b9bec6', linewidth=0.9, zorder=0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.xaxis.set_major_formatter(style.pct_formatter(100))
            ax.yaxis.set_major_formatter(style.pct_formatter(100))
            if i == 0 and j == 0:
                ax.annotate('better', xy=(0.10, 0.93), xytext=(0.36, 0.66), color=GRAY,
                            ha='center',
                            arrowprops=dict(arrowstyle='->', color=GRAY, linewidth=0.9))
            style.decorate(ax)
            ax.set_title(level_title(ds, labels))
            if i == len(levels) - 1:
                ax.set_xlabel('held-out inputs accepted (FPR)')
            if j == 0:
                ax.set_ylabel('seen inputs accepted (TPR)')
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.055))
    fig.tight_layout()
    style.save(fig, 'ood_roc.pdf')


def per_class(rows, levels=('1', 'all'), models=('nn', 'nn_matched', 'amo_ca')):
    """FPR@95TPR of every held-out class, mean and SEM over seeds. The horizontal lines are
    the means over the ten classes."""
    style.setup(0.80)
    fig, axes = plt.subplots(2, 2, squeeze=False, sharex='col', sharey=True)
    offsets = np.linspace(-0.28, 0.28, len(models))
    size = {m: 7.0 if MARKER[m] == '+' else 5.5 for m in models}
    edge = {m: MARKER[m] in ('x', '+') for m in models}
    for i, labels in enumerate(levels):
        for j, ds in enumerate(style.DATASETS):
            ax = axes[i][j]
            for k, m in enumerate(models):
                per = defaultdict(list)
                for r in rows:
                    if (r['dataset'], r['model'], r['labels']) == (ds, m, labels):
                        per[r['held_out']].append(score(r)['fpr95'])
                cls = sorted(per)
                mean = np.array([np.mean(per[c]) for c in cls])
                sem = np.array([np.std(per[c], ddof=1) / np.sqrt(len(per[c])) for c in cls])
                x = np.array(cls, dtype=float) + offsets[k]
                ax.axhline(mean.mean(), color=COLOR[m], linewidth=1.1, linestyle=LINESTYLE[m],
                           zorder=1)
                ax.errorbar(x, mean, yerr=sem, fmt='none', ecolor=COLOR[m], elinewidth=0.9,
                            zorder=2)
                ax.plot(x, mean, linestyle='none', marker=MARKER[m], color=COLOR[m],
                        markersize=size[m], markeredgewidth=1.6 if edge[m] else 0.6,
                        markeredgecolor=COLOR[m] if edge[m] else 'white', label=LABEL[m],
                        zorder=3)
            names = CLASS_NAMES[ds]
            ax.set_xticks(range(len(names)))
            if ds == 'mnist':
                ax.set_xticklabels(names)
            else:
                ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')
            ax.set_xlim(-0.6, len(names) - 0.4)
            ax.set_ylim(0, 1.04)
            ax.yaxis.set_major_formatter(style.pct_formatter(100))
            style.decorate(ax)
            ax.set_title(level_title(ds, labels))
            if i == len(levels) - 1:
                ax.set_xlabel('held-out class')
            if j == 0:
                ax.set_ylabel(f'FPR at 95{style.pct()} TPR')
    handles = [Line2D([0, 1], [0, 0], color=COLOR[m], linestyle=LINESTYLE[m], linewidth=1.1,
                      marker=MARKER[m], markersize=size[m],
                      markeredgewidth=1.6 if edge[m] else 0.6,
                      markeredgecolor=COLOR[m] if edge[m] else 'white') for m in models]
    fig.legend(handles, [LABEL[m] for m in models], loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.045), handlelength=4.2)
    fig.tight_layout()
    style.save(fig, 'ood_perclass.pdf')


def histograms(folds=(0, 1)):
    """log p(x) of the seen and the held-out test inputs, with the threshold that accepts
    95% of the seen inputs."""
    d = np.load(os.path.join(RESULTS, 'examples', 'ood_scores.npz'))
    style.setup(0.40)
    fig, axes = plt.subplots(1, len(folds), squeeze=False)
    for ax, h in zip(axes[0], folds):
        lpx, ood = d[f'log_px_{h}'], d[f'is_ood_{h}'].astype(bool)
        bins = np.linspace(np.percentile(lpx, 0.5), np.percentile(lpx, 99.5), 50)
        ax.hist(lpx[~ood], bins=bins, density=True, alpha=0.75, color='#2a78d6',
                label='seen classes')
        ax.hist(lpx[ood], bins=bins, density=True, alpha=0.75, color='#eb6834',
                label='held-out class')
        tau = np.percentile(lpx[~ood], 5.0)
        ax.axvline(tau, color='#52514e', linestyle=':', linewidth=1.3)
        print(f'digit {h} held out: AUROC {float(d[f"auroc_{h}"]):.3f}, '
              f'FPR@95TPR {float((lpx[ood] >= tau).mean()):.3f}')
        ax.set_title(f'digit {h} held out')
        ax.set_xlabel(r'$\log p(x)$')
        ax.set_yticks([])
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRAY)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    style.save(fig, 'oodhist_mnist.pdf')


if __name__ == '__main__':
    rows = [r for r in load('ood') if r['model'] in ORDER]
    auroc_fpr(rows)
    roc(rows)
    per_class(rows)
    histograms()
