# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""Figures of the screening experiment: screening_fpr95.pdf, screening_residual.pdf,
strip_mnist.pdf, score_population_mnist.pdf and worst_mistakes_mnist.pdf."""

import os
from collections import defaultdict

import matplotlib.ticker as mticker
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

import style
from style import COLOR, GRAY, LABEL, LINESTYLE, MARKER, plt
from amogmm import data
from amogmm.metrics import THRESHOLD
from amogmm.results import RESULTS, load

ORDER = ['nn', 'nn_ts', 'nn_matched', 'bare', 'bare_ca', 'amo', 'amo_ca']
WIDTH = {'nn': 2.0, 'nn_ts': 2.0, 'nn_matched': 1.4, 'bare': 2.0, 'bare_ca': 2.2, 'amo': 2.2,
         'amo_ca': 2.8}
EXAMPLES = ('nn', 'amo_ca')


def fpr95_curve(rows):
    per = defaultdict(dict)
    for r in rows:
        per[(r['dataset'], r['model'], r['labels'])][r['seed']] = \
            np.mean([c['fpr95'] for c in r['classes']])
    style.setup(0.45)
    fig, axes = plt.subplots(1, 2, squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], style.DATASETS):
        levels = sorted({k[2] for k in per if k[0] == ds}, key=lambda lv: style.count(lv, ds))
        for m in ORDER:
            pts = [(style.count(lv, ds), list(per[(ds, m, lv)].values())) for lv in levels
                   if (ds, m, lv) in per]
            if not pts:
                continue
            ax.errorbar([p[0] for p in pts], [np.mean(p[1]) for p in pts],
                        yerr=[np.std(p[1], ddof=1) / np.sqrt(len(p[1])) for p in pts],
                        label=LABEL[m], color=COLOR[m], marker=MARKER[m],
                        linestyle=LINESTYLE[m], linewidth=WIDTH[m], markersize=5.5,
                        capsize=2.5, elinewidth=0.9)
        style.log_label_axis(ax, ds, levels)
        ax.yaxis.set_major_formatter(style.pct_formatter(100))
        style.decorate(ax)
        ax.set_title(style.TITLE[ds])
        ax.set_xlabel('labeled examples per class')
    ax0 = axes[0][0]
    ax0.set_ylabel(f'FPR at 95{style.pct()} TPR')
    ax0.annotate('', xy=(0.93, 0.72), xytext=(0.93, 0.92), xycoords='axes fraction',
                 textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color=GRAY, linewidth=0.9))
    ax0.text(0.905, 0.82, 'better', transform=ax0.transAxes, ha='right', va='center', color=GRAY)
    fig.legend(*ax0.get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    style.save(fig, 'screening_fpr95.pdf')


def pooled_bins(rows, ds, m, labels):
    """Equal-mass bins pooled over the critical classes and seeds:
    (mean prediction, number of positives, number of points) per bin."""
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        if (r['dataset'], r['model'], r['labels']) != (ds, m, labels):
            continue
        for c in r['classes']:
            for i, (pred, pos, n) in enumerate(zip(c['bins']['pred'], c['bins']['pos'],
                                                   c['bins']['n'])):
                acc[i][0] += pred * n
                acc[i][1] += pos * n
                acc[i][2] += n
    keys = sorted(acc)
    return (np.array([acc[k][0] / acc[k][2] for k in keys]),
            np.array([acc[k][1] for k in keys]), np.array([acc[k][2] for k in keys]))


def wilson_upper(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    return np.minimum(((p + z * z / (2 * n)) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
                      / den, 1.0)


def break_marks(left, right, d=0.5):
    kw = dict(marker=[(-1, -d), (1, d)], markersize=7, linestyle='none', color=GRAY, mec=GRAY,
              mew=0.9, clip_on=False)
    left.plot([1], [0], transform=left.transAxes, **kw)
    right.plot([0], [0], transform=right.transAxes, **kw)


def residual(rows, levels=('1', 'all'), models=('nn', 'nn_ts', 'amo_ca')):
    """Observed positive rate divided by the mean prediction of every bin, on a logarithmic
    axis that is compressed beyond a factor of 100. Predictions below 1e-8 are drawn on a
    compressed segment left of a break. A bin without positives is drawn dashed, up to the
    upper end of the 95% Wilson interval of its rate."""
    split, xmin = 1e-8, 5e-8
    style.setup(0.95)
    fig = plt.figure()
    outer = fig.add_gridspec(len(levels), 2, wspace=0.22, hspace=0.32)
    bound_used = False
    mains = {}
    for i, labels in enumerate(levels):
        for j, ds in enumerate(style.DATASETS):
            pooled = {m: pooled_bins(rows, ds, m, labels) for m in models
                      if any((r['dataset'], r['model'], r['labels']) == (ds, m, labels)
                             for r in rows)}
            if any((pred < split).any() for pred, _, _ in pooled.values()):
                lo = min(pred[pred > 0].min() for pred, _, _ in pooled.values())
                tail_min = 10.0 ** (np.floor(np.log10(lo)) - 1)
                e_lo, e_hi = int(np.log10(tail_min)), int(np.log10(split))
                inner = outer[i, j].subgridspec(1, 2, width_ratios=[0.40, 1], wspace=0.06)
                ax_t = fig.add_subplot(inner[0, 0])
                ax = fig.add_subplot(inner[0, 1], sharey=ax_t)
            else:
                ax_t, ax = None, fig.add_subplot(outer[i, j])
            mains[(i, j)] = ax

            for m, (pred, npos, ntot) in pooled.items():
                zero = npos < 0.5
                with np.errstate(divide='ignore'):
                    ratio = np.log10((npos / ntot) / pred)
                    bound = np.log10(wilson_upper(npos, ntot) / pred)
                for seg, sel in ((ax, pred >= split), (ax_t, pred < split)):
                    if seg is None or not sel.any():
                        continue
                    meas = sel & ~zero
                    if meas.any():
                        seg.vlines(pred[meas], 0, ratio[meas], color=COLOR[m], linewidth=1.1,
                                   zorder=3)
                        seg.plot(pred[meas], ratio[meas], linestyle='none', marker=MARKER[m],
                                 color=COLOR[m], markersize=4.5,
                                 label=LABEL[m] if seg is ax else None, zorder=5)
                    lim = sel & zero
                    if lim.any():
                        bound_used = True
                        seg.vlines(pred[lim], 0, bound[lim], color=COLOR[m], linewidth=1.1,
                                   linestyle=(0, (2, 2)), zorder=3)
                        seg.plot(pred[lim], bound[lim], linestyle='none', marker=MARKER[m],
                                 color=COLOR[m], markersize=5.0, fillstyle='none',
                                 markeredgewidth=1.1, zorder=5)

            for seg in (ax, ax_t):
                if seg is None:
                    continue
                seg.axhline(0, color='#52514e', linewidth=0.9, zorder=2)
                seg.set_xscale('log')
                seg.set_yscale('symlog', linthresh=2.0, linscale=1.0)
                seg.set_ylim(-2.4, 45)
                seg.set_yticks([-2, -1, 0, 1, 2, 5, 10, 30])
                seg.set_yticklabels(['0.01', '0.1', '1', '10', '100', r'$10^{5}$', r'$10^{10}$',
                                     r'$10^{30}$'])
                seg.yaxis.set_minor_locator(mticker.NullLocator())
                seg.xaxis.set_minor_locator(mticker.NullLocator())
                style.decorate(seg)
            ax.set_xlim(xmin, 1.4)
            ax.set_xticks([1e-6, 1e-3, 1e0])
            ax.xaxis.set_major_formatter(style.pct_formatter(100))
            if ax_t is not None:
                ax_t.set_xlim(tail_min, split)
                ax_t.set_xticks([10.0 ** (e_lo + (e_hi - e_lo) // 2)])
                ax_t.xaxis.set_major_formatter(style.pct_formatter(100))
                ax_t.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
                break_marks(ax_t, ax)
            fs = plt.rcParams['xtick.labelsize']
            ax.text(0.97, 0.80, 'underestimation', transform=ax.transAxes, ha='right',
                    va='center', color=GRAY, fontsize=fs)
            ax.text(0.97, 0.035, 'overestimation', transform=ax.transAxes, ha='right',
                    va='center', color=GRAY, fontsize=fs)
            ax.set_title(f"{style.TITLE[ds]}, "
                         f"{'1 label per class' if labels == '1' else 'all labels'}")
            if i == len(levels) - 1:
                ax.set_xlabel(r'mean predicted $p(k^*|x)$ of a bin')
                ax.xaxis.set_label_coords(0.30 if ax_t is not None else 0.5, -0.14)
            if j == 0:
                (ax_t or ax).set_ylabel('observed / predicted rate')

    h, lab = mains[(0, 0)].get_legend_handles_labels()
    for hh, ll in zip(*mains[(len(levels) - 1, 0)].get_legend_handles_labels()):
        if ll not in lab:
            h.append(hh)
            lab.append(ll)
    if bound_used:
        h.append(tuple(Line2D([], [], linestyle=(0, (2, 2)), marker=MARKER[m], color=COLOR[m],
                              markersize=5.0, fillstyle='none', markeredgewidth=1.1,
                              linewidth=1.1) for m in ('nn', 'amo_ca')))
        lab.append('no positive observed, upper bound')
    fig.legend(h, lab, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    style.save(fig, 'screening_residual.pdf')


def examples():
    """Seed-1 probabilities of both models on the MNIST test set, with the test images."""
    p = np.load(os.path.join(RESULTS, 'examples', 'probabilities.npz'))
    X, _ = data.load('mnist')
    return {m: p[m] for m in EXAMPLES}, p['y'], X[p['test_index']].reshape(-1, 28, 28)


def fmt(v):
    if v >= 0.995:
        return '100' + style.pct()
    if v >= 0.01:
        return (f'{100 * v:.0f}' if v >= 0.095 else f'{100 * v:.1f}') + style.pct()
    if v >= 1e-4:
        return f'{100 * v:.2f}' + style.pct()
    return f'$10^{{{int(np.floor(np.log10(max(v, 1e-300) * 100)))}}}$' + style.pct()


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color('#c3cad3')


def rows_below(ax, texts, fs, step):
    for j, s in enumerate(texts):
        ax.text(0.5, -0.1 - step * j, s, transform=ax.transAxes, ha='center', va='top',
                fontsize=fs)


def row_names(ax, fs, step, gap=0.3):
    names = ['true label'] + [LABEL[m] for m in EXAMPLES]
    return [ax.text(-gap, -0.1 - step * j, s, transform=ax.transAxes, ha='right', va='top',
                    fontsize=fs) for j, s in enumerate(names)]


def strip(probs, y, images, k=9, idx=(4028, 9079, 1451, 4003, 12545, 13894, 10655)):
    """Hand-picked test images of 4s and 9s, sorted by the proposed model's p(k=9|x)."""
    p = {m: probs[m][:, k] for m in EXAMPLES}
    idx = sorted(idx, key=lambda i: p['amo_ca'][i])
    style.setup(0.3)
    fs = plt.rcParams['xtick.labelsize']
    fig, axes = plt.subplots(1, len(idx))
    fig.subplots_adjust(left=0.25, right=0.995, top=0.97, bottom=0.36, wspace=0.35)
    for ax, i in zip(axes, idx):
        ax.imshow(images[i], cmap='gray_r')
        frame(ax)
        rows_below(ax, [f'${y[i]}$'] + [fmt(p[m][i]) for m in EXAMPLES], fs, step=0.32)
    row_names(axes[0], fs, step=0.32)
    style.save(fig, 'strip_mnist.pdf')


def worst_mistakes(probs, y, images):
    """For every digit and both models, the misclassified test image whose true label gets
    the lowest probability."""
    p_true = {m: probs[m][np.arange(len(y)), y] for m in EXAMPLES}
    pred = {m: probs[m].argmax(1) for m in EXAMPLES}
    style.setup(0.44)
    fs = plt.rcParams['xtick.labelsize'] - 1
    fig = plt.figure()
    outer = fig.add_gridspec(2, 1, left=0.2, right=0.995, top=0.9, bottom=0.12, hspace=1.0)
    names, first = [], []
    for r, m in enumerate(EXAMPLES):
        inner = outer[r].subgridspec(1, 10, wspace=0.42)
        for k in range(10):
            sel = np.where((pred[m] != y) & (y == k))[0]
            i = sel[np.argmin(p_true[m][sel])]
            ax = fig.add_subplot(inner[0, k])
            ax.imshow(images[i], cmap='gray_r')
            frame(ax)
            rows_below(ax, [f'${k}$'] + [fmt(p_true[mm][i]) for mm in EXAMPLES], fs, step=0.36)
            if k == 0:
                names += row_names(ax, fs, step=0.36, gap=0.35)
                first.append(ax)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    renderer = fig.canvas.get_renderer()
    x_left = min(inv.transform_bbox(t.get_window_extent(renderer=renderer)).x0 for t in names)
    for r, ax in enumerate(first):
        fig.text(x_left, ax.get_position().y1 + 0.04,
                 f'({"ab"[r]}) Worst mistakes of the {LABEL[EXAMPLES[r]]}', ha='left',
                 va='bottom')
    style.save(fig, 'worst_mistakes_mnist.pdf')


def score_population(probs, y, split=1e-12):
    """All predictions p(k|x) of the test set, sorted ascending. The ticks below each panel
    mark the positives."""
    pops = {}
    for m in EXAMPLES:
        s = probs[m].T.ravel()
        positive = np.concatenate([y == k for k in range(probs[m].shape[1])])
        order = np.argsort(s, kind='stable')
        pops[m] = dict(x=np.linspace(0, 100, len(s)), s=s[order], positive=positive[order])
    line = {'nn': '--', 'amo_ca': '-'}
    style.setup(0.46)
    fig = plt.figure()
    outer = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.48, left=0.09,
                             right=0.995, top=0.92, bottom=0.14)

    ax_lin = fig.add_subplot(outer[0, 0])
    for m, p in pops.items():
        ax_lin.plot(p['x'], 100 * p['s'], color=COLOR[m], linestyle=line[m], linewidth=2.0,
                    label=LABEL[m])
    ax_lin.axhline(100 * THRESHOLD, color=GRAY, linestyle=':', linewidth=1.0)
    ax_lin.set_xlim(85, 100)
    ax_lin.set_ylim(-3, 103)
    ax_lin.set_xticks([85, 90, 95, 100])
    ax_lin.xaxis.set_major_formatter(style.pct_formatter())
    ax_lin.yaxis.set_major_formatter(style.pct_formatter())
    ax_lin.set_title(f'linear, top 15{style.pct()}')
    ax_lin.set_ylabel('predicted $p(k|x)$', labelpad=10)
    style.decorate(ax_lin)

    for col, m in ((1, 'nn'), (2, 'amo_ca')):
        p = pops[m]
        tail = p['s'].min() < split
        heights = [1, 0.45, 0.10] if tail else [1.45, 0.10]
        inner = outer[0, col].subgridspec(len(heights), 1, height_ratios=heights, hspace=0.07)
        ax = fig.add_subplot(inner[0, 0])
        segments = [ax]
        if tail:
            ax_t = fig.add_subplot(inner[1, 0], sharex=ax)
            segments.append(ax_t)
        ax_rug = fig.add_subplot(inner[-1, 0], sharex=ax)
        for seg in segments:
            seg.plot(p['x'], 100 * p['s'], color=COLOR[m], linestyle=line[m], linewidth=2.0)
            seg.set_yscale('log')
            seg.set_xlim(0, 100)
            seg.yaxis.set_major_formatter(style.pct_formatter())
            seg.yaxis.set_minor_locator(mticker.NullLocator())
            style.decorate(seg)
        ax.axhline(100 * THRESHOLD, color=GRAY, linestyle=':', linewidth=1.0)
        ax.set_ylim(100 * split, 200)
        ax.set_yticks([1e-6, 1e-2, 1e2])
        ax.text(2, 100 * THRESHOLD * 2.2, f'3{style.pct()} rule', fontsize=8, color=GRAY,
                va='bottom')
        ax.set_title(LABEL[m])
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        if tail:
            ax.spines['bottom'].set_visible(False)
            ax_t.set_ylim(100 * 1e-120, 100 * split)
            ax_t.set_yticks([1e-100, 1e-60, 1e-20])
            ax_t.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=7, linestyle='none',
                      color=GRAY, mec=GRAY, mew=0.9, clip_on=False)
            ax.plot([0], [0], transform=ax.transAxes, **kw)
            ax_t.plot([0], [1], transform=ax_t.transAxes, **kw)
        ax_rug.eventplot(p['x'][p['positive']], colors=COLOR[m], lineoffsets=0.5,
                         linelengths=1.0, linewidths=0.5)
        ax_rug.set_ylim(0, 1)
        ax_rug.set_yticks([])
        ax_rug.set_xticks([0, 50, 100])
        ax_rug.xaxis.set_major_formatter(style.pct_formatter())
        ax_rug.text(1.5, 0.5, 'positives', fontsize=7, color=GRAY, ha='left', va='center')
        for side in ('top', 'right', 'left'):
            ax_rug.spines[side].set_visible(False)
        ax_rug.spines['bottom'].set_color(GRAY)

    fig.supxlabel('share of all predictions, sorted ascending', y=-0.01)
    fig.legend(*ax_lin.get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    style.save(fig, 'score_population_mnist.pdf')


if __name__ == '__main__':
    rows = load('screening')
    fpr95_curve(rows)
    residual(rows)
    probs, y, images = examples()
    strip(probs, y, images)
    worst_mistakes(probs, y, images)
    score_population(probs, y)
