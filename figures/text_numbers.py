# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""Prints the numbers quoted in the text of Chapter 5, computed from the stored results.
Differences are paired by seed and reported as mean and SEM in percentage points."""

import os
from collections import defaultdict

import numpy as np

from amogmm.results import RESULTS, load
from amogmm.train import LABELS

TITLE = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10'}


def by_seed(rows, value, *keys):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[tuple(r[k] for k in keys)][r['seed']].append(value(r))
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in out.items()}


def mean(d):
    return float(np.mean(list(d.values())))


def paired(a, b):
    diff = np.array([a[s] - b[s] for s in sorted(a)])
    return 100 * diff.mean(), 100 * diff.std(ddof=1) / np.sqrt(len(diff))


def semi_supervised():
    print('== Section 5.2, semi-supervised learning')
    acc = by_seed(load('semi_supervised'), lambda r: r['test_acc'], 'dataset', 'model', 'labels')
    for ds in TITLE:
        a = lambda m, lv='1': acc[(ds, m, lv)]
        print(f'{TITLE[ds]}: + CA minus NN at 1 label/class {paired(a("amo_ca"), a("nn"))[0]:.1f} '
              f'+- {paired(a("amo_ca"), a("nn"))[1]:.1f}')
        diffs = [(lv, mean(a('amo_ca', lv)) - mean(a('nn', lv))) for lv in LABELS[ds]]
        cross = next(i for i, (_, d) in enumerate(diffs) if d < 0)
        print(f'  crossover between {diffs[cross - 1][0]} and {diffs[cross][0]} labels/class, '
              f'NN ahead by {-100 * diffs[-1][1]:.1f} with all labels')
        print(f'  Bare GMM {100 * mean(a("bare")):.1f}, + CA {100 * mean(a("amo_ca")):.1f} at 1 label/class')
        print(f'  amortized class-assignments: Amortized GMM +{paired(a("amo_ca"), a("amo"))[0]:.1f}, '
              f'Bare GMM +{paired(a("bare_ca"), a("bare"))[0]:.1f}, '
              f'encoder +{paired(a("amo_ca"), a("bare_ca"))[0]:.1f}')
        below = [lv for lv in LABELS[ds] if mean(a('amo', lv)) < mean(a('nn', lv))]
        print(f'  Amortized GMM below NN at: {", ".join(below)}')
        gap = max(abs(mean(a('nn_matched', lv)) - mean(a('nn', lv))) for lv in LABELS[ds])
        print(f'  largest |matched - NN|: {100 * gap:.2f}')

    d = np.load(os.path.join(RESULTS, 'examples', 'mechanism.npz'))
    from sklearn.neighbors import NearestNeighbors
    for m in ('amo', 'amo_ca'):
        cls = d[f'{m}_pkc'].argmax(0)
        _, idx = NearestNeighbors(n_neighbors=6).fit(d[f'{m}_means']).kneighbors(d[f'{m}_means'])
        print(f'5-NN class agreement of the components, {m}: '
              f'{100 * (cls[idx[:, 1:]] == cls[:, None]).mean():.0f}%')


def ood():
    print('== Section 5.3, out-of-distribution detection')
    rows = load('ood')
    score = lambda r: r['msp'] if r['model'].startswith('nn') else r['log_px']
    auroc = by_seed(rows, lambda r: score(r)['auroc'], 'dataset', 'model', 'labels')
    fpr = by_seed(rows, lambda r: score(r)['fpr95'], 'dataset', 'model', 'labels')
    msp = by_seed(rows, lambda r: r['msp']['auroc'], 'dataset', 'model', 'labels')
    fold = by_seed(rows, lambda r: score(r)['fpr95'], 'dataset', 'model', 'labels', 'held_out')
    fold_auroc = by_seed(rows, lambda r: score(r)['auroc'], 'dataset', 'model', 'labels', 'held_out')
    d = np.load(os.path.join(RESULTS, 'examples', 'ood_scores.npz'))
    for h in (0, 1):
        lpx, o = d[f'log_px_{h}'], d[f'is_ood_{h}']
        print(f'MNIST, digit {h} held out, seed 1, all labels: AUROC {float(d[f"auroc_{h}"]):.3f}, '
              f'{100 * (lpx[o] >= np.percentile(lpx[~o], 5)).mean():.0f}% of the held-out inputs pass')
    for ds in TITLE:
        k = lambda m, lv: (ds, m, lv)
        gap = paired(auroc[k('amo_ca', '1')], auroc[k('nn', '1')])
        print(f'{TITLE[ds]}: AUROC gap at 1 label/class {gap[0] / 100:.2f} +- {gap[1] / 100:.2f}')
        levels = sorted({key[2] for key in auroc if key[0] == ds})
        print(f'  + CA AUROC range over the supervision levels: '
              f'{min(mean(auroc[k("amo_ca", lv)]) for lv in levels):.3f} to '
              f'{max(mean(auroc[k("amo_ca", lv)]) for lv in levels):.3f}')
        for lv in ('100', 'all'):
            print(f'  FPR@95TPR at {lv}: NN {100 * mean(fpr[k("nn", lv)]):.0f}%, '
                  f'+ CA {100 * mean(fpr[k("amo_ca", lv)]):.0f}%')
        per_fold = {h: mean(fold[(ds, 'amo_ca', 'all', h)]) for h in range(10)}
        print(f'  + CA FPR@95TPR over the folds (all labels): {100 * min(per_fold.values()):.0f}% to '
              f'{100 * max(per_fold.values()):.0f}%')
        nn_better = sum(mean(fold[(ds, 'nn', 'all', h)]) < per_fold[h] for h in range(10))
        print(f'  folds where the fully supervised NN accepts fewer held-out inputs: {nn_better} of 10')
        print(f'  NN FPR@95TPR at 1 label/class over the folds: '
              f'{100 * min(mean(fold[(ds, "nn", "1", h)]) for h in range(10)):.0f}% to '
              f'{100 * max(mean(fold[(ds, "nn", "1", h)]) for h in range(10)):.0f}%')
        rises = [mean(fold_auroc[(ds, 'amo_ca', 'all', h)]) - mean(fold_auroc[(ds, 'amo_ca', '1', h)])
                 for h in range(10)]
        print('  + CA AUROC change per fold from 1 label/class to all: '
              + ', '.join(f'{x:+.3f}' for x in rises))
        print('  max softmax of the generative models at 1 label/class, AUROC: '
              + ', '.join(f'{m} {mean(msp[k(m, "1")]):.2f}' for m in ('bare', 'amo', 'amo_ca')))
        print(f'  all labels, AUROC: NN {mean(auroc[k("nn", "all")]):.2f}, '
              f'matched {mean(auroc[k("nn_matched", "all")]):.2f}')
    control = by_seed(load('ood_control'), lambda r: r['log_px']['auroc'], 'dataset', 'model', 'labels')
    for ds, m, lv in sorted(control):
        diff = mean(control[(ds, m, lv)]) - mean(auroc[(ds, 'bare', lv)])
        print(f'{TITLE[ds]}, {lv} label(s)/class: AUROC of Bare GMM + CA minus Bare GMM {diff:+.1e}')


def pooled(rows, ds, m, lv):
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        if (r['dataset'], r['model'], r['labels']) == (ds, m, lv):
            for c in r['classes']:
                for i, (p, q, n) in enumerate(zip(c['bins']['pred'], c['bins']['pos'], c['bins']['n'])):
                    acc[i][0] += p * n
                    acc[i][1] += q * n
                    acc[i][2] += n
    return [(a[0] / a[2], a[1], a[2]) for _, a in sorted(acc.items())]


def screening():
    print('== Section 5.4, high-sensitivity screening')
    rows = load('screening')
    fpr = by_seed(rows, lambda r: np.mean([c['fpr95'] for c in r['classes']]), 'dataset', 'model', 'labels')
    flag = by_seed(rows, lambda r: np.mean([c['flag95'] for c in r['classes']]), 'dataset', 'model', 'labels')
    for ds in TITLE:
        k = lambda m, lv='1': (ds, m, lv)
        print(f'{TITLE[ds]}: FPR@95TPR at 1 label/class NN {100 * mean(fpr[k("nn")]):.1f}%, '
              f'+ CA {100 * mean(fpr[k("amo_ca")]):.1f}%; flagged population NN '
              f'{100 * mean(flag[k("nn")]):.1f}%, + CA {100 * mean(flag[k("amo_ca")]):.1f}%')
        for lv in ('1', '10'):
            print(f'  Bare GMM {100 * mean(fpr[k("bare", lv)]):.1f}% -> Bare GMM + CA '
                  f'{100 * mean(fpr[k("bare_ca", lv)]):.1f}% at {lv} label(s)/class')
        for m in ('nn', 'amo_ca'):
            bins = pooled(rows, ds, m, 'all')
            n_points = [b[2] for b in bins]
            sens = [c['sens3'] for r in rows if (r['dataset'], r['model'], r['labels']) == (ds, m, 'all')
                    for c in r['classes']]
            mid = np.mean([np.sum([c['mid'] for c in r['classes']]) / 10 / np.sum(r['classes'][0]['bins']['n'])
                           for r in rows if (r['dataset'], r['model'], r['labels']) == (ds, m, 'all')])
            ratios = [np.log10((q / n) / p) for p, q, n in bins if q > 0]
            above = [abs(np.log10((q / n) / p)) for p, q, n in bins if q > 0 and p > 1e-3]
            print(f'  {m}, all labels: bins below 1e-4%: {sum(p < 1e-6 for p, _, _ in bins)} of 15, '
                  f'top bin predicted {100 * bins[-1][0]:.1f}% observed {100 * bins[-1][1] / bins[-1][2]:.1f}%, '
                  f'points per bin {min(n_points)} to {max(n_points)}')
            print(f'    3% rule catches {100 * np.mean(sens):.1f}% of the positives, share of the '
                  f'predictions in [0.1%, 50%]: {100 * mid:.1f}%')
            print(f'    log10(observed/predicted) per bin from {min(ratios):.1f} to {max(ratios):.1f}, '
                  f'largest factor for bins above 0.1%: {10 ** max(above):.2f}')

    d = np.load(os.path.join(RESULTS, 'examples', 'probabilities.npz'))
    for m in ('nn', 'amo_ca'):
        p = d[m]
        print(f'MNIST seed 1, all labels, {m}: {100 * (p < 1e-6).mean():.0f}% of the predictions below 1e-4%, '
              f'{100 * (p > 0.995).mean():.0f}% above 99.5%')
    p9 = {m: d[m][:, 9] for m in ('nn', 'amo_ca')}
    idx = sorted((4028, 9079, 1451, 4003, 12545, 13894, 10655), key=lambda i: p9['amo_ca'][i])
    print('strip, p(9|x) of the proposed model: ' + ', '.join(f'{100 * p9["amo_ca"][i]:.2g}%' for i in idx))
    print('strip, p(9|x) of the NN for the 4s: ' + ', '.join(f'{100 * p9["nn"][i]:.2g}%' for i in idx
                                                           if d['y'][i] == 4)
          + f'; for the middle 9: {100 * p9["nn"][idx[3]]:.1g}%')

    tail = load('screening_tail')
    agg = lambda f: np.array([f(r) for r in tail])
    sem = lambda v: v.std(ddof=1) / np.sqrt(len(v))
    for m in ('nn', 'amo_ca'):
        s = f'{m}_sweep'
        q = lambda f: (f'{100 * np.mean(agg(lambda r: r[s][f])):.1g}% '
                       f'({100 * np.median(agg(lambda r: r[s][f])):.1g}%)')
        extra = agg(lambda r: r[s]['extra_flagged'])
        print(f'{m}, mean (median) over seeds: positives missed by the 3% rule from {q("miss_p05")} '
              f'(5th percentile) to {q("miss_p95")} (95th percentile), median {q("miss_p50")}, '
              f'lowered threshold {q("t_low")}; flags {100 * extra.mean():.1f} +- '
              f'{100 * sem(extra):.1f}% more of the population')
    for ep in (20, 100, 500):
        t = lambda f: np.mean(agg(lambda r: r[f'nn_tail{ep}'][f]))
        print(f'NN after {ep} epochs: {100 * t("share_below"):.0f}% of the predictions below 1e-4%, '
              f'positive rate among them {100 * t("rate_below"):.3f}% '
              f'({t("rate_below") * 1e6:.1f} per million), accuracy {100 * t("acc"):.1f}%')


if __name__ == '__main__':
    semi_supervised()
    ood()
    screening()
