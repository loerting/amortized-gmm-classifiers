# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""The result tables of the appendix as LaTeX tabulars: e1_<dataset>.tex (accuracy),
e2_<dataset>.tex (out-of-distribution detection) and e3_<dataset>.tex (screening).

Uncertainties are SEMs over seeds, with folds averaged within a seed. The best model of a
row is bold, together with every model whose difference to it is not significant under a
paired t-test at alpha = 0.05."""

import os
from collections import defaultdict

import numpy as np
from scipy import stats

from amogmm.results import load
from amogmm.train import MODELS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'tables')
NAME = {'nn': 'NN', 'nn_matched': 'NN, matched', 'bare': 'Bare GMM', 'bare_ca': 'Bare + CA',
        'amo': 'Amo.\\ GMM', 'amo_ca': '+ CA', 'nn_ts': 'NN + temp.'}
DATASETS = ['mnist', 'cifar10']


def level_key(labels):
    return float('inf') if labels == 'all' else float(labels)


def fmt(mean, sem, digits=1, bold=False):
    body = f'{100 * mean:.{digits}f}$\\pm${100 * sem:.{digits}f}'
    return f'\\textbf{{{body}}}' if bold else body


def mean_sem(per_seed):
    v = np.array(list(per_seed.values()))
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def winners(per_seed, higher_is_better):
    """The best model and every model not significantly different from it."""
    names = [m for m, d in per_seed.items() if d]
    means = {m: np.mean(list(per_seed[m].values())) for m in names}
    best = (max if higher_is_better else min)(names, key=lambda m: means[m])
    out = {best}
    for m in names:
        seeds = sorted(set(per_seed[best]) & set(per_seed[m]))
        if m != best and len(seeds) > 1 and stats.ttest_rel(
                [per_seed[best][s] for s in seeds], [per_seed[m][s] for s in seeds]).pvalue >= 0.05:
            out.add(m)
    return out


def write(name, header, lines, align):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\\begin{tabular}{%s}\n\\toprule\n' % align)
        f.write(header + ' \\\\\n\\midrule\n')
        f.write(' \\\\\n'.join(lines))
        f.write(' \\\\\n\\bottomrule\n\\end{tabular}\n')
    print('wrote', os.path.join(OUT, name))


def accuracy_tables():
    per = defaultdict(dict)
    for r in load('semi_supervised'):
        per[(r['dataset'], r['model'], r['labels'])][r['seed']] = r['test_acc']
    for ds in DATASETS:
        lines = []
        for lv in sorted({k[2] for k in per if k[0] == ds}, key=level_key):
            win = winners({m: per[(ds, m, lv)] for m in MODELS}, True)
            lines.append(f'{lv} & ' + ' & '.join(fmt(*mean_sem(per[(ds, m, lv)]), bold=m in win)
                                                  for m in MODELS))
        write(f'e1_{ds}.tex', 'labels/class & ' + ' & '.join(NAME[m] for m in MODELS), lines,
              'l' + 'r' * len(MODELS))


def detection_tables():
    per = defaultdict(lambda: defaultdict(list))
    for r in load('ood'):
        s = r['msp'] if r['model'].startswith('nn') else r['log_px']
        per[(r['dataset'], r['model'], r['labels'])][r['seed']].append((s['auroc'], s['fpr95']))
    for ds in DATASETS:
        levels = sorted({k[2] for k in per if k[0] == ds}, key=level_key)
        models = [m for m in MODELS if any((ds, m, lv) in per for lv in levels)]
        lines = []
        for i, (metric, higher) in enumerate((('AUROC', True), ('FPR@95TPR', False))):
            lines.append(('\\midrule\n' if i else '')
                         + '\\multicolumn{%d}{l}{\\emph{%s}}' % (len(models) + 1, metric))
            for lv in levels:
                seeds = {m: {s: np.mean([x[i] for x in v]) for s, v in per[(ds, m, lv)].items()}
                         for m in models}
                win = winners(seeds, higher)
                lines.append(f'{lv} & ' + ' & '.join(fmt(*mean_sem(seeds[m]), bold=m in win)
                                                      for m in models))
        write(f'e2_{ds}.tex', 'labels/class & ' + ' & '.join(NAME[m] for m in models), lines,
              'l' + 'r' * len(models))


def log_ratio_error(rows):
    """Mean |log10(observed rate / mean prediction)| over the bins pooled over critical
    classes and seeds, using only bins with at least ten positives."""
    pooled = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        for c in r['classes']:
            for i, (pred, pos, n) in enumerate(zip(c['bins']['pred'], c['bins']['pos'],
                                                   c['bins']['n'])):
                pooled[i][0] += pred * n
                pooled[i][1] += pos * n
                pooled[i][2] += n
    vals = [abs(np.log10(pos / pred)) for pred, pos, n in pooled.values() if pos >= 10]
    return (float(np.mean(vals)), len(vals), len(pooled)) if vals else None


def screening_tables():
    rows = load('screening')
    models = ['nn', 'nn_ts', 'bare', 'bare_ca', 'amo', 'amo_ca']
    cell = defaultdict(list)
    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r['dataset'], r['model'], r['labels'])
        cell[key].append(r)
        per[key]['fpr95'][r['seed']] = np.mean([c['fpr95'] for c in r['classes']])
        per[key]['ece'][r['seed']] = np.mean([c['ece'] for c in r['classes']])
    for ds in DATASETS:
        lines = []
        for lv in ('1', '2', '5', '10', '100', 'all'):
            present = [m for m in models if (ds, m, lv) in per]
            win = {f: winners({m: per[(ds, m, lv)][f] for m in present}, False)
                   for f in ('fpr95', 'ece')}
            lre = {m: log_ratio_error(cell[(ds, m, lv)]) for m in present}
            best = min((m for m in present if lre[m]), key=lambda m: lre[m][0])
            for m in present:
                cells = [fmt(*mean_sem(per[(ds, m, lv)][f]), digits=d, bold=m in win[f])
                         for f, d in (('fpr95', 1), ('ece', 2))]
                body = f'{lre[m][0]:.1f} ({lre[m][1]}/{lre[m][2]})'
                cells.append(f'\\textbf{{{body}}}' if m == best else body)
                lines.append(f'{lv} & {NAME[m]} & ' + ' & '.join(cells))
        write(f'e3_{ds}.tex', 'labels/class & model & FPR@95TPR & ECE & LRE (bins)', lines,
              'llrrr')


if __name__ == '__main__':
    accuracy_tables()
    detection_tables()
    screening_tables()
