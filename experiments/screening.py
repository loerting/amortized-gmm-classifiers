# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""Experiment 3, high-sensitivity screening: every class takes a turn as the critical class
k*, and the probability p(k*|x) is evaluated for ranking (FPR@95TPR) and calibration
(equal-mass bins). nn_ts is the NN baseline with temperature scaling fitted on a labeled
validation split of 10% of the training split, run at 100 labels per class and with all
labels."""

import argparse
import os

from amogmm.metrics import apply_temperature, fit_temperature, screening
from amogmm.results import RESULTS, save
from amogmm.train import MODELS, accuracy, fit

TS_LABELS = ('100', 'all')


def evaluate(probs, y):
    return [dict(k=k, **screening(probs[:, k], (y == k).astype(int)))
            for k in range(probs.shape[1])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['mnist', 'cifar10'])
    p.add_argument('--models', nargs='+', default=list(MODELS) + ['nn_ts'])
    p.add_argument('--labels', nargs='+', default=['1', '2', '5', '10', '100', 'all'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 21)))
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--out', default=os.path.join(RESULTS, 'screening.json'))
    args = p.parse_args()

    rows = []
    for dataset in args.datasets:
        for labels in args.labels:
            for model in args.models:
                if model == 'nn_ts' and labels not in TS_LABELS:
                    continue
                for seed in args.seeds:
                    row = dict(dataset=dataset, model=model, labels=labels, seed=seed)
                    if model == 'nn_ts':
                        run = fit(dataset, 'nn', labels, seed, epochs=args.epochs, val_frac=0.1)
                        T = fit_temperature(run.log_proba(run.split.X_val), run.split.y_val)
                        probs = apply_temperature(run.log_proba(run.split.X_test), T)
                        row['temperature'] = T
                    else:
                        run = fit(dataset, model, labels, seed, epochs=args.epochs)
                        probs = run.proba(run.split.X_test)
                    row['test_acc'] = accuracy(probs, run.split.y_test)
                    row['classes'] = evaluate(probs, run.split.y_test)
                    rows.append(row)
                    print(f'{dataset} {labels:>4} {model:10} seed {seed:2}: acc {row["test_acc"]:.4f}',
                          flush=True)
                    save(rows, args.out)


if __name__ == '__main__':
    main()
