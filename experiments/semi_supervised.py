# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""Experiment 1, semi-supervised learning: test accuracy over the supervision level."""

import argparse
import os

from amogmm.results import RESULTS, save
from amogmm.train import LABELS, MODELS, accuracy, fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['mnist', 'cifar10'])
    p.add_argument('--models', nargs='+', default=list(MODELS))
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 21)))
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--out', default=os.path.join(RESULTS, 'semi_supervised.json'))
    args = p.parse_args()

    rows = []
    for dataset in args.datasets:
        for labels in args.labels or LABELS[dataset]:
            for model in args.models:
                for seed in args.seeds:
                    run = fit(dataset, model, labels, seed, epochs=args.epochs)
                    acc = accuracy(run.proba(run.split.X_test), run.split.y_test)
                    rows.append(dict(dataset=dataset, model=model, labels=labels, seed=seed,
                                     test_acc=acc))
                    print(f'{dataset} {labels:>4} {model:10} seed {seed:2}: {acc:.4f}', flush=True)
                    save(rows, args.out)


if __name__ == '__main__':
    main()
