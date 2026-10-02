# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""Experiment 2, out-of-distribution detection, leave-one-class-out: one class is removed
from the training split, labeled and unlabeled alike, and every test input is scored.
The generative models are scored by log p(x) and by their maximum softmax probability, the
baselines by their maximum softmax probability."""

import argparse
import os

from amogmm.metrics import detection
from amogmm.results import RESULTS, save
from amogmm.train import accuracy, fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', default=['mnist', 'cifar10'])
    p.add_argument('--models', nargs='+', default=['nn', 'nn_matched', 'bare', 'amo', 'amo_ca'])
    p.add_argument('--labels', nargs='+', default=['1', '10', '100', '500', 'all'])
    p.add_argument('--folds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 21)))
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--out', default=os.path.join(RESULTS, 'ood.json'))
    args = p.parse_args()

    rows = []
    for dataset in args.datasets:
        for labels in args.labels:
            for model in args.models:
                for held_out in args.folds:
                    for seed in args.seeds:
                        run = fit(dataset, model, labels, seed, epochs=args.epochs,
                                  held_out_class=held_out)
                        s = run.split
                        probs = run.proba(s.X_test)
                        msp = probs.max(1)
                        scores = {'msp': detection(msp[~s.is_ood], msp[s.is_ood])}
                        if not model.startswith('nn'):
                            lpx = run.log_px(s.X_test)
                            scores['log_px'] = detection(lpx[~s.is_ood], lpx[s.is_ood])
                        rows.append(dict(dataset=dataset, model=model, labels=labels,
                                         held_out=held_out, seed=seed,
                                         test_acc=accuracy(probs[~s.is_ood], s.y_test[~s.is_ood]),
                                         **scores))
                        main_score = scores.get('log_px', scores['msp'])
                        print(f'{dataset} {labels:>4} {model:10} held out {held_out} seed {seed:2}: '
                              f'AUROC {main_score["auroc"]:.4f} FPR@95TPR {main_score["fpr95"]:.4f}',
                              flush=True)
                        save(rows, args.out)


if __name__ == '__main__':
    main()
