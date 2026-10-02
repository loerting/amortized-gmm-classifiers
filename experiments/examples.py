# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""The single-seed MNIST runs behind the illustrative figures (seed 1).

mechanism.npz      component means and p(k|c) of the Amortized GMM with and without
                   amortized class-assignments, 10 labels per class
ood_scores.npz     log p(x) of the test set with the digit 0 and the digit 1 held out,
                   proposed model, all labels
probabilities.npz  p(k|x) of the test set, NN baseline and proposed model, all labels"""

import argparse
import os

import numpy as np
import torch

from amogmm.metrics import detection
from amogmm.results import RESULTS
from amogmm.train import fit

SEED = 1


def mechanism(out):
    arrays = {}
    for model in ('amo', 'amo_ca'):
        clf = fit('mnist', model, '10', SEED).clf
        with torch.no_grad():
            arrays[f'{model}_means'] = clf.means.cpu().numpy()
            arrays[f'{model}_pkc'] = clf._class_assignments().cpu().numpy()
    np.savez(os.path.join(out, 'mechanism.npz'), **arrays)


def ood_scores(out):
    arrays = {}
    for held_out in (0, 1):
        run = fit('mnist', 'amo_ca', 'all', SEED, held_out_class=held_out)
        lpx, ood = run.log_px(run.split.X_test), run.split.is_ood
        arrays[f'log_px_{held_out}'] = lpx
        arrays[f'is_ood_{held_out}'] = ood
        arrays[f'auroc_{held_out}'] = detection(lpx[~ood], lpx[ood])['auroc']
    np.savez(os.path.join(out, 'ood_scores.npz'), **arrays)


def probabilities(out):
    arrays = {}
    for model in ('nn', 'amo_ca'):
        run = fit('mnist', model, 'all', SEED)
        arrays[model] = run.proba(run.split.X_test)
    arrays['y'] = run.split.y_test
    arrays['test_index'] = run.split.test_index
    np.savez(os.path.join(out, 'probabilities.npz'), **arrays)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=os.path.join(RESULTS, 'examples'))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mechanism(args.out)
    ood_scores(args.out)
    probabilities(args.out)


if __name__ == '__main__':
    main()
