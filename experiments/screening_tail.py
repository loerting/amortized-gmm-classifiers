# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""The small predicted probabilities of Section 5.4, MNIST with all labels.

sweep: the positives the 3% rule misses, and the threshold that recovers nine in ten of
       them together with the share of the population it flags in addition.
tail:  share of predictions below 1e-6, the positive rate among them, test accuracy and
       the catch rate of the 3% rule, for the NN baseline after 20, 100 and 500 epochs."""

import argparse
import os

import numpy as np

from amogmm.metrics import THRESHOLD
from amogmm.results import RESULTS, save
from amogmm.train import accuracy, fit


def sweep(P, y):
    s = P.T.ravel()
    positive = np.concatenate([y == k for k in range(P.shape[1])])
    missed = s[positive & (s < THRESHOLD)]
    out = dict(catch3=float(1 - len(missed) / positive.sum()), n_missed=int(len(missed)))
    if len(missed):
        t_low = float(np.sort(missed)[::-1][int(np.ceil(0.9 * len(missed))) - 1])
        out.update(t_low=t_low, extra_flagged=float(((s >= t_low) & (s < THRESHOLD)).mean()),
                   miss_p05=float(np.percentile(missed, 5)), miss_p50=float(np.median(missed)),
                   miss_p95=float(np.percentile(missed, 95)), miss_min=float(missed.min()))
    return out


def tail(P, y):
    low = P < 1e-6
    positives = sum(int((low[:, k] & (y == k)).sum()) for k in range(P.shape[1]))
    return dict(acc=accuracy(P, y), share_below=float(low.mean()),
                rate_below=float(positives / max(int(low.sum()), 1)), n_below=int(low.sum()),
                catch3=float(np.mean([(P[y == k, k] >= THRESHOLD).mean()
                                      for k in range(P.shape[1])])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 21)))
    p.add_argument('--out', default=os.path.join(RESULTS, 'screening_tail.json'))
    args = p.parse_args()

    rows = []
    for seed in args.seeds:
        row = dict(dataset='mnist', seed=seed)
        for epochs in (20, 100, 500):
            run = fit('mnist', 'nn', 'all', seed, epochs=epochs)
            P, y = run.proba(run.split.X_test), run.split.y_test
            row[f'nn_tail{epochs}'] = tail(P, y)
        row['nn_sweep'] = sweep(P, y)
        run = fit('mnist', 'amo_ca', 'all', seed)
        row['amo_ca_sweep'] = sweep(run.proba(run.split.X_test), run.split.y_test)
        rows.append(row)
        print(f'seed {seed:2}: extra flagged nn {row["nn_sweep"].get("extra_flagged", 0):.4f} '
              f'amo_ca {row["amo_ca_sweep"].get("extra_flagged", 0):.4f}', flush=True)
        save(rows, args.out)


if __name__ == '__main__':
    main()
