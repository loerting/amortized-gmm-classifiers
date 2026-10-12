# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, roc_curve

N_BINS = 15
THRESHOLD = 0.03


def detection(s_in, s_out):
    """Seen inputs are the positives. FPR@95TPR is the fraction of held-out inputs whose
    score reaches the 5th percentile of the seen scores. The ROC curve is stored as the
    TPR on a grid of 101 false positive rates."""
    s_in, s_out = np.asarray(s_in, dtype=np.float64), np.asarray(s_out, dtype=np.float64)
    y = np.concatenate([np.ones_like(s_in), np.zeros_like(s_out)])
    s = np.concatenate([s_in, s_out])
    fpr, tpr, _ = roc_curve(y, s)
    tpr_grid = np.interp(np.linspace(0.0, 1.0, 101), fpr, tpr)
    return dict(auroc=float(roc_auc_score(y, s)),
                fpr95=float((s_out >= np.percentile(s_in, 5.0)).mean()),
                roc_tpr=[round(float(v), 4) for v in tpr_grid])


def screening(s, y):
    """One-vs-rest metrics of the critical-class probability s = p(k*|x) against the binary
    labels y."""
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y, dtype=np.int64)
    fpr, tpr, thr = roc_curve(y, s)
    i = min(int(np.searchsorted(tpr, 0.95, side='left')), len(thr) - 1)
    bins = {'pred': [], 'pos': [], 'n': []}
    ece = 0.0
    for b in np.array_split(np.argsort(s), N_BINS):
        pred, pos = float(s[b].mean()), float(y[b].mean())
        bins['pred'].append(pred)
        bins['pos'].append(pos)
        bins['n'].append(int(len(b)))
        ece += len(b) / len(s) * abs(pred - pos)
    return dict(fpr95=float(fpr[i]), flag95=float((s >= thr[i]).mean()), ece=float(ece),
                bins=bins, sens3=float((s[y == 1] >= THRESHOLD).mean()),
                mid=int(((s >= 0.001) & (s <= 0.5)).sum()))


def fit_temperature(logp, y):
    """Temperature T minimizing the negative log-likelihood of softmax(log p / T), with log p
    the log-softmax output of the network."""
    logits = np.asarray(logp, dtype=np.float64)

    def nll(T):
        z = logits / T
        z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method='bounded').x)


def apply_temperature(logp, T):
    z = np.asarray(logp, dtype=np.float64) / T
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)
