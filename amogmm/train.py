# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

import numpy as np
import torch

from . import data
from .models import AmortizedGMM, AmortizedGMMCA, BareGMM, BareGMMCA, NNClassifier, mlp

MODELS = ('nn', 'nn_matched', 'bare', 'bare_ca', 'amo', 'amo_ca')
NAMES = {'nn': 'NN baseline', 'nn_matched': 'NN, matched', 'bare': 'Bare GMM',
         'bare_ca': 'Bare GMM + CA', 'amo': 'Amortized GMM', 'amo_ca': 'Amortized GMM + CA',
         'nn_ts': 'NN + temp. scaling'}
LABELS = {'mnist': ['1', '2', '5', '10', '20', '50', '100', '200', '500', '1000', '2000',
                    '5050', 'all'],
          'cifar10': ['1', '2', '5', '10', '20', '50', '100', '200', '500', '1000', '2000',
                      'all']}


def labels_per_class(spec):
    return None if spec == 'all' else int(spec)


class Trained:
    def __init__(self, clf, split, device, batch_size):
        self.clf, self.split, self.device, self.batch_size = clf, split, device, batch_size

    def tensor(self, X):
        return torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)

    def proba(self, X):
        """p(k|x) in double precision as a numpy array."""
        X = self.tensor(X)
        if isinstance(self.clf, NNClassifier):
            return self.clf.predict_proba(X).cpu().numpy()
        return torch.cat([self.clf.predict_proba(X[b:b + self.batch_size])
                          for b in range(0, len(X), self.batch_size)]).double().cpu().numpy()

    def log_proba(self, X):
        """log p(k|x) of the NN baseline in double precision as a numpy array."""
        return self.clf.predict_log_proba(self.tensor(X)).cpu().numpy()

    def log_px(self, X):
        return self.clf.log_px(self.tensor(X)).cpu().numpy()


def fit(dataset, model, labels='all', seed=1, epochs=500, held_out_class=None, val_frac=0.0,
        C=200, reg_covar=1e-3, lr=1e-3, batch_size=4096, device=None):
    """Train one model on one seed. The seed fixes the split, the labeled subset and the
    initialization."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    split = data.split(dataset, seed, labels_per_class(labels), held_out_class, val_frac)
    X = torch.tensor(split.X_train, dtype=torch.float32, device=device)
    y = torch.tensor(split.y_train, dtype=torch.long, device=device)
    D, K = X.shape[1], split.K

    if model in ('nn', 'nn_matched'):
        clf = NNClassifier(D, C, K, matched=model == 'nn_matched', lr=lr, device=device)
        labeled = y != -1
        clf.fit(X[labeled], y[labeled], epochs, batch_size)
        return Trained(clf, split, device, batch_size)

    if C * D > 2e8 / batch_size:
        batch_size = max(256, int(2e8 / (C * D)))
    kw = dict(reg_covar=reg_covar, device=device)
    if model == 'bare':
        clf = BareGMM(C, D, K, **kw)
    elif model == 'bare_ca':
        clf = BareGMMCA(C, D, K, **kw)
    elif model in ('amo', 'amo_ca'):
        cls = AmortizedGMM if model == 'amo' else AmortizedGMMCA
        clf = cls(C, D, K, mlp(D, C), lr=lr, **kw)
    else:
        raise ValueError(f'unknown model {model}')
    clf.fit(X, y, epochs, batch_size, rng)
    return Trained(clf, split, device, batch_size)


def accuracy(probs, y):
    return float((probs.argmax(1) == y).mean())
