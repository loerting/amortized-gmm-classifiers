# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

import os
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get(
    'AMOGMM_DATA', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
DATASETS = ('mnist', 'cifar10')
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                 'ship', 'truck']

_cache = {}


def deskew(images):
    """Shear every image so that the principal axis of its pixel mass is vertical."""
    from scipy.ndimage import affine_transform
    out = np.empty_like(images)
    gy, gx = np.mgrid[0:28, 0:28]
    for i, im in enumerate(images):
        tot = im.sum()
        if tot <= 0:
            out[i] = im
            continue
        mu_y, mu_x = (gy * im).sum() / tot, (gx * im).sum() / tot
        mu11 = ((gx - mu_x) * (gy - mu_y) * im).sum() / tot
        mu02 = (((gy - mu_y) ** 2) * im).sum() / tot
        alpha = mu11 / mu02 if abs(mu02) > 1e-6 else 0.0
        out[i] = affine_transform(im, np.array([[1.0, 0.0], [alpha, 1.0]]),
                                  offset=np.array([0.0, -alpha * mu_y]), order=1,
                                  mode='constant')
    return out


def mnist_deskewed():
    """Deskewed MNIST images (70000, 784) in [0, 1] and their labels."""
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float64) / 255.0
    y = mnist.target.astype(int)
    return deskew(X.reshape(-1, 28, 28)).reshape(len(X), -1), y


def cifar10_images():
    """CIFAR-10 images (60000, 3072) in [0, 1], the five training batches followed by the
    test batch, and their labels."""
    root = os.path.join(DATA_DIR, 'cifar-10-batches-py')
    if not os.path.exists(root):
        os.makedirs(DATA_DIR, exist_ok=True)
        archive = os.path.join(DATA_DIR, 'cifar-10-python.tar.gz')
        if not os.path.exists(archive):
            urllib.request.urlretrieve(CIFAR_URL, archive)
        with tarfile.open(archive, 'r:gz') as tar:
            tar.extractall(path=DATA_DIR)
    X, y = [], []
    for name in [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']:
        with open(os.path.join(root, name), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        X.append(batch[b'data'])
        y.append(batch[b'labels'])
    return np.vstack(X).astype(np.float32) / 255.0, np.concatenate(y).astype(int)


def cifar10_dino():
    """384-dimensional class tokens of a frozen DINOv2 ViT-S/14 for every CIFAR-10 image."""
    images, y = cifar10_images()
    path = os.path.join(DATA_DIR, 'cifar10_dinov2_vits14.npy')
    if os.path.exists(path):
        return np.load(path), y

    import torch
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = images.reshape(-1, 3, 32, 32)
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), 256):
            b = torch.as_tensor(images[i:i + 256], dtype=torch.float32, device=device)
            b = F.interpolate(b, size=(224, 224), mode='bilinear', align_corners=False)
            feats.append(model((b - mean) / std).cpu().numpy())
    X = np.vstack(feats).astype(np.float32)
    np.save(path, X)
    return X, y


def load(dataset):
    if dataset not in _cache:
        _cache[dataset] = {'mnist': mnist_deskewed, 'cifar10': cifar10_dino}[dataset]()
    return _cache[dataset]


def fit_transform(X, dim=None):
    """Center, optionally project onto the first `dim` principal components, and divide by
    the root mean variance over all dimensions."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(0)
    Xc = X - mean
    if dim is None:
        W, var = None, Xc.var(0)
    else:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W, var = Vt.T[:, :dim], ((S ** 2) / max(len(X) - 1, 1))[:dim]
    scale = float(np.sqrt(max(var.mean(), 1e-12)))

    def transform(Z):
        Z = np.asarray(Z, dtype=np.float64) - mean
        return (Z if W is None else Z @ W) / scale

    return transform


def keep_labels(y, labels_per_class, seed):
    """Keep `labels_per_class` labels of every class and set the rest to -1."""
    rng = np.random.RandomState(seed)
    y_semi = np.full_like(y, -1)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < labels_per_class:
            raise ValueError(f'class {c} has only {len(idx)} examples')
        y_semi[rng.choice(idx, size=labels_per_class, replace=False)] = c
    return y_semi


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    K: int
    test_index: np.ndarray
    is_ood: np.ndarray = None
    X_val: np.ndarray = None
    y_val: np.ndarray = None


def split(dataset, seed, labels_per_class=None, held_out_class=None, val_frac=0.0):
    """Train/test split of one seed. `labels_per_class=None` keeps every training label.

    With `held_out_class`, that class is removed from the training split, the remaining
    labels are renumbered to 0..K-2, and held-out test points get the label -1.
    With `val_frac`, a fully labeled validation split is taken from the training split.
    The feature transform is fitted on the training split only."""
    X, y = load(dataset)
    K = len(np.unique(y))
    index = np.arange(len(y))
    Xtr, Xte, ytr, yte, _, ite = train_test_split(X, y, index, test_size=0.2,
                                                  random_state=seed, stratify=y)
    is_ood = None
    if held_out_class is not None:
        is_ood = yte == held_out_class
        keep = ytr != held_out_class
        Xtr, ytr = Xtr[keep], ytr[keep]
        remap = {c: i for i, c in enumerate(sorted(set(ytr.tolist())))}
        ytr = np.array([remap[c] for c in ytr])
        yte = np.array([remap.get(c, -1) for c in yte])
        K -= 1

    Xval, yval = None, None
    if val_frac > 0:
        Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=val_frac,
                                                random_state=seed + 7919, stratify=ytr)

    transform = fit_transform(Xtr, dim=30 if dataset == 'mnist' else None)
    if labels_per_class is not None:
        ytr = keep_labels(ytr, labels_per_class, seed)
    return Split(transform(Xtr), ytr, transform(Xte), yte, K, ite, is_ood,
                 None if Xval is None else transform(Xval), yval)
