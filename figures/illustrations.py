# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0
"""Figures of the introduction and the background chapter: fig_nn_boundary.pdf,
fig_nn_vs_optimal_boundary.pdf, fig_em_concept.pdf and fig_gmm_em.pdf."""

import warnings

import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

import style
from style import plt

TAB10 = plt.cm.tab10.colors
CLASS_NAMES = ['Class 0', 'Class 1']


def two_ellipses(n_samples=600, seed=42):
    """Two elongated Gaussian classes with equal priors, standardized per dimension. Returns
    the data, the labels and the standardized generative parameters."""
    means = np.array([[-1.0, 0.0], [3.0, -5.0]])
    covs = []
    for theta, scale in zip([np.pi / 4, np.pi / 8], [[3.0, 0.5], [0.5, 4.0]]):
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        covs.append(R @ np.diag(scale) @ R.T)
    np.random.seed(seed)
    X = np.vstack([np.random.multivariate_normal(means[c], covs[c], int(n_samples * 0.5))
                   for c in range(2)])
    y = np.repeat([0, 1], int(n_samples * 0.5))
    mean, std = X.mean(axis=0), X.std(axis=0)
    S = np.diag(1.0 / std)
    return (X - mean) / std, y, ((means - mean) / std, np.array([S @ c @ S for c in covs]))


def grid(X, margin=0.5, steps=200):
    xx, yy = torch.meshgrid(
        torch.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, steps, dtype=torch.float64),
        torch.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, steps, dtype=torch.float64),
        indexing='ij')
    return xx.numpy(), yy.numpy(), torch.stack([xx.ravel(), yy.ravel()], dim=-1)


def bayes_optimal(xx, yy, params):
    means, covs = params
    pos = np.dstack((xx, yy))
    return np.argmax(np.stack([0.5 * multivariate_normal(means[c], covs[c]).pdf(pos)
                               for c in range(2)], axis=2), axis=2)


def regions(ax, xx, yy, pred, **kwargs):
    cmap = mcolors.ListedColormap([mcolors.to_rgba(TAB10[c], alpha=0.2) for c in range(2)])
    ax.contourf(xx, yy, pred, cmap=cmap)
    ax.contour(xx, yy, pred, levels=[0.5], colors='white', linewidths=1.5, **kwargs)


def scatter(ax, X, y, legend=True, zorder=4):
    for c in range(2):
        ax.scatter(X[y == c, 0], X[y == c, 1], color=TAB10[c], edgecolor='white', linewidth=0.5,
                   s=20, label=CLASS_NAMES[c] if legend else None, zorder=zorder)


def blank(ax, title, ylabel=False):
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Feature 1')
    if ylabel:
        ax.set_ylabel('Feature 2')


def nn_boundary(n_samples=300, data_seed=1, torch_seed=0, epochs=(5, 20, 200)):
    """A ReLU network trained on all labels, at three stages of training, and its final
    boundary next to the Bayes-optimal one."""
    X, y, params = two_ellipses(n_samples, data_seed)
    xx, yy, G = grid(X)
    torch.manual_seed(torch_seed)
    net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, 2), nn.LogSoftmax(dim=1))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def boundary():
        with torch.no_grad():
            return net(G.float()).argmax(1).numpy().reshape(xx.shape)

    style.setup(0.33)
    fig, axes = plt.subplots(1, 3)
    for epoch in range(1, max(epochs) + 1):
        optimizer.zero_grad()
        nn.functional.nll_loss(net(Xt), yt).backward()
        optimizer.step()
        if epoch in epochs:
            i = epochs.index(epoch)
            regions(axes[i], xx, yy, boundary())
            scatter(axes[i], X, y, legend=i == 0)
            blank(axes[i], f'Epoch {epoch}', ylabel=i == 0)
    axes[0].legend(loc='upper left', framealpha=0.9, prop={'size': 8})
    plt.tight_layout()
    style.save(fig, 'fig_nn_boundary.pdf')

    style.setup(0.45)
    fig, axes = plt.subplots(1, 2)
    for ax, pred, title in ((axes[0], boundary(), 'Neural Network Boundary'),
                            (axes[1], bayes_optimal(xx, yy, params), 'Bayes-optimal Boundary')):
        regions(ax, xx, yy, pred)
        scatter(ax, X, y, legend=ax is axes[0])
        blank(ax, title, ylabel=ax is axes[0])
    axes[0].legend(loc='upper left', framealpha=0.9, prop={'size': 8})
    plt.tight_layout()
    style.save(fig, 'fig_nn_vs_optimal_boundary.pdf')


def em_concept():
    """Coordinate ascent of a Free Energy landscape over Theta and q from two initializations,
    one of which ends in a local optimum. Illustration inspired by the lecture Theoretical
    Foundations of Probabilistic Machine Learning and AI of Jörg Lücke."""
    style.setup(0.6)
    fig, ax = plt.subplots()
    theta = np.linspace(0.0, 4.0, 400)
    q = np.linspace(0.0, 4.0, 400)
    T, Q = np.meshgrid(theta, q)
    pos = np.dstack((T, Q))
    Z = (multivariate_normal([3.5, 3.6], [[0.25, 0.0], [0.0, 0.25]]).pdf(pos) * 2.5
         + multivariate_normal([2.6, 2.7], [[0.3, 0.1], [0.1, 0.3]]).pdf(pos) * 2.5
         + multivariate_normal([1.6, 1.0], [[0.8, 0.7], [0.7, 0.8]]).pdf(pos) * 1.2
         + multivariate_normal([0.7, 2.5], [[0.6, 0.5], [0.5, 0.6]]).pdf(pos) * 0.8)
    ax.contourf(T, Q, Z, levels=30, cmap=plt.cm.Purples, alpha=0.85)
    ax.contour(T, Q, Z, levels=15, colors='white', linewidths=0.4, alpha=0.7)
    q_best, t_best = np.unravel_index(np.argmax(Z), Z.shape)
    best = (theta[t_best], q[q_best])

    for start, (t0, q0) in enumerate([(0.2, 0.2), (3.0, 1.7)]):
        t_i, q_i = np.argmin(np.abs(theta - t0)), np.argmin(np.abs(q - q0))
        path = [(t_i, q_i)]
        for _ in range(8):
            q_i = np.argmax(Z[:, t_i])
            path.append((t_i, q_i))
            t_i = np.argmax(Z[q_i, :])
            path.append((t_i, q_i))
        path = np.array([[theta[a], q[b]] for a, b in path])
        for i in range(len(path) - 1):
            (x0, y0), (x1, y1) = path[i], path[i + 1]
            if np.hypot(x1 - x0, y1 - y0) < 0.02:
                break
            color = TAB10[2] if i % 2 == 0 else TAB10[3]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5, mutation_scale=12,
                                        shrinkA=0, shrinkB=0))
            if start == 0 and i < 2:
                ax.plot([], [], color=color, lw=1.5,
                        label='E-Step (Optimize $q$)' if i == 0 else r'M-Step (Optimize $\Theta$)')
        if np.hypot(*(path[-1] - best)) > 0.1:
            ax.scatter(*path[-1], color=TAB10[4], edgecolor='white', marker='*', s=200,
                       linewidths=0.5, label='Local Optimum' if start == 0 else None, zorder=6)
        ax.scatter(*path[0], color='white', edgecolor='black', marker='o', s=50, linewidths=0.5,
                   zorder=6, label='Initialization' if start == 0 else None)
    ax.scatter(*best, color=TAB10[8], edgecolor='white', marker='*', s=200, linewidths=0.5,
               label='Global Optimum', zorder=5)

    ax.set_xlabel(r'Model Parameters $\Theta$')
    ax.set_ylabel(r'Variational Distribution $q$')
    ax.set_xlim([0.0, 4.0])
    ax.set_ylim([0.0, 4.0])
    ax.set_xticks([])
    ax.set_yticks([])
    for xy, xytext in (((1.03, 0), (1.0, 0)), ((0, 1.03), (0, 1.0))):
        ax.annotate('', xy=xy, xytext=xytext,
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=1.0, mutation_scale=12),
                    annotation_clip=False)
    ax.legend(loc='lower right', framealpha=0.95, prop={'size': 10})
    plt.tight_layout()
    style.save(fig, 'fig_em_concept.pdf')


def gmm_em(iterations=(1, 8, 15)):
    """EM fitting a two-component GMM with full covariances; dotted is the Bayes-optimal
    boundary."""
    X, y, params = two_ellipses(600, 42)
    xx, yy, G = grid(X)
    optimal = bayes_optimal(xx, yy, params)
    gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=1, warm_start=True,
                          random_state=42, init_params='random')
    style.setup(0.33)
    fig, axes = plt.subplots(1, 3)
    pos = np.dstack((xx, yy))
    done = 0
    for ax, it in zip(axes, iterations):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            for _ in range(it - done):
                gmm.fit(X)
        done = it
        order = np.argsort(gmm.means_[:, 0])
        rank = np.empty(2, dtype=int)
        rank[order] = np.arange(2)
        regions(ax, xx, yy, rank[gmm.predict(G.numpy())].reshape(xx.shape))
        ax.contour(xx, yy, optimal, levels=[0.5], colors='white', linewidths=1.5,
                   linestyles='dotted', alpha=0.8)
        scatter(ax, X, y, legend=ax is axes[0], zorder=1)
        for c in order:
            ax.contour(xx, yy, multivariate_normal(gmm.means_[c], gmm.covariances_[c]).pdf(pos),
                       levels=4, colors=['#666666'], alpha=1.0, linewidths=0.6)
        blank(ax, f'EM Iteration {it}', ylabel=ax is axes[0])
    axes[0].legend(loc='upper left', framealpha=0.9, prop={'size': 8})
    plt.tight_layout()
    style.save(fig, 'fig_gmm_em.pdf')


if __name__ == '__main__':
    nn_boundary()
    em_concept()
    gmm_em()
