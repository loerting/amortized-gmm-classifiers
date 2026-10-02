# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from amogmm.train import NAMES  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
TEXTWIDTH_PT = 418.25
GRAY = '#8d8b86'
GRID = dict(color='#d8d7d2', linewidth=0.6)
DATASETS = ['mnist', 'cifar10']
TITLE = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10'}
ALL_COUNT = {'mnist': 5600, 'cifar10': 4800}
COLOR = {'nn': '#52514e', 'nn_matched': '#52514e', 'nn_ts': '#8a70c9', 'bare': '#eda100',
         'bare_ca': '#1baf7a', 'amo': '#2a78d6', 'amo_ca': '#eb6834'}
MARKER = {'nn': 'x', 'nn_matched': '+', 'nn_ts': 'P', 'bare': 'o', 'bare_ca': 'D', 'amo': 's',
          'amo_ca': 'v'}
LINESTYLE = {'nn': '--', 'nn_matched': ':', 'nn_ts': ':', 'bare': '-', 'bare_ca': '-.',
             'amo': '-', 'amo_ca': '-.'}
LABEL = NAMES


def setup(aspect_ratio=0.45, fontsize=11):
    """Figure width of the thesis text width, LaTeX text in the document font size."""
    width = TEXTWIDTH_PT / 72.27
    usetex = shutil.which('latex') is not None
    plt.rcParams.update({
        'figure.figsize': (width, width * aspect_ratio),
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'text.usetex': usetex,
        'text.latex.preamble': r'\usepackage{lmodern} \usepackage{anyfontsize} \usepackage{amsmath}',
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'] if usetex else ['DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': fontsize,
        'figure.titlesize': fontsize,
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
        'legend.fontsize': fontsize - 1,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333',
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
    })


def pct():
    return r'\%' if plt.rcParams['text.usetex'] else '%'


def pct_label(v):
    if v == 0:
        return r'$0\%$'
    if abs(v) >= 0.01:
        return f'${v:g}\\%$'
    return f'$10^{{{int(round(np.log10(abs(v))))}}}\\%$'


def pct_formatter(scale=1.0):
    """Ticks in percent. `scale` converts the data to percent."""
    return FuncFormatter(lambda v, _: pct_label(v * scale))


def decorate(ax, spine_color=GRAY):
    ax.grid(True, axis='y', **GRID)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(spine_color)


def count(labels, dataset):
    """Position of a supervision level on the logarithmic axis."""
    return ALL_COUNT[dataset] if labels == 'all' else int(labels)


def log_label_axis(ax, dataset, levels, keep=None):
    xs = [count(lv, dataset) for lv in levels]
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels(['all' if lv == 'all' else f'{x}' if keep is None or x in keep else ''
                        for x, lv in zip(xs, levels)])
    ax.minorticks_off()
    return xs


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)
