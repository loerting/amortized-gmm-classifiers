# Neural Network Classifiers Trained as Amortized Mixture Models

Code for the master's thesis *Neural Network Classifiers Trained as Amortized Mixture Models*
by Leonhard Kohl-Lörting, written at the Artificial Intelligence Lab of the University of
Innsbruck and supervised by Univ.-Prof. Dr. Jörg Lücke (2026). It contains the model, the three
experiments of the thesis, the results they produced, and the code for every figure and table.

## The model

A class $k$ generates a mixture component $c$, and the component generates the observation
$x$ from a Gaussian $\mathcal{N}(x; \mu_c, \Sigma)$ with a diagonal covariance shared by all
components. The model is fitted to labeled and unlabeled data by variational EM:

- an encoder $q_\Phi(c|x)$ (an MLP ending in a softmax over the $C$ components) replaces the
  exact posterior and takes one gradient step on the Free Energy per batch,
- the means, the covariance and the component prior are updated in closed form,
- the class-assignments $p(k|c) = \mathrm{softmax}_k f_\Psi(\mu_c)$ are computed by a small
  network from the component means, so that nearby components share their label evidence.

After training, the prediction $p(k|x) = \sum_c q_\Phi(c|x)\,p(k|c)$ is an ordinary
feedforward network, and $\log p(x)$ of the mixture scores how plausible an input is.

## Installation

```bash
git clone https://github.com/loerting/amortized-gmm-classifiers.git
cd amortized-gmm-classifiers
pip install -e .
```

This installs the package `amogmm` and its dependencies, including the Gaussian mixture
library [emmi](https://github.com/variational-sublinear-clustering/emmi) at the commit used for
the thesis. The figures are typeset with LaTeX (`latex`, `lmodern`, `anyfontsize`) when it is
available.

Tested with Python 3.12, PyTorch 2.6.0 (CUDA 12.4), NumPy 1.26.4, SciPy 1.14.1,
scikit-learn 1.5.0 and matplotlib 3.9.2. The thesis experiments ran on NVIDIA L40S GPUs.

## Data

Both datasets are downloaded on first use.

- **MNIST** is fetched from OpenML through scikit-learn, deskewed, and projected onto its first
  30 principal components, fitted on the training split.
- **CIFAR-10** is downloaded to `data/`, and the class tokens of a frozen DINOv2 ViT-S/14 are
  extracted once and cached in `data/cifar10_dinov2_vits14.npy` (a GPU is recommended). The
  features of a different GPU differ slightly from the ones used in the thesis.

Set `AMOGMM_DATA` to keep the data elsewhere.

## Models

| code         | thesis               |
|--------------|----------------------|
| `nn`         | NN baseline          |
| `nn_matched` | NN, matched          |
| `bare`       | Bare GMM             |
| `bare_ca`    | Bare GMM + CA        |
| `amo`        | Amortized GMM        |
| `amo_ca`     | Amortized GMM + CA   |
| `nn_ts`      | NN + temp. scaling   |

`amogmm.train.fit(dataset, model, labels, seed)` trains one model, where `dataset` is `mnist`
or `cifar10` and `labels` is the number of labels per class or `'all'`. The seed fixes the
train/test split, the labeled subset and the initialization.

## Experiments

Each script runs the full grid of the thesis by default (20 seeds, 500 epochs) and writes its
results to `results/`, one row per training run.

| command                              | output                         | thesis               |
|--------------------------------------|--------------------------------|----------------------|
| `python experiments/semi_supervised.py` | `results/semi_supervised.json` | Section 5.2      |
| `python experiments/ood.py`          | `results/ood.json`             | Section 5.3          |
| `python experiments/screening.py`    | `results/screening.json`       | Section 5.4          |
| `python experiments/screening_tail.py` | `results/screening_tail.json` | Section 5.4         |
| `python experiments/examples.py`     | `results/examples/*.npz`       | Figures 5.2, 5.4, 5.5, 5.9, 5.10, 5.13 |

The full grid has about 15,000 training runs. A generative model takes between half a minute
and a few minutes per run on a GPU. `semi_supervised.py`, `ood.py` and `screening.py` accept
`--datasets`, `--models`, `--labels` and `--seeds` (`ood.py` also `--folds`), so the grid can be
split into jobs, each writing to its own file such as `results/ood-part3.json`. All files `results/<name>.json` and
`results/<name>-*.json` are read together, and a run that appears twice is taken from the
later file.

The detection control with the Bare GMM + CA in Section 5.3 is not part of the grid:

```bash
python experiments/ood.py --labels 1 --models bare_ca --out results/ood_control.json
```

## Figures and tables

```bash
python figures/illustrations.py     # Figures 1.1, 1.2, 3.2, 3.3
python figures/semi_supervised.py   # Figures 5.2, 5.3, 5.4
python figures/ood.py               # Figures 5.5 to 5.8
python figures/screening.py         # Figures 5.9 to 5.13
python figures/result_tables.py     # Tables B.1 to B.6
python figures/text_numbers.py      # the numbers quoted in the text of Chapter 5
```

The illustration of the EM algorithm (Figure 3.2) is inspired by the lecture *Theoretical
Foundations of Probabilistic Machine Learning and AI* of Jörg Lücke.

The outputs are written to `figures/output/`. They are computed from the files in `results/`,
which hold the runs of the thesis, so the figures can be regenerated without training anything.

## Results

- `semi_supervised.json`: test accuracy of every model, supervision level and seed.
- `ood.json`: for every held-out class, the AUROC, the FPR@95TPR and the ROC curve (TPR on a
  grid of 101 false positive rates) of the maximum softmax probability (`msp`) and, for the
  generative models, of `log_px`.
- `screening.json`: for every critical class, the FPR@95TPR, the share of the test set flagged
  at 95% TPR, the ECE, the fifteen equal-mass bins of the reliability diagram, the share of
  positives caught by the 3% rule, and the number of predictions between 0.1% and 50%.
- `screening_tail.json`: the small probabilities of the NN baseline after 20, 100 and 500
  epochs, and the threshold that recovers the positives missed by the 3% rule.
- `examples/`: the single-seed runs behind the illustrative figures.

Training is deterministic for a fixed GPU and software version. On a different GPU, rerunning
gives per-seed accuracies that differ in the third or fourth decimal.

## License

Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of
Innsbruck.

Licensed under the Academic Free License version 3.0, see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{kohl-loerting2026amortized,
  author       = {Kohl-L{\"o}rting, Leonhard and Kahlke, Till and L{\"u}cke, J{\"o}rg},
  title        = {Neural Network Classifiers Trained as Amortized Mixture Models},
  howpublished = {Artificial Intelligence Lab, University of Innsbruck},
  year         = {2026}
}
```
