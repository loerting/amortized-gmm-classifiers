# Copyright (C) 2026 Leonhard Kohl-Lörting, Artificial Intelligence Lab of the University of Innsbruck.
# Licensed under the Academic Free License version 3.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from emmi import Gaussian


def mlp(in_dim, out_dim, hidden=(256, 128), log_softmax=True):
    layers = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.SiLU()]
        in_dim = h
    layers += [nn.Linear(in_dim, out_dim), nn.LogSoftmax(dim=1) if log_softmax else nn.SiLU()]
    return nn.Sequential(*layers)


def assignment_network(D, K, hidden=64):
    return nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, K))


class MatchedNet(nn.Module):
    """Softmax over C units followed by a bias-free C x K matrix of categorical
    distributions, the inference architecture of the amortized GMM."""

    def __init__(self, in_dim, C, K):
        super().__init__()
        self.encoder = mlp(in_dim, C)
        self.W = nn.Parameter(torch.randn(C, K))

    def forward(self, x):
        q = torch.exp(self.encoder(x))
        return torch.log(q @ torch.softmax(self.W, dim=1) + 1e-10)

    def log_proba_double(self, x):
        """log p(k|x) = log sum_c q(c|x) p(k|c) in double precision."""
        return torch.logsumexp(self.encoder(x).double()[:, :, None]
                               + torch.log_softmax(self.W.double(), dim=1)[None], dim=1)


class NNClassifier:
    """Discriminative baseline, trained on the cross-entropy of the labeled points."""

    def __init__(self, D, C, K, matched=False, lr=1e-3, device=None):
        self.net = (MatchedNet(D, C, K) if matched else
                    nn.Sequential(mlp(D, C, log_softmax=False), nn.Linear(C, K),
                                  nn.LogSoftmax(dim=1))).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.device = device

    def fit(self, X, y, epochs, batch_size):
        self.net.train()
        for _ in range(epochs):
            idx = torch.randperm(len(X), device=self.device)
            for s in range(0, len(X), batch_size):
                b = idx[s:s + batch_size]
                self.optimizer.zero_grad()
                F.nll_loss(self.net(X[b]), y[b], ignore_index=-1).backward()
                self.optimizer.step()

    def predict_log_proba(self, X):
        """log p(k|x) in double precision."""
        self.net.eval()
        with torch.no_grad():
            if isinstance(self.net, MatchedNet):
                return self.net.log_proba_double(X)
            return self.net(X).double()

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(X).double().exp()


class _SemiSupervisedGMM(Gaussian):
    """Gaussian mixture with a diagonal covariance shared by all components, fitted on
    partially labeled data. Unlabeled points carry the label -1."""

    def __init__(self, C, D, K, reg_covar=1e-3, device=None, dtype=torch.float32):
        super().__init__(C=C, D=D, covariance_type='diagonal', shared=True,
                         reg_covar=reg_covar, dtype=dtype, device=device)
        self.K = K

    def fit(self, X, y, epochs, batch_size, rng):
        self._allocate(X, batch_size, save_pjc=False)
        self.initialize(X, rng=rng)
        self._precompute(X)
        for _ in range(epochs):
            self._auxiliary()
            for i, (Xb, pjc) in enumerate(self._dl):
                self._step(Xb, y[self._dl.start[i]:self._dl.end[i]], pjc)
            self._M_step_finalize(len(X))
        self._auxiliary()

    def _class_assignments(self):
        """p(k|c) as a (K, C) tensor."""
        raise NotImplementedError

    def _mvn(self):
        reg = torch.eye(self.D, dtype=self.dtype, device=self.device) * 1e-6
        return torch.distributions.MultivariateNormal(self.means, self.covariances + reg)

    def log_px(self, X, batch_size=512):
        """log p(x) = log sum_c p(c) N(x; mu_c, Sigma)."""
        out = []
        with torch.no_grad():
            for b in range(0, len(X), batch_size):
                Xb = X[b:b + batch_size]
                pjc = torch.empty((len(Xb), self.C), dtype=self.dtype, device=self.device)
                self._E_step_ljs(Xb, pjc)
                out.append(torch.logsumexp(pjc, dim=1))
        return torch.cat(out)


class _TabularAssignments:
    """p(c|k) and p(k) stored as free parameters, updated in closed form from the
    responsibility counts N_ck."""

    def _init_table(self):
        self.phi = torch.ones(self.K, dtype=self.dtype, device=self.device) / self.K
        self.pi_ck = torch.ones((self.K, self.C), dtype=self.dtype, device=self.device) / self.C
        self._N_ck = torch.zeros((self.K, self.C), dtype=self.dtype, device=self.device)

    def _class_assignments(self):
        p_ck = self.pi_ck * self.phi.unsqueeze(1)
        return p_ck / (p_ck.sum(dim=0).unsqueeze(0) + 1e-10)

    def initialize(self, X, rng=None, verbose=False):
        super().initialize(X, rng=rng, verbose=verbose)
        self.prior = (self.pi_ck * self.phi.unsqueeze(1)).sum(dim=0)

    def _M_step_finalize(self, N):
        super()._M_step_finalize(N)
        s = 1e-6
        N_k = self._N_ck.sum(dim=1)
        self.phi = (N_k + s) / (N_k.sum() + s * self.K)
        self.pi_ck = (self._N_ck + s) / (N_k.unsqueeze(1) + s * self.C)
        self.prior = (self.pi_ck * self.phi.unsqueeze(1)).sum(dim=0)
        self._N_ck.zero_()


class BareGMM(_TabularAssignments, _SemiSupervisedGMM):
    """Exact EM with the exact posterior and a class-assignment table."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_table()

    def _step(self, X, y, pjc):
        self._E_step_ljs(X, pjc)
        log_p_xc = pjc.clone()
        log_p_x_given_c = log_p_xc - torch.log(self.prior.unsqueeze(0) + 1e-10)
        labeled = y != -1
        unlabeled = ~labeled
        q = torch.zeros_like(log_p_xc)
        if labeled.any():
            y_l = y[labeled]
            log_joint = log_p_x_given_c[labeled] + torch.log(self.pi_ck[y_l] + 1e-10)
            log_joint -= torch.logsumexp(log_joint, dim=1, keepdim=True)
            q_l = torch.exp(log_joint)
            q[labeled] = q_l
            self._N_ck += F.one_hot(y_l, num_classes=self.K).to(self.dtype).T @ q_l
        if unlabeled.any():
            log_joint = log_p_xc[unlabeled]
            q_u = torch.exp(log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True))
            q[unlabeled] = q_u
            self._N_ck += self._class_assignments() * q_u.sum(dim=0).unsqueeze(0)
        pjc.copy_(q)
        self._M_step_update(X, pjc)

    def predict_log_proba(self, X):
        """log p(k|x) in double precision. With few labels the class posteriors of many inputs
        differ by less than the resolution of log p(x, k) in single precision."""
        with torch.no_grad():
            var = self.variance.double() + 1e-6
            mu = self.means.double()
            prec = 1.0 / var
            X = X.double()
            maha = (X ** 2) @ prec.T - 2 * X @ (mu * prec).T + (mu ** 2 * prec).sum(-1)[None]
            log_p_x_given_c = -0.5 * (maha + torch.log(var).sum(-1)[None]
                                      + self.D * np.log(2 * np.pi))
            log_p_xk = torch.logsumexp(log_p_x_given_c[:, None, :]
                                       + torch.log(self.pi_ck.double() + 1e-10)[None]
                                       + torch.log(self.phi.double() + 1e-10)[None, :, None], 2)
            return log_p_xk - torch.logsumexp(log_p_xk, 1, keepdim=True)

    def predict_proba(self, X):
        return self.predict_log_proba(X).exp()

    def log_px(self, X, batch_size=512):
        with torch.no_grad():
            mvn = self._mvn()
            log_pc = torch.log((self.pi_ck * self.phi.unsqueeze(1)).sum(0) + 1e-10)
            return torch.cat([torch.logsumexp(mvn.log_prob(X[b:b + batch_size].unsqueeze(1))
                                              + log_pc.unsqueeze(0), dim=1)
                              for b in range(0, len(X), batch_size)])


class BareGMMCA(_SemiSupervisedGMM):
    """Exact EM with the exact posterior and amortized class-assignments
    p(k|c) = softmax_k f_psi(mu_c). psi is fitted by gradient ascent on
    sum_{c,k} N_ck log p_psi(k|c) after every M-step, with N_ck from the labeled points."""

    def __init__(self, C, D, K, assignment_lr=1e-3, assignment_steps=20, **kwargs):
        super().__init__(C, D, K, **kwargs)
        self.f_psi = assignment_network(D, K).to(device=self.device, dtype=self.dtype)
        self.psi_optimizer = torch.optim.Adam(self.f_psi.parameters(), lr=assignment_lr)
        self.assignment_steps = assignment_steps
        self._N_ck = torch.zeros((K, C), dtype=self.dtype, device=self.device)

    def _class_assignments(self):
        return torch.softmax(self.f_psi(self.means.detach()), dim=-1).T

    def _step(self, X, y, pjc):
        self._E_step_ljs(X, pjc)
        log_p_xc = pjc.clone()
        labeled = y != -1
        unlabeled = ~labeled
        q = torch.zeros_like(log_p_xc)
        with torch.no_grad():
            log_p_k_given_c = torch.log(self._class_assignments() + 1e-10)
        if labeled.any():
            y_l = y[labeled]
            log_joint = log_p_xc[labeled] + log_p_k_given_c[y_l]
            log_joint = log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)
            q_l = torch.exp(log_joint)
            q[labeled] = q_l
            self._N_ck += F.one_hot(y_l, num_classes=self.K).to(self.dtype).T @ q_l
        if unlabeled.any():
            log_joint = log_p_xc[unlabeled]
            q[unlabeled] = torch.exp(log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True))
        pjc.copy_(q)
        self._M_step_update(X, pjc)

    def _M_step_finalize(self, N):
        super()._M_step_finalize(N)
        if self._N_ck.sum() > 0:
            target = self._N_ck.detach()
            for _ in range(self.assignment_steps):
                self.psi_optimizer.zero_grad()
                log_p = torch.log_softmax(self.f_psi(self.means.detach()), dim=-1).T
                (-(target * log_p).sum() / target.sum()).backward()
                self.psi_optimizer.step()
        self._N_ck.zero_()

    def predict_proba(self, X):
        with torch.no_grad():
            log_pxc = self._mvn().log_prob(X.unsqueeze(1)) + torch.log(self.prior + 1e-10)
            q = torch.exp(log_pxc - torch.logsumexp(log_pxc, dim=1, keepdim=True))
            return q @ self._class_assignments().T

    def log_px(self, X, batch_size=512):
        with torch.no_grad():
            mvn = self._mvn()
            log_pc = torch.log(self.prior + 1e-10)
            return torch.cat([torch.logsumexp(mvn.log_prob(X[b:b + batch_size].unsqueeze(1))
                                              + log_pc.unsqueeze(0), dim=1)
                              for b in range(0, len(X), batch_size)])


class _AmortizedGMM(_SemiSupervisedGMM):
    """Variational EM with the encoder q_Phi(c|x) in place of the exact posterior. The
    encoder takes one gradient step on the Free Energy per batch (partial E-step), the
    Gaussians are updated in closed form from the encoder's responsibilities."""

    def __init__(self, C, D, K, encoder, lr=1e-3, **kwargs):
        super().__init__(C, D, K, **kwargs)
        self.encoder = encoder.to(self.device, self.dtype)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def _log_joint_labeled(self, log_p_x_given_c, log_pc, p_kc, y):
        raise NotImplementedError

    def _psi_step(self, q_sup, p_kc, y):
        return None

    def _step(self, X, y, pjc):
        log_q = self.encoder(X)
        q = torch.exp(log_q)
        self._E_step_ljs(X, pjc)
        log_p_xc = pjc.clone()
        log_pc = torch.log(self.prior.unsqueeze(0) + 1e-10)
        log_p_x_given_c = log_p_xc - log_pc
        labeled = y != -1
        unlabeled = ~labeled
        free_energy = torch.zeros(X.shape[0], dtype=self.dtype, device=self.device)
        p_kc = self._class_assignments()
        q_m = torch.zeros_like(q)
        psi_term = None

        if labeled.any():
            y_l = y[labeled]
            log_joint = self._log_joint_labeled(log_p_x_given_c[labeled], log_pc, p_kc, y_l)
            log_q_sup = log_q[labeled] + torch.log(p_kc[y_l] + 1e-10)
            log_q_sup = log_q_sup - torch.logsumexp(log_q_sup, dim=1, keepdim=True)
            q_sup = torch.exp(log_q_sup)
            free_energy[labeled] = torch.sum(q_sup * (log_joint.detach() - log_q_sup), dim=1)
            q_m[labeled] = q_sup.detach()
            psi_term = self._psi_step(q_sup, p_kc, y_l)

        if unlabeled.any():
            q_u, log_q_u = q[unlabeled], log_q[unlabeled]
            free_energy[unlabeled] = torch.sum(q_u * (log_p_xc[unlabeled].detach() - log_q_u),
                                               dim=1)
            q_m[unlabeled] = q_u.detach()

        loss = -free_energy.mean()
        if psi_term is not None:
            loss = loss - psi_term.mean()
        loss.backward()
        self.encoder_optimizer.step()
        self.encoder_optimizer.zero_grad()
        self._after_backward()

        with torch.no_grad():
            self._accumulate(q_m, p_kc, labeled, y)
            pjc.copy_(q_m)
            self._M_step_update(X, pjc)

    def _after_backward(self):
        pass

    def _accumulate(self, q_m, p_kc, labeled, y):
        pass

    def predict_proba(self, X):
        with torch.no_grad():
            return torch.exp(self.encoder(X)) @ self._class_assignments().T


class AmortizedGMM(_TabularAssignments, _AmortizedGMM):
    """Amortized GMM with a class-assignment table."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_table()

    def _log_joint_labeled(self, log_p_x_given_c, log_pc, p_kc, y):
        return (log_p_x_given_c + torch.log(self.pi_ck[y] + 1e-10)
                + torch.log(self.phi[y] + 1e-10).unsqueeze(1))

    def _accumulate(self, q_m, p_kc, labeled, y):
        if labeled.any():
            Y = F.one_hot(y[labeled], num_classes=self.K).to(self.dtype)
            self._N_ck += Y.T @ q_m[labeled]
        if (~labeled).any():
            self._N_ck += p_kc.detach() * q_m[~labeled].sum(dim=0).unsqueeze(0)


class AmortizedGMMCA(_AmortizedGMM):
    """Amortized GMM with amortized class-assignments p(k|c) = softmax_k f_psi(mu_c), the
    proposed model. psi takes one gradient step per batch on
    sum_c q_sup(c) log p_psi(k=y|c) over the labeled points."""

    def __init__(self, C, D, K, encoder, lr=1e-3, **kwargs):
        super().__init__(C, D, K, encoder, lr=lr, **kwargs)
        self.f_psi = assignment_network(D, K).to(device=self.device, dtype=self.dtype)
        self.psi_optimizer = torch.optim.Adam(self.f_psi.parameters(), lr=lr)

    def _class_assignments(self):
        return torch.softmax(self.f_psi(self.means.detach()), dim=-1).T

    def _log_joint_labeled(self, log_p_x_given_c, log_pc, p_kc, y):
        return log_p_x_given_c + log_pc + torch.log(p_kc + 1e-10)[y].detach()

    def _psi_step(self, q_sup, p_kc, y):
        return torch.sum(q_sup.detach() * torch.log(p_kc + 1e-10)[y], dim=1)

    def _after_backward(self):
        self.psi_optimizer.step()
        self.psi_optimizer.zero_grad()
