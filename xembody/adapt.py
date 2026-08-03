"""A gated corrector: target-arm action -> source-arm action.

Trained on pairs where both arms sat at the SAME TCP, so the source arm's
action is a free label for the target arm's observation.

WHAT THE DATA SAYS THE MODEL SHOULD AND SHOULD NOT DO (450 pairs, two suites):

  approach (phase 0.0-0.2)   0.0% disagreement  -> DO NOT correct
  contact  (phase 0.4-0.6)   26-27%             -> this is the job
  gripper channel            1/450 disagreements-> leave it alone entirely

So the loss is weighted by the source command magnitude: correcting a
near-zero command chases ill-conditioned angles, not behaviour. And the
gripper channel is passed through untouched -- the discrete decision already
transfers, and a regressor would only add noise to it.

DISTRIBUTION SHIFT. Pairs come from MATCHED poses; at deployment the arm is
wherever it drifted to. One round of this is round one of DAgger, not a fix.
"""

import numpy as np

from .pairs import GRIP, WV


def featurise(a_target, tcp):
    """Inputs available at run time. NOT proprioception -- the policy ignores
    it (state swap changed the output by cos 1.000), so feeding it here would
    invite the model to fit noise."""
    a = np.atleast_2d(np.asarray(a_target, np.float32))
    t = np.atleast_2d(np.asarray(tcp, np.float32))
    n = np.linalg.norm(a[:, WV], axis=1, keepdims=True)
    d = a[:, WV] / np.maximum(n, 1e-6)
    return np.concatenate([a[:, :6], d, n, t], axis=1).astype(np.float32)


class Corrector:
    """Small MLP on [action, direction, magnitude, tcp] -> corrected 6-vector."""

    def __init__(self, hidden=64, seed=0):
        self.hidden, self.seed, self.net, self.mu, self.sd = hidden, seed, None, None, None

    def fit(self, X, Y, W, epochs=400, lr=1e-3, val_frac=0.2, verbose=True):
        import torch
        from torch import nn

        g = torch.Generator().manual_seed(self.seed)
        X = torch.as_tensor(X); Y = torch.as_tensor(Y); W = torch.as_tensor(W)
        self.mu, self.sd = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = (X - self.mu) / self.sd
        n = len(X); perm = torch.randperm(n, generator=g)
        nv = int(n * val_frac); vi, ti = perm[:nv], perm[nv:]
        self.net = nn.Sequential(nn.Linear(Xn.shape[1], self.hidden), nn.SiLU(),
                                 nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                                 nn.Linear(self.hidden, 6))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        best, best_state = float("inf"), None
        for ep in range(epochs):
            self.net.train()
            # predict a RESIDUAL: identity is the right prior, since most
            # samples already agree
            pred = X[ti, :6] + self.net(Xn[ti])
            loss = ((pred - Y[ti]) ** 2).sum(1)
            loss = (loss * W[ti]).sum() / W[ti].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 20 == 0 or ep == epochs - 1:
                self.net.eval()
                with torch.no_grad():
                    pv = X[vi, :6] + self.net(Xn[vi])
                    vl = (((pv - Y[vi]) ** 2).sum(1) * W[vi]).sum() / W[vi].sum()
                if vl.item() < best:
                    best = vl.item()
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                if verbose and ep % 100 == 0:
                    print(f"    epoch {ep:>4}  train {loss.item():.5f}  val {vl.item():.5f}")
        if best_state: self.net.load_state_dict(best_state)
        return best

    def __call__(self, a_target, tcp):
        import torch
        X = torch.as_tensor(featurise(a_target, tcp))
        with torch.no_grad():
            out = X[:, :6] + self.net((X - self.mu) / self.sd)
        a = np.atleast_2d(np.asarray(a_target, np.float32)).copy()
        a[:, :6] = out.numpy()
        return a            # gripper channel passed through untouched
