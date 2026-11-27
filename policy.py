#!/usr/bin/env python3
"""
Step 4: the learned policy.

Inputs (24), exactly the information the MPC uses:
    e_y, e_psi, v, w                       path-frame state and wheel speeds
    kappa(s), kappa preview x 8            curvature now and 0.4 ... 3.2 m ahead
    d(s), d preview x 8                    corridor half-width now and ahead
    v_c,prev, w_c,prev                     last applied command (the MPC limits how fast it changes)
Outputs (2): v_c, w_c  -- or, for a "delta" network, the change (dv_c, dw_c) from the last command,
clipped to the rate limits (cannot break them by construction).

Network: a small multilayer perceptron (2 hidden layers x 64, tanh), trained with scikit-learn on
standardised inputs and outputs. At run time the forward pass is plain numpy (a few small matrix
products), so its timing is comparable with the MPC's. Outputs are clipped to the input limits,
the "projection" of Chen et al. (2018) in its simplest form.
"""
import json
import os
import time

import numpy as np

import sim

FEATURES = (["e_y", "e_psi", "v", "w", "kappa"] + [f"kappa_{i}" for i in range(8)]
            + ["d"] + [f"d_{i}" for i in range(8)] + ["v_c_prev", "w_c_prev"])
HIDDEN = (64, 64)
UPREV = slice(FEATURES.index("v_c_prev"), FEATURES.index("w_c_prev") + 1)
RATE = np.array([sim.LIMITS["acc_max"], sim.LIMITS["alpha_max"]]) * sim.DT_CTRL   # max change per step


def features(X, info, u_prev):
    return np.concatenate([[info["e_y"], info["e_psi"], X[3], X[4], info["kappa"]], info["preview"],
                           [info["d"]], info["d_preview"], u_prev])


class Policy:
    """Numpy MLP: y = W3 tanh(W2 tanh(W1 x' + b1) + b2) + b3, with standardisation around it."""

    def __init__(self, params):
        self.p = {k: np.asarray(v) for k, v in params.items()}
        self.u_prev = np.zeros(2)
        self.times = []

    @classmethod
    def fit(cls, Xf, U, seed=0, max_iter=400, delta=False):
        """delta=False: the network outputs the command u.  delta=True: it outputs the CHANGE
        du = u - u_prev, clipped at run time to the rate limits, so its commands can never change
        faster than allowed (a rate-feasible output layer, cf. Chen et al. 2018)."""
        from sklearn.neural_network import MLPRegressor
        if delta:
            U = U - Xf[:, UPREV]
        xm, xs = Xf.mean(0), Xf.std(0) + 1e-6
        um, us = U.mean(0), U.std(0) + 1e-6
        net = MLPRegressor(hidden_layer_sizes=HIDDEN, activation="tanh", solver="adam", learning_rate_init=1e-3,
                           batch_size=256, max_iter=max_iter, early_stopping=True, validation_fraction=0.1,
                           n_iter_no_change=25, random_state=seed)
        net.fit((Xf - xm) / xs, (U - um) / us)
        params = dict(xm=xm, xs=xs, um=um, us=us, delta=float(delta))
        for i, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
            params[f"W{i}"], params[f"b{i}"] = W, b
        pol = cls(params)
        pol.train_info = dict(n_samples=int(len(Xf)), epochs=int(net.n_iter_),
                              best_val_r2=float(net.best_validation_score_))
        return pol

    def predict(self, f):
        p = self.p
        h = (f - p["xm"]) / p["xs"]
        n_layers = sum(1 for k in p if k.startswith("W"))
        for i in range(n_layers):
            h = h @ p[f"W{i}"] + p[f"b{i}"]
            if i < n_layers - 1:
                h = np.tanh(h)
        out = h * p["us"] + p["um"]
        if float(p.get("delta", 0.0)):
            return f[UPREV] + np.clip(out, -RATE, RATE)
        return out

    def reset(self):
        self.u_prev = np.zeros(2)
        self.times = []

    def __call__(self, P, X, info):
        t0 = time.perf_counter()
        u = sim.clip_command(self.predict(features(X, info, self.u_prev)))
        self.times.append(time.perf_counter() - t0)
        self.u_prev = u
        return u

    def save(self, path):
        json.dump({k: v.tolist() for k, v in self.p.items()}, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))


def record_mpc(P, robot, ctrl, X0=None, s_init=0.0, t_max=None):
    """Run the MPC in closed loop and record (features, MPC command) at every control step."""
    feats, labels = [], []

    def recorder(P_, X, info):
        f = features(X, info, ctrl.u_prev.copy())
        u = ctrl(P_, X, info)
        feats.append(f)
        labels.append(u.copy())
        return u

    log = sim.run(P, robot, recorder, X0=X0, s_init=s_init, t_max=t_max)
    return np.array(feats), np.array(labels), log
