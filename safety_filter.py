#!/usr/bin/env python3
"""
Step 6: predictive safety filter (proposal section 5.5).

Every control period the network proposes u_NN = (v_c, w_c). The filter asks: if that command is held
for the next N = 20 steps (2 s), does the robot stay inside the corridor, the input limits and the
rate limits, and end in a safe state (heading along the path, turning only as much as the path needs)?
If yes, u_NN passes unchanged. If not, it solves a small QP for the smallest change:

    min   || W (u_0 - u_NN) ||^2  +  eps * sum_k>0 || du_k ||^2  +  rho1 * sum s_k  +  rho2 * sum s_k^2
    s.t.  z_{k+1} = zbar_{k+1} + A_k (z_k - zbar_k) + B_k (u_k - ubar_k)   (linearised along the nominal rollout)
          u_min <= u_k <= u_max,   |u_k - u_{k-1}| <= du_max   (u_{-1} = last applied command)
          -d_k - s_k <= e_y,k <= d_k + s_k,   s_k >= 0,   k = 1 ... N
          |e_psi,N| <= TERM_PSI + s_N,   |w_N - a v_N kappa_N / b| <= TERM_W + s_N      (terminal safe set)

The terminal set is what makes a short-horizon filter safe beyond its horizon (Wabersich & Zeilinger):
a state that is aligned with the path and turning with it can be held there. The QP is re-linearised
up to SQP_ITERS times around its own latest solution.

The model is the MPC's path-frame model (the fold's a, b and tau), so the filter knows no more than
the MPC. Safety check = u_NN held for N steps. If that is unsafe, the QP is linearised around a backup trajectory
(u_NN for one step, then a simple steer-back-to-the-path law) and re-linearised around its own solution;
(A_k, B_k) are exact Jacobians (CasADi). States
are eliminated (condensed QP: 2N inputs + N slacks), solved with OSQP.

Slack s keeps the QP feasible when no input can keep the robot inside (it is already at or past the
edge); its use is logged. Recovery: if the robot is already outside the corridor, the term that keeps
u_0 close to u_NN is nearly switched off, so the QP only minimises the slack, i.e. returns inside as
fast as the limits allow.

Guarantee, stated honestly: constraint satisfaction holds for the nominal linearised model within the
0.8 s horizon. No recursive-feasibility proof, no robustness to model error; step 7 tests both.
"""
import time

import casadi as ca
import numpy as np
import osqp
import scipy.sparse as sp

import sim

N = 20                                  # horizon: 20 x 0.1 s = 2 s (as the MPC). 0.8 s was too short:
                                        # unwinding a full-rate turn under the rate limit takes 1.5-2 s
SQP_ITERS = 3                           # re-linearise around the latest solution (far from the nominal
                                        # rollout after a bad command, one linearisation is not enough)
TERM_PSI = 0.10                         # terminal set: |e_psi_N| <= 0.10 rad and the turn rate within
TERM_W = 0.15                           # 0.15 rad/s of what the path itself needs (a v kappa / b)
DT = sim.DT_CTRL
W_U = np.array([1.0, 1.0])              # weights on (v_c, w_c) deviation from u_NN
EPS = 1e-2                              # small weight on later inputs (keeps the QP well posed)
RHO1, RHO2 = 1e2, 1e3                   # slack penalties (linear + quadratic); 1e3/1e4 made OSQP hit its
                                        # iteration limit (badly scaled QP)
K_EY, K_EPSI = 1.0, 2.5                 # backup controller gains (see _backup)
RECOVERY_W = 1e-3                       # weight on |u_0 - u_NN| while outside the corridor
TOL = 1e-6


class SafetyFilter:
    def __init__(self, a, b, tau=0.2, n=N):
        self.a, self.b, self.tau, self.n = a, b, tau, n
        L = sim.LIMITS
        self.u_lo = np.array([L["v_c_min"], -L["w_c_max"]])
        self.u_hi = np.array([L["v_c_max"], L["w_c_max"]])
        self.du = np.array([L["acc_max"], L["alpha_max"]]) * DT
        self._build_model()
        self.reset()

    # ------------------------------------------------------------------ model and its Jacobians
    def _build_model(self):
        z, u, k = ca.SX.sym("z", 5), ca.SX.sym("u", 2), ca.SX.sym("k")

        def f(z, u):
            sdot = self.a * z[3] * ca.cos(z[2]) / (1 - k * z[1])
            return ca.vertcat(sdot, self.a * z[3] * ca.sin(z[2]), self.b * z[4] - k * sdot,
                              (u[0] - z[3]) / self.tau, (u[1] - z[4]) / self.tau)
        h = DT / 2
        zn = z
        for _ in range(2):                                   # RK4, two sub-steps per period (as the MPC)
            k1 = f(zn, u); k2 = f(zn + h / 2 * k1, u); k3 = f(zn + h / 2 * k2, u); k4 = f(zn + h * k3, u)
            zn = zn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.F = ca.Function("F", [z, u, k], [zn])
        self.FA = ca.Function("FA", [z, u, k], [ca.jacobian(zn, z)])
        self.FB = ca.Function("FB", [z, u, k], [ca.jacobian(zn, u)])

    def reset(self):
        self.stats = dict(steps=0, active=0, slack_steps=0, max_slack=0.0, recovery=0, qp_fail=0,
                          times=[], dev_v=[], dev_w=[])

    # --------------------------------------------------------------------------- one decision
    def _horizon(self, info, s_pred):
        rel = np.concatenate([[0.0], sim.PREVIEW_DIST])
        kap_tab = np.concatenate([[info["kappa"]], info["preview"]])
        d_tab = np.concatenate([[info["d"]], info["d_preview"]])
        ds = np.clip(s_pred - info["s"], 0, rel[-1])
        kap = np.interp(ds, rel, kap_tab)
        d = d_tab[np.clip(np.searchsorted(rel, ds, side="right") - 1, 0, len(rel) - 1)]
        return kap, d

    def _rollout(self, z0, U, info):
        """Nominal rollout with inputs U (n x 2); curvature from the preview at the predicted progress."""
        Z = np.zeros((self.n + 1, 5)); Z[0] = z0
        kap = np.zeros(self.n)
        for it in range(2):                                  # 2 passes: progress -> curvature -> progress
            s_pred = Z[:-1, 0] if it else z0[0] + self.a * max(z0[3], 0) * DT * np.arange(self.n)
            kap, d = self._horizon(info, s_pred)
            for k in range(self.n):
                Z[k + 1] = np.array(self.F(Z[k], U[k], kap[k])).ravel()
        return Z, kap, d

    def _terminal_ref(self, Z, kap):
        return self.a * Z[-1, 3] * kap[-1] / self.b      # turn rate the path needs at the horizon end

    def _backup(self, z0, u_first, u_prev, info):
        """Linearisation point when u_NN is unsafe: u_NN (rate-limited) for the first step, then a simple
        stabilising law back to the path, w_c = a v kappa / b - K_EY e_y - K_EPSI e_psi, speed held.
        Starting the SQP near a safe trajectory keeps the linearisation accurate (holding a bad
        command for 2 s, e.g. a full-rate turn, would linearise around a circle)."""
        Z = np.zeros((self.n + 1, 5)); Z[0] = z0
        U = np.zeros((self.n, 2))
        prev = np.asarray(u_prev, float)
        kap, d = self._horizon(info, z0[0] + self.a * max(z0[3], 0) * DT * np.arange(self.n))
        for k in range(self.n):
            if k == 0:
                u = np.clip(u_first, prev - self.du, prev + self.du)
            else:
                e_y, e_psi, v = Z[k, 1], Z[k, 2], Z[k, 3]
                w = self.a * v * kap[k] / self.b - K_EY * e_y - K_EPSI * e_psi
                u = np.clip(np.array([prev[0], w]), prev - self.du, prev + self.du)
            u = np.clip(u, self.u_lo, self.u_hi)
            U[k] = u
            prev = u
            Z[k + 1] = np.array(self.F(Z[k], u, kap[k])).ravel()
        return U

    def _safe(self, Z, U, u_prev, d, kap):
        ok_in = np.all(np.abs(Z[1:, 1]) <= d + TOL)
        ok_t = abs(Z[-1, 2]) <= TERM_PSI + TOL and abs(Z[-1, 4] - self._terminal_ref(Z, kap)) <= TERM_W + TOL
        ok_u = np.all(U >= self.u_lo - TOL) and np.all(U <= self.u_hi + TOL)
        dU = np.diff(np.vstack([u_prev, U]), axis=0)
        ok_r = np.all(np.abs(dU) <= self.du + TOL)
        return ok_in and ok_t and ok_u and ok_r

    def __call__(self, X, info, u_nn, u_prev):
        t0 = time.perf_counter()
        self.stats["steps"] += 1
        z0 = np.array([info["s"], info["e_y"], info["e_psi"], X[3], X[4]])
        Ubar = np.tile(np.asarray(u_nn, float), (self.n, 1))
        Z, kap, d = self._rollout(z0, Ubar, info)
        if self._safe(Z, Ubar, u_prev, d, kap):              # inactive: the network's command is safe
            self.stats["times"].append(time.perf_counter() - t0)
            return np.asarray(u_nn, float)

        outside = abs(z0[1]) > info["d"]
        self.stats["active"] += 1
        self.stats["recovery"] += int(outside)
        n, nu = self.n, 2 * self.n
        u_ref = np.asarray(u_nn, float)
        U_lin = self._backup(z0, u_ref, u_prev, info)          # linearisation point (inputs)
        Z, kap, d = self._rollout(z0, U_lin, info)
        smax, ok = 0.0, False
        for it in range(SQP_ITERS):
            if it > 0:
                Z, kap, d = self._rollout(z0, U_lin, info)
            x, status = None, None
            x, status = self._qp(Z, U_lin, kap, d, z0, u_ref, u_prev, outside)
            if x is None:
                break
            ok = True
            step = x[:nu].reshape(n, 2)
            U_lin = np.clip(U_lin + step, self.u_lo, self.u_hi)
            smax = float(np.max(x[nu:]))
            if np.max(np.abs(step)) < 1e-3:
                break
        if not ok:
            self.stats["qp_fail"] += 1
            u = np.clip(u_ref, np.maximum(self.u_lo, u_prev - self.du),
                        np.minimum(self.u_hi, u_prev + self.du))           # fallback: clip to limits
        else:
            u = U_lin[0]
            if smax > 1e-4:
                self.stats["slack_steps"] += 1
                self.stats["max_slack"] = max(self.stats["max_slack"], smax)
        u = np.clip(u, self.u_lo, self.u_hi)
        self.stats["dev_v"].append(abs(u[0] - u_nn[0])); self.stats["dev_w"].append(abs(u[1] - u_nn[1]))
        self.stats["times"].append(time.perf_counter() - t0)
        return u

    def _qp(self, Z, Ulin, kap, d, z0, u_ref, u_prev, outside):
        """One condensed QP around the rollout (Z, Ulin). Decision x = [du_0..du_{N-1}, s_1..s_N],
        u_k = Ulin_k + du_k. Returns (x, status) or (None, status)."""
        n, nu = self.n, 2 * self.n
        A = [np.array(self.FA(Z[k], Ulin[k], kap[k])) for k in range(n)]
        B = [np.array(self.FB(Z[k], Ulin[k], kap[k])) for k in range(n)]
        Phi = np.zeros((5, nu))
        G = np.zeros((n, nu))
        for k in range(n):
            Phi = A[k] @ Phi
            Phi[:, 2 * k:2 * k + 2] += B[k]
            G[k] = Phi[1]
        G_psiN, G_wN, G_vN = Phi[2], Phi[4], Phi[3]          # sensitivities of the terminal state
        ebar = Z[1:, 1]
        w0 = (RECOVERY_W if outside else 1.0) * W_U
        Pd = np.concatenate([w0, np.full(nu - 2, EPS), np.full(n, RHO2)])
        P = sp.diags(2 * Pd, format="csc")
        q = np.concatenate([2 * w0 * (Ulin[0] - u_ref), np.zeros(nu - 2), np.full(n, RHO1)])
        Z0 = sp.csc_matrix((nu, n))
        rows, lo, hi = [], [], []
        rows.append(sp.hstack([sp.eye(nu), Z0]))
        lo.append(np.tile(self.u_lo, n) - Ulin.ravel()); hi.append(np.tile(self.u_hi, n) - Ulin.ravel())
        D = sp.eye(nu) - sp.eye(nu, k=-2)
        rows.append(sp.hstack([D, Z0]))
        base = np.diff(np.vstack([u_prev, Ulin]), axis=0).ravel()
        lo.append(-np.tile(self.du, n) - base); hi.append(np.tile(self.du, n) - base)
        Gs = sp.csc_matrix(G)
        rows.append(sp.hstack([Gs, -sp.eye(n)])); lo.append(np.full(n, -np.inf)); hi.append(d - ebar)
        rows.append(sp.hstack([Gs, sp.eye(n)])); lo.append(-d - ebar); hi.append(np.full(n, np.inf))
        eN = sp.csc_matrix(([1.0], ([0], [n - 1])), shape=(1, n))          # terminal rows share slack s_N
        rows.append(sp.hstack([sp.csc_matrix(G_psiN), -eN])); lo.append([-np.inf]); hi.append([TERM_PSI - Z[-1, 2]])
        rows.append(sp.hstack([sp.csc_matrix(G_psiN), eN])); lo.append([-TERM_PSI - Z[-1, 2]]); hi.append([np.inf])
        k_ab = self.a * kap[-1] / self.b
        g_w = G_wN - k_ab * G_vN                                            # d/du of (w_N - a v_N kappa_N / b)
        w_gap = Z[-1, 4] - k_ab * Z[-1, 3]
        rows.append(sp.hstack([sp.csc_matrix(g_w), -eN])); lo.append([-np.inf]); hi.append([TERM_W - w_gap])
        rows.append(sp.hstack([sp.csc_matrix(g_w), eN])); lo.append([-TERM_W - w_gap]); hi.append([np.inf])
        rows.append(sp.hstack([sp.csc_matrix((n, nu)), sp.eye(n)])); lo.append(np.zeros(n)); hi.append(np.full(n, np.inf))
        Aqp = sp.vstack(rows, format="csc")
        solver = osqp.OSQP()
        solver.setup(P, q, Aqp, np.concatenate([np.atleast_1d(v) for v in lo]),
                     np.concatenate([np.atleast_1d(v) for v in hi]), verbose=False,
                     eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polish=True, scaling=15)
        res = solver.solve()
        status = res.info.status
        if status not in ("solved", "solved inaccurate") or res.x is None or np.any(~np.isfinite(res.x)):
            return None, status
        return res.x, status

    def summary(self):
        t = np.array(self.stats["times"]) * 1e3
        s = self.stats
        return dict(filter_steps=s["steps"], filter_active_pct=round(100 * s["active"] / max(s["steps"], 1), 2),
                    recovery_steps=s["recovery"], slack_steps=s["slack_steps"], max_slack_m=round(s["max_slack"], 4),
                    qp_failures=s["qp_fail"],
                    mean_dev_v=round(float(np.mean(s["dev_v"])), 4) if s["dev_v"] else 0.0,
                    mean_dev_w=round(float(np.mean(s["dev_w"])), 4) if s["dev_w"] else 0.0,
                    filter_ms_median=round(float(np.median(t)), 3) if len(t) else 0.0,
                    filter_ms_p99=round(float(np.percentile(t, 99)), 3) if len(t) else 0.0)


class SafeController:
    """Any proposing controller (the network, or a test command) followed by the filter.
    The proposer is told the command actually applied, so the network's u_prev input stays truthful."""

    def __init__(self, proposer, filt):
        self.proposer, self.filt = proposer, filt
        self.u_prev = np.zeros(2)
        self.times = []

    def reset(self):
        self.u_prev = np.zeros(2)
        self.times = []
        self.filt.reset()
        if hasattr(self.proposer, "reset"):
            self.proposer.reset()

    def __call__(self, P, X, info):
        t0 = time.perf_counter()
        u_nn = np.asarray(self.proposer(P, X, info), float)
        u = self.filt(X, info, u_nn, self.u_prev)
        self.u_prev = u
        if hasattr(self.proposer, "u_prev"):
            self.proposer.u_prev = u.copy()
        self.times.append(time.perf_counter() - t0)
        return u
