#!/usr/bin/env python3
"""
Step 3: nonlinear MPC in the path frame (teacher and baseline).

State  z = [s, e_y, e_psi, v, w]   progress, lateral error, heading error, wheel speed, wheel turn rate
Input  u = [v_c, w_c]              commanded wheel speed and turn rate
Model  (the controller's model; a, b from the training drives of the fold, tau nominal)
    s'     = a v cos(e_psi) / (1 - kappa e_y)
    e_y'   = a v sin(e_psi)
    e_psi' = b w - kappa s'
    v'     = (v_c - v) / tau,   w' = (w_c - w) / tau
Cost   sum_k  q_y e_y^2 + q_psi e_psi^2 + q_v (v_c - v_ref)^2 + r_dv dv_c^2 + r_dw dw_c^2 + rho_1 sigma + rho_2 sigma^2
Hard   0 <= v_c <= v_max, |w_c| <= w_max, |dv_c| <= acc_max dt, |dw_c| <= alpha_max dt (also vs. the last applied command)
Soft   |e_y,k| <= d_k + sigma_k, sigma_k >= 0 (heavy penalty; keeps the problem feasible, slack use is logged)

Information: the MPC sees exactly what the network will see: the current path-frame state and the
curvature and corridor previews at 0.4 ... 3.2 m ahead (sim.PREVIEW_DIST). Curvature and corridor at
each horizon step are read from those previews at the progress predicted by the previous solution
(shifted), so the optimisation problem itself has fixed parameters.
"""
import time

import casadi as ca
import numpy as np

import sim

N = 20                 # horizon steps
DT = sim.DT_CTRL       # 0.1 s -> 2 s horizon
WEIGHTS = dict(q_y=20.0, q_psi=3.0, q_v=1.0, r_dv=2.0, r_dw=8.0, qN_y=40.0, qN_psi=6.0, rho_1=1e3, rho_2=1e4)
V_REF = 1.6            # target commanded speed (true speed 1.4 m/s), a margin below the 1.8 m/s limit


class MPC:
    def __init__(self, a, b, tau=0.2, weights=None, v_ref=V_REF):
        self.a, self.b, self.tau = a, b, tau
        self.w = dict(WEIGHTS, **(weights or {}))
        self.v_ref = v_ref
        L = sim.LIMITS
        self.lim = L
        self._build()
        self.reset()

    # -------------------------------------------------------------------------------- problem
    def _f(self, z, u, kap):
        s, ey, epsi, v, w = z[0], z[1], z[2], z[3], z[4]
        sdot = self.a * v * ca.cos(epsi) / (1 - kap * ey)
        return ca.vertcat(sdot, self.a * v * ca.sin(epsi), self.b * w - kap * sdot,
                          (u[0] - v) / self.tau, (u[1] - w) / self.tau)

    def _rk4(self, z, u, kap, h=DT, n=2):
        h = h / n
        for _ in range(n):
            k1 = self._f(z, u, kap)
            k2 = self._f(z + h / 2 * k1, u, kap)
            k3 = self._f(z + h / 2 * k2, u, kap)
            k4 = self._f(z + h * k3, u, kap)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    def _build(self):
        W, L = self.w, self.lim
        opti = ca.Opti()
        Z = opti.variable(5, N + 1)
        U = opti.variable(2, N)
        SIG = opti.variable(N)
        z0 = opti.parameter(5)
        kap = opti.parameter(N)          # curvature at each step (from the preview)
        dlim = opti.parameter(N)         # corridor half-width at each step (from the preview)
        u_prev = opti.parameter(2)       # command applied in the previous period
        cost = 0
        opti.subject_to(Z[:, 0] == z0)
        for k in range(N):
            opti.subject_to(Z[:, k + 1] == self._rk4(Z[:, k], U[:, k], kap[k]))
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            cost += (W["q_y"] * Z[1, k + 1] ** 2 + W["q_psi"] * Z[2, k + 1] ** 2
                     + W["q_v"] * (U[0, k] - self.v_ref) ** 2 + W["r_dv"] * du[0] ** 2 + W["r_dw"] * du[1] ** 2
                     + W["rho_1"] * SIG[k] + W["rho_2"] * SIG[k] ** 2)
            opti.subject_to(opti.bounded(L["v_c_min"], U[0, k], L["v_c_max"]))
            opti.subject_to(opti.bounded(-L["w_c_max"], U[1, k], L["w_c_max"]))
            opti.subject_to(opti.bounded(-L["acc_max"] * DT, du[0], L["acc_max"] * DT))
            opti.subject_to(opti.bounded(-L["alpha_max"] * DT, du[1], L["alpha_max"] * DT))
            opti.subject_to(SIG[k] >= 0)
            opti.subject_to(Z[1, k + 1] <= dlim[k] + SIG[k])
            opti.subject_to(-Z[1, k + 1] <= dlim[k] + SIG[k])
        cost += W["qN_y"] * Z[1, N] ** 2 + W["qN_psi"] * Z[2, N] ** 2
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"print_level": 0, "max_iter": 200, "tol": 1e-6, "acceptable_tol": 1e-4,
                     "warm_start_init_point": "yes", "sb": "yes"})
        self.opti, self.Z, self.U, self.SIG = opti, Z, U, SIG
        self.p_z0, self.p_kap, self.p_d, self.p_uprev = z0, kap, dlim, u_prev

    # -------------------------------------------------------------------------------- runtime
    def reset(self):
        self.Z_guess = None
        self.U_guess = None
        self.u_prev = np.zeros(2)
        self.stats = dict(solves=0, failures=0, fallback=0, slack_steps=0, times=[])
        self.last = None

    def _horizon_params(self, info, s_pred):
        """Curvature and corridor at each predicted step, read from the previews only."""
        rel = np.concatenate([[0.0], sim.PREVIEW_DIST])
        kap_tab = np.concatenate([[info["kappa"]], info["preview"]])
        d_tab = np.concatenate([[info["d"]], info["d_preview"]])
        ds = np.clip(s_pred - info["s"], 0, rel[-1])
        kap = np.interp(ds, rel, kap_tab)
        d = d_tab[np.clip(np.searchsorted(rel, ds, side="right") - 1, 0, len(rel) - 1)]
        return kap, d

    def __call__(self, P, X, info):
        z0 = np.array([info["s"], info["e_y"], info["e_psi"], X[3], X[4]])
        if self.Z_guess is None:      # cold start: constant speed guess along the path
            v0 = max(X[3], 0.3)
            self.Z_guess = np.tile(z0[:, None], (1, N + 1))
            self.Z_guess[0] = z0[0] + self.a * v0 * DT * np.arange(N + 1)
            self.U_guess = np.tile(np.array([[max(X[3], 0.0)], [0.0]]), (1, N))
        kap, d = self._horizon_params(info, self.Z_guess[0, 1:])
        o = self.opti
        o.set_value(self.p_z0, z0)
        o.set_value(self.p_kap, kap)
        o.set_value(self.p_d, d)
        o.set_value(self.p_uprev, self.u_prev)
        o.set_initial(self.Z, self.Z_guess)
        o.set_initial(self.U, self.U_guess)
        o.set_initial(self.SIG, 0)
        t0 = time.perf_counter()
        try:
            sol = o.solve()
            Zs, Us, Ss = sol.value(self.Z), sol.value(self.U), sol.value(self.SIG)
            ok = True
        except RuntimeError:
            ok = False
        self.stats["times"].append(time.perf_counter() - t0)
        self.stats["solves"] += 1
        if ok:
            u = Us[:, 0]
            if np.max(Ss) > 1e-4:
                self.stats["slack_steps"] += 1
            # shift the solution one step for the next warm start
            self.Z_guess = np.hstack([Zs[:, 1:], Zs[:, -1:]])
            self.Z_guess[0, -1] = Zs[0, -1] + (Zs[0, -1] - Zs[0, -2])
            self.U_guess = np.hstack([Us[:, 1:], Us[:, -1:]])
            self.last = dict(Z=Zs, U=Us, sigma=Ss)
        else:
            # fall back to the next step of the last good plan (or a gentle stop if none)
            self.stats["failures"] += 1
            if self.U_guess is not None:
                u = self.U_guess[:, 0]
                self.stats["fallback"] += 1
                self.Z_guess = np.hstack([self.Z_guess[:, 1:], self.Z_guess[:, -1:]])
                self.U_guess = np.hstack([self.U_guess[:, 1:], self.U_guess[:, -1:]])
            else:
                u = np.array([0.0, 0.0])
        self.u_prev = sim.clip_command(u)
        return self.u_prev

    def summary(self):
        t = np.array(self.stats["times"]) * 1e3
        return dict(solves=self.stats["solves"], failures=self.stats["failures"],
                    slack_steps=self.stats["slack_steps"],
                    solve_ms_median=round(float(np.median(t)), 2), solve_ms_p99=round(float(np.percentile(t, 99)), 2),
                    solve_ms_max=round(float(t.max()), 2))
