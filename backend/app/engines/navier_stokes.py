"""Navier-Stokes + B3/S23 engine with dwell enrichment."""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._lookup import lookup_direction
from ._temporal import EngineSnapshot, compute_dwell_s, dwell_bin
from .base import Engine, EngineContext, EngineOutput

WINDOW_TICKS = 48
K_NODES = 16
SIM_STEPS = 8
DT = 0.15
VISCOSITY = 0.05
AUTOMATON_DENSITY = 0.5
AUTOMATON_FORCING = 0.25
SEED = 0xC0FFEE


def _build_graph(prices):
    n = len(prices)
    if n < 4:
        return None
    try:
        qs = np.quantile(prices, np.linspace(0, 1, K_NODES + 1))
        qs[-1] += 1e-9
        assign = np.clip(np.digitize(prices, qs) - 1, 0, K_NODES - 1)
    except Exception:
        return None

    mass = np.zeros(K_NODES)
    price_sum = np.zeros(K_NODES)
    vel_sum = np.zeros(K_NODES)
    for i, a in enumerate(assign):
        mass[a] += 1.0
        price_sum[a] += prices[i]
        if i > 0:
            vel_sum[a] += prices[i] - prices[i - 1]

    used = np.where(mass > 0)[0]
    if len(used) < 3:
        return None

    remap = -np.ones(K_NODES, dtype=np.int64)
    remap[used] = np.arange(len(used))
    a_idx = remap[assign]
    N = len(used)

    x = price_sum[used] / np.maximum(mass[used], 1.0)
    v = vel_sum[used] / np.maximum(mass[used], 1.0)
    span = max(x.max() - x.min(), 1e-9)
    x = (x - x.min()) / span

    adj = np.zeros((N, N))
    for i in range(n - 1):
        a, b = int(a_idx[i]), int(a_idx[i + 1])
        if a != b:
            adj[a, b] += 1.0
    row = adj.sum(axis=1, keepdims=True)
    adj = np.divide(adj, row, out=np.zeros_like(adj), where=row > 0)

    return {"x": x, "vel": v, "adj": adj, "N": N}


def _ns_step(u, p, adj, nu, forcing, dt):
    neigh = adj @ u
    conv = (neigh - u) * 0.5
    grad_p = adj @ p - p
    grad_p2 = np.stack([grad_p, np.zeros_like(grad_p)], axis=-1)
    laplace = neigh - u
    u_new = u + dt * (-conv - grad_p2 + nu * laplace + forcing)
    div = adj @ u_new[:, 0] - u_new[:, 0]
    p_new = p + dt * div
    return u_new, p_new


def _seed_auto(N, rng):
    return (rng.random(N) < AUTOMATON_DENSITY).astype(float)


def _auto_step(s, adj):
    n_count = adj @ s * max(adj.shape[0] - 1, 1)
    born = (s < 0.5) & (n_count > 2.5) & (n_count < 3.5)
    survive = (s > 0.5) & (n_count > 1.5) & (n_count < 3.5)
    return (born | survive).astype(float)


def _auto_forcing(s, adj):
    neigh = adj @ s
    gate = np.clip(np.abs(s - neigh), 0.0, 1.0)
    fx = AUTOMATON_FORCING * gate * (neigh - s)
    fy = np.zeros_like(fx)
    return np.stack([fx, fy], axis=-1)


def _sig(u_mean_x, p_mean, density, activity):
    if u_mean_x > 0.05:
        vb = 2
    elif u_mean_x < -0.05:
        vb = 0
    else:
        vb = 1
    if p_mean < -0.5:
        pb = 0
    elif p_mean > 0.5:
        pb = 2
    else:
        pb = 1
    db = min(int(density * 4), 3)
    if activity < -0.05:
        ab = 0
    elif activity > 0.05:
        ab = 2
    else:
        ab = 1
    return f"ns:v{vb}_p{pb}_d{db}_a{ab}"


class NavierStokesAutomatonEngine(Engine):
    name = "navier_stokes"

    def _core(self, window):
        if len(window) < 8:
            return None
        prices = np.array([p for _, p in window[-WINDOW_TICKS:]], dtype=float)
        g = _build_graph(prices)
        if g is None:
            return None
        adj = g["adj"]
        N = int(g["N"])

        vel_x = g["vel"] / max(abs(g["vel"]).max(), 1e-9)
        u = np.stack([vel_x, np.zeros_like(vel_x)], axis=-1)
        p = g["mass"] if "mass" in g else np.zeros(N)
        # If mass is missing, derive from node count proxy
        if "mass" not in g:
            p = np.zeros(N)
        p = g["vel"] if False else p
        # Use normalized velocity as pressure proxy
        p = vel_x - 0.5

        rng = np.random.default_rng(SEED + N + int(abs(prices[-1]) * 100) % 1000)
        s = _seed_auto(N, rng)
        prev_density = float(s.mean())

        for _ in range(SIM_STEPS):
            forcing = _auto_forcing(s, adj)
            u, p = _ns_step(u, p, adj, VISCOSITY, forcing, DT)
            s = _auto_step(s, adj)

        u_mean_x = float(u[:, 0].mean())
        p_mean = float(p.mean())
        density = float(s.mean())
        activity = density - prev_density
        sig = _sig(u_mean_x, p_mean, density, activity)
        return EngineSnapshot(
            signature=sig,
            features={
                "u_mean_x": u_mean_x,
                "p_mean": p_mean,
                "density": density,
            },
        )

    def _snapshot_for_window(self, window):
        return self._core(window)

    def run(self, ctx):
        raise NotImplementedError("NS engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 8:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )
        snap = self._core(ctx.recent_ticks)
        if snap is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "snapshot failed"},
            )
        dwell_s = compute_dwell_s(
            self._snapshot_for_window, ctx.recent_ticks, snap.signature, max_back=6
        )
        full_sig = f"{snap.signature}_dw{dwell_bin(dwell_s)}"
        raw: Dict[str, Any] = {"dwell_s": dwell_s, **snap.features}

        try:
            lookup = await lookup_direction(full_sig)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=full_sig,
                raw_state={**raw, "lookup": "insufficient_history"},
            )
        direction, occ, conf = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=conf,
            pattern_signature=full_sig,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occ,
                    "confidence": conf,
                    "chosen_direction": direction,
                },
            },
        )
