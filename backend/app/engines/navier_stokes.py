"""Navier-Stokes + B3/S23 Cellular Automaton Engine.

Combines two classical systems on a graph built from recent ticks:
  - A simplified Navier-Stokes fluid layer tracking velocity and pressure
    over the graph.
  - A discrete B3/S23 cellular automaton (Conway's Life rule) whose state
    injects localized forcing into the fluid.

The signature is a coarse quantization of the final fluid+automaton state.
Direction is learned from history via the shared pattern-lookup mechanism.

Signature space: 3 (velocity) x 3 (pressure) x 4 (density) x 3 (activity)
= 108 states. Fast to learn.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._lookup import lookup_direction
from .base import Engine, EngineContext, EngineOutput

# --- Parameters -------------------------------------------------------------

WINDOW_TICKS = 48
K_NODES = 16            # coarse graph for the simulation
SIM_STEPS = 8           # NS + automaton iterations per round
DT = 0.15               # integration step
VISCOSITY = 0.05        # fixed (learned from local volatility if you want)
AUTOMATON_DENSITY = 0.5  # initial fraction of "alive" cells
AUTOMATON_FORCING = 0.25  # how strongly the automaton pushes the fluid

# Initial automaton seed (deterministic, gives a fixed graph-state pattern).
SEED = 0xC0FFEE


# --- Graph construction -----------------------------------------------------

def _build_graph(prices: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Nodes = quantile-bins of price. Edges = consecutive-tick transitions.

    Returns dict with:
      x        : [N] mean normalized price per node
      mass     : [N] count of ticks per node (proxy for pressure)
      vel      : [N] mean price change per node (proxy for velocity)
      adj      : [N, N] row-normalized adjacency (transition probabilities)
    """
    n = len(prices)
    if n < 4:
        return None

    try:
        qs = np.quantile(prices, np.linspace(0, 1, K_NODES + 1))
        qs[-1] += 1e-9
        assign = np.clip(np.digitize(prices, qs) - 1, 0, K_NODES - 1)
    except Exception:
        return None

    mass = np.zeros(K_NODES, dtype=np.float64)
    price_sum = np.zeros(K_NODES, dtype=np.float64)
    vel_sum = np.zeros(K_NODES, dtype=np.float64)
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
    m = mass[used]
    v = vel_sum[used] / np.maximum(mass[used], 1.0)

    # Normalize x to [0, 1] for numerical stability
    span = max(x.max() - x.min(), 1e-9)
    x = (x - x.min()) / span

    # Adjacency from consecutive transitions
    adj = np.zeros((N, N), dtype=np.float64)
    for i in range(n - 1):
        a, b = int(a_idx[i]), int(a_idx[i + 1])
        if a != b:
            adj[a, b] += 1.0
    row_sum = adj.sum(axis=1, keepdims=True)
    adj = np.divide(adj, row_sum, out=np.zeros_like(adj), where=row_sum > 0)

    return {"x": x, "mass": m, "vel": v, "adj": adj, "N": np.array([N])}


# --- Navier-Stokes step -----------------------------------------------------

def _ns_step(
    u: np.ndarray,      # [N, 2] velocity (x-component: price, y-component: mass)
    p: np.ndarray,      # [N]    pressure
    adj: np.ndarray,    # [N, N] row-normalized adjacency
    nu: float,          # scalar viscosity
    forcing: np.ndarray,  # [N, 2] external forcing from automaton
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Convection: (u · ∇) u ≈ adj @ u - u
    neighbor_u = adj @ u
    conv = (neighbor_u - u) * 0.5

    # Pressure gradient: ∇p ≈ adj @ p - p (scalar broadcast to both channels)
    grad_p = adj @ p - p
    grad_p2 = np.stack([grad_p, np.zeros_like(grad_p)], axis=-1)

    # Laplacian: Δu = adj @ u - u (already have it as neighbor_u - u)
    laplace_u = neighbor_u - u

    # Euler step
    u_new = u + dt * (-conv - grad_p2 + nu * laplace_u + forcing)

    # Pressure correction: proportional to divergence of u_new
    div = (adj @ u_new[:, 0] - u_new[:, 0])
    p_new = p + dt * div

    return u_new, p_new


# --- B3/S23 automaton -------------------------------------------------------

def _seed_automaton(N: int, rng: np.random.Generator) -> np.ndarray:
    """Initial binary state array."""
    return (rng.random(N) < AUTOMATON_DENSITY).astype(np.float64)


def _automaton_step(
    s: np.ndarray,   # [N] binary 0/1
    adj: np.ndarray,  # [N, N] row-normalized
) -> np.ndarray:
    """One B3/S23 update step (discrete)."""
    # Neighbor count = adj @ s (row-normalized -> weighted neighbor density)
    # Scale by average node degree to get an approximate integer count.
    neighbor_count = adj @ s * max(adj.shape[0] - 1, 1)

    # B3/S23:
    #   born  if dead and count ≈ 3
    #   survives if alive and count ∈ {2, 3}
    #   dies otherwise
    born = (s < 0.5) & (neighbor_count > 2.5) & (neighbor_count < 3.5)
    survive = (s > 0.5) & (neighbor_count > 1.5) & (neighbor_count < 3.5)
    s_new = (born | survive).astype(np.float64)
    return s_new


def _automaton_forcing(
    s: np.ndarray, adj: np.ndarray
) -> np.ndarray:
    """Convert automaton state to a 2D forcing vector per node.

    x-component: pushes toward the neighbor mean (fluid flows toward dense
                 automaton regions).
    y-component: zero (mass is not forced directly).

    The forcing is gated by the local "alive gradient": how different
    this cell's state is from its neighborhood.
    """
    neigh_s = adj @ s
    gate = np.clip(np.abs(s - neigh_s), 0.0, 1.0)
    direction_x = neigh_s - s
    fx = AUTOMATON_FORCING * gate * direction_x
    fy = np.zeros_like(fx)
    return np.stack([fx, fy], axis=-1)


# --- Signature --------------------------------------------------------------

def _quantize(value: float, bins: List[float]) -> int:
    for i in range(len(bins) - 1, -1, -1):
        if value >= bins[i]:
            return i
    return 0


def _signature(
    u_mean_x: float,
    p_mean: float,
    density: float,
    activity: float,
) -> str:
    # Velocity direction bin: negative, flat, positive
    if u_mean_x > 0.05:
        vel_bin = 2
    elif u_mean_x < -0.05:
        vel_bin = 0
    else:
        vel_bin = 1

    # Pressure bin: low / mid / high (based on relative magnitude)
    if p_mean < -0.5:
        p_bin = 0
    elif p_mean > 0.5:
        p_bin = 2
    else:
        p_bin = 1

    # Density bin: 0-3
    d_bin = min(int(density * 4), 3)

    # Activity bin: 0=mostly deaths, 1=stable, 2=mostly births
    if activity < -0.05:
        act_bin = 0
    elif activity > 0.05:
        act_bin = 2
    else:
        act_bin = 1

    return f"ns:v{vel_bin}_p{p_bin}_d{d_bin}_a{act_bin}"


# --- Engine -----------------------------------------------------------------

class NavierStokesAutomatonEngine(Engine):
    name = "navier_stokes"

    def run(self, ctx: EngineContext) -> EngineOutput:
        raise NotImplementedError("Navier-Stokes engine is async; use run_async()")

    async def run_async(self, ctx: EngineContext) -> EngineOutput:
        if not ctx.recent_ticks or len(ctx.recent_ticks) < 8:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "insufficient ticks"},
            )

        window = ctx.recent_ticks[-WINDOW_TICKS:]
        prices = np.array([p for _, p in window], dtype=np.float64)

        g = _build_graph(prices)
        if g is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=None,
                raw_state={"reason": "graph build failed"},
            )

        adj = g["adj"]
        N = int(g["N"][0])

        # Initialize fields
        # Velocity: 2 components. x = normalized local price change per node.
        vel_x = g["vel"] / max(abs(g["vel"]).max(), 1e-9)
        u = np.stack([vel_x, np.zeros_like(vel_x)], axis=-1)

        # Pressure: normalized mass per node
        p = g["mass"] / max(g["mass"].max(), 1e-9) - 0.5

        # Viscosity: fixed scalar (could be adaptive; v1 keeps it simple)
        nu = VISCOSITY

        # Automaton state (seeded deterministically from N and price stats)
        rng = np.random.default_rng(SEED + N + int(abs(prices[-1]) * 100) % 1000)
        s = _seed_automaton(N, rng)

        prev_density = float(s.mean())

        for _ in range(SIM_STEPS):
            forcing = _automaton_forcing(s, adj)
            u, p = _ns_step(u, p, adj, nu, forcing, DT)
            s = _automaton_step(s, adj)

        # Extract final state features
        u_mean_x = float(u[:, 0].mean())
        p_mean = float(p.mean())
        density = float(s.mean())
        activity = density - prev_density

        signature = _signature(u_mean_x, p_mean, density, activity)

        raw: Dict[str, Any] = {
            "u_mean_x": u_mean_x,
            "u_std_x": float(u[:, 0].std()),
            "p_mean": p_mean,
            "density": density,
            "activity": activity,
            "N": N,
            "steps": SIM_STEPS,
        }

        try:
            lookup = await lookup_direction(signature)
        except Exception as e:
            raw["lookup_error"] = str(e)
            lookup = None

        if lookup is None:
            return EngineOutput(
                engine=self.name,
                direction="NO_SIGNAL",
                pattern_signature=signature,
                raw_state={**raw, "lookup": "insufficient_history"},
            )

        direction, occurrences, confidence = lookup
        return EngineOutput(
            engine=self.name,
            direction=direction,
            confidence=confidence,
            pattern_signature=signature,
            raw_state={
                **raw,
                "lookup": {
                    "occurrences": occurrences,
                    "confidence": confidence,
                    "chosen_direction": direction,
                },
            },
        )
