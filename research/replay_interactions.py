"""Fast chronological bootstrap for the adaptive interaction memory.

The bootstrap is deliberately staged instead of enumerating every 2-, 3-, and
4-way combination independently for every row. Pair support is discovered
first, then only combinations whose lower-order subsets are already supported
are expanded. This preserves the research objective while avoiding the
~tens-of-millions-of-combinations trap of a naive replay.

Reads a CSV with columns named `<engine>_direction` and
`<engine>_signature`, plus `next_round_outcome`.

Timestamps are ignored as predictive features. Session/round IDs are used
only for ordering/continuity metadata when present.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd


def parse_signature_features(engine: str, signature: str) -> List[str]:
    """Mirror the live structural parser; exact signature remains atomic."""
    out = [f"sig.{engine}={signature}"]
    if ":" not in signature:
        return out
    ns, body = signature.split(":", 1)
    parts = body.split("_")

    def add_prefixed(prefixes: Sequence[str]) -> None:
        # Longest-first avoids e.g. pair -> p=air or dircode -> d=ircode.
        ordered = sorted(prefixes, key=len, reverse=True)
        for part in parts:
            for prefix in ordered:
                if part.startswith(prefix) and len(part) > len(prefix):
                    out.append(f"feat.{engine}.{prefix}={part[len(prefix):]}")
                    break

    if ns == "traj":
        if parts:
            out.append(f"feat.{engine}.shape={parts[0]}")
        if len(parts) >= 2:
            out.append(f"feat.{engine}.model={parts[1]}")
        if len(parts) >= 3:
            out.append(f"feat.{engine}.state={parts[2]}")
        for part in parts[3:]:
            if part.startswith("dw") and len(part) > 2:
                out.append(f"feat.{engine}.dw={part[2:]}")

    elif ns == "crt":
        m = re.fullmatch(r"(\d+)-(\d+)-(\d+)(?:_dw(\d+))?", body)
        if m:
            a, b, c, dw = m.groups()
            out += [
                f"feat.{engine}.a={a}",
                f"feat.{engine}.b={b}",
                f"feat.{engine}.c={c}",
                f"feat.{engine}.sum={int(a)+int(b)+int(c)}",
                f"feat.{engine}.repeated={str(len({a,b,c})<3).lower()}",
            ]
            if dw is not None:
                out.append(f"feat.{engine}.dw={dw}")

    elif ns == "trig":
        add_prefixed(("rel", "cx", "tr", "dir", "sg", "dw"))
    elif ns == "proj":
        add_prefixed(("a", "c", "h", "dw"))
    elif ns == "ent":
        add_prefixed(("p", "r", "d", "dw", "v"))
    elif ns == "mom":
        add_prefixed(("r", "m", "s", "c", "dw"))
    elif ns == "cg":
        add_prefixed(("f", "s", "q", "dw"))
    elif ns == "ns":
        add_prefixed(("v", "p", "d", "a", "dw"))
    elif ns == "nsflow":
        if parts:
            out.append(f"feat.{engine}.flow={parts[0]}")
        if len(parts) >= 2 and parts[1].startswith("p") and len(parts[1]) > 1:
            out.append(f"feat.{engine}.p={parts[1][1:]}")
    elif ns == "sf":
        add_prefixed(("m", "n1", "n2", "d", "dw"))
    elif ns == "lab":
        if parts:
            out.append(f"feat.{engine}.result={parts[0]}")
        add_prefixed(("len", "sum", "dw", "v"))
    elif ns == "cs":
        for i, part in enumerate(parts):
            out.append(f"feat.{engine}.part{i}={part}")
        if "body" in parts:
            out.append(f"feat.{engine}.form=body")
        elif parts:
            out.append(f"feat.{engine}.form={parts[0]}")
        if "mid" in parts:
            out.append(f"feat.{engine}.dircode=mid")
        if len(parts) >= 2 and parts[1] in {"mid", "bull", "bear", "hi", "lo"}:
            out.append(f"feat.{engine}.dircode={parts[1]}")
        m = re.search(r"(?:^|_)st([A-Za-z0-9]+)$", body)
        if m:
            out.append(f"feat.{engine}.st={m.group(1)}")
        for token in reversed(parts):
            if token in {"D", "U"}:
                out.append(f"feat.{engine}.candle_dir={token}")
                break
    elif ns == "trail":
        add_prefixed(("nt", "pev", "t", "dw"))
        for token in parts:
            if token in {"UD", "DU", "DD", "UU"}:
                out.append(f"feat.{engine}.pair={token}")
                break

    return sorted(set(out))


def row_atoms(row: pd.Series) -> List[str]:
    atoms: List[str] = []
    for col in row.index:
        if not col.endswith("_signature"):
            continue
        value = row[col]
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            continue
        engine = col[:-10]
        atoms.extend(parse_signature_features(engine, value.strip()))
        dcol = engine + "_direction"
        direction = row[dcol] if dcol in row.index else None
        if isinstance(direction, str) and direction.strip().upper() in {"UP", "DOWN"}:
            atoms.append(f"dir.{engine}={direction.strip().upper()}")
    return sorted(set(atoms))


def _atom_engine(atom: str) -> str:
    if atom.startswith("sig."):
        rest = atom[4:]
        return rest.split("=", 1)[0]
    if atom.startswith("dir."):
        rest = atom[4:]
        return rest.split("=", 1)[0]
    if atom.startswith("feat."):
        rest = atom[5:]
        return rest.split(".", 1)[0]
    return "unknown"


def _atom_kind(atom: str) -> str:
    if atom.startswith("sig."):
        return "exact_signature"
    if atom.startswith("dir."):
        return "direction"
    return "structural"


def select_research_atoms(atoms: Sequence[str], max_atoms: int = 28) -> List[str]:
    """Match the deterministic engine-diverse live atom selector."""
    priority = {
        "model": 100, "state": 95, "dw": 90, "a": 90, "b": 90, "c": 90,
        "sum": 85, "repeated": 80, "rel": 90, "dir": 88, "pair": 88,
        "form": 88, "dircode": 85, "m": 88, "v": 80, "p": 85,
        "flow": 85, "f": 80, "q": 80, "h": 75, "r": 75, "s": 75,
        "t": 70, "cx": 65, "tr": 65, "sg": 65, "n1": 70, "n2": 70,
    }
    grouped: Dict[str, List[str]] = defaultdict(list)
    for atom in atoms:
        grouped[_atom_engine(atom)].append(atom)

    selected: List[str] = []
    for engine in sorted(grouped):
        sigs = [a for a in grouped[engine] if _atom_kind(a) == "exact_signature"]
        if sigs:
            selected.append(sigs[0])
    for engine in sorted(grouped):
        dirs = [a for a in grouped[engine] if _atom_kind(a) == "direction"]
        if dirs:
            selected.append(dirs[0])

    remaining = []
    for engine in sorted(grouped):
        for atom in grouped[engine]:
            if _atom_kind(atom) != "structural":
                continue
            if atom.startswith("feat."):
                body = atom[5:]
                field = body.split("=", 1)[0].rsplit(".", 1)[-1]
            else:
                field = "unknown"
            remaining.append((priority.get(field, 10), engine, atom))
    remaining.sort(key=lambda x: (-x[0], x[1], x[2]))
    for _, _, atom in remaining:
        if atom not in selected:
            selected.append(atom)
        if len(selected) >= max_atoms:
            break
    return sorted(set(selected))[:max_atoms]


def key_for(combo: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(combo)))


def score_rule(up: int, down: int) -> float:
    n = up + down
    if n <= 0:
        return 0.0
    p = up / n
    return abs(p - 0.5) * (n ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="interaction_replay_stats.json")
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--max-atoms", type=int, default=28)
    ap.add_argument("--max-pairs", type=int, default=5000)
    ap.add_argument("--max-triples", type=int, default=5000)
    ap.add_argument("--max-quads", type=int, default=5000)
    ap.add_argument("--higher-order-atoms", type=int, default=14)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "next_round_outcome" not in df.columns:
        raise SystemExit("CSV must contain next_round_outcome")

    eligible_rows: List[Tuple[List[str], str]] = []
    skips = Counter()
    for _, row in df.iterrows():
        outcome = row["next_round_outcome"]
        if outcome not in {"UP", "DOWN"}:
            skips["unresolved_outcome"] += 1
            continue
        atoms = select_research_atoms(row_atoms(row), max_atoms=args.max_atoms)
        sig_engines = {_atom_engine(a) for a in atoms if _atom_kind(a) == "exact_signature"}
        if len(sig_engines) < 2:
            skips["fewer_than_two_signature_engines"] += 1
            continue
        eligible_rows.append((atoms, outcome))

    # Stage 1: all supported pair statistics. This is the only fully dense
    # stage; 28 atoms yield only 378 pairs per row.
    pair_stats: Dict[Tuple[str, ...], List[int]] = defaultdict(lambda: [0, 0])
    for atoms, outcome in eligible_rows:
        for combo in itertools.combinations(atoms, 2):
            key = key_for(combo)
            pair_stats[key][0] += 1
            pair_stats[key][1] += int(outcome == "UP")

    frequent_pairs = {k for k, (n, _) in pair_stats.items() if n >= args.min_support}
    top_pairs = sorted(
        frequent_pairs,
        key=lambda k: (-score_rule(pair_stats[k][1], pair_stats[k][0] - pair_stats[k][1]), -pair_stats[k][0], k),
    )[:args.max_pairs]
    pair_whitelist = set(top_pairs)

    def lower_pairs_supported(combo: Tuple[str, ...]) -> bool:
        return all(key_for(pair) in pair_whitelist for pair in itertools.combinations(combo, 2))

    # Stage 2/3/4: expand only through well-supported lower-order structure.
    higher_stats: Dict[int, Dict[Tuple[str, ...], List[int]]] = {
        order: defaultdict(lambda: [0, 0]) for order in range(3, args.max_order + 1)
    }
    max_per_order = {3: args.max_triples, 4: args.max_quads}

    for atoms, outcome in eligible_rows:
        higher_atoms = atoms[:min(len(atoms), args.higher_order_atoms)]
        for order in range(3, args.max_order + 1):
            if order not in higher_stats:
                continue
            generated = 0
            for combo in itertools.combinations(higher_atoms, order):
                if not lower_pairs_supported(combo):
                    continue
                key = key_for(combo)
                rec = higher_stats[order][key]
                rec[0] += 1
                rec[1] += int(outcome == "UP")
                generated += 1
                if generated >= max_per_order.get(order, 5000):
                    break

    # Consolidate supported rules from all mined orders.
    rules = []
    for key, (n, up) in pair_stats.items():
        if key not in pair_whitelist or n < args.min_support:
            continue
        rules.append((key, n, up, n-up))
    for order, table in higher_stats.items():
        for key, (n, up) in table.items():
            if n >= args.min_support:
                rules.append((key, n, up, n-up))

    serialized = []
    for key, n, up, down in rules:
        canonical = json.dumps(list(key), separators=(",", ":"), ensure_ascii=True)
        h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        p_up = up / n
        serialized.append({
            "hash": h,
            "key": list(key),
            "n": n,
            "up": up,
            "down": down,
            "p_up": p_up,
            "p_down": 1.0 - p_up,
        })
    serialized.sort(key=lambda r: (-score_rule(r["up"], r["down"]), -r["n"], r["key"]))

    payload = {
        "source": str(Path(args.csv).resolve()),
        "timestamps_used_as_features": False,
        "selection": "shared_live_research_atom_selector",
        "max_order": args.max_order,
        "min_support": args.min_support,
        "max_atoms": args.max_atoms,
        "max_pairs": args.max_pairs,
        "max_triples": args.max_triples,
        "max_quads": args.max_quads,
        "higher_order_atoms": args.higher_order_atoms,
        "eligible_rows": len(eligible_rows),
        "skips": dict(skips),
        "pair_candidates": len(pair_stats),
        "frequent_pairs": len(frequent_pairs),
        "retained_pairs": len(pair_whitelist),
        "rules": serialized,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "eligible_rows": len(eligible_rows),
        "pair_candidates": len(pair_stats),
        "frequent_pairs": len(frequent_pairs),
        "retained_pairs": len(pair_whitelist),
        "supported_rules": len(serialized),
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
