"""Chronological offline replay for the adaptive interaction memory.

Reads a CSV with columns named `<engine>_direction` and
`<engine>_signature`, plus `next_round_outcome`.

Timestamps are intentionally ignored as features. Session/round IDs are used
only for ordering and continuity checks when present.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence

import pandas as pd


def parse_signature_features(engine: str, signature: str) -> List[str]:
    """Return structural atoms used only for exploratory replay."""
    out = [f"sig.{engine}={signature}"]
    if ":" not in signature:
        return out
    ns, body = signature.split(":", 1)
    parts = body.split("_")
    for i, part in enumerate(parts):
        if ns == "traj":
            if i == 0:
                out.append(f"feat.{engine}.shape={part}")
            elif i == 1:
                out.append(f"feat.{engine}.model={part}")
            elif i == 2:
                out.append(f"feat.{engine}.state={part}")
            elif part.startswith("dw"):
                out.append(f"feat.{engine}.dw={part[2:]}")
        elif ns == "crt":
            m = re.fullmatch(r"(\d+)-(\d+)-(\d+)(?:_dw(\d+))?", body)
            if m:
                a,b,c,dw=m.groups()
                out += [f"feat.{engine}.a={a}",f"feat.{engine}.b={b}",f"feat.{engine}.c={c}",
                        f"feat.{engine}.sum={int(a)+int(b)+int(c)}",
                        f"feat.{engine}.repeated={str(len({a,b,c})<3).lower()}"]
                if dw is not None:
                    out.append(f"feat.{engine}.dw={dw}")
                break
        else:
            for prefix in ("rel","cx","tr","dir","sg","dw","a","c","h","p","r","d","v","m","s","f","q","t"):
                if part.startswith(prefix) and len(part)>len(prefix):
                    out.append(f"feat.{engine}.{prefix}={part[len(prefix):]}")
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
        # Signature counts even when its direction field is NO_SIGNAL/empty.
        atoms.extend(parse_signature_features(engine, value))
        dcol = engine + "_direction"
        if dcol in row.index and isinstance(row[dcol], str) and row[dcol] in {"UP", "DOWN"}:
            atoms.append(f"dir.{engine}={row[dcol]}")
    return sorted(set(atoms))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="interaction_replay_stats.json")
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--max-atoms", type=int, default=28)
    ap.add_argument("--max-candidates", type=int, default=6000)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "next_round_outcome" not in df.columns:
        raise SystemExit("CSV must contain next_round_outcome")

    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"key": None, "n": 0, "up": 0, "down": 0})
    n_eligible = 0
    n_skipped = Counter()

    for _, row in df.iterrows():
        outcome = row["next_round_outcome"]
        if outcome not in {"UP", "DOWN"}:
            n_skipped["unresolved_outcome"] += 1
            continue
        atoms = row_atoms(row)
        engines = {a.split(".",1)[1].split("=",1)[0] for a in atoms if a.startswith("sig.")}
        if len(engines) < 2:
            n_skipped["fewer_than_two_signature_engines"] += 1
            continue
        n_eligible += 1
        atoms = atoms[:args.max_atoms]
        generated = 0
        for order in range(2, args.max_order + 1):
            for combo in itertools.combinations(atoms, order):
                key = json.dumps(sorted(set(combo)), separators=(",", ":"), ensure_ascii=True)
                h = __import__('hashlib').sha256(key.encode()).hexdigest()
                rec = stats[h]
                rec["key"] = json.loads(key)
                rec["n"] += 1
                rec["up"] += int(outcome == "UP")
                rec["down"] += int(outcome == "DOWN")
                generated += 1
                if generated >= args.max_candidates:
                    break
            if generated >= args.max_candidates:
                break

    # Store only supported interactions in the bootstrap file.
    rules = []
    for h, rec in stats.items():
        if rec["n"] < args.min_support:
            continue
        p_up = rec["up"] / rec["n"]
        rules.append({**rec, "hash": h, "p_up": p_up, "p_down": 1-p_up})
    rules.sort(key=lambda r: (abs(r["p_up"]-0.5) * (r["n"] ** 0.5), r["n"]), reverse=True)

    payload = {
        "source": str(Path(args.csv).resolve()),
        "timestamps_used_as_features": False,
        "max_order": args.max_order,
        "min_support": args.min_support,
        "eligible_rows": n_eligible,
        "skips": dict(n_skipped),
        "rules": rules,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "eligible_rows": n_eligible,
        "candidate_interactions": len(stats),
        "supported_rules": len(rules),
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
