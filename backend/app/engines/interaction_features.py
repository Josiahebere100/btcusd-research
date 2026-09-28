"""Dynamic full-state feature extraction for the interaction research engines.

Design rules
------------
* Exact signatures are preserved verbatim and are atomic identities.
* A NO_SIGNAL direction is never treated as a directional vote.
* A NO_SIGNAL engine's signature remains contextual state.
* No engine names are hard-coded into state extraction.
* Structural features are additive; they never replace exact tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # production package
    from .base import EngineContext, EngineOutput
except ImportError:  # standalone test/import convenience
    EngineContext = Any  # type: ignore
    EngineOutput = Any  # type: ignore


_DIRECTION_VALUES = {"UP", "DOWN", "NO_SIGNAL"}


def _clean_direction(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip().upper()
        if value in _DIRECTION_VALUES:
            return value
    return "NO_SIGNAL"


def _valid_signature(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value.strip() not in {
        "-", "NO_SIGNAL", "NO SIGNAL", "NONE", "NULL", "NAN"
    }


def _parse_namespace(signature: str) -> Tuple[str, str, List[str]]:
    if ":" not in signature:
        return "unknown", signature, [signature]
    namespace, body = signature.split(":", 1)
    return namespace, body, body.split("_")


def _add_prefixed_fields(
    out: Dict[str, str],
    parts: Sequence[str],
    prefixes: Iterable[str],
) -> None:
    # Match longer prefixes before shorter prefixes.
    # This prevents "dw0" from being interpreted as "d" + "w0".
    ordered_prefixes = sorted(prefixes, key=len, reverse=True)

    for part in parts:
        for prefix in ordered_prefixes:
            if part.startswith(prefix) and len(part) > len(prefix):
                out[prefix.rstrip("_")] = part[len(prefix):]
                break


def parse_signature_features(engine: str, signature: str) -> Dict[str, str]:
    """Parse known structural syntax without changing the exact signature."""
    namespace, body, parts = _parse_namespace(signature)
    f: Dict[str, str] = {
        "namespace": namespace,
        "exact": signature,
    }

    if namespace == "traj":
        if parts:
            f["shape"] = parts[0]
        if len(parts) >= 2:
            f["model"] = parts[1]
        if len(parts) >= 3:
            f["state"] = parts[2]
        _add_prefixed_fields(f, parts[3:], ("dw",))

    elif namespace == "crt":
        m = re.fullmatch(r"(\d+)-(\d+)-(\d+)(?:_dw(\d+))?", body)
        if m:
            a, b, c, dw = m.groups()
            f.update(a=a, b=b, c=c, sum=str(int(a) + int(b) + int(c)),
                     repeated=str(len({a, b, c}) < 3).lower())
            if dw is not None:
                f["dw"] = dw

    elif namespace == "trig":
        _add_prefixed_fields(f, parts, ("rel", "cx", "tr", "dir", "sg", "dw"))

    elif namespace == "proj":
        _add_prefixed_fields(f, parts, ("a", "c", "h", "dw"))

    elif namespace == "ent":
        _add_prefixed_fields(f, parts, ("p", "r", "d", "dw", "v"))

    elif namespace == "mom":
        _add_prefixed_fields(f, parts, ("r", "m", "s", "c", "dw"))

    elif namespace == "cg":
        _add_prefixed_fields(f, parts, ("f", "s", "q", "dw"))

    elif namespace == "ns":
        _add_prefixed_fields(f, parts, ("v", "p", "d", "a", "dw"))

    elif namespace == "nsflow":
        if parts:
            f["flow"] = parts[0]
        if len(parts) >= 2 and parts[1].startswith("p"):
            f["p"] = parts[1][1:]

    elif namespace == "sf":
        _add_prefixed_fields(f, parts, ("m", "n1", "n2", "d", "dw"))

    elif namespace == "lab":
        # Examples: lab:loss:len2_sum3_dw0_v2
        if parts:
            f["result"] = parts[0]
        _add_prefixed_fields(f, parts[1:], ("len", "sum", "dw", "v"))

    elif namespace == "cs":
        # Preserve all positional content and expose common tokens.
        for i, part in enumerate(parts):
            f[f"part{i}"] = part
        if "body" in parts:
            f["form"] = "body"
        elif parts:
            f["form"] = parts[0]
        if "mid" in parts:
            f["dircode"] = "mid"
        if len(parts) >= 2 and parts[1] in {"mid", "bull", "bear", "hi", "lo"}:
            f["dircode"] = parts[1]
        m = re.search(r"(?:^|_)st([A-Za-z0-9]+)$", body)
        if m:
            f["st"] = m.group(1)
        if parts:
            # Some existing candlestick signatures encode D/U as a token.
            for token in reversed(parts):
                if token in {"D", "U"}:
                    f["candle_dir"] = token
                    break

    elif namespace == "trail":
        _add_prefixed_fields(f, parts, ("nt", "pev", "t", "dw"))
        for token in parts:
            if token in {"UD", "DU", "DD", "UU"}:
                f["pair"] = token

    return f


@dataclass(frozen=True)
class StateAtom:
    key: str
    value: str
    source: str
    engine: str
    kind: str
    direction: Optional[str] = None

    @property
    def canonical(self) -> str:
        return f"{self.key}={self.value}"


def extract_observed_state(outputs: Sequence[EngineOutput]) -> List[StateAtom]:
    """Turn arbitrary EngineOutput objects into exact/contextual atoms.

    Signature atoms are emitted for EVERY engine with a valid signature,
    including engines whose direction is NO_SIGNAL.
    Direction atoms are emitted ONLY for UP/DOWN outputs.
    """
    atoms: List[StateAtom] = []
    for output in outputs:
        engine = str(getattr(output, "engine", "")).strip()
        if not engine:
            continue
        direction = _clean_direction(getattr(output, "direction", None))
        signature = getattr(output, "pattern_signature", None)
        if not _valid_signature(signature):
            continue
        assert isinstance(signature, str)

        atoms.append(StateAtom(
            key=f"sig.{engine}", value=signature,
            source=engine, engine=engine, kind="exact_signature", direction=direction,
        ))

        if direction in {"UP", "DOWN"}:
            atoms.append(StateAtom(
                key=f"dir.{engine}", value=direction,
                source=engine, engine=engine, kind="direction", direction=direction,
            ))

        structural = parse_signature_features(engine, signature)
        for field, value in structural.items():
            if field == "exact":
                continue
            atoms.append(StateAtom(
                key=f"feat.{engine}.{field}", value=str(value),
                source=engine, engine=engine, kind="structural", direction=direction,
            ))

    # Deduplicate exact same atoms while preserving deterministic order.
    unique: Dict[str, StateAtom] = {a.canonical: a for a in atoms}
    return [unique[k] for k in sorted(unique)]



def select_research_atoms(atoms_obj: Sequence[StateAtom], max_atoms: int = 28) -> List[str]:
    """Select a deterministic, engine-diverse bounded research atom set.

    The selector is shared by live prediction and offline replay so the
    historical bootstrap uses the same contextual vocabulary that the live
    miner sees. Exact signatures are retained even when their engine emitted
    NO_SIGNAL; directional atoms are only UP/DOWN.
    """
    priority = {
        "model": 100, "state": 95, "dw": 90, "a": 90, "b": 90, "c": 90,
        "sum": 85, "repeated": 80, "rel": 90, "dir": 88, "pair": 88,
        "form": 88, "dircode": 85, "m": 88, "v": 80, "p": 85,
        "flow": 85, "f": 80, "q": 80, "h": 75, "r": 75, "s": 75,
        "t": 70, "cx": 65, "tr": 65, "sg": 65, "n1": 70, "n2": 70,
        "namespace": 20, "result": 20, "exact": 0,
    }
    by_engine: Dict[str, List[StateAtom]] = {}
    for atom in atoms_obj:
        by_engine.setdefault(atom.engine, []).append(atom)

    selected: List[str] = []
    for engine in sorted(by_engine):
        sigs = [a for a in by_engine[engine] if a.kind == "exact_signature"]
        if sigs:
            selected.append(sigs[0].canonical)

    for engine in sorted(by_engine):
        dirs = [a for a in by_engine[engine] if a.kind == "direction"]
        if dirs:
            selected.append(dirs[0].canonical)

    remaining = []
    for engine in sorted(by_engine):
        for atom in by_engine[engine]:
            if atom.kind != "structural":
                continue
            field = atom.key.rsplit(".", 1)[-1]
            remaining.append((priority.get(field, 10), engine, atom.canonical))
    remaining.sort(key=lambda x: (-x[0], x[1], x[2]))
    for _, _, canonical in remaining:
        if canonical not in selected:
            selected.append(canonical)
        if len(selected) >= max_atoms:
            break
    return sorted(set(selected))[:max_atoms]

def active_signature_engine_count(outputs: Sequence[EngineOutput]) -> int:
    engines = set()
    for o in outputs:
        if _valid_signature(getattr(o, "pattern_signature", None)):
            engines.add(str(getattr(o, "engine", "")))
    return len(engines)


def directional_engine_count(outputs: Sequence[EngineOutput]) -> int:
    return sum(
        1 for o in outputs
        if _clean_direction(getattr(o, "direction", None)) in {"UP", "DOWN"}
        and _valid_signature(getattr(o, "pattern_signature", None))
    )


def canonical_state_json(outputs: Sequence[EngineOutput]) -> str:
    triples = []
    for o in outputs:
        engine = str(getattr(o, "engine", ""))
        signature = getattr(o, "pattern_signature", None)
        if not engine or not _valid_signature(signature):
            continue
        direction = _clean_direction(getattr(o, "direction", None))
        triples.append([engine, direction, signature])
    triples.sort()
    return json.dumps(triples, separators=(",", ":"), ensure_ascii=True)


def atom_values(atoms: Sequence[StateAtom]) -> List[str]:
    return [a.canonical for a in atoms]
