"""Registry of all FlashInfer contest kernel definitions.

Definitions are loaded from the contest JSON files at:
  /data02/henryg/mlsys2026-flashinfer-contest/data/flashinfer-trace/definitions/

The old hand-crafted Python definitions are kept for reference but the
primary DEFINITIONS dict is populated from the actual contest JSON files.

Usage:
    from flashinfer.definitions import DEFINITIONS, list_definitions

    spec = DEFINITIONS["gdn_decode_qk4_v8_d128_k_last"]
    print(list_definitions())
"""

from __future__ import annotations

from pathlib import Path

from flashinfer.adapter import DefinitionSpec

# ---------------------------------------------------------------------------
# Contest JSON root (absolute path on this machine)
# ---------------------------------------------------------------------------
_CONTEST_ROOT = Path("/data02/henryg/mlsys2026-flashinfer-contest/data/flashinfer-trace/definitions")

DEFINITIONS: dict[str, DefinitionSpec] = {}


def _load_contest_definitions() -> None:
    """Discover and load all contest JSON definitions."""
    if not _CONTEST_ROOT.exists():
        return
    from flashinfer.definitions.loader import load_contest_json
    for json_path in sorted(_CONTEST_ROOT.rglob("*.json")):
        try:
            spec = load_contest_json(json_path)
            DEFINITIONS[spec.key] = spec
        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to load contest definition {json_path}: {exc}")


_load_contest_definitions()


def list_definitions() -> list[str]:
    """Return a list of all registered definition keys."""
    return list(DEFINITIONS.keys())
