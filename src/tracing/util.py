"""Canonical JSON + hashing helpers shared by the logger and ingest layers.

Canonicalization (recursive key-sorting) is what makes args_hash/result_hash
order-of-keys-agnostic — see docs/determinism_index.md for why that matters
when comparing tool calls across runs.
"""

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
