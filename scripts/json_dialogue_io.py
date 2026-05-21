"""
Helpers for reading dialogue_history.json and similar artifacts.

Some runs may accidentally concatenate two JSON documents (e.g. double write
or resume), which makes json.loads fail with JSONDecodeError: Extra data.
We parse the first complete top-level JSON value and ignore benign trailing text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_first_value(text: str) -> Any:
    text = text.lstrip("\ufeff").strip()
    if not text:
        raise ValueError("empty JSON document")
    decoder = json.JSONDecoder()
    obj, idx = decoder.raw_decode(text)
    return obj


def load_json_first_value_from_path(path: Path) -> Any:
    return load_json_first_value(path.read_text(encoding="utf-8"))
