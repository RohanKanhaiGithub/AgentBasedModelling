"""Load model settings from YAML.

The input is a YAML path. The output is a Config object with access to nested
settings and the main simulation values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def dt(self) -> float:
        return float(self.get("paper", "dt_s"))

    @property
    def duration_s(self) -> float:
        return float(self.get("paper", "duration_s"))

    @property
    def n_robots(self) -> int:
        return int(self.get("paper", "n_robots"))
