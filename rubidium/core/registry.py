"""Rubidium base registry object"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RubidiumRegistry:
    """Shared registry for live pattern objects."""

    def __init__(self, config: Any) -> None:
        self.printer = config.get_printer()
        self.base_section = config
        self.patterns: Dict[str, Any] = {}

    def register_pattern(self, name: str, pattern: Any, *, config: Optional[Any] = None) -> None:
        if name in self.patterns:
            msg = f"Duplicate pattern '{name}'"
            if config is not None:
                raise config.error(msg)
            raise RuntimeError(msg)
        self.patterns[name] = pattern

    def lookup_pattern(self, name: str, default=None) -> Optional[Any]:
        return self.patterns.get(name, default)

    def get_status(self, eventtime=None) -> Dict[str, Any]:
        return {
            "patterns": sorted(self.patterns.keys()),
        }
