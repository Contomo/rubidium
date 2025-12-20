# rubidium/core/configview.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class ConfigView:
    """Read options from an override section with fallback to a base section."""

    base: Any
    override: Any

    # -----------------------------------------------------------------
    # existence

    @staticmethod
    def _has_option(sec: Any, name: str) -> bool:
        has_opt = getattr(sec, "has_option", None)
        if callable(has_opt):
            return bool(has_opt(name))
        getter = getattr(sec, "get", None)
        if callable(getter):
            return getter(name, None) is not None
        return False

    def _get_from(self, method: str, name: str, default, **kwargs):
        o = self.override
        b = self.base

        og = getattr(o, method, None)
        if callable(og) and self._has_option(o, name):
            return og(name, default, **kwargs)

        bg = getattr(b, method, None)
        if callable(bg) and self._has_option(b, name):
            return bg(name, default, **kwargs)

        return default

    # -----------------------------------------------------------------
    # primitives

    def get_str(self, name: str, default: str) -> str:
        return str(self._get_from("get", name, default)).strip()

    def get_str_opt(self, name: str) -> Optional[str]:
        v = self._get_from("get", name, None)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def get_int(self, name: str, default: int, **kwargs) -> int:
        return int(self._get_from("getint", name, default, **kwargs)) # type: ignore

    def get_float(self, name: str, default: float, **kwargs) -> float:
        return float(self._get_from("getfloat", name, default, **kwargs)) # type: ignore

    def get_bool(self, name: str, default: bool) -> bool:
        return bool(self._get_from("getboolean", name, default))

    # -----------------------------------------------------------------
    # "required" variants

    def require_str(self, name: str, *, where: str) -> str:
        v = self.get_str_opt(name)
        if not v:
            raise RuntimeError(f"rubidium: missing required option '{name}' in {where}")
        return v

    def require_int(self, name: str, *, where: str, **kwargs) -> int:
        if not (self._has_option(self.override, name) or self._has_option(self.base, name)):
            raise RuntimeError(f"rubidium: missing required option '{name}' in {where}")
        return self.get_int(name, 0, **kwargs)

    def require_float(self, name: str, *, where: str, **kwargs) -> float:
        if not (self._has_option(self.override, name) or self._has_option(self.base, name)):
            raise RuntimeError(f"rubidium: missing required option '{name}' in {where}")
        return self.get_float(name, 0.0, **kwargs)

    def require_bool(self, name: str, *, where: str) -> bool:
        if not (self._has_option(self.override, name) or self._has_option(self.base, name)):
            raise RuntimeError(f"rubidium: missing required option '{name}' in {where}")
        return self.get_bool(name, False)


def config_dir_from_printer(printer) -> Optional[Path]:
    """Return the printer's config directory if available, else None."""
    try:
        args = printer.get_start_args()
        if isinstance(args, dict):
            cfg = args.get("config_file")
            if cfg:
                return Path(str(cfg)).expanduser().resolve().parent
    except Exception:
        pass
    return None
