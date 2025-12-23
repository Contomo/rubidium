# rubidium/core/configview.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


@dataclass(slots=True)
class ConfigView:
    base: Any
    override: Any

    @staticmethod
    def _has(sec, name: str) -> bool:
        try:
            sec.get(name)
            return True
        except Exception:
            return False

    def _pick(self, name: str) -> Any:
        if self._has(self.override, name):
            return self.override
        if self._has(self.base, name):
            return self.base
        return self.override

    # ---------------- primitives

    def get_str(self, name: str, default: str) -> str:
        sec = self._pick(name)
        return str(sec.get(name, default)).strip()

    def get_int(self, name: str, default: int, **kwargs) -> int:
        sec = self._pick(name)
        return int(sec.getint(name, default, **kwargs))

    def get_float(self, name: str, default: float, **kwargs) -> float:
        sec = self._pick(name)
        return float(sec.getfloat(name, default, **kwargs))

    def get_bool(self, name: str, default: bool) -> bool:
        sec = self._pick(name)
        return bool(sec.getboolean(name, default))

    # --------------------------------------------------------------------------------------------------

    def get_str_list(
        self, name: str, default: Tuple[str, ...] = (), *, sep: str = ",", count: Optional[int] = None
    ) -> Tuple[str, ...]:
        sec = self._pick(name)
        return tuple(sec.getlist(name, default, sep=sep, count=count))

    def get_int_list(
        self, name: str, default: Tuple[int, ...] = (), *, sep: str = ",", count: Optional[int] = None, **kwargs
    ) -> Tuple[int, ...]:
        sec = self._pick(name)
        return tuple(sec.getintlist(name, default, sep=sep, count=count, **kwargs))

    def get_float_list(
        self, name: str, default: Tuple[float, ...] = (), *, sep: str = ",", count: Optional[int] = None, **kwargs
    ) -> Tuple[float, ...]:
        sec = self._pick(name)
        return tuple(sec.getfloatlist(name, default, sep=sep, count=count, **kwargs))

    # --------------------------------------------------------------------------------------------------

    def get_str_opt(self, name: str) -> Optional[str]:
        sec = self._pick(name)
        v = sec.get(name, None)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    def get_int_opt(self, name: str, **kwargs) -> Optional[int]:
        sec = self._pick(name)
        v = sec.getint(name, None, **kwargs)
        if v is None:
            return None
        s = int(v)
        return s if s else None
    
    # --------------------------------------------------------------------------------------------------

    def get_str_list_opt(
        self, name: str, *, sep: str = ",", count: Optional[int] = None
    ) -> Optional[Tuple[str, ...]]:
        sec = self._pick(name)
        v = sec.getlist(name, None, sep=sep, count=count)
        return None if v is None else tuple(v)

    def get_int_list_opt(
        self, name: str, *, sep: str = ",", count: Optional[int] = None, **kwargs
    ) -> Optional[Tuple[int, ...]]:
        sec = self._pick(name)
        v = sec.getintlist(name, None, sep=sep, count=count, **kwargs)
        return None if v is None else tuple(v)

    def get_float_list_opt(
        self, name: str, *, sep: str = ",", count: Optional[int] = None, **kwargs
    ) -> Optional[Tuple[float, ...]]:
        sec = self._pick(name)
        v = sec.getfloatlist(name, None, sep=sep, count=count, **kwargs)
        return None if v is None else tuple(v)

    # ---------------- required variants

    def require_str(self, name: str) -> str:
        if self._has(self.override, name):
            return str(self.override.get(name)).strip()
        if self._has(self.base, name):
            return str(self.base.get(name)).strip()
        return str(self.override.get(name)).strip()

    def require_int(self, name: str, **kwargs) -> int:
        if self._has(self.override, name):
            return int(self.override.getint(name, **kwargs))
        if self._has(self.base, name):
            return int(self.base.getint(name, **kwargs))
        return int(self.override.getint(name, **kwargs))

    def require_float(self, name: str, **kwargs) -> float:
        if self._has(self.override, name):
            return float(self.override.getfloat(name, **kwargs))
        if self._has(self.base, name):
            return float(self.base.getfloat(name, **kwargs))
        return float(self.override.getfloat(name, **kwargs))

    def require_bool(self, name: str) -> bool:
        if self._has(self.override, name):
            return bool(self.override.getboolean(name))
        if self._has(self.base, name):
            return bool(self.base.getboolean(name))
        return bool(self.override.getboolean(name))


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
