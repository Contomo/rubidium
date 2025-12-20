from __future__ import annotations

from typing import Any, Protocol


class GCodeCommand(Protocol):
    def get(self, name: str, default: str | None = ...) -> str: ...
    def get_int(
        self,
        name: str,
        default: int = ...,
        *,
        minval: int | None = ...,
        maxval: int | None = ...,
    ) -> int: ...
    def get_float(
        self,
        name: str,
        default: float = ...,
        *,
        minval: float | None = ...,
        maxval: float | None = ...,
        above: float | None = ...,
        below: float | None = ...,
    ) -> float: ...
    def respond_info(self, msg: str) -> None: ...
    def error(self, msg: str) -> Exception: ...

class ConfigWrapper(Protocol):
    def get_printer(self) -> Any: ...
    def get_name(self) -> str: ...

    def get(self, option: str, default: str | None = ...) -> str: ...

    def getint(
        self,
        option: str,
        default: int = ...,
        *,
        minval: int | None = ...,
        maxval: int | None = ...,
    ) -> int: ...

    def getfloat(
        self,
        option: str,
        default: float = ...,
        *,
        minval: float | None = ...,
        maxval: float | None = ...,
        above: float | None = ...,
        below: float | None = ...,
    ) -> float: ...
    def error(self, msg: str) -> Exception: ...
