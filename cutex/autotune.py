"""Declarative autotune metadata, intentionally independent of CUDA imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class AutotuneError(RuntimeError):
    """Raised when no autotune candidate can be compiled and measured."""


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class Config:
    """One compile-time configuration in an autotune search space."""

    kwargs: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    pre_hook: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", _copy_mapping(self.kwargs))
        if self.name is not None and not self.name.strip():
            raise ValueError("Config.name cannot be blank")

    @property
    def label(self) -> str:
        return self.name or repr(dict(self.kwargs))


@dataclass(frozen=True)
class AutotuneSpec:
    configs: tuple[Config, ...]
    key: tuple[str, ...]
    warmup: int = 5
    rep: int = 30
    cache_results: bool = True
    force_retune: bool = False
    cache_path: Path | None = None
    do_bench: Callable[..., float] | None = None


def autotune(
    *,
    configs: Sequence[Config],
    key: Sequence[str],
    warmup: int = 5,
    rep: int = 30,
    cache_results: bool = True,
    force_retune: bool = False,
    cache_path: str | PathLike[str] | None = None,
    do_bench: Callable[..., float] | None = None,
):
    """Attach an autotune search space to a ``@cute.jit`` host function.

    The decorator only stores metadata. :func:`cutex.compile` performs the
    actual compilation, CUDA-event benchmarking, and persistent cache lookup.
    """

    normalized_configs = tuple(configs)
    normalized_key = tuple(str(name) for name in key)
    if not normalized_configs:
        raise ValueError("autotune requires at least one Config")
    if any(not name for name in normalized_key):
        raise ValueError("autotune key names cannot be blank")
    if len(set(normalized_key)) != len(normalized_key):
        raise ValueError("autotune key names must be unique")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if rep < 1:
        raise ValueError("rep must be at least one")

    normalized_path = None if cache_path is None else Path(cache_path)
    spec = AutotuneSpec(
        configs=normalized_configs,
        key=normalized_key,
        warmup=warmup,
        rep=rep,
        cache_results=cache_results,
        force_retune=force_retune,
        cache_path=normalized_path,
        do_bench=do_bench,
    )

    def decorator(kernel):
        setattr(kernel, "__cutex_autotune__", spec)
        return kernel

    return decorator


def read_autotune_spec(kernel) -> AutotuneSpec | None:
    spec = getattr(kernel, "__cutex_autotune__", None)
    if spec is not None:
        return spec
    call = getattr(kernel, "__call__", None)
    return getattr(call, "__cutex_autotune__", None)

