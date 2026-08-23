"""CuTeDSL compilation plus persistent first-use autotuning."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .autotune import AutotuneError, AutotuneSpec, Config, read_autotune_spec
from .benchmark import cuda_benchmark


CACHE_FORMAT_VERSION = 1


@dataclass
class CompiledKernel:
    """Callable result of an autotuned CuTeDSL compilation."""

    compiled: Callable[..., Any]
    best_config: Config
    timings_us: dict[str, float]
    cache_hit: str | None = None
    cache_path: Path | None = None

    def __call__(self, *args: Any, **kwargs: Any):
        return self.compiled(*args, **kwargs)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "best_config": {
                "name": self.best_config.name,
                "kwargs": dict(self.best_config.kwargs),
            },
            "candidate_timings_us": dict(self.timings_us),
            "cache_hit": self.cache_hit,
            "cache_path": None if self.cache_path is None else str(self.cache_path),
        }


def _signature(kernel):
    try:
        return inspect.signature(kernel)
    except (TypeError, ValueError):
        return inspect.signature(kernel.__call__)


def _resolve_key_values(
    kernel, spec: AutotuneSpec, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    try:
        bound = _signature(kernel).bind_partial(*args, **kwargs)
        values = dict(bound.arguments)
    except (TypeError, ValueError):
        params = list(_signature(kernel).parameters)
        values = dict(zip(params, args))
        values.update(kwargs)

    missing = [name for name in spec.key if name not in values]
    if missing:
        available = ", ".join(sorted(values)) or "none"
        raise AutotuneError(
            f"missing autotune key field {missing[0]!r}; available fields: {available}"
        )
    return {name: _json_value(values[name]) for name in spec.key}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    item = getattr(value, "value", None)
    if item is not None and item is not value:
        return _json_value(item)
    return str(value)


def _is_constexpr(annotation: Any) -> bool:
    name = getattr(annotation, "__name__", None)
    if name == "Constexpr":
        return True
    return str(annotation).endswith(".Constexpr") or str(annotation).endswith("Constexpr'>")


def _runtime_arguments(
    kernel, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Drop compile-time Constexpr parameters from a compiled-function call."""

    params = list(_signature(kernel).parameters.values())
    runtime_args: list[Any] = []
    for index, value in enumerate(args):
        if index >= len(params) or not _is_constexpr(params[index].annotation):
            runtime_args.append(value)

    by_name = {param.name: param for param in params}
    runtime_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name not in by_name or not _is_constexpr(by_name[name].annotation)
    }
    return tuple(runtime_args), runtime_kwargs


def _kernel_identifier(kernel) -> str:
    for target in (kernel, getattr(kernel, "__wrapped__", None), getattr(kernel, "fn", None)):
        if target is None:
            continue
        module = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
        if module and qualname:
            return f"{module}.{qualname}"
    cls = type(kernel)
    return f"{cls.__module__}.{cls.__qualname__}"


def _read_cache(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    if payload.get("version") != CACHE_FORMAT_VERSION:
        return []
    entries = payload.get("entries")
    return entries if isinstance(entries, list) else []


def _write_cache(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CACHE_FORMAT_VERSION, "entries": entries}
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _same_entry(entry: dict[str, Any], kernel_id: str, key_values: dict[str, Any]) -> bool:
    return entry.get("kernel") == kernel_id and entry.get("key") == key_values


def _find_cached_config(
    entries: list[dict[str, Any]],
    kernel_id: str,
    key_values: dict[str, Any],
    configs: tuple[Config, ...],
) -> tuple[Config, dict[str, float]] | None:
    for entry in entries:
        if not _same_entry(entry, kernel_id, key_values):
            continue
        cached = entry.get("best_config", {})
        for config in configs:
            if cached.get("name") == config.name and cached.get("kwargs") == dict(config.kwargs):
                timings = {
                    str(name): float(value)
                    for name, value in entry.get("candidate_timings_us", {}).items()
                }
                return config, timings
    return None


def _save_best(
    path: Path,
    entries: list[dict[str, Any]],
    kernel_id: str,
    key_values: dict[str, Any],
    config: Config,
    timings_us: dict[str, float],
) -> None:
    retained = [
        entry for entry in entries if not _same_entry(entry, kernel_id, key_values)
    ]
    retained.append(
        {
            "kernel": kernel_id,
            "key": key_values,
            "best_config": {"name": config.name, "kwargs": dict(config.kwargs)},
            "candidate_timings_us": timings_us,
        }
    )
    _write_cache(path, retained)


def compile(
    kernel,
    *args: Any,
    verbose: bool = False,
    force_retune: bool | None = None,
    cache_path: str | os.PathLike[str] | None = None,
    do_bench: Callable[..., float] | None = None,
    _compile_fn: Callable[..., Any] | None = None,
    **kwargs: Any,
):
    """Compile a CuTeDSL function and tune any attached :func:`autotune` spec.

    A custom ``do_bench`` receives ``(compiled, *runtime_args, warmup=, rep=,
    **runtime_kwargs)`` and must return microseconds. The private ``_compile_fn``
    hook exists so the orchestration can be unit-tested without a local GPU.
    """

    if _compile_fn is None:
        import cutlass.cute as cute

        _compile_fn = cute.compile

    spec = read_autotune_spec(kernel)
    if spec is None:
        return _compile_fn(kernel, *args, **kwargs)

    effective_force = spec.force_retune if force_retune is None else force_retune
    effective_path = Path(cache_path) if cache_path is not None else spec.cache_path
    key_values = _resolve_key_values(kernel, spec, args, kwargs)
    kernel_id = _kernel_identifier(kernel)
    entries = _read_cache(effective_path) if effective_path is not None else []

    cached = None
    if spec.cache_results and not effective_force:
        cached = _find_cached_config(entries, kernel_id, key_values, spec.configs)
    if cached is not None:
        config, timings_us = cached
        if verbose:
            print(f"[cutex.autotune] disk cache hit: {config.label}")
        if config.pre_hook is not None:
            config.pre_hook(kernel, *args, **kwargs)
        compile_kwargs = {**kwargs, **dict(config.kwargs)}
        compiled = _compile_fn(kernel, *args, **compile_kwargs)
        return CompiledKernel(
            compiled=compiled,
            best_config=config,
            timings_us=timings_us,
            cache_hit="disk",
            cache_path=effective_path,
        )

    runtime_args, runtime_kwargs = _runtime_arguments(kernel, args, kwargs)
    benchmark_fn = do_bench or spec.do_bench
    failures: list[tuple[str, Exception]] = []
    candidates: list[tuple[float, Config, Callable[..., Any]]] = []
    timings_us: dict[str, float] = {}

    for config in spec.configs:
        try:
            if config.pre_hook is not None:
                config.pre_hook(kernel, *args, **kwargs)
            compile_kwargs = {**kwargs, **dict(config.kwargs)}
            compiled = _compile_fn(kernel, *args, **compile_kwargs)
            if benchmark_fn is None:
                timing = cuda_benchmark(
                    compiled,
                    *runtime_args,
                    warmup=spec.warmup,
                    rep=spec.rep,
                    **runtime_kwargs,
                ).median_us
            else:
                timing = float(
                    benchmark_fn(
                        compiled,
                        *runtime_args,
                        warmup=spec.warmup,
                        rep=spec.rep,
                        **runtime_kwargs,
                    )
                )
            if timing < 0:
                raise ValueError("benchmark returned a negative duration")
            timings_us[config.label] = timing
            candidates.append((timing, config, compiled))
            if verbose:
                print(f"[cutex.autotune] {config.label}: {timing:.3f} us")
        except Exception as exc:  # one invalid candidate should not abort the search
            failures.append((config.label, exc))
            if verbose:
                print(f"[cutex.autotune] rejected {config.label}: {type(exc).__name__}: {exc}")

    if not candidates:
        details = "\n".join(
            f"- {label}: {type(exc).__name__}: {exc}" for label, exc in failures
        )
        raise AutotuneError(f"all autotune candidates failed\n{details}")

    _, best_config, best_compiled = min(candidates, key=lambda item: item[0])
    if verbose:
        print(f"[cutex.autotune] selected: {best_config.label}")

    if spec.cache_results and effective_path is not None:
        _save_best(
            effective_path,
            entries,
            kernel_id,
            key_values,
            best_config,
            timings_us,
        )

    return CompiledKernel(
        compiled=best_compiled,
        best_config=best_config,
        timings_us=timings_us,
        cache_path=effective_path,
    )
