"""CUTLASS IKET profiling orchestration and artifact transfer helpers."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


class IketTraceError(RuntimeError):
    """Raised when ``run-iket`` fails or does not produce a usable trace."""


@dataclass(frozen=True)
class IketTraceResult:
    """Files and process metadata produced by one ``run-iket`` invocation."""

    output_dir: Path
    command: tuple[str, ...]
    returncode: int
    artifacts: tuple[Path, ...]

    def to_metadata(self, *, volume_root: str | Path) -> dict[str, Any]:
        root = Path(volume_root)
        files = []
        for path in self.artifacts:
            relative_path = path.relative_to(self.output_dir).as_posix()
            files.append(
                {
                    "kind": _artifact_kind(path),
                    "relative_path": f"iket/{relative_path}",
                    "volume_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                }
            )
        return {
            "backend": "cutlass_iket",
            "enabled": True,
            "instrumentation": "in_kernel",
            "benchmark_instrumented": False,
            "enabled_cluster": None,
            "capture_scope": "all_ctas",
            "files": files,
        }


def build_run_iket_command(
    output_dir: str | Path,
    workload_args: Sequence[str],
    *,
    postprocess: str = "all",
    python_executable: str | None = None,
) -> list[str]:
    """Build the standalone profiler command used inside the Modal container."""

    if postprocess not in {"all", "perfetto", "json"}:
        raise ValueError("postprocess must be one of: all, perfetto, json")
    command = [
        "run-iket",
        "--output-dir",
        str(Path(output_dir)),
        "--clobber",
        "profile",
        "--postprocess",
        postprocess,
    ]
    command.extend(
        [
            "--",
            python_executable or sys.executable,
            "-m",
            "cutex.iket_worker",
            *(str(argument) for argument in workload_args),
        ]
    )
    return command


def run_iket_profile(
    output_dir: str | Path,
    workload_args: Sequence[str],
    *,
    timeout_seconds: float = 10 * 60,
) -> IketTraceResult:
    """Run one deterministic, single-launch workload under CUTLASS IKET."""

    destination = Path(output_dir)
    command = build_run_iket_command(
        destination,
        workload_args,
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise IketTraceError(
            "run-iket was not found; install a CUTLASS DSL release that ships IKET"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise IketTraceError(
            f"run-iket exceeded the {timeout_seconds:g}s timeout"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "run-iket.log"
    log_path.write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        output_tail = (completed.stdout + "\n" + completed.stderr)[-4000:]
        raise IketTraceError(
            f"run-iket failed with exit code {completed.returncode}\n{output_tail}"
        )

    perfetto = sorted(destination.rglob("*.pftrace"))
    trace_json = sorted(destination.rglob("*.trace.json"))
    if not trace_json:
        trace_json = sorted(
            path
            for path in destination.rglob("*.json")
            if "trace" in path.name.lower()
        )
    if not perfetto or not trace_json:
        produced = ", ".join(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        )
        raise IketTraceError(
            "run-iket succeeded but did not produce both Perfetto and JSON traces; "
            f"files: {produced or 'none'}"
        )

    artifacts = tuple(dict.fromkeys([*perfetto, *trace_json, log_path]))
    return IketTraceResult(
        output_dir=destination,
        command=tuple(command),
        returncode=completed.returncode,
        artifacts=artifacts,
    )


def download_iket_artifacts(
    volume: Any,
    files: Iterable[dict[str, Any]],
    destination: str | Path,
) -> list[Path]:
    """Stream selected IKET files from a Modal Volume into a local run folder."""

    output_root = Path(destination)
    downloaded = []
    for artifact in files:
        relative = PurePosixPath(str(artifact["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe IKET artifact path: {relative}")
        target = output_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file_handle:
            for chunk in volume.read_file(str(artifact["volume_path"])):
                file_handle.write(chunk)
        downloaded.append(target)
    return downloaded


def disabled_iket_metadata() -> dict[str, Any]:
    return {
        "backend": "cutlass_iket",
        "enabled": False,
        "instrumentation": "in_kernel",
        "benchmark_instrumented": False,
        "enabled_cluster": None,
        "capture_scope": None,
        "files": [],
    }


def _artifact_kind(path: Path) -> str:
    if path.suffix == ".pftrace":
        return "perfetto"
    if path.name.endswith(".trace.json") or path.suffix == ".json":
        return "json"
    if path.suffix == ".log":
        return "log"
    return "file"
