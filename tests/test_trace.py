import subprocess
import sys

import pytest

from cutex.trace import (
    build_run_iket_command,
    download_iket_artifacts,
    run_iket_profile,
)


def test_build_run_iket_command_profiles_one_worker_process(tmp_path):
    command = build_run_iket_command(
        tmp_path / "iket",
        ["dense-gemm", "--m", "512", "--n", "512", "--k", "512"],
    )

    assert command[:4] == [
        "run-iket",
        "--output-dir",
        str(tmp_path / "iket"),
        "--clobber",
    ]
    assert "--enabled-cluster" not in command
    separator = command.index("--")
    assert command[separator + 1 : separator + 4] == [
        sys.executable,
        "-m",
        "cutex.iket_worker",
    ]


def test_build_run_iket_command_can_reserve_event_buffer(tmp_path):
    command = build_run_iket_command(
        tmp_path / "iket",
        ["dense-gemm", "--implementation", "manual_pipeline_v9"],
        max_ts_cnt_per_warp=2048,
    )

    profile = command.index("profile")
    separator = command.index("--")
    assert command[profile + 1 : separator] == [
        "--max-ts-cnt-per-warp",
        "2048",
        "--postprocess",
        "all",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_ts_cnt_per_warp": 0}, "must be positive"),
    ],
)
def test_build_run_iket_command_validates_sampling_options(
    tmp_path, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        build_run_iket_command(tmp_path / "iket", ["dense-gemm"], **kwargs)


def test_run_iket_profile_requires_perfetto_and_json(monkeypatch, tmp_path):
    output_dir = tmp_path / "iket"

    def fake_run(command, **_kwargs):
        output_dir.mkdir(parents=True)
        (output_dir / "kernel.pftrace").write_bytes(b"perfetto")
        (output_dir / "iket_pid_1.trace.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "worker ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_iket_profile(
        output_dir,
        ["dense-gemm"],
        instrumented_cluster=(32, 32, 0),
        max_ts_cnt_per_warp=2048,
    )
    metadata = result.to_metadata(volume_root=tmp_path)

    assert metadata["backend"] == "cutlass_iket"
    assert metadata["benchmark_instrumented"] is False
    assert metadata["enabled_cluster"] == [32, 32, 0]
    assert metadata["capture_scope"] == "single_cluster_in_kernel"
    assert metadata["max_ts_cnt_per_warp"] == 2048
    assert {item["kind"] for item in metadata["files"]} == {
        "perfetto",
        "json",
        "log",
    }
    assert (output_dir / "run-iket.log").read_text(encoding="utf-8").endswith(
        "worker ok\n\nSTDERR\n"
    )


def test_download_iket_artifacts_streams_volume_files(tmp_path):
    class FakeVolume:
        def read_file(self, path):
            assert path == "runs/1/iket/kernel.pftrace"
            yield b"abc"
            yield b"def"

    files = [
        {
            "relative_path": "iket/kernel.pftrace",
            "volume_path": "runs/1/iket/kernel.pftrace",
        }
    ]
    downloaded = download_iket_artifacts(FakeVolume(), files, tmp_path)

    assert downloaded == [tmp_path / "iket" / "kernel.pftrace"]
    assert downloaded[0].read_bytes() == b"abcdef"


def test_download_iket_artifacts_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        download_iket_artifacts(
            object(),
            [{"relative_path": "../escape", "volume_path": "ignored"}],
            tmp_path,
        )
