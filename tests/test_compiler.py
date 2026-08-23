from dataclasses import dataclass

from cutex import Config, autotune
from cutex.compiler import CompiledKernel, compile


@dataclass
class FakeCompiled:
    tile: int

    def __call__(self, *_args, **_kwargs):
        return self.tile


def test_compile_selects_and_persists_best_config(tmp_path):
    @autotune(
        configs=[
            Config({"tile": 32}, name="tile32"),
            Config({"tile": 128}, name="tile128"),
        ],
        key=["m"],
        warmup=1,
        rep=3,
    )
    def kernel(m, tile=32):
        return m + tile

    compile_calls = []

    def fake_compile(_kernel, *_args, **kwargs):
        compile_calls.append(kwargs["tile"])
        return FakeCompiled(kwargs["tile"])

    def fake_benchmark(compiled, *_args, **_kwargs):
        return {32: 8.0, 128: 3.0}[compiled.tile]

    cache_path = tmp_path / "autotune.json"
    first = compile(
        kernel,
        1024,
        cache_path=cache_path,
        do_bench=fake_benchmark,
        _compile_fn=fake_compile,
    )
    assert isinstance(first, CompiledKernel)
    assert first.best_config.name == "tile128"
    assert first.cache_hit is None
    assert first(1024) == 128
    assert cache_path.exists()
    assert compile_calls == [32, 128]

    compile_calls.clear()
    second = compile(
        kernel,
        1024,
        cache_path=cache_path,
        do_bench=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disk cache should skip benchmarking")
        ),
        _compile_fn=fake_compile,
    )
    assert second.best_config.name == "tile128"
    assert second.cache_hit == "disk"
    assert compile_calls == [128]


def test_force_retune_ignores_persistent_choice(tmp_path):
    @autotune(
        configs=[Config({"tile": 1}, name="one"), Config({"tile": 2}, name="two")],
        key=["m"],
        rep=1,
    )
    def kernel(m, tile=1):
        return m + tile

    def fake_compile(_kernel, *_args, **kwargs):
        return FakeCompiled(kwargs["tile"])

    cache_path = tmp_path / "autotune.json"
    compile(
        kernel,
        8,
        cache_path=cache_path,
        do_bench=lambda compiled, *_args, **_kwargs: float(compiled.tile),
        _compile_fn=fake_compile,
    )
    retuned = compile(
        kernel,
        8,
        cache_path=cache_path,
        force_retune=True,
        do_bench=lambda compiled, *_args, **_kwargs: float(3 - compiled.tile),
        _compile_fn=fake_compile,
    )
    assert retuned.best_config.name == "two"
    assert retuned.cache_hit is None

