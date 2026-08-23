from cutex.autotune import Config, autotune, read_autotune_spec


def test_autotune_decorator_records_validated_spec():
    @autotune(
        configs=[Config({"tile": 64}, name="tile64")],
        key=["m", "n"],
        warmup=2,
        rep=7,
    )
    def kernel(m, n, tile=64):
        return m + n + tile

    spec = read_autotune_spec(kernel)
    assert spec is not None
    assert spec.key == ("m", "n")
    assert spec.configs[0].kwargs == {"tile": 64}
    assert spec.warmup == 2
    assert spec.rep == 7


def test_autotune_rejects_empty_search_space():
    try:
        autotune(configs=[], key=["m"])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty autotune search space should fail")

