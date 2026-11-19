from coffee.infer.bench import decode_throttled, median_ms, parse_throttled


def test_clean_throttle_has_no_flags() -> None:
    assert parse_throttled("throttled=0x0\n") == 0
    assert decode_throttled(0) == []


def test_throttle_bits_are_named() -> None:
    flags = decode_throttled(0x50005)
    assert "under-voltage now" in flags
    assert "throttled now" in flags
    assert "under-voltage has occurred" in flags
    assert "throttling has occurred" in flags


def test_median_of_even_and_odd_counts() -> None:
    assert median_ms([300, 100, 200]) == 200
    assert median_ms([100, 400]) == 250
