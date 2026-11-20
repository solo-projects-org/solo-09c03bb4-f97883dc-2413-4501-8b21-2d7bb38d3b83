import math

import pytest

from app.analysis import FeasibleRange
from app.formatting import grams, interval, pct, prefill_high, prefill_low


def test_pct_trims_float_noise():
    assert pct(80.0) == "80%"
    assert pct(4.699999999999) == "4.7%"
    assert pct(0.0) == "0%"
    assert pct(100.0) == "100%"


def test_grams_formats_cleanly():
    assert grams(50.0) == "50"
    assert grams(12.5) == "12.5"


def test_interval_uses_full_range():
    assert interval(FeasibleRange(4.69, 4.83)) == "4.69% ～ 4.83%"


@pytest.mark.parametrize(
    "low,high",
    [(4.690000001, 4.8299999), (0.0, 100.0), (12.34567, 12.34568), (0.0, 0.0)],
)
def test_prefill_interval_stays_inside_allowed_range(low, high):
    range_ = FeasibleRange(low, high)
    low_text = prefill_low(range_)
    high_text = prefill_high(range_)
    low_value = float(low_text)
    high_value = float(high_text)
    assert 0.0 <= low_value <= high_value <= 100.0
    assert low_value <= low + 1e-12
    assert high_value >= high - 1e-12
    # The prefill must still overlap the allowed range so it is accepted.
    assert not (high_value < low or low_value > high)


def test_prefill_never_escapes_zero_to_hundred():
    assert prefill_low(FeasibleRange(0.0, 0.0)) == "0"
    assert prefill_high(FeasibleRange(100.0, 100.0)) == "100"
    assert math.isfinite(float(prefill_high(FeasibleRange(99.99999999, 100.0))))
