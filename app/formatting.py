"""Formatting helpers shared by templates."""

from __future__ import annotations

import math
from fractions import Fraction

from app.analysis import FeasibleRange


def frac(value: Fraction) -> str:
    """Render an exact fraction, with integers shown without denominator."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def approx(value: Fraction, places: int = 6) -> str:
    """Short decimal approximation of an exact fraction for display."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    scale = 10**places
    rounded = (value.numerator * scale + value.denominator // 2) // value.denominator
    whole, tail = divmod(rounded, scale)
    text = f"{whole}.{tail:0{places}d}".rstrip("0").rstrip(".")
    if text.endswith("."):
        text += "0"
    return sign + text


def pct(value: float) -> str:
    """Format a moisture percentage without noisy float tails."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return f"{text}%"


def interval(range_: FeasibleRange) -> str:
    return f"{pct(range_.low)} ～ {pct(range_.high)}"


def grams(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def prefill_low(range_: FeasibleRange) -> str:
    """Round *down* to 4 dp so the prefilled interval stays inside the range."""
    value = math.floor(range_.low * 10_000) / 10_000
    return _plain(max(0.0, value))


def prefill_high(range_: FeasibleRange) -> str:
    """Round *up* to 4 dp so the prefilled interval stays inside the range."""
    value = math.ceil(range_.high * 10_000) / 10_000
    return _plain(min(100.0, value))


def _plain(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text
