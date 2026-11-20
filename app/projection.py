"""Exact two-variable projection of the batch moisture feasibility system.

The batch model is a linear system over one unknown moisture ``x_s`` per
original sample (see :mod:`app.analysis`).  When an external laboratory
rechecks only two of the samples, the whole-batch system has to be projected
onto those two variables: we need *exactly* the linear inequalities in
``(x_a, x_b)`` that hold if and only if the remaining samples can take some
moistures making every report compatible.

The projection is computed with Fourier–Motzkin elimination followed by exact
half-plane clipping.  Every coefficient is a :class:`fractions.Fraction`, so
the exported inequalities are exact rational consequences of the stored
decimal gram masses and measured intervals -- no floating point noise.  The
clipping step also yields the irredundant facet description and handles the
degenerate outcomes (a single feasible point or a line segment) explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Sequence

from .analysis import Report

ZERO = Fraction(0)
ONE = Fraction(1)
MIN_X = Fraction(0)
MAX_X = Fraction(100)


class ProjectionError(ValueError):
    """User-facing problem while building or consuming a projection."""


class ProjectionInfeasible(ProjectionError):
    """The whole-batch system has no solution, so no package can be built."""


@dataclass(frozen=True)
class Inequality:
    """``a * x1 + b * x2 <= c`` with exact rational coefficients."""

    a: Fraction
    b: Fraction
    c: Fraction

    def lhs(self, x1: Fraction, x2: Fraction) -> Fraction:
        return self.a * x1 + self.b * x2

    def satisfied_by(self, x1: Fraction, x2: Fraction) -> bool:
        return self.lhs(x1, x2) <= self.c


@dataclass(frozen=True)
class Projection:
    samples: tuple[str, str]
    inequalities: tuple[Inequality, ...]


# ---------------------------------------------------------------------------
# Canonical inequality handling
# ---------------------------------------------------------------------------

def _lcm(left: int, right: int) -> int:
    return left // gcd(left, right) * right


def _oriented_key(
    coeffs: Sequence[Fraction], rhs: Fraction
) -> tuple[tuple[Fraction, ...], Fraction] | None:
    """Primitive integer reduction that *keeps* the inequality direction.

    Unlike :func:`_canonical`, opposite half-planes stay distinct:
    ``x <= 1`` and ``-x <= -1`` remain different keys, while scalar multiples
    describing the same half-plane (``2x <= 2`` and ``x <= 1``) collapse.
    """
    denominator = 1
    for value in (*coeffs, rhs):
        denominator = _lcm(denominator, value.denominator)
    int_coeffs = [int(value * denominator) for value in coeffs]
    int_rhs = int(rhs * denominator)
    common = 0
    for value in (*int_coeffs, int_rhs):
        common = gcd(common, abs(value))
    if common > 0:
        int_coeffs = [value // common for value in int_coeffs]
        int_rhs //= common
    if all(value == 0 for value in int_coeffs):
        if int_rhs < 0:
            raise ProjectionInfeasible("线性系统出现 0 <= 负数 的矛盾约束")
        return None
    return tuple(Fraction(value) for value in int_coeffs), Fraction(int_rhs)


# ---------------------------------------------------------------------------
# Batch system construction
# ---------------------------------------------------------------------------

def _as_fraction(value: float) -> Fraction:
    # str(float) is the shortest decimal that round-trips, i.e. the decimal
    # value originally submitted for the gram mass / measured interval.
    return Fraction(str(value))


def _build_system(
    reports: Sequence[Report],
) -> tuple[list[tuple[list[Fraction], Fraction]], list[str]]:
    """Return constraints ``A . x <= b`` over all samples of the batch."""
    order: list[str] = []
    seen: set[str] = set()
    for report in reports:
        for use in report.samples:
            if use.sample_no not in seen:
                seen.add(use.sample_no)
                order.append(use.sample_no)
    index = {sample_no: i for i, sample_no in enumerate(order)}
    n = len(order)
    constraints: list[tuple[list[Fraction], Fraction]] = []

    # Each sample moisture is pinned to the closed 0%..100% box.
    for i in range(n):
        upper = [ZERO] * n
        lower = [ZERO] * n
        upper[i] = ONE
        lower[i] = -ONE
        constraints.append((upper, MAX_X))
        constraints.append((lower, ZERO))

    for report in reports:
        total = sum((_as_fraction(use.grams) for use in report.samples), ZERO)
        if total <= 0:
            raise ProjectionError(
                f"报告 {report.report_no} 取用总质量必须大于 0"
            )
        weighted = [ZERO] * n
        for use in report.samples:
            weighted[index[use.sample_no]] += _as_fraction(use.grams) / total
        low = _as_fraction(report.low)
        high = _as_fraction(report.high)
        # weighted . x <= high
        constraints.append((list(weighted), high))
        # -weighted . x <= -low
        constraints.append(([-value for value in weighted], -low))
    return constraints, order


# ---------------------------------------------------------------------------
# Fourier-Motzkin elimination
# ---------------------------------------------------------------------------

def _dedupe(
    constraints: list[tuple[list[Fraction], Fraction]],
) -> list[tuple[list[Fraction], Fraction]]:
    # Direction must be preserved: -x <= 0 (the lower box bound x >= 0) is
    # the opposite half-plane of x <= 0 and must never be merged with it.
    unique: set[tuple[tuple[Fraction, ...], Fraction]] = set()
    result: list[tuple[list[Fraction], Fraction]] = []
    for coeffs, rhs in constraints:
        canonical = _oriented_key(coeffs, rhs)
        if canonical is None:
            continue
        key = (canonical[0], canonical[1])
        if key not in unique:
            unique.add(key)
            result.append((list(canonical[0]), canonical[1]))
    return result


def _eliminate(
    constraints: list[tuple[list[Fraction], Fraction]],
    n_variables: int,
    eliminated: int,
) -> list[tuple[list[Fraction], Fraction]]:
    """Project out one variable via Fourier-Motzkin."""
    lowers: list[tuple[list[Fraction], Fraction]] = []
    uppers: list[tuple[list[Fraction], Fraction]] = []
    zeroes: list[tuple[list[Fraction], Fraction]] = []
    for coeffs, rhs in constraints:
        pivot = coeffs[eliminated]
        if pivot == 0:
            zeroes.append((coeffs, rhs))
        elif pivot > 0:
            # z + u . x <= b'
            uppers.append(([c / pivot for c in coeffs], rhs / pivot))
        else:
            # -z + l . x <= b'
            scale = -pivot
            lowers.append(([c / scale for c in coeffs], rhs / scale))

    combined: list[tuple[list[Fraction], Fraction]] = list(zeroes)
    for lower_coeffs, lower_rhs in lowers:
        for upper_coeffs, upper_rhs in uppers:
            merged = [
                lower_coeffs[j] + upper_coeffs[j] for j in range(n_variables)
            ]
            merged[eliminated] = ZERO
            combined.append((merged, lower_rhs + upper_rhs))
    # Canonicalising after every step keeps the pairwise blow-up in check.
    return _dedupe(combined)


def fourier_motzkin(
    constraints: list[tuple[list[Fraction], Fraction]],
    n_variables: int,
    keep: Sequence[int],
) -> list[Inequality]:
    """Eliminate every variable except those in ``keep`` (ordered)."""
    keep_set = set(keep)
    active = [
        ([Fraction(c) for c in coeffs], Fraction(rhs))
        for coeffs, rhs in constraints
    ]
    for eliminated in range(n_variables):
        if eliminated in keep_set:
            continue
        active = _eliminate(active, n_variables, eliminated)
    kept = list(keep)
    projected: list[Inequality] = []
    for coeffs, rhs in active:
        projected.append(
            Inequality(
                a=coeffs[kept[0]],
                b=coeffs[kept[1]],
                c=rhs,
            )
        )
    return projected


# ---------------------------------------------------------------------------
# Exact 2D half-plane clipping and facet extraction
# ---------------------------------------------------------------------------

Point = tuple[Fraction, Fraction]


def _intersection(
    start: Point, end: Point, inequality: Inequality
) -> Point:
    a, b, c = inequality.a, inequality.b, inequality.c
    f_start = a * start[0] + b * start[1]
    f_end = a * end[0] + b * end[1]
    t = (c - f_start) / (f_end - f_start)
    return (
        start[0] + t * (end[0] - start[0]),
        start[1] + t * (end[1] - start[1]),
    )


def _clip_polygon(polygon: list[Point], inequality: Inequality) -> list[Point]:
    if not polygon:
        return []
    result: list[Point] = []
    start = polygon[-1]
    start_inside = inequality.satisfied_by(*start)
    for end in polygon:
        end_inside = inequality.satisfied_by(*end)
        if end_inside:
            if not start_inside:
                result.append(_intersection(start, end, inequality))
            result.append(end)
        elif start_inside:
            result.append(_intersection(start, end, inequality))
        start = end
        start_inside = end_inside
    return _dedupe_points(result)


def _dedupe_points(points: list[Point]) -> list[Point]:
    result: list[Point] = []
    for point in points:
        if not result or result[-1] != point:
            result.append(point)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def _canonical_line(
    a: Fraction, b: Fraction, c: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    key = _oriented_key([a, b], c)
    assert key is not None
    (ca, cb), cc = key
    return ca, cb, cc


def _point_facets(point: Point) -> tuple[Inequality, ...]:
    x, y = point
    # x1 = x and x2 = y, each equality split into two closed half-planes.
    return (
        Inequality(ONE, ZERO, x),
        Inequality(-ONE, ZERO, -x),
        Inequality(ZERO, ONE, y),
        Inequality(ZERO, -ONE, -y),
    )


def _segment_facets(start: Point, end: Point) -> tuple[Inequality, ...]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    # A normal to the segment direction (dx, dy) is (-dy, dx); the supporting
    # line is n . p = n . start.
    nx, ny, level = _canonical_line(
        -dy, dx, -dy * start[0] + dx * start[1]
    )
    # Caps along the segment direction: d . start <= d . p <= d . end.
    t_low = dx * start[0] + dy * start[1]
    t_high = dx * end[0] + dy * end[1]
    facets = [
        Inequality(nx, ny, level),
        Inequality(-nx, -ny, -level),
        Inequality(dx, dy, t_high),
        Inequality(-dx, -dy, -t_low),
    ]
    return tuple(_dedupe_inequalities(facets))


def _dedupe_inequalities(
    inequalities: Sequence[Inequality],
) -> list[Inequality]:
    seen: set[tuple[Fraction, Fraction, Fraction]] = set()
    result: list[Inequality] = []
    for inequality in inequalities:
        key = _oriented_key(
            [inequality.a, inequality.b], inequality.c
        )
        if key is None:
            continue
        (a, b), c = key
        full_key = (a, b, c)
        if full_key not in seen:
            seen.add(full_key)
            result.append(Inequality(a, b, c))
    return result


def _polygon_facets(polygon: list[Point]) -> tuple[Inequality, ...]:
    unique = _dedupe_points(polygon)
    if len(unique) == 1:
        return _point_facets(unique[0])

    # Detect a degenerate (collinear) polygon and reduce it to its segment.
    if len(unique) >= 2:
        p0, p1 = unique[0], unique[1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        collinear = all(
            dx * (p[1] - p0[1]) == dy * (p[0] - p0[0]) for p in unique[2:]
        )
        if collinear:
            # Keep the two extreme points along the segment direction.
            direction = (dx, dy)
            scored = sorted(
                unique,
                key=lambda p: direction[0] * p[0] + direction[1] * p[1],
            )
            return _segment_facets(scored[0], scored[-1])

    vertices = _drop_collinear(unique)
    interior = (
        sum((p[0] for p in vertices), ZERO) / len(vertices),
        sum((p[1] for p in vertices), ZERO) / len(vertices),
    )
    facets: list[Inequality] = []
    count = len(vertices)
    for i in range(count):
        start = vertices[i]
        end = vertices[(i + 1) % count]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        a, b = -dy, dx
        c = a * start[0] + b * start[1]
        # Orient the half-plane towards the polygon interior.
        if a * interior[0] + b * interior[1] > c:
            a, b, c = -a, -b, -c
        ca, cb, cc = _canonical_line(a, b, c)
        facets.append(Inequality(ca, cb, cc))
    return tuple(_dedupe_inequalities(facets))


def _drop_collinear(points: list[Point]) -> list[Point]:
    """Remove vertices lying in the middle of a straight polygon edge."""
    result: list[Point] = []
    count = len(points)
    for i, point in enumerate(points):
        previous = points[(i - 1) % count]
        following = points[(i + 1) % count]
        d1x = point[0] - previous[0]
        d1y = point[1] - previous[1]
        d2x = following[0] - point[0]
        d2y = following[1] - point[1]
        if d1x * d2y == d1y * d2x:
            # Collinear with its two neighbours: the corner is redundant.
            continue
        result.append(point)
    return result or points


def projection_polygon(
    inequalities: Sequence[Inequality],
) -> list[Point]:
    """Clip the 0%..100% box by every inequality; empty list means infeasible."""
    polygon: list[Point] = [
        (MIN_X, MIN_X),
        (MAX_X, MIN_X),
        (MAX_X, MAX_X),
        (MIN_X, MAX_X),
    ]
    for inequality in inequalities:
        polygon = _clip_polygon(polygon, inequality)
        if not polygon:
            return []
    return polygon


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def project_batch(
    reports: Sequence[Report], selected: tuple[str, str]
) -> Projection:
    """Project the whole-batch feasibility system onto two selected samples.

    Raises :class:`ProjectionInfeasible` when the submitted batch itself has
    no compatible assignment, and :class:`ProjectionError` for bad selection.
    """
    sample_a, sample_b = selected
    if sample_a == sample_b:
        raise ProjectionError("必须选择两个不同的原样")
    constraints, order = _build_system(reports)
    index = {sample_no: i for i, sample_no in enumerate(order)}
    if sample_a not in index or sample_b not in index:
        raise ProjectionError("所选原样不在该批次中")
    kept = (index[sample_a], index[sample_b])
    two_d = fourier_motzkin(constraints, len(order), kept)
    polygon = projection_polygon(two_d)
    if not polygon:
        raise ProjectionInfeasible(
            "整批化验单互相矛盾，不存在满足全部报告的原样含水率组合"
        )
    facets = _polygon_facets(polygon)
    return Projection(
        samples=(sample_a, sample_b),
        inequalities=tuple(facets),
    )
