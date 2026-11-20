"""Feasibility analysis for blended mineral-powder moisture reports.

Model
-----
Every original sample (原样) has one unknown moisture percentage ``x_s`` in
``[0, 100]`` shared by all reports that use it.  A report ``r`` declares, for
each original sample ``s``, the mass ``m[r, s]`` (grams) taken into the blend.
Its blended moisture is the mass-weighted average::

    y_r = sum_s m[r, s] * x_s / sum_s m[r, s]

Each report supplies a closed measured interval ``[low_r, high_r]``.  The batch
is compatible exactly when there is an assignment of sample moistures that
puts every ``y_r`` inside its interval.  That is a linear feasibility problem
and is solved with HiGHS through :mod:`scipy.optimize.linprog`.

A report is a *single-correction candidate* when the batch is infeasible but
becomes feasible after that one report is dropped (i.e. replaced by a retest).
For such a candidate the feasible range of its own blended moisture under all
remaining reports is also computed; any retest interval that overlaps that
range restores compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog

MIN_MOISTURE = 0.0
MAX_MOISTURE = 100.0
# Percentage-point relaxation used to absorb floating point noise in LP output.
# It is far below any laboratory resolution (reports are quoted to 0.01%), so
# it never turns a genuinely contradictory batch into a compatible one.
FEASIBILITY_TOL = 1e-9


class AnalysisError(RuntimeError):
    """Raised when the LP solver fails in an unexpected way."""


@dataclass(frozen=True)
class SampleUse:
    sample_no: str
    grams: float


@dataclass(frozen=True)
class Report:
    report_no: str
    samples: tuple[SampleUse, ...]
    low: float
    high: float


@dataclass(frozen=True)
class FeasibleRange:
    low: float
    high: float


@dataclass(frozen=True)
class Candidate:
    report_no: str
    allowed: FeasibleRange


@dataclass(frozen=True)
class BatchAnalysis:
    feasible: bool
    candidates: tuple[Candidate, ...]
    sample_ranges: tuple[tuple[str, FeasibleRange], ...]
    sample_count: int


class MixtureModel:
    """Normalised matrix form of a batch of reports."""

    def __init__(self, reports: Sequence[Report]):
        if not reports:
            raise AnalysisError("没有可分析的报告")
        self.reports = tuple(reports)
        sample_order: list[str] = []
        seen: set[str] = set()
        for report in reports:
            for use in report.samples:
                if use.sample_no not in seen:
                    seen.add(use.sample_no)
                    sample_order.append(use.sample_no)
        self.sample_nos = tuple(sample_order)
        self.index = {sample_no: i for i, sample_no in enumerate(self.sample_nos)}
        self.n_samples = len(self.sample_nos)

        rows = np.zeros((len(reports), self.n_samples), dtype=float)
        lows = np.empty(len(reports), dtype=float)
        highs = np.empty(len(reports), dtype=float)
        for r, report in enumerate(reports):
            total = sum(use.grams for use in report.samples)
            if total <= 0:
                raise AnalysisError(f"报告 {report.report_no} 取用总质量必须大于 0")
            for use in report.samples:
                rows[r, self.index[use.sample_no]] += use.grams / total
            lows[r] = report.low
            highs[r] = report.high
        self.rows = rows
        self.lows = lows
        self.highs = highs

    # -- low level ---------------------------------------------------------
    def _constraints(
        self, skip_reports: Iterable[int] = ()
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        skip = set(skip_reports)
        keep = [i for i in range(len(self.reports)) if i not in skip]
        if not keep:
            return None, None
        selected = self.rows[keep]
        upper = self.highs[keep] + FEASIBILITY_TOL
        lower = -self.lows[keep] + FEASIBILITY_TOL
        return (
            np.vstack([selected, -selected]),
            np.concatenate([upper, lower]),
        )

    def _solve(
        self,
        objective: np.ndarray,
        skip_reports: Iterable[int] = (),
    ) -> float | None:
        a_ub, b_ub = self._constraints(skip_reports)
        result = linprog(
            c=objective,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[(MIN_MOISTURE, MAX_MOISTURE)] * self.n_samples,
            method="highs",
        )
        if result.status == 0:
            return float(result.fun)
        if result.status == 2:  # infeasible
            return None
        raise AnalysisError(f"线性规划求解失败：{result.message}")

    def _check_feasible(self, skip_reports: Iterable[int] = ()) -> bool:
        value = self._solve(np.zeros(self.n_samples), skip_reports)
        return value is not None

    # -- public ------------------------------------------------------------
    def feasible(self) -> bool:
        return self._check_feasible()

    def feasible_without(self, report_index: int) -> bool:
        return self._check_feasible([report_index])

    def feasible_without_group(self, report_indices: Iterable[int]) -> bool:
        return self._check_feasible(report_indices)

    def blended_range_without(self, report_index: int) -> FeasibleRange | None:
        """Range of report's blended moisture under all *other* reports."""
        row = self.rows[report_index]
        low = self._solve(row, [report_index])
        if low is None:
            return None
        negative_high = self._solve(-row, [report_index])
        if negative_high is None:
            return None
        return FeasibleRange(
            low=_clamp(low), high=_clamp(-negative_high)
        )

    def sample_range(self, sample_index: int) -> FeasibleRange | None:
        objective = np.zeros(self.n_samples)
        objective[sample_index] = 1.0
        low = self._solve(objective)
        if low is None:
            return None
        negative_high = self._solve(-objective)
        if negative_high is None:
            return None
        return FeasibleRange(low=_clamp(low), high=_clamp(-negative_high))


def _clamp(value: float) -> float:
    if abs(value) < 1e-9:
        value = 0.0
    if abs(value - MAX_MOISTURE) < 1e-9:
        value = MAX_MOISTURE
    return max(MIN_MOISTURE, min(MAX_MOISTURE, value))


def model_for(reports: Sequence[Report]) -> MixtureModel:
    return MixtureModel(reports)


def analyze(reports: Sequence[Report]) -> BatchAnalysis:
    """Analyse a complete batch.

    If compatible, also report the feasible moisture range for every original
    sample.  If contradictory, report every report whose individual correction
    (retest) can restore compatibility together with the blended-moisture range
    the other reports allow for it.
    """
    model = MixtureModel(reports)
    if model.feasible():
        ranges: list[tuple[str, FeasibleRange]] = []
        for i, sample_no in enumerate(model.sample_nos):
            sample_range = model.sample_range(i)
            if sample_range is None:
                raise AnalysisError("整批可行，但原样可行范围求解失败")
            ranges.append((sample_no, sample_range))
        return BatchAnalysis(
            feasible=True,
            candidates=(),
            sample_ranges=tuple(ranges),
            sample_count=model.n_samples,
        )

    candidates: list[Candidate] = []
    for i, report in enumerate(model.reports):
        if not model.feasible_without(i):
            continue
        allowed = model.blended_range_without(i)
        if allowed is None:
            raise AnalysisError(
                f"移除报告 {report.report_no} 后可行，但允许范围求解失败"
            )
        candidates.append(Candidate(report_no=report.report_no, allowed=allowed))
    return BatchAnalysis(
        feasible=False,
        candidates=tuple(candidates),
        sample_ranges=(),
        sample_count=model.n_samples,
    )


def is_candidate(reports: Sequence[Report], report_no: str) -> bool:
    """Whether correcting only ``report_no`` can make the batch compatible.

    A correction can only relax/remove one report's interval, and dropping that
    interval entirely is the most permissive correction there is.  So the check
    is: the batch must be contradictory now, and must be feasible without the
    report.
    """
    model = MixtureModel(reports)
    if model.feasible():
        return False
    for i, report in enumerate(model.reports):
        if report.report_no == report_no:
            return model.feasible_without(i)
    return False


def replacement_is_compatible(
    reports: Sequence[Report], report_no: str, low: float, high: float
) -> bool:
    """Check the whole batch with one report's interval replaced."""
    replaced: list[Report] = []
    found = False
    for report in reports:
        if report.report_no == report_no:
            found = True
            replaced.append(
                Report(
                    report_no=report.report_no,
                    samples=report.samples,
                    low=low,
                    high=high,
                )
            )
        else:
            replaced.append(report)
    if not found:
        return False
    return MixtureModel(replaced).feasible()


def retest_rejection(
    reports: Sequence[Report], report_no: str, low: float, high: float
) -> str | None:
    """Decide whether a retest may be stored.

    Returns ``None`` when the replacement makes the whole batch compatible,
    otherwise a code explaining why the retest must be refused.
    """
    model = MixtureModel(reports)
    if model.feasible():
        return "already_feasible"
    index = next(
        (i for i, report in enumerate(model.reports) if report.report_no == report_no),
        None,
    )
    if index is None:
        return "unknown_report"
    if not model.feasible_without(index):
        return "not_candidate"
    if not replacement_is_compatible(reports, report_no, low, high):
        return "conflict"
    return None


def minimal_correction_size(
    reports: Sequence[Report], max_group: int = 2
) -> int | None:
    """Smallest number of reports that must be corrected, searched up to ``max_group``.

    Only used to phrase the "expand retesting" hint, so the search is kept
    small on purpose: single corrections have already been ruled out by the
    time this runs, and combinations blow up quickly.  ``2`` means "two are
    enough"; ``None`` means "even ``max_group`` are not enough", i.e. at least
    more than that.
    """
    model = MixtureModel(reports)
    report_count = len(reports)
    for size in range(1, min(max_group, report_count) + 1):
        for group in combinations(range(report_count), size):
            if model.feasible_without_group(group):
                return size
    return None
