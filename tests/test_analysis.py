import math

import pytest

from app.analysis import (
    MAX_MOISTURE,
    Report,
    SampleUse,
    analyze,
    is_candidate,
    minimal_correction_size,
    replacement_is_compatible,
)


def report(no, samples, low, high):
    return Report(
        report_no=no,
        samples=tuple(SampleUse(sample_no=s, grams=g) for s, g in samples),
        low=low,
        high=high,
    )


def test_compatible_batch_passes_and_reports_sample_ranges():
    reports = [
        report("R1", [("A", 10), ("B", 10)], 40, 60),
        report("R2", [("A", 10), ("B", 30)], 30, 50),
    ]
    result = analyze(reports)
    assert result.feasible
    assert result.candidates == ()
    ranges = dict(result.sample_ranges)
    assert ranges["A"].low <= 20 <= ranges["A"].high
    assert ranges["B"].low <= 60 <= ranges["B"].high
    for low, high in ((r.low, r.high) for r in ranges.values()):
        assert 0 <= low <= high <= 100


def test_weighting_is_mass_based():
    # A alone is 10%, B alone is 90%; the 3:1 blend must be 30%.
    reports = [
        report("R1", [("A", 10)], 9.9, 10.1),
        report("R2", [("B", 10)], 89.9, 90.1),
        report("R3", [("A", 30), ("B", 10)], 29.9, 30.1),
    ]
    assert analyze(reports).feasible


def test_conflict_lists_only_reports_whose_single_correction_suffices():
    reports = [
        report("R1", [("A", 10), ("B", 10)], 10, 20),
        report("R2", [("A", 10), ("B", 10)], 80, 90),
        report("R3", [("A", 10), ("B", 10)], 80, 90),
    ]
    result = analyze(reports)
    assert not result.feasible
    # Dropping R1 leaves R2 and R3 which agree; dropping R2 still leaves R1 vs R3.
    assert [c.report_no for c in result.candidates] == ["R1"]
    allowed = result.candidates[0].allowed
    assert math.isclose(allowed.low, 80, abs_tol=1e-6)
    assert math.isclose(allowed.high, 90, abs_tol=1e-6)


def test_multiple_single_correction_candidates_with_disjoint_pairs():
    # R1 and R2 pin sample A at different values, so at least one of them is
    # wrong; R3's blend agrees with neither pin.  Both R1 and R2 therefore are
    # single-correction candidates (removing either leaves a consistent pair).
    reports = [
        report("R1", [("A", 10)], 10, 10),
        report("R2", [("A", 10)], 20, 20),
        report("R3", [("A", 10), ("B", 90)], 18, 19),
    ]
    result = analyze(reports)
    assert not result.feasible
    candidates = {c.report_no: c.allowed for c in result.candidates}
    assert set(candidates) == {"R1", "R2"}
    # Removing R1 leaves A pinned at 20, so R1's retest must read exactly 20%.
    assert math.isclose(candidates["R1"].low, 20, abs_tol=1e-6)
    assert math.isclose(candidates["R1"].high, 20, abs_tol=1e-6)
    # Removing R2 leaves A pinned at 10; R1's own blend is just A, i.e. 10%.
    assert math.isclose(candidates["R2"].low, 10, abs_tol=1e-6)
    assert math.isclose(candidates["R2"].high, 10, abs_tol=1e-6)


def test_no_single_correction_hint_counts_group_size():
    reports = [
        report("R1", [("A", 10)], 10, 20),
        report("R2", [("A", 10)], 40, 50),
        report("R3", [("A", 10)], 80, 90),
    ]
    result = analyze(reports)
    assert not result.feasible
    assert result.candidates == ()
    assert minimal_correction_size(reports, max_group=3) == 2


def test_three_disagreeing_camps_require_three_corrections():
    reports = [
        report("R1", [("A", 10)], 10, 20),
        report("R2", [("A", 10)], 10, 20),
        report("R3", [("A", 10)], 40, 50),
        report("R4", [("A", 10)], 40, 50),
        report("R5", [("A", 10)], 80, 90),
        report("R6", [("A", 10)], 80, 90),
    ]
    result = analyze(reports)
    assert not result.feasible
    assert result.candidates == ()
    assert minimal_correction_size(reports, max_group=3) is None


def test_candidate_detection_depends_on_weighted_blend():
    reports = [
        report("R1", [("A", 10)], 0, 10),
        report("R2", [("B", 10)], 90, 100),
        report("R3", [("A", 90), ("B", 10)], 9, 19),
    ]
    assert analyze(reports).feasible
    assert not is_candidate(reports, "R1")

    contradictory = reports[:2] + [
        report("R3", [("A", 90), ("B", 10)], 50, 60)
    ]
    assert not analyze(contradictory).feasible
    assert is_candidate(contradictory, "R3")


def test_replacement_boundaries():
    reports = [
        report("R1", [("A", 10), ("B", 10)], 10, 20),
        report("R2", [("A", 10), ("B", 10)], 80, 90),
    ]
    assert replacement_is_compatible(reports, "R1", 80, 90)
    assert replacement_is_compatible(reports, "R1", 80, 100)
    # Any interval overlapping [80, 90] restores compatibility.
    assert replacement_is_compatible(reports, "R1", 81, 90)
    assert not replacement_is_compatible(reports, "R1", 91, 100)
    assert not replacement_is_compatible(reports, "R1", 0, 79.9)


def test_unknown_report_is_not_replaceable():
    reports = [report("R1", [("A", 1)], 1, 2)]
    assert not replacement_is_compatible(reports, "R9", 0, 100)


def test_zero_gram_report_is_rejected():
    from app.analysis import AnalysisError, MixtureModel

    with pytest.raises(AnalysisError):
        MixtureModel([report("R1", [("A", 0)], 1, 2)])


def test_degenerate_single_sample_scale_limit():
    # A is pinned to 0% and B to 100%, so any equal-mass blend is exactly 50%.
    reports = [
        report("R1", [("A", 10)], 0, 0),
        report("R2", [("B", 10)], 100, 100),
        report("R3", [("A", 10), ("B", 10)], 10, 20),
    ]
    result = analyze(reports)
    assert not result.feasible
    candidates = {c.report_no: c.allowed for c in result.candidates}
    # R1's own reading can never be questioned: its blend is pure A.
    assert "R1" not in candidates
    assert math.isclose(candidates["R3"].low, 50, abs_tol=1e-6)
    assert math.isclose(candidates["R3"].high, 50, abs_tol=1e-6)
    assert math.isclose(candidates["R2"].low, 20, abs_tol=1e-6)
    assert math.isclose(candidates["R2"].high, 40, abs_tol=1e-6)
    assert MAX_MOISTURE == 100.0


def test_blend_conflict_needs_two_corrections():
    # A fixed at 10%, B fixed at 70%.  The 1:1 blend must be 40% (claims 25%)
    # and the 1:2 blend must be 50% (claims 40%).  Removing any one report
    # leaves at least one impossible claim on the *same* two pinned samples,
    # so two reports must be retested.
    reports = [
        report("R1", [("A", 10)], 10, 10),
        report("R2", [("B", 10)], 70, 70),
        report("R3", [("A", 10), ("B", 10)], 25, 25),
        report("R4", [("A", 10), ("B", 20)], 40, 40),
    ]
    result = analyze(reports)
    assert not result.feasible
    assert result.candidates == ()
    assert minimal_correction_size(reports, max_group=3) == 2
