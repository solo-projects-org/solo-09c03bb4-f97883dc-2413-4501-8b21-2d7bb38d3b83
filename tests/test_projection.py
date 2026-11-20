"""Tests for the exact two-variable Fourier-Motzkin projection."""

from fractions import Fraction

import pytest

from app.analysis import Report, SampleUse
from app.projection import (
    Inequality,
    ProjectionError,
    ProjectionInfeasible,
    project_batch,
)


def report(no, samples, low, high):
    return Report(
        report_no=no,
        samples=tuple(SampleUse(sample_no=s, grams=g) for s, g in samples),
        low=low,
        high=high,
    )


def holds(projection, x1, x2):
    return all(
        inequality.satisfied_by(Fraction(str(x1)), Fraction(str(x2)))
        for inequality in projection.inequalities
    )


def test_two_sample_box_projection_keeps_both_directions():
    # With no reports other than the moisture box, both variables keep their
    # full 0..100 range: x <= 100 and -x <= 0 must both survive.
    reports = [report("R1", [("A", 10), ("B", 10)], 0, 100)]
    projection = project_batch(reports, ("A", "B"))
    assert Inequality(Fraction(1), Fraction(0), Fraction(100)) in (
        projection.inequalities
    )
    assert Inequality(Fraction(-1), Fraction(0), Fraction(0)) in (
        projection.inequalities
    )
    assert Inequality(Fraction(0), Fraction(1), Fraction(100)) in (
        projection.inequalities
    )
    assert Inequality(Fraction(0), Fraction(-1), Fraction(0)) in (
        projection.inequalities
    )
    assert holds(projection, 0, 0)
    assert holds(projection, 100, 100)
    assert not holds(projection, -1, 50)
    assert not holds(projection, 50, 101)


def test_point_projection_when_both_samples_pinned():
    reports = [
        report("R1", [("A", 10)], 10, 10),
        report("R2", [("B", 10)], 70, 70),
        report("R3", [("A", 10), ("B", 10)], 40, 40),
    ]
    projection = project_batch(reports, ("A", "B"))
    # A point collapses to four half-planes: x1=10 and x2=70, each split in two.
    keys = {(i.a, i.b, i.c) for i in projection.inequalities}
    assert keys == {
        (Fraction(1), Fraction(0), Fraction(10)),
        (Fraction(-1), Fraction(0), Fraction(-10)),
        (Fraction(0), Fraction(1), Fraction(70)),
        (Fraction(0), Fraction(-1), Fraction(-70)),
    }
    assert holds(projection, 10, 70)
    assert not holds(projection, 10.0000001, 70)
    assert not holds(projection, 10, 69.9999999)


def test_segment_projection_preserves_equality_and_caps():
    # Equal-mass blend pinned at 50% means x_A + x_B = 100 inside the box.
    reports = [report("R1", [("A", 10), ("B", 10)], 50, 50)]
    projection = project_batch(reports, ("A", "B"))
    keys = {(i.a, i.b, i.c) for i in projection.inequalities}
    assert (Fraction(1), Fraction(1), Fraction(100)) in keys
    assert (Fraction(-1), Fraction(-1), Fraction(-100)) in keys
    # Endpoint caps: 0 <= x_A <= 100 and 0 <= x_B <= 100.
    assert (Fraction(1), Fraction(-1), Fraction(100)) in keys
    assert (Fraction(-1), Fraction(1), Fraction(100)) in keys
    for point in ((0, 100), (50, 50), (100, 0)):
        assert holds(projection, *point)
    for point in ((0, 99.9), (50, 51), (100, 0.1)):
        assert not holds(projection, *point)


def test_polygon_projection_matches_weighted_intervals():
    # A+B equal mass must lie in [40, 60]: -100 <= x_A + x_B... i.e.
    # x_A + x_B <= 120 and -(x_A + x_B) <= -80.
    reports = [report("R1", [("A", 10), ("B", 10)], 40, 60)]
    projection = project_batch(reports, ("A", "B"))
    keys = {(i.a, i.b, i.c) for i in projection.inequalities}
    assert (Fraction(1), Fraction(1), Fraction(120)) in keys
    assert (Fraction(-1), Fraction(-1), Fraction(-80)) in keys
    assert holds(projection, 40, 40)
    assert holds(projection, 60, 60)
    assert not holds(projection, 61, 60)
    assert not holds(projection, 0, 79)


def test_fractional_gram_coefficients_are_exact():
    # Masses 30 and 20 give the exact weighted row (3/5)x_A + (2/5)x_B <= 41.
    reports = [report("R1", [("A", 30), ("B", 20)], 41, 41)]
    projection = project_batch(reports, ("A", "B"))
    keys = {(i.a, i.b, i.c) for i in projection.inequalities}
    # Primitive integer form: 3 x_A + 2 x_B <= 205.
    assert (Fraction(3), Fraction(2), Fraction(205)) in keys
    assert (Fraction(-3), Fraction(-2), Fraction(-205)) in keys


def test_elimination_of_other_samples_keeps_projection_sound():
    # Four samples; reports pin C and D indirectly, and constrain A and B.
    reports = [
        report("R1", [("C", 10)], 20, 20),
        report("R2", [("D", 10)], 80, 80),
        report("R3", [("A", 10), ("C", 10)], 20, 30),
        report("R4", [("B", 10), ("D", 10)], 70, 80),
        report("R5", [("A", 10), ("B", 10)], 40, 60),
    ]
    projection = project_batch(reports, ("A", "B"))
    # C=20 and R3 => 20 <= (A+20)/2 <= 30 => 20 <= A <= 40.
    assert holds(projection, 20, 60)
    assert holds(projection, 40, 80)
    assert not holds(projection, 19.9, 60)
    # D=80 and R4 => 70 <= (B+80)/2 <= 80 => 60 <= B <= 80.
    assert not holds(projection, 30, 59.9)


def test_infeasible_batch_is_rejected():
    reports = [
        report("R1", [("A", 10)], 10, 20),
        report("R2", [("A", 10)], 40, 50),
        report("R3", [("B", 10), ("A", 10)], 10, 12),
    ]
    with pytest.raises(ProjectionInfeasible):
        project_batch(reports, ("A", "B"))


def test_repeated_sample_selection_is_rejected():
    reports = [report("R1", [("A", 10), ("B", 10)], 0, 100)]
    with pytest.raises(ProjectionError, match="不同的原样"):
        project_batch(reports, ("A", "A"))


def test_unknown_sample_selection_is_rejected():
    reports = [report("R1", [("A", 10), ("B", 10)], 0, 100)]
    with pytest.raises(ProjectionError, match="不在该批次"):
        project_batch(reports, ("A", "Z"))


def test_selection_order_is_preserved():
    reports = [report("R1", [("A", 10), ("B", 10)], 0, 100)]
    first = project_batch(reports, ("A", "B"))
    second = project_batch(reports, ("B", "A"))
    assert first.samples == ("A", "B")
    assert second.samples == ("B", "A")
    # Swapping selection swaps the two coefficient columns.
    for inequality in first.inequalities:
        swapped = Inequality(inequality.b, inequality.a, inequality.c)
        assert swapped in second.inequalities


def test_boundary_points_of_closed_intervals_are_accepted():
    # A single-blend report pins the value exactly; the boundary is included.
    reports = [report("R1", [("A", 10), ("B", 10)], 50, 50)]
    projection = project_batch(reports, ("A", "B"))
    assert holds(projection, 0, 100)
    assert holds(projection, 100, 0)


def test_two_vertex_projection_forms_a_segment_with_working_caps():
    # A pinned at 40 and the equal-mass blend pinned at 45 gives B = 50:
    # the feasible region collapses to the single point (40, 50).  A relaxed
    # blend interval instead yields a segment whose caps are exercised.
    reports = [
        report("R1", [("A", 10)], 40, 40),
        report("R2", [("A", 10), ("B", 10)], 45, 50),
    ]
    projection = project_batch(reports, ("A", "B"))
    # A = 40 fixed; 90 <= 40 + B <= 100 -> B in [50, 60].
    assert holds(projection, 40, 50)
    assert holds(projection, 40, 60)
    assert not holds(projection, 40, 49.9)
    assert not holds(projection, 40, 60.1)
    assert not holds(projection, 40.1, 55)


def test_decimal_inputs_are_taken_at_face_value():
    # Stored decimals must be interpreted exactly, not as binary floats.
    reports = [
        report("R1", [("A", 10)], 3.15, 3.15),
        report("R2", [("B", 10)], 7.25, 7.25),
    ]
    projection = project_batch(reports, ("A", "B"))
    assert holds(projection, 3.15, 7.25)
    assert not holds(projection, 3.1500000000001, 7.25)
