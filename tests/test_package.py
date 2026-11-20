"""Tests for review-package building, serialization, parsing and verification."""

import json
from fractions import Fraction

import pytest

from app.analysis import Report, SampleUse
from app.package import (
    CheckResult,
    PackageError,
    ReviewPackage,
    build_package,
    check_readings,
    fraction_text,
    load_package_bytes,
    package_document,
    package_filename,
    package_json,
    parse_fraction,
    parse_package,
)
from app.projection import ProjectionInfeasible


def report(no, samples, low, high):
    return Report(
        report_no=no,
        samples=tuple(SampleUse(sample_no=s, grams=g) for s, g in samples),
        low=low,
        high=high,
    )


PINNED = [
    report("R1", [("A", 10)], 10, 10),
    report("R2", [("B", 10)], 70, 70),
    report("R3", [("A", 10), ("B", 10)], 40, 40),
]


def make_package():
    return build_package(
        PINNED, batch_no=5, revision=3, selected=("A", "B")
    )


# ---------------------------------------------------------------------------
# Fraction helpers
# ---------------------------------------------------------------------------

def test_fraction_text_omits_unit_denominator():
    assert fraction_text(Fraction(4)) == "4"
    assert fraction_text(Fraction(-2, 3)) == "-2/3"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (3, Fraction(3)),
        (3.5, Fraction(7, 2)),
        ("3.15", Fraction(315, 100)),
        ("17/4", Fraction(17, 4)),
        (" -2/5 ", Fraction(-2, 5)),
    ],
)
def test_parse_fraction_accepts_decimals_and_strings(raw, expected):
    assert parse_fraction(raw, "x") == expected


@pytest.mark.parametrize("raw", [True, False, None, "", "abc", "1/0", object()])
def test_parse_fraction_rejects_bad_values(raw):
    with pytest.raises(PackageError):
        parse_fraction(raw, "x")


# ---------------------------------------------------------------------------
# Building and serialization
# ---------------------------------------------------------------------------

def test_build_package_carries_batch_identity_and_exact_coefficients():
    package = make_package()
    assert package.batch_no == 5
    assert package.revision == 3
    assert package.samples == ("A", "B")
    assert all(
        isinstance(inequality.c, Fraction) for inequality in package.inequalities
    )


def test_build_package_rejects_infeasible_batch():
    bad = [
        report("R1", [("A", 10)], 10, 20),
        report("R2", [("A", 10)], 40, 50),
        report("R3", [("B", 10)], 0, 100),
    ]
    with pytest.raises(ProjectionInfeasible):
        build_package(bad, batch_no=1, revision=1, selected=("A", "B"))


def test_document_contains_only_the_four_required_keys():
    document = package_document(make_package())
    assert set(document) == {"batch_no", "revision", "samples", "inequalities"}
    assert document["batch_no"] == 5
    assert document["revision"] == 3
    assert document["samples"] == ["A", "B"]
    for inequality in document["inequalities"]:
        assert set(inequality) == {"a", "b", "c"}
        for key in ("a", "b", "c"):
            assert isinstance(inequality[key], str)


def test_package_json_round_trips_through_parser():
    package = make_package()
    parsed = parse_package(json.loads(package_json(package)))
    assert parsed == package


def test_package_filename_is_safe_and_informative():
    name = package_filename(make_package())
    assert name == "review_batch-5_rev-3_A_B.json"
    weird = build_package(
        [report("R1", [("YP/01", 10), ("YP 02", 10)], 0, 100)],
        batch_no=1,
        revision=1,
        selected=("YP/01", "YP 02"),
    )
    assert "/" not in package_filename(weird)
    assert " " not in package_filename(weird)


# ---------------------------------------------------------------------------
# Parsing uploaded packages
# ---------------------------------------------------------------------------

def test_parse_rejects_extra_top_level_fields():
    document = package_document(make_package())
    document["secret"] = 1
    with pytest.raises(PackageError, match="未知字段"):
        parse_package(document)


def test_parse_rejects_extra_inequality_fields():
    document = package_document(make_package())
    document["inequalities"][0]["op"] = ">="
    with pytest.raises(PackageError, match="op"):
        parse_package(document)


def test_parse_accepts_explicit_leq_operator():
    document = package_document(make_package())
    for inequality in document["inequalities"]:
        inequality["op"] = "<="
    parsed = parse_package(document)
    assert parsed.inequalities == make_package().inequalities


def test_parse_rejects_wrong_types_and_missing_fields():
    base = package_document(make_package())
    bad = dict(base)
    bad["batch_no"] = "5"
    with pytest.raises(PackageError):
        parse_package(bad)
    bad = dict(base)
    bad["revision"] = 0
    with pytest.raises(PackageError):
        parse_package(bad)
    bad = dict(base)
    bad["samples"] = ["A"]
    with pytest.raises(PackageError):
        parse_package(bad)
    bad = dict(base)
    bad["samples"] = ["A", "A"]
    with pytest.raises(PackageError, match="不能相同"):
        parse_package(bad)
    bad = dict(base)
    bad["inequalities"] = []
    with pytest.raises(PackageError):
        parse_package(bad)
    bad = dict(base)
    bad["inequalities"][0] = {"a": "0", "b": "0", "c": "1"}
    with pytest.raises(PackageError, match="同时为 0"):
        parse_package(bad)


def test_parse_rejects_booleans_as_integers():
    document = package_document(make_package())
    document["batch_no"] = True
    with pytest.raises(PackageError):
        parse_package(document)


def test_load_package_bytes_reports_json_errors():
    with pytest.raises(PackageError, match="合法 JSON"):
        load_package_bytes(b"{oops", "x.json")
    with pytest.raises(PackageError, match="UTF-8"):
        load_package_bytes(b"\xff\xfe", "x.json")
    with pytest.raises(PackageError, match="2 MB"):
        load_package_bytes(b"x" * (2 * 1024 * 1024 + 1), "big.json")


def test_loaded_package_accepts_bom():
    payload = ("﻿" + package_json(make_package())).encode("utf-8")
    parsed = load_package_bytes(payload, "p.json")
    assert parsed == make_package()


# ---------------------------------------------------------------------------
# Reading verification
# ---------------------------------------------------------------------------

def test_check_readings_accepts_the_feasible_point():
    result = check_readings(make_package(), "10", "70")
    assert isinstance(result, CheckResult)
    assert result.accepted
    assert result.readings == (Fraction(10), Fraction(70))
    assert all(row.holds for row in result.rows)


def test_check_readings_rejects_a_point_outside_the_projection():
    result = check_readings(make_package(), "10.1", "70")
    assert not result.accepted
    violated = [row.index for row in result.rows if not row.holds]
    assert violated


def test_check_readings_accepts_fraction_inputs():
    result = check_readings(make_package(), "100/10", "700/10")
    assert result.accepted


@pytest.mark.parametrize(
    "first,second",
    [("", "70"), ("10", ""), ("abc", "70"), ("-0.1", "70"), ("100.1", "70")],
)
def test_check_readings_validates_ranges(first, second):
    with pytest.raises(PackageError):
        check_readings(make_package(), first, second)


def test_boundary_readings_are_accepted_on_closed_intervals():
    package = build_package(
        [report("R1", [("A", 10), ("B", 10)], 50, 50)],
        batch_no=1,
        revision=1,
        selected=("A", "B"),
    )
    assert check_readings(package, "0", "100").accepted
    assert check_readings(package, "100", "0").accepted
    assert not check_readings(package, "0.0001", "100").accepted


def test_review_package_is_self_contained_dataclass():
    package = make_package()
    assert package.projection.samples == ("A", "B")
    assert isinstance(package, ReviewPackage)
