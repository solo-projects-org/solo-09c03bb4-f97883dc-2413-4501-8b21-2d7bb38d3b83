import json

import pytest

from app.parsing import (
    MAX_REPORTS,
    UploadError,
    parse_documents,
    parse_upload,
)


def doc(*reports):
    return {"batch_name": "T", "reports": list(reports)}


def simple(report_no, samples=None, interval=(1, 2)):
    return {
        "report_no": report_no,
        "samples": samples or [{"sample_no": "YP-01", "grams": 50}],
        "interval": list(interval),
    }


def test_accepts_canonical_document():
    parsed = parse_documents([("a.json", doc(simple("R1"), simple("R2")))])
    assert parsed.name == "T"
    assert [r.report_no for r in parsed.reports] == ["R1", "R2"]
    assert parsed.reports[0].low == 1 and parsed.reports[0].high == 2


def test_accepts_bare_array_and_single_object_across_files():
    parsed = parse_documents(
        [
            ("a.json", [simple("R1")]),
            ("b.json", simple("R2")),
        ]
    )
    assert [r.report_no for r in parsed.reports] == ["R1", "R2"]


def test_accepts_alternative_field_spellings():
    document = {
        "reports": [
            {
                "报告号": "R1",
                "原样": [{"原样编号": "YP-01", "取用克数": 30}],
                "含水率区间": [3.1, 3.4],
            },
            {
                "reportNo": "R2",
                "samples": {"YP-01": 20},
                "moisture": {"low": 3.0, "high": 3.5},
            },
            {
                "report_no": "R3",
                "sample_grams": {"YP-01": 10},
                "low": 2.9,
                "high": 3.6,
            },
        ]
    }
    parsed = parse_documents([("a.json", document)])
    assert [r.report_no for r in parsed.reports] == ["R1", "R2", "R3"]
    assert parsed.reports[1].samples[0].grams == 20


def test_rejects_duplicate_report_numbers():
    with pytest.raises(UploadError, match="重复"):
        parse_documents([("a.json", doc(simple("R1"), simple("R1")))])


def test_rejects_duplicate_sample_within_report():
    bad = simple(
        "R1",
        [
            {"sample_no": "YP-01", "grams": 10},
            {"sample_no": "YP-01", "grams": 20},
        ],
    )
    with pytest.raises(UploadError, match="重复出现原样"):
        parse_documents([("a.json", doc(bad))])


def test_rejects_interval_outside_zero_to_hundred():
    with pytest.raises(UploadError, match="0%"):
        parse_documents([("a.json", doc(simple("R1", interval=(-1, 5))))])
    with pytest.raises(UploadError, match="0%"):
        parse_documents([("a.json", doc(simple("R1", interval=(90, 101))))])


def test_rejects_inverted_interval():
    with pytest.raises(UploadError, match="下限不能大于上限"):
        parse_documents([("a.json", doc(simple("R1", interval=(5, 4))))])


def test_rejects_zero_or_negative_grams():
    with pytest.raises(UploadError, match="大于 0"):
        parse_documents(
            [("a.json", doc(simple("R1", [{"sample_no": "YP-01", "grams": 0}])))]
        )


def test_rejects_non_finite_numbers():
    payload = json.loads(
        '{"reports":[{"report_no":"R1","samples":[{"sample_no":"A","grams":1}],'
        '"low":NaN,"high":2}]}'
    )
    with pytest.raises(UploadError, match="有限"):
        parse_documents([("a.json", payload)])


def test_rejects_more_than_24_reports():
    reports = [simple(f"R{i}") for i in range(MAX_REPORTS + 1)]
    with pytest.raises(UploadError, match="最多 24"):
        parse_documents([("a.json", doc(*reports))])


def test_rejects_more_than_12_samples():
    samples = [{"sample_no": f"YP-{i:02d}", "grams": 1} for i in range(13)]
    with pytest.raises(UploadError, match="最多 12"):
        parse_documents([("a.json", doc(simple("R1", samples)))])


def test_parse_upload_combines_files_and_paste():
    files = [("a.json", json.dumps(simple("R1")).encode())]
    pasted = json.dumps([simple("R2")])
    parsed = parse_upload(files, pasted, "手工批次")
    assert parsed.name == "手工批次"
    assert [r.report_no for r in parsed.reports] == ["R1", "R2"]


def test_parse_upload_reports_bad_json_line():
    with pytest.raises(UploadError, match="不是合法 JSON"):
        parse_upload([("bad.json", b"{oops")], None, None)


def test_parse_upload_requires_something():
    with pytest.raises(UploadError, match="请上传"):
        parse_upload([], "   ", None)
