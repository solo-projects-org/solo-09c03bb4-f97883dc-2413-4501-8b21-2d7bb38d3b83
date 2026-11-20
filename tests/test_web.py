import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def conn(client):
    connection = db.connect()
    try:
        yield connection
    finally:
        connection.close()


CONFLICT = {
    "batch_name": "冲突批次",
    "reports": [
        {
            "report_no": "R1",
            "samples": [{"sample_no": "A", "grams": 10}, {"sample_no": "B", "grams": 10}],
            "interval": [10, 20],
        },
        {
            "report_no": "R2",
            "samples": [{"sample_no": "A", "grams": 10}, {"sample_no": "B", "grams": 10}],
            "interval": [80, 90],
        },
        {
            "report_no": "R3",
            "samples": [{"sample_no": "A", "grams": 10}, {"sample_no": "B", "grams": 10}],
            "interval": [80, 90],
        },
    ],
}

COMPATIBLE = {
    "batch_name": "相容批次",
    "reports": [
        {
            "report_no": "C1",
            "samples": [{"sample_no": "A", "grams": 10}],
            "interval": [10, 20],
        },
        {
            "report_no": "C2",
            "samples": [{"sample_no": "A", "grams": 10}],
            "interval": [12, 18],
        },
    ],
}


def upload(client, document, name=None):
    files = [
        ("files", ("batch.json", json.dumps(document).encode(), "application/json"))
    ]
    data = {"batch_name": name or "", "pasted": ""}
    return client.post("/batches", files=files, data=data, follow_redirects=False)


def batch_id_from(response) -> int:
    location = response.headers["location"]
    assert location.startswith("/batches/")
    return int(location.rstrip("/").split("/")[-1])


def add_batch(conn, document):
    from app.parsing import parse_documents

    parsed = parse_documents([("x.json", document)])
    return db.create_batch(conn, parsed.name, parsed.reports)


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "上传同批化验单" in response.text


def test_upload_compatible_batch_shows_pass(client):
    response = upload(client, COMPATIBLE)
    assert response.status_code == 303
    batch_id = batch_id_from(response)
    page = client.get(f"/batches/{batch_id}")
    assert page.status_code == 200
    assert "通过" in page.text
    assert "整批相容" in page.text


def test_upload_conflict_lists_candidates(client):
    response = upload(client, CONFLICT)
    batch_id = batch_id_from(response)
    page = client.get(f"/batches/{batch_id}")
    assert "互相矛盾" in page.text
    # Only R1 can be corrected alone in this batch.
    assert "R1" in page.text
    assert "其余报告允许的混样含水率" in page.text
    assert "80% ～ 90%" in page.text


def test_invalid_upload_shows_error_and_creates_nothing(client, conn):
    document = {
        "reports": [
            {
                "report_no": "R1",
                "samples": [{"sample_no": "A", "grams": 0}],
                "interval": [1, 2],
            }
        ]
    }
    response = upload(client, document)
    assert response.status_code == 400
    assert "未能建立批次" in response.text
    assert db.list_batches(conn) == []


def test_upload_rejects_more_than_24_reports(client):
    reports = [
        {
            "report_no": f"R{i}",
            "samples": [{"sample_no": "A", "grams": 1}],
            "interval": [1, 2],
        }
        for i in range(25)
    ]
    response = upload(client, {"reports": reports})
    assert response.status_code == 400
    assert "最多 24" in response.text


def test_successful_retest_is_saved_and_updates_conclusion(client, conn):
    batch_id = batch_id_from(upload(client, CONFLICT))
    before = db.get_batch(conn, batch_id)
    assert before.revision == 1
    assert not __import__("app.analysis", fromlist=["analyze"]).analyze(
        before.to_domain()
    ).feasible

    response = client.post(
        f"/batches/{batch_id}/retests",
        data={"revision": "1", "report_no": "R1", "low": "80", "high": "90"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "notice=retest_saved" in response.headers["location"]

    after = db.get_batch(conn, batch_id)
    assert after.revision == 2
    r1 = next(r for r in after.reports if r.report_no == "R1")
    assert (r1.low, r1.high) == (80, 90)
    assert (r1.original_low, r1.original_high) == (10, 20)
    events = db.list_retests(conn, batch_id)
    assert len(events) == 1
    assert events[0].report_no == "R1"
    assert (events[0].new_low, events[0].new_high) == (80, 90)

    page = client.get(f"/batches/{batch_id}")
    assert "通过" in page.text
    assert "原报 10% ～ 20%" in page.text
    assert "复测记录" in page.text


def test_conflicting_retest_is_rejected_and_state_kept(client, conn):
    batch_id = batch_id_from(upload(client, CONFLICT))
    response = client.post(
        f"/batches/{batch_id}/retests",
        data={"revision": "1", "report_no": "R1", "low": "70", "high": "79"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "notice=conflict" in response.headers["location"]
    after = db.get_batch(conn, batch_id)
    assert after.revision == 1
    r1 = next(r for r in after.reports if r.report_no == "R1")
    assert (r1.low, r1.high) == (10, 20)
    assert db.list_retests(conn, batch_id) == []

    page = client.get(response.headers["location"])
    assert "仍不相容" in page.text
    assert "保持现状" in page.text


def test_revision_mismatch_is_rejected(client, conn):
    batch_id = batch_id_from(upload(client, CONFLICT))
    # Simulate a change made elsewhere between page render and submit.
    client.post(
        f"/batches/{batch_id}/retests",
        data={"revision": "1", "report_no": "R1", "low": "80", "high": "90"},
        follow_redirects=False,
    )
    response = client.post(
        f"/batches/{batch_id}/retests",
        data={"revision": "1", "report_no": "R1", "low": "81", "high": "90"},
        follow_redirects=False,
    )
    assert "notice=stale" in response.headers["location"]
    after = db.get_batch(conn, batch_id)
    assert after.revision == 2
    assert len(db.list_retests(conn, batch_id)) == 1
    page = client.get(response.headers["location"])
    assert "已有更正" in page.text
    assert "未保存" in page.text


def test_non_candidate_report_cannot_be_retested(client, conn):
    batch_id = batch_id_from(upload(client, CONFLICT))
    response = client.post(
        f"/batches/{batch_id}/retests",
        data={"revision": "1", "report_no": "R2", "low": "10", "high": "20"},
        follow_redirects=False,
    )
    assert "notice=not_candidate" in response.headers["location"]
    after = db.get_batch(conn, batch_id)
    assert after.revision == 1
    assert db.list_retests(conn, batch_id) == []


def test_retest_on_compatible_batch_is_rejected(client, conn):
    batch_id = batch_id_from(upload(client, COMPATIBLE))
    response = client.post(
        f"/batches/{batch_id}/retests",
        data={"revision": "1", "report_no": "C1", "low": "10", "high": "20"},
        follow_redirects=False,
    )
    assert "notice=already_feasible" in response.headers["location"]
    page = client.get(response.headers["location"])
    assert "现在已经相容" in page.text
    assert db.list_retests(conn, batch_id) == []
    assert db.get_batch(conn, batch_id).revision == 1


@pytest.mark.parametrize(
    "low,high",
    [("101", "110"), ("-1", "5"), ("90", "80"), ("abc", "90"), ("nan", "90")],
)
def test_invalid_retest_values_are_rejected(client, conn, low, high):
    batch_id = batch_id_from(upload(client, CONFLICT))
    response = client.post(
        f"/batches/{batch_id}/retests",
        data={"revision": "1", "report_no": "R1", "low": low, "high": high},
        follow_redirects=False,
    )
    assert "notice=invalid" in response.headers["location"]
    after = db.get_batch(conn, batch_id)
    assert after.revision == 1
    assert db.list_retests(conn, batch_id) == []


def test_unknown_batch_returns_404(client):
    page = client.get("/batches/9999")
    assert page.status_code == 404


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_persistence_across_app_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "persist.db"))
    with TestClient(app) as first:
        response = upload(first, CONFLICT)
        batch_id = batch_id_from(response)
        first.post(
            f"/batches/{batch_id}/retests",
            data={"revision": "1", "report_no": "R1", "low": "80", "high": "90"},
            follow_redirects=False,
        )
    with TestClient(app) as second:
        page = second.get(f"/batches/{batch_id}")
        assert "通过" in page.text
        assert "复测记录" in page.text
        home = second.get("/")
        assert "冲突批次" in home.text


# ---------------------------------------------------------------------------
# External-laboratory review packages
# ---------------------------------------------------------------------------

PINNED_BATCH = {
    "batch_name": "钉死批次",
    "reports": [
        {
            "report_no": "P1",
            "samples": [{"sample_no": "A", "grams": 10}],
            "interval": [10, 10],
        },
        {
            "report_no": "P2",
            "samples": [{"sample_no": "B", "grams": 10}],
            "interval": [70, 70],
        },
        {
            "report_no": "P3",
            "samples": [
                {"sample_no": "A", "grams": 10},
                {"sample_no": "B", "grams": 10},
            ],
            "interval": [40, 40],
        },
    ],
}


def test_batch_page_offers_package_panel_for_supported_batches(client):
    batch_id = batch_id_from(upload(client, PINNED_BATCH))
    page = client.get(f"/batches/{batch_id}")
    assert "外协复核出包" in page.text
    assert "下载复核 JSON" in page.text
    assert "/verify" in page.text


def test_package_download_returns_self_contained_json(client):
    batch_id = batch_id_from(upload(client, PINNED_BATCH))
    response = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "1", "sample_a": "A", "sample_b": "B"},
        follow_redirects=False,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert f"batch-{batch_id}" in disposition
    document = response.json()
    assert set(document) == {"batch_no", "revision", "samples", "inequalities"}
    assert document["batch_no"] == batch_id
    assert document["revision"] == 1
    assert document["samples"] == ["A", "B"]
    assert document["inequalities"]


def test_package_round_trip_through_verify_page_accepts_and_rejects(client):
    batch_id = batch_id_from(upload(client, PINNED_BATCH))
    package_response = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "1", "sample_a": "A", "sample_b": "B"},
        follow_redirects=False,
    )
    payload = package_response.content

    accepted = client.post(
        "/verify",
        files={"file": ("review.json", payload, "application/json")},
        data={"first": "10", "second": "70"},
    )
    assert accepted.status_code == 200
    assert "通过" in accepted.text
    assert "满足全部不等式" in accepted.text

    rejected = client.post(
        "/verify",
        files={"file": ("review.json", payload, "application/json")},
        data={"first": "10.1", "second": "70"},
    )
    assert rejected.status_code == 200
    assert "拒绝" in rejected.text
    assert "违反" in rejected.text


def test_verify_accepts_pasted_package_without_original_batch(client):
    batch_id = batch_id_from(upload(client, PINNED_BATCH))
    package_response = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "1", "sample_a": "A", "sample_b": "B"},
        follow_redirects=False,
    )
    text = package_response.text
    response = client.post(
        "/verify",
        data={"file": "", "pasted": text, "first": "10", "second": "70"},
    )
    assert response.status_code == 200
    assert "通过" in response.text


def test_verify_rejects_readings_outside_zero_to_hundred(client):
    batch_id = batch_id_from(upload(client, PINNED_BATCH))
    payload = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "1", "sample_a": "A", "sample_b": "B"},
        follow_redirects=False,
    ).content
    response = client.post(
        "/verify",
        files={"file": ("review.json", payload, "application/json")},
        data={"first": "101", "second": "70"},
    )
    assert response.status_code == 400
    assert "0%～100%" in response.text


def test_verify_without_package_shows_error(client):
    response = client.post(
        "/verify", data={"first": "10", "second": "70"}
    )
    assert response.status_code == 400
    assert "请上传复核 JSON" in response.text


def test_verify_rejects_package_with_unknown_field(client):
    bad = {
        "batch_no": 1,
        "revision": 1,
        "samples": ["A", "B"],
        "inequalities": [{"a": "1", "b": "0", "c": "100"}],
        "extra": 1,
    }
    response = client.post(
        "/verify",
        files={"file": ("x.json", json.dumps(bad).encode(), "application/json")},
        data={"first": "1", "second": "2"},
    )
    assert response.status_code == 400
    assert "未知字段" in response.text


def test_verify_get_page_renders(client):
    response = client.get("/verify")
    assert response.status_code == 200
    assert "外协复核核验" in response.text


def test_shipped_example_package_verifies(client):
    example = (
        Path(__file__).resolve().parent.parent
        / "app"
        / "static"
        / "examples"
        / "review_package_example.json"
    )
    payload = example.read_bytes()
    accepted = client.post(
        "/verify",
        files={"file": ("review.json", payload, "application/json")},
        data={"first": "3.9", "second": "4.0"},
    )
    assert "通过" in accepted.text
    rejected = client.post(
        "/verify",
        files={"file": ("review.json", payload, "application/json")},
        data={"first": "5", "second": "5"},
    )
    assert "拒绝" in rejected.text


def test_infeasible_batch_refuses_package_with_reason(client):
    batch_id = batch_id_from(upload(client, CONFLICT))
    # CONFLICT has two samples (A, B) but is contradictory.
    response = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "1", "sample_a": "A", "sample_b": "B"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "notice=package_rejected" in response.headers["location"]
    page = client.get(response.headers["location"])
    assert "复核出包已拒绝" in page.text
    assert "互相矛盾" in page.text


def test_stale_revision_blocks_package(client):
    batch_id = batch_id_from(upload(client, CONFLICT))
    response = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "99", "sample_a": "A", "sample_b": "B"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "notice=stale" in response.headers["location"]


def test_invalid_sample_choice_blocks_package(client):
    batch_id = batch_id_from(upload(client, PINNED_BATCH))
    response = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "1", "sample_a": "A", "sample_b": "NOPE"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "notice=package_invalid" in response.headers["location"]


def test_same_sample_choice_blocks_package(client):
    batch_id = batch_id_from(upload(client, PINNED_BATCH))
    response = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "1", "sample_a": "A", "sample_b": "A"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "notice=package_invalid" in response.headers["location"]


def test_package_forced_off_for_batch_with_more_than_four_samples(client):
    five_samples = {
        "batch_name": "五原样批",
        "reports": [
            {
                "report_no": "F1",
                "samples": [
                    {"sample_no": f"S{i}", "grams": 10} for i in range(1, 6)
                ],
                "interval": [10, 90],
            }
        ],
    }
    batch_id = batch_id_from(upload(client, five_samples))
    page = client.get(f"/batches/{batch_id}")
    assert "不能出包" in page.text
    # The hidden panel must not be bypassable by a direct POST.
    response = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "1", "sample_a": "S1", "sample_b": "S2"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "notice=package_invalid" in response.headers["location"]


def test_package_carries_current_revision_after_retest_correction(client):
    # CONFLICT becomes compatible after retesting R1; the revision is bumped
    # and the next package must carry the new revision number.
    batch_id = batch_id_from(upload(client, CONFLICT))
    retest = client.post(
        f"/batches/{batch_id}/retests",
        data={"revision": "1", "report_no": "R1", "low": "80", "high": "90"},
        follow_redirects=False,
    )
    assert retest.status_code == 303
    payload = client.post(
        f"/batches/{batch_id}/packages",
        data={"revision": "2", "sample_a": "A", "sample_b": "B"},
        follow_redirects=False,
    ).json()
    assert payload["revision"] == 2
    assert payload["batch_no"] == batch_id
    # The package built from corrected data must accept the corrected reading.
    verify = client.post(
        "/verify",
        files={"file": ("p.json", json.dumps(payload).encode(), "application/json")},
        data={"first": "85", "second": "85"},
    )
    assert verify.status_code == 200
    assert "通过" in verify.text
