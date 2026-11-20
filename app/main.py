"""FastAPI web application for mineral-powder blend moisture reconciliation."""

from __future__ import annotations

import math
import sqlite3
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import db
from .analysis import (
    MAX_MOISTURE,
    MIN_MOISTURE,
    AnalysisError,
    BatchAnalysis,
    analyze,
    minimal_correction_size,
    retest_rejection,
)
from .formatting import grams, interval, pct, prefill_high, prefill_low
from .parsing import MAX_FILES, MAX_REPORTS, UploadError, parse_upload

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
TEMPLATES.env.filters["pct"] = pct
TEMPLATES.env.filters["interval"] = interval
TEMPLATES.env.filters["grams"] = grams
TEMPLATES.env.filters["prefill_low"] = prefill_low
TEMPLATES.env.filters["prefill_high"] = prefill_high

NOTICE_TEXT = {
    "retest_saved": "复测区间已保存：换用后整批相容，结论已更新。",
    "stale": (
        "页面依据的数据已有更正，本次复测提交未保存。"
        "为避免覆盖他人的更正，请核对下方最新结果后重新提交。"
    ),
    "conflict": (
        "复测区间换入后整批仍不相容，本次复测提交未保存，保持现状。"
    ),
    "not_candidate": (
        "该报告已不能通过单独更正恢复相容（可能已被更正或整批已变化），"
        "本次复测提交未保存，保持现状。"
    ),
    "already_feasible": (
        "整批现在已经相容，无需再更正该报告，本次复测提交未保存。"
    ),
    "invalid": "复测区间无效，未保存：请填写 0%～100% 内且下限不大于上限的闭区间。",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    conn = db.connect()
    try:
        db.init_db(conn)
    finally:
        conn.close()
    yield


app = FastAPI(title="矿粉混样化验单核对", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def get_conn() -> AsyncIterator[sqlite3.Connection]:
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]


def render(
    request: Request,
    template: str,
    context: dict,
    status_code: int = 200,
) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request, template, context, status_code=status_code
    )


def redirect_to_batch(batch_id: int, **params: object) -> RedirectResponse:
    url = f"/batches/{batch_id}"
    clean = {k: v for k, v in params.items() if v is not None}
    if clean:
        url += "?" + urllib.parse.urlencode(clean)
    return RedirectResponse(url, status_code=303)


def submitted_range(request: Request) -> dict[str, str] | None:
    """Echo the submitted retest interval back on the form after a rejection."""
    low = request.query_params.get("low")
    high = request.query_params.get("high")
    if low is None or high is None:
        return None
    code = request.query_params.get("notice")
    if code == "retest_saved":
        return None
    try:
        if not math.isfinite(float(low)) or not math.isfinite(float(high)):
            return None
    except ValueError:
        return None
    return {"low": low, "high": high}


def build_analysis(batch: db.StoredBatch) -> BatchAnalysis:
    try:
        return analyze(batch.to_domain())
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def notice_from_query(request: Request) -> dict | None:
    code = request.query_params.get("notice")
    if not code:
        return None
    text = NOTICE_TEXT.get(code)
    if text is None:
        return None
    return {
        "code": code,
        "text": text,
        "report_no": request.query_params.get("report"),
        "range": (
            f"{request.query_params.get('low')}% ～ {request.query_params.get('high')}%"
            if request.query_params.get("low") is not None
            and request.query_params.get("high") is not None
            else None
        ),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request, conn: Conn) -> HTMLResponse:
    return render(
        request,
        "index.html",
        {
            "batches": db.list_batches(conn),
            "max_reports": MAX_REPORTS,
            "pasted": "",
            "batch_name": "",
            "error": None,
        },
    )


@app.post("/batches", response_class=HTMLResponse)
async def create_batch(
    request: Request,
    conn: Conn,
    files: Annotated[list[UploadFile], File()] = [],
    pasted: Annotated[str, Form()] = "",
    batch_name: Annotated[str, Form()] = "",
) -> HTMLResponse:
    named = [upload for upload in files if upload.filename]
    if len(named) > MAX_FILES:
        return render(
            request,
            "index.html",
            {
                "batches": db.list_batches(conn),
                "max_reports": MAX_REPORTS,
                "pasted": pasted,
                "batch_name": batch_name,
                "error": f"单次最多上传 {MAX_FILES} 个文件",
            },
            status_code=400,
        )
    uploads: list[tuple[str, bytes]] = []
    for upload in named:
        uploads.append((upload.filename or "", await upload.read()))
    try:
        parsed = parse_upload(uploads, pasted, batch_name)
    except UploadError as exc:
        return render(
            request,
            "index.html",
            {
                "batches": db.list_batches(conn),
                "max_reports": MAX_REPORTS,
                "pasted": pasted,
                "batch_name": batch_name,
                "error": str(exc),
            },
            status_code=400,
        )
    batch_id = db.create_batch(conn, parsed.name, parsed.reports)
    return redirect_to_batch(batch_id)


@app.get("/batches/{batch_id}", response_class=HTMLResponse)
def batch_detail(
    request: Request, conn: Conn, batch_id: int
) -> HTMLResponse:
    batch = db.get_batch(conn, batch_id)
    if batch is None:
        return render(
            request,
            "not_found.html",
            {},
            status_code=404,
        )
    analysis = build_analysis(batch)
    candidate_names = {c.report_no for c in analysis.candidates}
    retests = db.list_retests(conn, batch_id)
    minimal_group = None
    if not analysis.feasible and not analysis.candidates:
        minimal_group = minimal_correction_size(batch.to_domain(), max_group=2)
    return render(
        request,
        "batch.html",
        {
            "batch": batch,
            "analysis": analysis,
            "candidate_names": candidate_names,
            "retests": retests,
            "notice": notice_from_query(request),
            "submitted": submitted_range(request),
            "minimal_group": minimal_group,
        },
    )


@app.post("/batches/{batch_id}/retests")
def submit_retest(
    conn: Conn,
    batch_id: int,
    revision: Annotated[int, Form()],
    report_no: Annotated[str, Form()],
    low: Annotated[str, Form()],
    high: Annotated[str, Form()],
) -> RedirectResponse:
    batch = db.get_batch(conn, batch_id)
    if batch is None:
        return RedirectResponse("/", status_code=303)
    report_no = report_no.strip()
    if batch.revision != revision:
        return redirect_to_batch(batch_id, notice="stale", report=report_no)
    try:
        low_value = float(low)
        high_value = float(high)
    except ValueError:
        return redirect_to_batch(
            batch_id, notice="invalid", report=report_no, low=low, high=high
        )
    if (
        not math.isfinite(low_value)
        or not math.isfinite(high_value)
        or low_value < MIN_MOISTURE
        or high_value > MAX_MOISTURE
        or low_value > high_value
    ):
        return redirect_to_batch(
            batch_id, notice="invalid", report=report_no, low=low, high=high
        )
    try:
        db.apply_retest(
            conn,
            batch_id,
            expected_revision=revision,
            report_no=report_no,
            new_low=low_value,
            new_high=high_value,
            verdict=retest_rejection,
        )
    except db.RetestRejected as exc:
        code = {
            "stale": "stale",
            "missing_batch": "stale",
            "unknown_report": "not_candidate",
            "already_feasible": "already_feasible",
            "not_candidate": "not_candidate",
            "conflict": "conflict",
        }.get(exc.code, "conflict")
        return redirect_to_batch(
            batch_id, notice=code, report=report_no, low=low, high=high
        )
    return redirect_to_batch(batch_id, notice="retest_saved", report=report_no)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
