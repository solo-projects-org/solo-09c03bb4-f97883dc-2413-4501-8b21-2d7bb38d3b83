"""Parsing and validation of uploaded JSON laboratory reports."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .analysis import MAX_MOISTURE, MIN_MOISTURE, Report, SampleUse

MAX_REPORTS = 24
MAX_SAMPLES = 12
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_FILES = 24
MAX_REPORT_NO_LEN = 64
MAX_SAMPLE_NO_LEN = 64

REPORT_NO_KEYS = ("report_no", "reportNo", "报告号", "化验单号")
SAMPLE_NO_KEYS = ("sample_no", "sampleNo", "原样编号", "样品编号")
GRAMS_KEYS = ("grams", "mass", "weight", "取用克数", "取用质量")
LOW_KEYS = ("low", "low_percent", "min", "下限", "含水率下限")
HIGH_KEYS = ("high", "high_percent", "max", "上限", "含水率上限")


class UploadError(ValueError):
    """User-facing upload problem."""


@dataclass(frozen=True)
class ParsedBatch:
    name: str
    reports: tuple[Report, ...]


def _get(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _text(value: Any, label: str, max_len: int) -> str:
    if not isinstance(value, str):
        raise UploadError(f"{label}必须是字符串")
    text = value.strip()
    if not text:
        raise UploadError(f"{label}不能为空")
    if len(text) > max_len:
        raise UploadError(f"{label}长度不能超过 {max_len} 个字符")
    return text


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UploadError(f"{label}必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise UploadError(f"{label}必须是有限数字")
    return number


def _interval(report: dict[str, Any], where: str) -> tuple[float, float]:
    interval = report.get("interval", report.get("moisture_interval", report.get("含水率区间")))
    if interval is not None:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise UploadError(f"{where}的含水率区间必须是包含两个数字的数组")
        low = _number(interval[0], f"{where}的含水率下限")
        high = _number(interval[1], f"{where}的含水率上限")
    else:
        moisture = report.get("moisture", report.get("含水率"))
        if isinstance(moisture, dict):
            low = _number(_get(moisture, LOW_KEYS), f"{where}的含水率下限")
            high = _number(_get(moisture, HIGH_KEYS), f"{where}的含水率上限")
        else:
            low_value = _get(report, LOW_KEYS)
            high_value = _get(report, HIGH_KEYS)
            if low_value is None or high_value is None:
                raise UploadError(
                    f"{where}缺少实测含水率区间（low/high 或 interval）"
                )
            low = _number(low_value, f"{where}的含水率下限")
            high = _number(high_value, f"{where}的含水率上限")
    if low < MIN_MOISTURE or high > MAX_MOISTURE:
        raise UploadError(f"{where}的含水率必须位于 0%～100% 之间")
    if low > high:
        raise UploadError(f"{where}的含水率下限不能大于上限")
    return low, high


def _samples(report: dict[str, Any], where: str) -> tuple[SampleUse, ...]:
    raw = report.get("samples", report.get("sample_grams", report.get("原样")))
    if raw is None:
        raw = report.get("原样取用克数")
    if isinstance(raw, dict):
        # Accept a flat mapping {"YP-01": 50, ...} for the sample masses.
        raw = [{"sample_no": k, "grams": v} for k, v in raw.items()]
    if not isinstance(raw, list) or not raw:
        raise UploadError(f"{where}至少需要一个原样的取用克数")
    uses: list[SampleUse] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise UploadError(f"{where}的第 {index} 个原样必须是对象")
        sample_no = _text(
            _get(item, SAMPLE_NO_KEYS), f"{where}第 {index} 个原样的原样编号", MAX_SAMPLE_NO_LEN
        )
        if sample_no in seen:
            raise UploadError(f"{where}中重复出现原样 {sample_no}")
        seen.add(sample_no)
        grams = _number(_get(item, GRAMS_KEYS), f"{where}原样 {sample_no} 的取用克数")
        if grams <= 0:
            raise UploadError(f"{where}原样 {sample_no} 的取用克数必须大于 0")
        uses.append(SampleUse(sample_no=sample_no, grams=grams))
    return tuple(uses)


def _reports_from_document(document: Any, source: str) -> tuple[list[dict], str | None]:
    batch_name: str | None = None
    if isinstance(document, list):
        return document, None
    if not isinstance(document, dict):
        raise UploadError(f"{source}必须是 JSON 对象或报告数组")
    if "reports" in document or "化验单" in document:
        raw = document.get("reports", document.get("化验单"))
        name = document.get("batch_name", document.get("批次名称"))
        if name is not None:
            batch_name = _text(name, f"{source}的批次名称", 128)
        if not isinstance(raw, list):
            raise UploadError(f"{source}的 reports 必须是数组")
        return raw, batch_name
    # A single report object.
    return [document], None


def parse_documents(
    documents: Iterable[tuple[str, Any]],
    default_name: str | None = None,
) -> ParsedBatch:
    raw_reports: list[dict] = []
    batch_name: str | None = None
    for source, document in documents:
        reports, name = _reports_from_document(document, source)
        if name and batch_name is None:
            batch_name = name
        for item in reports:
            if not isinstance(item, dict):
                raise UploadError(f"{source}中的每份报告都必须是对象")
            raw_reports.append(item)

    if not raw_reports:
        raise UploadError("没有找到任何化验单")
    if len(raw_reports) > MAX_REPORTS:
        raise UploadError(f"同批化验单最多 {MAX_REPORTS} 份，收到 {len(raw_reports)} 份")

    reports: list[Report] = []
    seen_reports: set[str] = set()
    for index, raw in enumerate(raw_reports, start=1):
        where = f"第 {index} 份报告"
        report_no = _text(
            _get(raw, REPORT_NO_KEYS), f"{where}的报告号", MAX_REPORT_NO_LEN
        )
        if report_no in seen_reports:
            raise UploadError(f"报告号 {report_no} 重复")
        seen_reports.add(report_no)
        low, high = _interval(raw, f"报告 {report_no}")
        samples = _samples(raw, f"报告 {report_no}")
        reports.append(Report(report_no=report_no, samples=samples, low=low, high=high))

    sample_nos = {use.sample_no for report in reports for use in report.samples}
    if len(sample_nos) > MAX_SAMPLES:
        raise UploadError(
            f"同批最多 {MAX_SAMPLES} 个原样，收到 {len(sample_nos)} 个"
        )
    for report in reports:
        total = sum(use.grams for use in report.samples)
        if not math.isfinite(total) or total <= 0:
            raise UploadError(f"报告 {report.report_no} 的取用总质量无效")

    name = (default_name or "").strip() or batch_name
    if not name:
        name = f"批次 {reports[0].report_no} 等 {len(reports)} 份"
    if len(name) > 128:
        name = name[:128]

    return ParsedBatch(name=name, reports=tuple(reports))


def load_json_bytes(data: bytes, filename: str) -> Any:
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(f"文件 {filename} 超过 2 MB 限制")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadError(f"文件 {filename} 必须是 UTF-8 编码的 JSON") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise UploadError(
            f"文件 {filename} 不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列）"
        ) from exc


def parse_upload(
    files: Sequence[tuple[str, bytes]],
    pasted_text: str | None,
    batch_name: str | None,
) -> ParsedBatch:
    if len(files) > MAX_FILES:
        raise UploadError(f"单次最多上传 {MAX_FILES} 个文件")
    total = sum(len(data) for _, data in files)
    if total > MAX_TOTAL_BYTES:
        raise UploadError("上传文件合计超过 10 MB 限制")
    documents: list[tuple[str, Any]] = []
    for filename, data in files:
        documents.append((filename, load_json_bytes(data, filename)))
    if pasted_text and pasted_text.strip():
        try:
            documents.append(("粘贴内容", json.loads(pasted_text)))
        except json.JSONDecodeError as exc:
            raise UploadError(
                f"粘贴的内容不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列）"
            ) from exc
    if not documents:
        raise UploadError("请上传 JSON 文件或粘贴 JSON 内容")
    return parse_documents(documents, default_name=batch_name)


def load_example(path: Path) -> ParsedBatch:
    document = json.loads(path.read_text(encoding="utf-8"))
    return parse_documents([(path.name, document)])
