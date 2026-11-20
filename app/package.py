"""Two-sample review packages: export, parse and verification of readings.

A package is the self-contained JSON handed to an external laboratory.  It
intentionally contains only the batch number, revision, the two selected
sample numbers and the exact linear inequalities (fractional coefficients
allowed) that their two moisture readings must satisfy.  The receiving side
needs neither the original batch nor database access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from .analysis import Report
from .projection import Inequality, Projection, project_batch

PACKAGE_VERSION = 1
BATCH_NO_KEY = "batch_no"
REVISION_KEY = "revision"
SAMPLES_KEY = "samples"
INEQUALITIES_KEY = "inequalities"
A_KEY = "a"
B_KEY = "b"
C_KEY = "c"
OPERATOR_KEY = "op"
PACKAGE_KEYS = {
    BATCH_NO_KEY,
    REVISION_KEY,
    SAMPLES_KEY,
    INEQUALITIES_KEY,
}
INEQUALITY_KEYS = {A_KEY, B_KEY, C_KEY}
OPERATOR_KEYS = {OPERATOR_KEY}


class PackageError(ValueError):
    """User-facing problem with an uploaded package or entered readings."""


@dataclass(frozen=True)
class ReviewPackage:
    batch_no: int
    revision: int
    samples: tuple[str, str]
    inequalities: tuple[Inequality, ...]

    @property
    def projection(self) -> Projection:
        return Projection(samples=self.samples, inequalities=self.inequalities)


@dataclass(frozen=True)
class CheckRow:
    index: int
    inequality: Inequality
    lhs: Fraction
    holds: bool


@dataclass(frozen=True)
class CheckResult:
    package: ReviewPackage
    readings: tuple[Fraction, Fraction]
    rows: tuple[CheckRow, ...]

    @property
    def accepted(self) -> bool:
        return all(row.holds for row in self.rows)


# ---------------------------------------------------------------------------
# Fraction helpers
# ---------------------------------------------------------------------------

def fraction_text(value: Fraction) -> str:
    """Serialize a fraction as ``"p/q"`` (integers without denominator)."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_fraction(value: Any, label: str) -> Fraction:
    """Accept ints, finite floats and exact decimal/fraction strings."""
    if isinstance(value, bool):
        raise PackageError(f"{label}必须是数字或分数字符串")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise PackageError(f"{label}必须是有限数字")
        return Fraction(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise PackageError(f"{label}不能为空")
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise PackageError(f"{label}不是合法的数字或分数") from exc
    raise PackageError(f"{label}必须是数字或分数字符串")


# ---------------------------------------------------------------------------
# Building and serializing
# ---------------------------------------------------------------------------

def build_package(
    reports: Sequence[Report],
    *,
    batch_no: int,
    revision: int,
    selected: tuple[str, str],
) -> ReviewPackage:
    projection = project_batch(reports, selected)
    return ReviewPackage(
        batch_no=int(batch_no),
        revision=int(revision),
        samples=projection.samples,
        inequalities=projection.inequalities,
    )


def package_document(package: ReviewPackage) -> dict[str, Any]:
    return {
        BATCH_NO_KEY: package.batch_no,
        REVISION_KEY: package.revision,
        SAMPLES_KEY: list(package.samples),
        INEQUALITIES_KEY: [
            {
                A_KEY: fraction_text(inequality.a),
                B_KEY: fraction_text(inequality.b),
                C_KEY: fraction_text(inequality.c),
            }
            for inequality in package.inequalities
        ],
    }


def package_json(package: ReviewPackage) -> str:
    return json.dumps(
        package_document(package), ensure_ascii=False, indent=2
    )


def package_filename(package: ReviewPackage) -> str:
    sample_a, sample_b = package.samples
    safe_a = _safe(sample_a)
    safe_b = _safe(sample_b)
    return (
        f"review_batch-{package.batch_no}_rev-{package.revision}"
        f"_{safe_a}_{safe_b}.json"
    )


def _safe(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return cleaned or "sample"


# ---------------------------------------------------------------------------
# Parsing an uploaded package
# ---------------------------------------------------------------------------

def parse_package(document: Any) -> ReviewPackage:
    if not isinstance(document, dict):
        raise PackageError("JSON 顶层必须是对象")
    extra = set(document) - PACKAGE_KEYS
    if extra:
        raise PackageError(f"JSON 含未知字段：{', '.join(sorted(extra))}")
    for key in PACKAGE_KEYS:
        if key not in document:
            raise PackageError(f"JSON 缺少字段 {key}")

    batch_no = document[BATCH_NO_KEY]
    revision = document[REVISION_KEY]
    if isinstance(batch_no, bool) or not isinstance(batch_no, int) or batch_no < 1:
        raise PackageError("batch_no 必须是正整数批次号")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise PackageError("revision 必须是正整数修订号")

    samples = document[SAMPLES_KEY]
    if not isinstance(samples, list) or len(samples) != 2:
        raise PackageError("samples 必须是恰好两个原样编号的数组")
    sample_nos: list[str] = []
    for index, raw in enumerate(samples, start=1):
        if not isinstance(raw, str) or not raw.strip():
            raise PackageError(f"第 {index} 个原样编号必须是非空字符串")
        sample_nos.append(raw.strip())
    if sample_nos[0] == sample_nos[1]:
        raise PackageError("两个原样编号不能相同")

    raw_inequalities = document[INEQUALITIES_KEY]
    if not isinstance(raw_inequalities, list) or not raw_inequalities:
        raise PackageError("inequalities 必须是非空数组")
    if len(raw_inequalities) > 200:
        raise PackageError("inequalities 数量超过 200 条")
    inequalities: list[Inequality] = []
    for index, raw in enumerate(raw_inequalities, start=1):
        if not isinstance(raw, dict):
            raise PackageError(f"第 {index} 条不等式必须是对象")
        extra = (set(raw) - INEQUALITY_KEYS) - OPERATOR_KEYS
        if extra:
            raise PackageError(
                f"第 {index} 条不等式含未知字段：{', '.join(sorted(extra))}"
            )
        if OPERATOR_KEY in raw and raw[OPERATOR_KEY] != "<=":
            raise PackageError(
                f"第 {index} 条不等式的 {OPERATOR_KEY} 只允许为 \"<=\""
            )
        for key in INEQUALITY_KEYS:
            if key not in raw:
                raise PackageError(f"第 {index} 条不等式缺少字段 {key}")
        a = parse_fraction(raw[A_KEY], f"第 {index} 条不等式的 {A_KEY}")
        b = parse_fraction(raw[B_KEY], f"第 {index} 条不等式的 {B_KEY}")
        c = parse_fraction(raw[C_KEY], f"第 {index} 条不等式的 {C_KEY}")
        if a == 0 and b == 0:
            raise PackageError(f"第 {index} 条不等式不能两个系数同时为 0")
        inequalities.append(Inequality(a=a, b=b, c=c))

    return ReviewPackage(
        batch_no=batch_no,
        revision=revision,
        samples=(sample_nos[0], sample_nos[1]),
        inequalities=tuple(inequalities),
    )


def load_package_bytes(data: bytes, filename: str) -> ReviewPackage:
    if len(data) > 2 * 1024 * 1024:
        raise PackageError(f"文件 {filename} 超过 2 MB 限制")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PackageError(f"文件 {filename} 必须是 UTF-8 编码的 JSON") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PackageError(
            f"文件 {filename} 不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列）"
        ) from exc
    return parse_package(document)


# ---------------------------------------------------------------------------
# Verifying two readings
# ---------------------------------------------------------------------------

def check_readings(
    package: ReviewPackage, first: str, second: str
) -> CheckResult:
    x1 = _reading_fraction(first, package.samples[0])
    x2 = _reading_fraction(second, package.samples[1])
    rows = tuple(
        CheckRow(
            index=index,
            inequality=inequality,
            lhs=inequality.lhs(x1, x2),
            holds=inequality.satisfied_by(x1, x2),
        )
        for index, inequality in enumerate(package.inequalities, start=1)
    )
    return CheckResult(
        package=package,
        readings=(x1, x2),
        rows=rows,
    )


def _reading_fraction(raw: str, sample_no: str) -> Fraction:
    if raw is None or not str(raw).strip():
        raise PackageError(f"请填写原样 {sample_no} 的含水率读数")
    try:
        value = Fraction(str(raw).strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise PackageError(
            f"原样 {sample_no} 的读数不是合法数字（如 3.15 或 17/4）"
        ) from exc
    if value < 0 or value > 100:
        raise PackageError(
            f"原样 {sample_no} 的含水率读数必须位于 0%～100% 闭区间内"
        )
    return value
