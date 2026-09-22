"""外部 CSV 数据导入实现。

所有数据级错误聚合为一次 :class:`DataImportError` 抛出，并精确定位到
``文件名:行号``（表头为物理第 1 行）。
"""

import csv
import hashlib
import io
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.turbine import Turbine
from ..core.wind_resource import WindResource, WindSector, _gamma_lanczos

# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


@dataclass
class ImportProblem:
    """单条导入问题，可定位到文件与数据行。"""

    filepath: str
    message: str
    line: Optional[int] = None
    column: Optional[str] = None

    def __str__(self) -> str:
        loc = self.filepath
        if self.line is not None:
            loc += f":{self.line}"
        if self.column:
            loc += f"（列 {self.column}）"
        return f"{loc}: {self.message}"


class DataImportError(ValueError):
    """CSV 导入错误，可包含多条行级问题。"""

    def __init__(self, problems: list[ImportProblem] | ImportProblem) -> None:
        if isinstance(problems, ImportProblem):
            problems = [problems]
        self.problems = problems
        super().__init__("\n".join(str(p) for p in problems))


# ---------------------------------------------------------------------------
# 路径与指纹工具
# ---------------------------------------------------------------------------


def normalize_relative_path(path: str, base_dir: Optional[str] = None) -> str:
    """按配置文件所在目录解析相对路径。

    绝对路径原样返回；相对路径基于 ``base_dir``（通常是配置文件目录）解析；
    ``base_dir`` 为 None 时基于当前工作目录。
    """
    if os.path.isabs(path):
        return os.path.normpath(path)
    if base_dir is None:
        base_dir = os.getcwd()
    return os.path.normpath(os.path.join(base_dir, path))


def _read_bytes(path: str) -> bytes:
    if not os.path.isfile(path):
        raise DataImportError(ImportProblem(path, "文件不存在或不是普通文件"))
    with open(path, "rb") as f:
        return f.read()


def _fingerprint(raw: bytes) -> str:
    """对文件原始字节计算 SHA-256，作为可重复识别的数据指纹。"""
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# 表头与数值解析
# ---------------------------------------------------------------------------

_SECTOR_ALIASES = {
    "start": (
        "start", "start_angle", "direction_start", "angle_start", "from",
        "dir_start", "boundary_start", "起始角", "起始角度", "风向起始",
        "风向起始角", "起始方向", "下限", "开始角度",
    ),
    "end": (
        "end", "end_angle", "direction_end", "angle_end", "to",
        "dir_end", "boundary_end", "结束角", "结束角度", "风向结束",
        "风向结束角", "结束方向", "上限", "终止角度",
    ),
    "center": (
        "center", "center_angle", "direction", "direction_center",
        "mid", "angle", "sector_center", "中心角", "中心角度", "扇区中心",
        "风向", "风向中心", "中心", "代表风向",
    ),
    "width": (
        "width", "sector_width", "direction_width", "angle_width",
        "sector", "扇区宽度", "宽度", "角宽", "角度宽度", "扇区角宽",
    ),
    "frequency": (
        "frequency", "freq", "probability", "prob", "p", "f",
        "频率", "风频", "风向频率", "扇区频率", "频率占比", "频率(%)",
        "频率（%）", "风频(%)", "风频（%）", "百分比",
    ),
    "mean_speed": (
        "mean_speed", "avg_speed", "average_speed", "vmean", "v_mean",
        "mean_wind_speed", "speed", "mean", "average", "ws_mean",
        "平均风速", "均值风速", "风速均值", "平均风", "风速", "均值",
        "平均风速(m/s)", "平均风速（m/s）",
    ),
    "weibull_k": (
        "k", "weibull_k", "k_value", "shape", "shape_factor",
        "weibull_shape", "k参数", "威布尔k", "形状参数", "形状因子",
    ),
    "weibull_c": (
        "c", "weibull_c", "c_value", "scale", "scale_factor",
        "weibull_scale", "c参数", "威布尔c", "尺度参数", "尺度因子",
    ),
}

_CURVE_ALIASES = {
    "wind_speed": (
        "wind_speed", "speed", "v", "ws", "wind", "windspeed", "u",
        "风速", "风速(m/s)", "风速（m/s）", "来流风速", "风速度",
    ),
    "power": (
        "power", "p", "power_kw", "power_mw", "kw", "mw", "output",
        "power_output", "active_power", "功率", "功率(kw)", "功率（kw）",
        "功率(mw)", "功率（mw）", "输出功率", "有功功率",
    ),
    "ct": (
        "ct", "ct_curve", "c_t", "thrust", "thrust_coefficient",
        "thrust_coeff", "thrust_curve", "推力系数", "推力系数曲线",
        "推力", "ct值",
    ),
}

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "")


def _header_variants(name: str) -> list[str]:
    """生成表头的候选归一化形式：原样、去除百分号与括号单位后的形式。"""
    norm = _normalize_header(name)
    variants = [norm]
    stripped = norm.replace("（%）", "").replace("(%)", "").rstrip("%")
    stripped = re.sub(r"[（(][^）)]*[）)]", "", stripped)
    if stripped and stripped != norm:
        variants.append(stripped)
    return variants


def _resolve_columns(
    header: list[str],
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """把表头解析为 {规范字段: 原始表头名}。"""
    normalized: dict[str, str] = {}
    for h in header:
        for variant in _header_variants(h):
            normalized.setdefault(variant, h)
    resolved: dict[str, str] = {}
    for canonical, names in aliases.items():
        for alias in names:
            key = _normalize_header(alias)
            if key in normalized:
                resolved[canonical] = normalized[key]
                break
    return resolved


def _read_table(
    path: str,
    raw: bytes,
) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """读取 CSV，返回 (表头, [(物理行号, 单元格列表), ...])。

    支持 UTF-8 BOM、逗号/分号/Tab/竖线分隔、空行与 ``#`` 注释行。
    """
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DataImportError(
            ImportProblem(path, "文件不是有效的 UTF-8 文本编码")
        )

    first_line = ""
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            first_line = line
            break

    delimiter = ","
    if first_line:
        candidates = {d: first_line.count(d) for d in (",", ";", "\t", "|")}
        best = max(candidates, key=candidates.get)
        if candidates[best] > 0:
            delimiter = best

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows: list[tuple[int, list[str]]] = []
    header: Optional[list[str]] = None
    for row in reader:
        line_no = reader.line_num
        stripped = [c.strip() for c in row]
        if not any(stripped):
            continue
        if stripped[0].startswith("#"):
            continue
        if header is None:
            header = stripped
            continue
        rows.append((line_no, stripped))

    if header is None:
        raise DataImportError(ImportProblem(path, "CSV 文件为空或缺少表头"))

    return header, rows


def _parse_number(
    raw_value: str,
    filepath: str,
    line: int,
    column: str,
) -> tuple[float, str, bool]:
    """解析数值单元格，返回 (数值, 单位后缀, 是否带百分号)。"""
    token = raw_value.strip().lower().replace(" ", "")
    if not token:
        raise DataImportError(
            ImportProblem(filepath, "单元格为空", line, column)
        )
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    match = _NUMBER_RE.match(token)
    if not match:
        raise DataImportError(
            ImportProblem(filepath, f"无法解析数值 '{raw_value}'", line, column)
        )
    suffix = token[match.end():].strip()
    value = float(match.group())
    if not np.isfinite(value):
        raise DataImportError(
            ImportProblem(filepath, "数值必须有限", line, column)
        )
    return value, suffix, is_percent


def _parse_angle(
    raw_value: str,
    filepath: str,
    line: int,
    column: str,
) -> float:
    value, suffix, _ = _parse_number(raw_value, filepath, line, column)
    if suffix not in ("", "deg", "degree", "°", "度"):
        raise DataImportError(
            ImportProblem(
                filepath,
                f"角度单元格带未知单位 '{raw_value}'（应为度）",
                line,
                column,
            )
        )
    return float(value)


def _try_parse_number(
    raw_value: str,
    filepath: str,
    line: int,
    column: str,
    problems: list[ImportProblem],
) -> Optional[tuple[float, str, bool]]:
    """容错数值解析：失败时把行级问题加入 problems 并返回 None。"""
    try:
        return _parse_number(raw_value, filepath, line, column)
    except DataImportError as exc:
        problems.extend(exc.problems)
        return None


def _try_parse_angle(
    raw_value: str,
    filepath: str,
    line: int,
    column: str,
    problems: list[ImportProblem],
) -> Optional[float]:
    """容错角度解析：失败时把行级问题加入 problems 并返回 None。"""
    try:
        return _parse_angle(raw_value, filepath, line, column)
    except DataImportError as exc:
        problems.extend(exc.problems)
        return None


# ---------------------------------------------------------------------------
# 扇区几何校验：跨零、重叠、缺口
# ---------------------------------------------------------------------------

_ANGLE_EPS = 1e-6


def _sector_parts(sec: dict) -> list[tuple[float, float]]:
    """把扇区拆成 [0, 360) 上的一个或两个半开区间 [a, b)。"""
    if sec["crosses_zero"]:
        if sec["width"] >= 360.0:
            return [(0.0, 360.0)]
        return [(sec["start"], 360.0), (0.0, sec["end"])]
    return [(sec["start"], sec["end"])]


def _interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _validate_coverage(
    sectors: list[dict],
    filepath: str,
    problems: list[ImportProblem],
) -> None:
    """标记跨零扇区，校验整圈覆盖：扇区两两不重叠、无缺口。"""
    for sec in sectors:
        # 整圈单扇区（width=360）不算跨越；其余 start>end 即跨零
        sec["crosses_zero"] = (
            sec["width"] < 360.0 and sec["start"] > sec["end"] + _ANGLE_EPS
        )

    parts = [_sector_parts(s) for s in sectors]

    # 成对重叠检查
    for i in range(len(sectors)):
        for j in range(i + 1, len(sectors)):
            overlap = 0.0
            for a in parts[i]:
                for b in parts[j]:
                    overlap += _interval_overlap(a, b)
            if overlap > _ANGLE_EPS:
                problems.append(
                    ImportProblem(
                        filepath,
                        f"扇区与第 {sectors[j]['line']} 行扇区重叠，"
                        f"重叠角度宽度 {overlap:g}°",
                        sectors[i]["line"],
                    )
                )

    # 缺口检查：合并所有区间
    flat = sorted((a, b) for sec_parts in parts for a, b in sec_parts)
    if not flat:
        return
    merged: list[list[float]] = []
    for a, b in flat:
        if merged and a <= merged[-1][1] + _ANGLE_EPS:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    gaps: list[tuple[float, float]] = []
    if merged[0][0] > _ANGLE_EPS:
        gaps.append((0.0, merged[0][0]))
    for prev, nxt in zip(merged, merged[1:]):
        if nxt[0] - prev[1] > _ANGLE_EPS:
            gaps.append((prev[1], nxt[0]))
    if 360.0 - merged[-1][1] > _ANGLE_EPS:
        gaps.append((merged[-1][1], 360.0))

    for a, b in gaps:
        problems.append(
            ImportProblem(
                filepath,
                f"风向角度 [{a:g}°, {b:g}°) 未被任何扇区覆盖"
                f"（存在 {b - a:g}° 缺口）",
            )
        )


# ---------------------------------------------------------------------------
# Weibull 推导
# ---------------------------------------------------------------------------

_GAMMA_MIN = 0.8856008  # Gamma(x) 的全局最小值（x ≈ 1.4616）
_K_AT_GAMMA_MIN = 2.166  # 1 + 1/k = 1.4616 时的 k（下分支端点）


def _weibull_mean(k: float, c: float) -> float:
    return float(c * _gamma_lanczos(np.asarray(1.0 + 1.0 / k)))


def _weibull_c_from_mean(k: float, mean: float) -> float:
    return float(mean / _gamma_lanczos(np.asarray(1.0 + 1.0 / k)))


def _weibull_k_from_mean_c(mean: float, c: float) -> float:
    """由均值与 c 反推 k，取下分支 k ∈ (1, 2.166]（常见风电 k 范围）。"""
    ratio = mean / c
    if ratio >= 1.0:
        raise ValueError(
            f"平均风速 {mean:g} 必须小于 Weibull c 参数 {c:g}"
            "（均值 = c·Γ(1+1/k) < c）"
        )
    if ratio < _GAMMA_MIN - 1e-4:
        raise ValueError(
            f"平均风速/c = {ratio:.4f} 超出 Weibull 可行范围 [0.8856, 1)，"
            "无法由均值与 c 反推 k，请直接提供 k 参数"
        )

    def gamma_ratio(k: float) -> float:
        return float(_gamma_lanczos(np.asarray(1.0 + 1.0 / k)))

    lo, hi = 1.0 + 1e-6, _K_AT_GAMMA_MIN
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if gamma_ratio(mid) > ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 风资源扇区表导入
# ---------------------------------------------------------------------------

_FREQ_STRICT_TOL = 2e-3  # strict 策略下允许的频率闭合差（分数）


def load_wind_resource_csv(
    path: str,
    frequency_normalization: str = "strict",
    base_dir: Optional[str] = None,
) -> WindResource:
    """从测风塔扇区表 CSV 加载风资源。

    Parameters
    ----------
    path : str
        CSV 路径，相对路径按 ``base_dir`` 解析
    frequency_normalization : str
        频率归一化策略：

        - ``"strict"``（默认）：原始频率和必须为 1（容差 2e-3），闭合差
          校正到最大频率扇区；百分比列/单元格自动除以 100
        - ``"normalize"``：按比例缩放使总和为 1
    base_dir : str, optional
        相对路径解析基准目录（配置文件所在目录）

    Returns
    -------
    WindResource
        带 ``source`` 溯源信息的风资源对象
    """
    if frequency_normalization not in ("strict", "normalize"):
        raise DataImportError(
            ImportProblem(
                path,
                f"未知的频率归一化策略 '{frequency_normalization}'，"
                "可选 'strict' 或 'normalize'",
            )
        )

    resolved = normalize_relative_path(path, base_dir)
    raw = _read_bytes(resolved)
    fingerprint = _fingerprint(raw)

    header, rows = _read_table(resolved, raw)
    columns = _resolve_columns(header, _SECTOR_ALIASES)
    problems: list[ImportProblem] = []

    if "frequency" not in columns:
        problems.append(
            ImportProblem(resolved, f"缺少频率列，可用表头: {header}", 1)
        )
    if not any(k in columns for k in ("start", "end", "center", "width")):
        problems.append(
            ImportProblem(resolved, "缺少角度列：需要 起始角/结束角，或 中心角/扇区宽度", 1)
        )
    if not any(k in columns for k in ("mean_speed", "weibull_k", "weibull_c")):
        problems.append(
            ImportProblem(resolved, "缺少风速列：需要 平均风速，或 Weibull k/c 参数", 1)
        )
    if problems:
        raise DataImportError(problems)

    freq_col = columns["frequency"]
    header_percent = "%" in freq_col

    parsed_sectors: list[dict] = []
    percent_cell_count = 0

    for line, cells in rows:
        row = dict(zip(header, cells))

        def cell(key: str) -> Optional[str]:
            col = columns.get(key)
            if col is None:
                return None
            value = row.get(col)
            return value if value is not None and value.strip() else None

        row_invalid = False

        # ---- 角度 -----------------------------------------------------------
        angle_values = {}
        for key in ("start", "end", "center", "width"):
            raw_angle = cell(key)
            if raw_angle is not None:
                parsed_angle = _try_parse_angle(
                    raw_angle, resolved, line, columns[key], problems
                )
                if parsed_angle is None:
                    row_invalid = True
                else:
                    angle_values[key] = parsed_angle

        sec = _build_sector_geometry(angle_values, resolved, line, problems)
        if sec is None or row_invalid:
            continue

        # ---- 频率（允许逐单元格使用 % 后缀）--------------------------------
        raw_freq = cell("frequency")
        parsed_freq = _try_parse_number(
            raw_freq, resolved, line, freq_col, problems
        )
        if parsed_freq is None:
            continue
        freq_value, freq_suffix, freq_pct = parsed_freq
        if freq_suffix:
            problems.append(
                ImportProblem(resolved, f"频率单元格带未知后缀 '{raw_freq}'", line, freq_col)
            )
        if freq_value < 0.0:
            problems.append(
                ImportProblem(resolved, f"频率不能为负（当前 {freq_value:g}）", line, freq_col)
            )
        if freq_pct:
            percent_cell_count += 1

        # ---- 风速 / Weibull -------------------------------------------------
        values: dict[str, float] = {}
        wind_parse_failed = False
        for key in ("mean_speed", "weibull_k", "weibull_c"):
            raw_v = cell(key)
            if raw_v is not None:
                parsed_v = _try_parse_number(
                    raw_v, resolved, line, columns[key], problems
                )
                if parsed_v is None:
                    wind_parse_failed = True
                    continue
                num, suffix, _ = parsed_v
                allowed = ("", "m/s", "ms", "米/秒") if key != "weibull_k" else ("",)
                if suffix not in allowed:
                    problems.append(
                        ImportProblem(
                            resolved,
                            f"{columns[key]} 列带未知单位 '{raw_v}'"
                            "（风速应为 m/s，k 无单位）",
                            line,
                            columns[key],
                        )
                    )
                values[key] = num

        if wind_parse_failed:
            continue

        mean_speed, weibull_k, weibull_c, derived = _resolve_weibull(
            values.get("mean_speed"),
            values.get("weibull_k"),
            values.get("weibull_c"),
            resolved,
            line,
            problems,
        )

        sec.update(
            line=line,
            raw_frequency=freq_value,
            freq_percent=freq_pct or header_percent,
            mean_speed=mean_speed,
            weibull_k=weibull_k,
            weibull_c=weibull_c,
            derived=derived,
        )
        parsed_sectors.append(sec)

    if not parsed_sectors:
        problems.append(ImportProblem(resolved, "没有任何有效数据行"))
        raise DataImportError(problems)

    # ---- 几何覆盖校验 ------------------------------------------------------
    _validate_coverage(parsed_sectors, resolved, problems)

    # 原始频率合计（保留 CSV 中的数值量级，用于溯源展示）
    raw_freq_sum = float(sum(s["raw_frequency"] for s in parsed_sectors))

    # ---- 频率归一化（逐单元格 % 或整列 %）---------------------------------
    fr_freqs = np.array(
        [
            s["raw_frequency"] * (0.01 if s["freq_percent"] else 1.0)
            for s in parsed_sectors
        ]
    )
    frequency_input = (
        "percent" if (header_percent or percent_cell_count > 0) else "fraction"
    )

    closure = 0.0
    normalization_note = "原始频率已归一"
    total = float(fr_freqs.sum())
    if total <= 0.0:
        problems.append(ImportProblem(resolved, "频率之和为 0，无法归一化"))
    elif frequency_normalization == "strict":
        if abs(total - 1.0) > _FREQ_STRICT_TOL:
            problems.append(
                ImportProblem(
                    resolved,
                    f"频率之和为 {total:.6f}（原始值合计 {raw_freq_sum:g}），"
                    f"超出 strict 容差 ±{_FREQ_STRICT_TOL:g}；"
                    "如确认按比例归一化，请在配置中设置 "
                    '"frequency_normalization": "normalize"',
                )
            )
        else:
            closure = 1.0 - total
            if abs(closure) > 0.0:
                idx = int(np.argmax(fr_freqs))
                fr_freqs[idx] += closure
                normalization_note = (
                    f"闭合差 {closure:+.6f} 已校正到第 "
                    f"{parsed_sectors[idx]['line']} 行（最大频率扇区）"
                )
    else:  # normalize
        fr_freqs = fr_freqs / total
        normalization_note = f"按比例归一化到 1.0（原始合计 {total:g}）"

    if problems:
        raise DataImportError(problems)

    # ---- 构建 WindSector ---------------------------------------------------
    sectors = []
    for sec, freq in zip(parsed_sectors, fr_freqs):
        sectors.append(
            WindSector(
                direction_center=sec["center"],
                direction_width=sec["width"],
                frequency=float(freq),
                mean_speed=sec["mean_speed"],
                weibull_k=sec["weibull_k"],
                weibull_c=sec["weibull_c"],
                direction_start=sec["start"],
                direction_end=sec["end"],
            )
        )

    source = {
        "kind": "wind_resource",
        "path": path,
        "resolved_path": resolved,
        "fingerprint": fingerprint,
        "byte_size": len(raw),
        "n_sectors": len(sectors),
        "frequency_input": frequency_input,
        "raw_frequency_sum": raw_freq_sum,
        "normalized_frequency_sum": float(fr_freqs.sum()),
        "frequency_normalization": frequency_normalization,
        "normalization_note": normalization_note,
        "closure_correction": closure,
        "columns": {k: columns[k] for k in columns},
        "unequal_widths": bool(np.ptp([s.direction_width for s in sectors]) > 1e-9),
        "sector_widths": [float(s.direction_width) for s in sectors],
        "crosses_zero_sectors": [
            s["line"] for s in parsed_sectors if s.get("crosses_zero")
        ],
        "derived_c_rows": [s["line"] for s in parsed_sectors if s.get("derived") == "c"],
        "derived_mean_rows": [s["line"] for s in parsed_sectors if s.get("derived") == "mean"],
        "default_k_rows": [s["line"] for s in parsed_sectors if s.get("derived") == "default_k"],
        "solved_k_rows": [s["line"] for s in parsed_sectors if s.get("derived") == "k"],
    }

    return WindResource(sectors, source=source)


def _norm_start(angle: float) -> float:
    """起始角归一化到 [0, 360)。"""
    return angle % 360.0


def _norm_end(angle: float) -> float:
    """结束角归一化到 (0, 360]；0 与 360 统一表示正北（顺时针到达）。"""
    e = angle % 360.0
    return 360.0 if e == 0.0 else e


def _clockwise_span(start: float, end: float) -> float:
    """从 start 顺时针到 end 的角度跨度，end ∈ (0,360]，start ∈ [0,360)。"""
    return end - start if end >= start else end + 360.0 - start


def _build_sector_geometry(
    angle_values: dict[str, float],
    filepath: str,
    line: int,
    problems: list[ImportProblem],
) -> Optional[dict]:
    """根据行内可用角度字段计算 start/end/width/center。"""
    start = angle_values.get("start")
    end = angle_values.get("end")
    center = angle_values.get("center")
    width = angle_values.get("width")

    try:
        if start is not None and end is not None:
            s = _norm_start(start)
            e = _norm_end(end)
            if width is not None:
                if not (0.0 < width <= 360.0):
                    raise ValueError(f"扇区宽度必须在 (0, 360] 范围内，当前 {width:g}°")
                w = float(width)
                inferred_e = s + w
                inferred_e = inferred_e if inferred_e <= 360.0 else inferred_e % 360.0
                if abs(inferred_e - e) > 1e-6:
                    raise ValueError(
                        f"起始角 {start:g}°、结束角 {end:g}° 与宽度 {width:g}° 不一致"
                    )
            else:
                w = _clockwise_span(s, e)
                if w <= 0.0:
                    raise ValueError(
                        f"起始角 {start:g}° 与结束角 {end:g}° 形成的扇区宽度为 0"
                    )
            c = (s + w / 2.0) % 360.0
            final_e = e if s + w > 360.0 else s + w
            result = {"start": s, "end": final_e, "width": w, "center": c}
        elif start is not None and width is not None:
            if not (0.0 < width <= 360.0):
                raise ValueError(f"扇区宽度必须在 (0, 360] 范围内，当前 {width:g}°")
            s = _norm_start(start)
            e = s + width
            e = e if e <= 360.0 else e % 360.0
            result = {
                "start": s,
                "end": e,
                "width": float(width),
                "center": (s + width / 2.0) % 360.0,
            }
        elif end is not None and width is not None:
            if not (0.0 < width <= 360.0):
                raise ValueError(f"扇区宽度必须在 (0, 360] 范围内，当前 {width:g}°")
            e = _norm_end(end)
            s = (e - width) % 360.0
            result = {
                "start": s,
                "end": e,
                "width": float(width),
                "center": (s + width / 2.0) % 360.0,
            }
        elif center is not None and width is not None:
            if not (0.0 < width <= 360.0):
                raise ValueError(f"扇区宽度必须在 (0, 360] 范围内，当前 {width:g}°")
            c = center % 360.0
            s = _norm_start(c - width / 2.0)
            e = s + width
            e = e if e <= 360.0 else e % 360.0
            result = {
                "start": s,
                "end": e,
                "width": float(width),
                "center": c,
            }
        else:
            raise ValueError(
                "角度信息不足：需要 起始角+结束角、起始角+宽度、"
                "结束角+宽度 或 中心角+宽度"
            )

        if center is not None and abs(result["center"] - center % 360.0) > 1e-6:
            raise ValueError(
                f"中心角 {center:g}° 与起止角度推导的中心 "
                f"{result['center']:g}° 不一致"
            )
    except ValueError as exc:
        problems.append(ImportProblem(filepath, str(exc), line))
        return None

    return result


def _resolve_weibull(
    mean_v: Optional[float],
    k_v: Optional[float],
    c_v: Optional[float],
    filepath: str,
    line: int,
    problems: list[ImportProblem],
) -> tuple[float, float, float, Optional[str]]:
    """根据行内提供的均值/k/c 补全 Weibull 参数。

    返回 (mean, k, c, derived 标记)。
    """
    if k_v is not None and k_v <= 0.0:
        problems.append(
            ImportProblem(filepath, f"Weibull k 必须为正（当前 {k_v:g}）", line)
        )
    if c_v is not None and c_v <= 0.0:
        problems.append(
            ImportProblem(filepath, f"Weibull c 必须为正（当前 {c_v:g}）", line)
        )
    if mean_v is not None and mean_v <= 0.0:
        problems.append(
            ImportProblem(filepath, f"平均风速必须为正（当前 {mean_v:g}）", line)
        )

    if mean_v is not None and k_v is not None and c_v is not None:
        check = _weibull_mean(k_v, c_v)
        if abs(check - mean_v) > max(0.05 * mean_v, 0.1):
            problems.append(
                ImportProblem(
                    filepath,
                    f"平均风速 {mean_v:g} m/s 与 k={k_v:g}、c={c_v:g} "
                    f"推导的均值 {check:.2f} m/s 不一致（偏差超过 5%），"
                    "请检查单位或列错位",
                    line,
                )
            )
        return mean_v, k_v, c_v, None

    if mean_v is not None and k_v is not None:
        return mean_v, k_v, _weibull_c_from_mean(k_v, mean_v), "c"

    if k_v is not None and c_v is not None:
        return _weibull_mean(k_v, c_v), k_v, c_v, "mean"

    if mean_v is not None and c_v is not None:
        try:
            k_solved = _weibull_k_from_mean_c(mean_v, c_v)
            return mean_v, k_solved, c_v, "k"
        except ValueError as exc:
            problems.append(ImportProblem(filepath, str(exc), line))
            return mean_v, 2.0, c_v, "k"

    if mean_v is not None:
        # 仅有均值：采用风电行业常用默认 k=2.0
        default_k = 2.0
        return mean_v, default_k, _weibull_c_from_mean(default_k, mean_v), "default_k"

    if k_v is not None:
        problems.append(
            ImportProblem(
                filepath,
                "仅提供 Weibull k 无法确定分布，还需要平均风速或 c 参数",
                line,
            )
        )
        return float("nan"), k_v, float("nan"), None

    problems.append(
        ImportProblem(
            filepath,
            "缺少风速信息：需要 平均风速，或 Weibull k/c，或 平均风速+k",
            line,
        )
    )
    return float("nan"), 2.0, float("nan"), None


# ---------------------------------------------------------------------------
# 机组功率 / 推力曲线导入
# ---------------------------------------------------------------------------


def _speed_from_value(value: float, suffix: str, header: str) -> float:
    token = suffix or _normalize_header(header)
    if "km/h" in token or "kph" in token or "kmh" in token:
        return value / 3.6
    return value


def _load_curve_file(
    path: str,
    base_dir: Optional[str],
    needed: tuple[str, ...],
    role: str,
) -> tuple[str, bytes, dict[str, str], list[tuple[int, dict[str, str]]]]:
    """读取曲线 CSV，返回 (绝对路径, 原始字节, 列映射, [(行号, {字段: 原始文本})])。"""
    resolved = normalize_relative_path(path, base_dir)
    raw = _read_bytes(resolved)
    header, rows = _read_table(resolved, raw)
    columns = _resolve_columns(header, _CURVE_ALIASES)

    problems = []
    for key in needed:
        if key not in columns:
            problems.append(
                ImportProblem(resolved, f"{role}文件缺少必要列，可用表头: {header}", 1, columns.get(key))
            )
    if problems:
        raise DataImportError(problems)

    parsed = []
    for line, cells in rows:
        row = dict(zip(header, cells))
        record = {}
        missing = []
        for key in columns:
            raw_cell = row.get(columns[key], "")
            if not raw_cell.strip():
                if key in needed:
                    missing.append(columns[key])
                continue
            record[key] = raw_cell
        if missing:
            problems.append(
                ImportProblem(resolved, f"必要列为空: {', '.join(missing)}", line)
            )
        elif record:
            parsed.append((line, record))

    if problems:
        raise DataImportError(problems)
    return resolved, raw, columns, parsed


def _validate_speed_series(
    speeds: list[tuple[int, float]], filepath: str, column: str
) -> list[ImportProblem]:
    """校验风速非负、有限、严格单调递增。"""
    problems: list[ImportProblem] = []
    prev_v = -np.inf
    prev_line: Optional[int] = None
    for line, v in speeds:
        if not np.isfinite(v) or v < 0.0:
            problems.append(
                ImportProblem(filepath, f"风速必须为非负有限数值（当前 {v:g}）", line, column)
            )
            continue
        if v <= prev_v:
            problems.append(
                ImportProblem(
                    filepath,
                    f"风速必须严格单调递增：{v:g} m/s 不大于前一数据点 "
                    f"({prev_v:g} m/s，第 {prev_line} 行)",
                    line,
                    column,
                )
            )
        prev_v, prev_line = v, line
    return problems


def load_custom_turbine(
    spec: dict,
    base_dir: Optional[str] = None,
) -> Turbine:
    """从厂商 CSV 加载自定义机组。

    spec 支持的键：

    - ``name``：机型名称（必需）
    - ``hub_height`` / ``rotor_diameter``：轮毂高度 / 叶轮直径 m（必需）
    - ``curve_csv`` 或 ``path``：单文件，含 风速/功率(/推力系数) 列
    - ``power_curve_csv`` + ``thrust_curve_csv``：功率与推力曲线分文件
    - ``power_unit``：``"auto"``（默认，按表头/后缀识别，否则按 kW）、``"kw"``、``"mw"``
    - ``thrust_coefficient``：无推力曲线时的恒定 Ct
    """
    problems: list[ImportProblem] = []
    config_label = "<自定义机组配置>"

    name = str(spec.get("name", "")).strip()
    if not name:
        problems.append(ImportProblem(config_label, "custom_turbine.name 为必填项"))

    hub_height = spec.get("hub_height")
    rotor_diameter = spec.get("rotor_diameter")
    if not isinstance(hub_height, (int, float)) or hub_height <= 0:
        problems.append(
            ImportProblem(config_label, "custom_turbine.hub_height 必须为正数（米）")
        )
    if not isinstance(rotor_diameter, (int, float)) or rotor_diameter <= 0:
        problems.append(
            ImportProblem(config_label, "custom_turbine.rotor_diameter 必须为正数（米）")
        )

    configured_unit = str(spec.get("power_unit", "auto")).lower()
    if configured_unit not in ("auto", "kw", "mw"):
        problems.append(
            ImportProblem(config_label, f"未知功率单位配置 '{configured_unit}'，可选 auto/kw/mw")
        )

    combined_path = spec.get("curve_csv") or spec.get("path")
    power_path = spec.get("power_curve_csv")
    thrust_path = spec.get("thrust_curve_csv")

    file_entries: list[dict] = []
    power_records: list[tuple[int, dict]] = []
    thrust_records: list[tuple[int, dict]] = []
    power_columns: dict[str, str] = {}
    power_resolved = ""

    if combined_path:
        power_resolved, raw, power_columns, power_records = _load_curve_file(
            combined_path, base_dir, ("wind_speed", "power"), "机组曲线"
        )
        file_entries.append(
            {
                "role": "power_curve",
                "path": combined_path,
                "resolved_path": power_resolved,
                "fingerprint": _fingerprint(raw),
                "byte_size": len(raw),
                "n_rows": len(power_records),
            }
        )
        if "ct" in power_columns:
            thrust_records = [
                (line, rec) for line, rec in power_records if "ct" in rec
            ]
    else:
        if not power_path:
            problems.append(
                ImportProblem(
                    config_label,
                    "需要 curve_csv（单文件）或 power_curve_csv（功率曲线文件）",
                )
            )
        else:
            power_resolved, raw, power_columns, power_records = _load_curve_file(
                power_path, base_dir, ("wind_speed", "power"), "功率曲线"
            )
            file_entries.append(
                {
                    "role": "power_curve",
                    "path": power_path,
                    "resolved_path": power_resolved,
                    "fingerprint": _fingerprint(raw),
                    "byte_size": len(raw),
                    "n_rows": len(power_records),
                }
            )
        if thrust_path:
            t_resolved, t_raw, _, t_records = _load_curve_file(
                thrust_path, base_dir, ("wind_speed", "ct"), "推力系数曲线"
            )
            thrust_records = t_records
            file_entries.append(
                {
                    "role": "thrust_curve",
                    "path": thrust_path,
                    "resolved_path": t_resolved,
                    "fingerprint": _fingerprint(t_raw),
                    "byte_size": len(t_raw),
                    "n_rows": len(t_records),
                }
            )

    if problems:
        raise DataImportError(problems)

    thrust_file_entry = next(
        (e for e in file_entries if e["role"] == "thrust_curve"), None
    )
    ct_source_path = (
        thrust_file_entry["resolved_path"] if thrust_file_entry else power_resolved
    )

    # ---- 功率曲线：kW/MW 单位识别 ------------------------------------------
    speed_header = power_columns["wind_speed"]
    power_header = power_columns["power"]
    norm_power_header = _normalize_header(power_header)
    header_unit = "mw" if "mw" in norm_power_header else (
        "kw" if "kw" in norm_power_header else None
    )

    speeds_p: list[tuple[int, float]] = []
    power_points: list[tuple[float, float]] = []
    explicit_mw_cells = 0
    for line, rec in power_records:
        parsed_speed = _try_parse_number(
            rec["wind_speed"], power_resolved, line, speed_header, problems
        )
        parsed_power = _try_parse_number(
            rec["power"], power_resolved, line, power_header, problems
        )
        if parsed_speed is None or parsed_power is None:
            continue

        v_speed, s_speed, _ = parsed_speed
        speed = _speed_from_value(v_speed, s_speed, speed_header)
        speeds_p.append((line, speed))

        p_val, p_suffix, _ = parsed_power
        raw_power = rec["power"]
        if p_val < 0.0:
            problems.append(
                ImportProblem(power_resolved, f"功率不能为负（当前 {p_val:g}）", line, power_header)
            )

        cell_unit = None
        if p_suffix == "mw":
            cell_unit = "mw"
            explicit_mw_cells += 1
        elif p_suffix == "kw":
            cell_unit = "kw"
        elif p_suffix:
            problems.append(
                ImportProblem(
                    power_resolved,
                    f"功率单元格带未知单位 '{raw_power}'（支持 kW/MW）",
                    line,
                    power_header,
                )
            )

        if configured_unit in ("kw", "mw"):
            unit = configured_unit
        elif cell_unit is not None:
            unit = cell_unit
        elif header_unit is not None:
            unit = header_unit
        else:
            unit = "kw"

        power_points.append((speed, p_val * 1000.0 if unit == "mw" else p_val))

    problems.extend(_validate_speed_series(speeds_p, power_resolved, speed_header))
    if len(power_points) < 2:
        problems.append(ImportProblem(power_resolved, "功率曲线至少需要两个有效数据点"))

    if configured_unit in ("kw", "mw"):
        resolved_unit = configured_unit
    elif explicit_mw_cells > 0:
        resolved_unit = "mw"
    elif header_unit is not None:
        resolved_unit = header_unit
    else:
        resolved_unit = "kw"

    # ---- 推力曲线 / 恒定 Ct ------------------------------------------------
    thrust_points: list[tuple[float, float]] = []
    if thrust_records:
        speeds_t: list[tuple[int, float]] = []
        for line, rec in thrust_records:
            parsed_speed = _try_parse_number(
                rec["wind_speed"], ct_source_path, line, "wind_speed", problems
            )
            parsed_ct = _try_parse_number(
                rec["ct"], ct_source_path, line, "ct", problems
            )
            if parsed_speed is None or parsed_ct is None:
                continue

            v_speed, s_speed, _ = parsed_speed
            speed = _speed_from_value(v_speed, s_speed, speed_header)
            speeds_t.append((line, speed))

            ct_val, ct_suffix, ct_pct = parsed_ct
            if ct_pct:
                ct_val /= 100.0
            if ct_suffix:
                problems.append(
                    ImportProblem(
                        ct_source_path,
                        f"推力系数单元格带未知后缀 '{rec['ct']}'（Ct 无单位，可用 % 表示百分比）",
                        line,
                    )
                )
            if not (0.0 <= ct_val < 1.0):
                problems.append(
                    ImportProblem(
                        ct_source_path,
                        f"推力系数 Ct 必须在 [0, 1) 范围内（当前 {ct_val:g}）",
                        line,
                    )
                )
            thrust_points.append((speed, ct_val))

        problems.extend(_validate_speed_series(speeds_t, ct_source_path, "wind_speed"))
        if len(thrust_points) < 2:
            problems.append(
                ImportProblem(ct_source_path, "推力系数曲线至少需要两个有效数据点")
            )

    constant_ct = spec.get("thrust_coefficient")
    if constant_ct is not None:
        if not isinstance(constant_ct, (int, float)) or not (0.0 < constant_ct < 1.0):
            problems.append(
                ImportProblem(
                    config_label,
                    f"thrust_coefficient 必须在 (0, 1) 范围内（当前 {constant_ct}）",
                )
            )

    if not thrust_points and constant_ct is None:
        problems.append(
            ImportProblem(
                power_resolved,
                "未提供推力系数曲线，需要在 CSV 中给出 ct 列，"
                "或在配置 custom_turbine.thrust_coefficient 中指定恒定 Ct",
            )
        )

    if problems:
        raise DataImportError(problems)

    power_curve = np.asarray(power_points, dtype=np.float64)
    thrust_curve = np.asarray(thrust_points, dtype=np.float64) if thrust_points else None

    turbine = Turbine(
        name=name,
        hub_height=float(hub_height),
        rotor_diameter=float(rotor_diameter),
        thrust_coefficient=float(constant_ct) if thrust_curve is None else None,
        power_curve=power_curve,
        thrust_curve=thrust_curve,
    )

    source = {
        "kind": "turbine",
        "name": name,
        "files": file_entries,
        "power_unit": resolved_unit,
        "rated_power_kw": turbine.rated_power,
        "cut_in_speed": turbine.cut_in_speed,
        "rated_speed": turbine.rated_speed,
        "cut_out_speed": turbine.cut_out_speed,
        "power_speed_range": [float(power_curve[0, 0]), float(power_curve[-1, 0])],
        "thrust": "curve" if thrust_curve is not None else f"constant:{constant_ct:g}",
    }
    if thrust_curve is not None:
        source["thrust_speed_range"] = [
            float(thrust_curve[0, 0]),
            float(thrust_curve[-1, 0]),
        ]
    turbine.source = source
    return turbine
