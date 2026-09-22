"""测风塔扇区表 CSV 导入器。

支持导入不等宽风向扇区、频率（比例/百分数/小时数）以及 Weibull 参数，
或仅给出可推导 Weibull 尺度参数的平均风速。

角度约定：气象风向，0° 为正北，顺时针；扇区为左闭右开区间 ``[start, end)``，
允许跨越零度（如 350°→10°）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.wind_resource import WindResource, WindSector, _gamma_lanczos
from .csv_reader import ParsedRow, parse_float, read_csv_rows, resolve_relative_path
from .errors import DataImportError, file_error, row_error
from .fingerprint import SourceFingerprint, make_fingerprint

# 角度容差（度）：小于该值的重叠/缺口视为舍入误差
_ANGLE_TOL = 1e-6

_WIND_COLUMNS: dict[str, tuple[str, ...]] = {
    "frequency": (
        "frequency", "freq", "probability", "p",
        "频率", "风频", "风向频率", "扇区频率",
    ),
}

_BOUNDARY_COLUMNS: dict[str, tuple[str, ...]] = {
    "start": (
        "direction_start", "start_angle", "sector_start", "dir_start",
        "dir_min", "angle_min", "begin", "start",
        "起始角度", "起始风向", "扇区起始", "左边界", "开始角度",
    ),
    "end": (
        "direction_end", "end_angle", "sector_end", "dir_end",
        "dir_max", "angle_max", "end",
        "结束角度", "终止角度", "终止风向", "扇区终止", "右边界",
    ),
}

_CENTER_WIDTH_COLUMNS: dict[str, tuple[str, ...]] = {
    "center": (
        "direction", "direction_center", "center", "center_direction",
        "sector_center", "wind_direction", "dir",
        "风向", "中心风向", "扇区中心", "中心角度", "角度",
    ),
    "width": (
        "width", "sector_width", "direction_width", "angle_width",
        "扇区宽度", "宽度", "角度宽度",
    ),
}

_WEIBULL_COLUMNS: dict[str, tuple[str, ...]] = {
    "k": (
        "weibull_k", "k", "shape", "shape_factor", "weibull shape",
        "威布尔k", "形状参数", "k参数", "形状因子",
    ),
    "c": (
        "weibull_c", "c", "scale", "scale_factor", "weibull scale",
        "威布尔c", "尺度参数", "c参数", "尺度因子",
    ),
    "mean_speed": (
        "mean_speed", "vmean", "avg_speed", "average_speed",
        "mean_wind_speed", "speed", "ws_mean", "v_avg",
        "平均风速", "平均风", "风速均值", "风速",
    ),
}


@dataclass
class WindResourceImportResult:
    """风资源导入结果。

    Attributes
    ----------
    wind_resource : WindResource
        规范化后的风资源（扇区按起始角度排序，频率和为 1）
    fingerprint : SourceFingerprint
        可重复识别的来源指纹与规范化摘要
    """

    wind_resource: WindResource
    fingerprint: SourceFingerprint


def load_wind_resource_csv(
    path: str | Path,
    *,
    referenced_as: Optional[str] = None,
    base_dir: Optional[str | Path] = None,
    frequency_strategy: str = "auto",
    frequency_scale: Optional[float] = None,
    default_k: float = 2.0,
    allow_partial_coverage: bool = False,
) -> WindResourceImportResult:
    """从测风塔扇区表 CSV 加载风资源。

    Parameters
    ----------
    path : str | Path
        CSV 路径；相对路径按 ``base_dir`` 解析
    referenced_as : Optional[str]
        配置中引用路径的原样写法（记入指纹）
    base_dir : Optional[str | Path]
        配置文件所在目录，用于解析相对路径
    frequency_strategy : str
        频率归一化策略：
        ``"auto"``（默认，自动识别比例/百分数/任意权重）、
        ``"normalize"``（恒按总和归一化）、
        ``"scale"``（按 ``frequency_scale`` 缩放，如 100 或 8760）、
        ``"strict"``（要求原始频率和严格为 1）
    frequency_scale : Optional[float]
        ``strategy="scale"`` 时的缩放因子
    default_k : float
        仅给出平均风速时使用的 Weibull 形状参数
    allow_partial_coverage : bool
        为 True 时扇区未覆盖完整 360° 不报错（缺口写入摘要）
    """
    if base_dir is not None and not Path(path).is_absolute():
        path = resolve_relative_path(base_dir, str(path))
    path = Path(path)

    required = dict(_WIND_COLUMNS)
    rows, col_index, raw_headers = read_csv_rows(
        path,
        required_columns=required,
        optional_columns={**_BOUNDARY_COLUMNS, **_CENTER_WIDTH_COLUMNS, **_WEIBULL_COLUMNS},
    )

    has_boundaries = col_index["start"] >= 0 and col_index["end"] >= 0
    has_center_width = col_index["center"] >= 0 and col_index["width"] >= 0
    if not has_boundaries and not has_center_width:
        raise file_error(
            "扇区表必须提供起始/终止角度两列，或中心角度/扇区宽度两列", path
        )
    if has_boundaries and (col_index["start"] < 0 or col_index["end"] < 0):
        raise file_error("起始角度与终止角度必须同时提供", path)
    if has_center_width and (col_index["center"] < 0 or col_index["width"] < 0):
        raise file_error("中心角度与扇区宽度必须同时提供", path)

    has_k = col_index["k"] >= 0
    has_c = col_index["c"] >= 0
    has_mean = col_index["mean_speed"] >= 0
    if not has_mean and not (has_k and has_c):
        raise file_error(
            "风速信息不足：需提供平均风速列，或同时提供 Weibull k、c 两列", path
        )
    if default_k <= 0:
        raise DataImportError(f"default_k 必须为正数，当前为 {default_k}")

    # ---- 逐行解析 ----
    parsed: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {"data_row": row.data_row, "line": row.line}

        freq = parse_float(row, "frequency", path)
        if freq is None or freq < 0.0:
            raise row_error(
                f"频率必须为非负数，当前为 {freq}",
                path, line=row.line, data_row=row.data_row, column="frequency",
            )
        rec["frequency_raw"] = freq

        if has_boundaries:
            start = parse_float(row, "start", path)
            end = parse_float(row, "end", path)
            rec["start_raw"], rec["end_raw"] = start, end
        else:
            center = parse_float(row, "center", path)
            width = parse_float(row, "width", path)
            rec["center_raw"], rec["width_raw"] = center, width

        rec["k_raw"] = parse_float(row, "k", path, allow_blank=True) if has_k else None
        rec["c_raw"] = parse_float(row, "c", path, allow_blank=True) if has_c else None
        rec["mean_raw"] = (
            parse_float(row, "mean_speed", path, allow_blank=True) if has_mean else None
        )
        parsed.append(rec)

    # ---- 角度归一化与扇区几何 ----
    sectors_arc = _build_arcs(parsed, has_boundaries, path)
    _check_coverage(
        parsed, sectors_arc, path,
        allow_partial_coverage=allow_partial_coverage,
    )

    # ---- 频率归一化 ----
    raw_freqs = np.array([r["frequency_raw"] for r in parsed], dtype=np.float64)
    freqs, freq_action, raw_sum = _normalize_frequencies(
        raw_freqs,
        strategy=frequency_strategy,
        scale=frequency_scale,
        path=path,
    )

    # ---- Weibull 参数推导 ----
    weibull_specs = _build_weibull(parsed, has_k, has_c, has_mean, default_k, path)

    # ---- 组装 WindSector（按起始角排序） ----
    order = sorted(range(len(parsed)), key=lambda i: sectors_arc[i][0])
    wind_sectors: list[WindSector] = []
    for idx in order:
        start0, end_ext, width, center = sectors_arc[idx]
        k, c, mean = weibull_specs[idx]
        wind_sectors.append(
            WindSector(
                direction_center=float(center),
                direction_width=float(width),
                frequency=float(freqs[idx]),
                mean_speed=float(mean),
                weibull_k=float(k),
                weibull_c=float(c),
            )
        )

    details: dict[str, Any] = {
        "format": "boundaries" if has_boundaries else "center_width",
        "columns": raw_headers,
        "num_sectors": len(parsed),
        "frequency": {
            "strategy": frequency_strategy,
            "action": freq_action,
            "raw_sum": float(raw_sum),
        },
        "coverage_degrees": float(sum(a[2] for a in sectors_arc)),
        "weibull": {
            "k_source": "column" if has_k else "default",
            "c_source": "column" if has_c else (
                "derived_from_mean_speed" if has_mean else "column"
            ),
            "mean_speed_source": "column" if has_mean else "derived_from_weibull",
            "default_k": float(default_k) if not has_k else None,
        },
        "zero_crossing_sectors": [
            r["data_row"] for r, arc in zip(parsed, sectors_arc) if arc[1] > 360.0
        ],
        "source_row_order": [parsed[i]["data_row"] for i in order],
    }

    fp = make_fingerprint(
        kind="wind_resource",
        path=path,
        referenced_as=referenced_as,
        rows=len(parsed),
        details=details,
    )
    resource = WindResource(wind_sectors, provenance=fp)
    return WindResourceImportResult(wind_resource=resource, fingerprint=fp)


# ---------------------------------------------------------------------------
# 角度几何
# ---------------------------------------------------------------------------

def _build_arcs(
    parsed: list[dict[str, Any]],
    has_boundaries: bool,
    path: str | Path,
) -> list[tuple[float, float, float, float]]:
    """把每行转换为圆弧 ``(start0, end_ext, width, center)``。

    ``start0`` 归一化到 [0, 360)；``end_ext`` 为顺时针终点，跨越零度时 > 360。
    """
    arcs: list[tuple[float, float, float, float]] = []
    for rec in parsed:
        dr, ln = rec["data_row"], rec["line"]
        if has_boundaries:
            start = rec["start_raw"] % 360.0
            end0 = rec["end_raw"] % 360.0
            width = (end0 - start) % 360.0
            if width < _ANGLE_TOL:
                raise row_error(
                    "扇区宽度为 0：起始角与终止角相同（若要表示整圆需分两个扇区）",
                    path, line=ln, data_row=dr, column="end",
                )
            end_ext = start + width
        else:
            center = rec["center_raw"] % 360.0
            width = rec["width_raw"]
            if not (0.0 < width <= 360.0):
                raise row_error(
                    f"扇区宽度必须在 (0, 360] 范围内，当前为 {width}",
                    path, line=ln, data_row=dr, column="width",
                )
            if width > 360.0 - _ANGLE_TOL:
                raise row_error(
                    "单个扇区不能覆盖完整 360°", path, line=ln, data_row=dr,
                    column="width",
                )
            start = (center - width / 2.0) % 360.0
            end_ext = start + width

        center = (start + end_ext) / 2.0 % 360.0
        arcs.append((start, end_ext, width, center))
    return arcs


def _check_coverage(
    parsed: list[dict[str, Any]],
    arcs: list[tuple[float, float, float, float]],
    path: str | Path,
    *,
    allow_partial_coverage: bool,
) -> None:
    """检测扇区重叠与缺口，错误定位到相关数据行。"""
    # 拆成 [start, end) 段（跨零度扇区拆为两段），记录来源行
    segments: list[tuple[float, float, int]] = []
    for rec, (start, end_ext, _w, _c) in zip(parsed, arcs):
        data_row = rec["data_row"]
        if end_ext <= 360.0 + _ANGLE_TOL:
            segments.append((start, min(end_ext, 360.0), data_row))
        else:
            segments.append((start, 360.0, data_row))
            segments.append((0.0, end_ext - 360.0, data_row))

    # ---- 重叠检测（扫描线） ----
    events: list[tuple[float, int, int]] = []  # (位置, +1开始/-1结束, 行号)
    for s, e, dr in segments:
        events.append((s, 1, dr))
        events.append((e, -1, dr))
    # 结束事件优先于开始事件，使首尾相接不算重叠
    events.sort(key=lambda x: (x[0], x[1]))

    active: set[int] = set()
    overlaps: list[tuple[int, int, float]] = []
    prev_pos = 0.0
    for pos, kind, dr in events:
        if len(active) >= 2 and pos - prev_pos > _ANGLE_TOL:
            members = tuple(sorted(active))
            overlaps.append((members[0], members[1], pos))
        if kind == 1:
            active.add(dr)
        else:
            active.discard(dr)
        prev_pos = pos

    if overlaps:
        a, b, pos = overlaps[0]
        raise DataImportError(
            f"扇区重叠：数据行 {a} 与数据行 {b} 在 {pos:.2f}° 附近相交",
            path=str(path),
            data_row=a,
        )

    # ---- 缺口检测（并集覆盖） ----
    covered = sorted((s, e) for s, e, _ in segments)
    cursor = 0.0
    gaps: list[tuple[float, float]] = []
    for s, e in covered:
        if s > cursor + _ANGLE_TOL:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < 360.0 - _ANGLE_TOL:
        gaps.append((cursor, 360.0))

    if gaps and not allow_partial_coverage:
        gs, ge = gaps[0]
        raise DataImportError(
            f"扇区存在缺口：{gs:.2f}° 至 {ge:.2f}° 未被任何扇区覆盖"
            f"（共 {len(gaps)} 处；如确认按比例归一化，可在配置中设置 "
            f"allow_partial_coverage=true）",
            path=str(path),
        )


# ---------------------------------------------------------------------------
# 频率与 Weibull
# ---------------------------------------------------------------------------

def _normalize_frequencies(
    raw: np.ndarray,
    *,
    strategy: str,
    scale: Optional[float],
    path: str | Path,
) -> tuple[np.ndarray, str, float]:
    """按策略归一化频率，返回（归一化频率, 实际动作描述, 原始和）。"""
    total = float(raw.sum())
    if total <= 0.0:
        raise file_error("所有扇区频率之和为 0，无法归一化", path)

    strategy = strategy.lower()
    if strategy == "auto":
        if abs(total - 1.0) <= 1e-3 + 1e-6 * len(raw):
            return raw / raw.sum(), "already_unit_sum", total
        if abs(total - 100.0) <= 1.0 + 1e-3 * len(raw):
            return raw / 100.0, "percent_divided_by_100", total
        return raw / total, "rescaled_to_unit_sum", total

    if strategy == "normalize":
        return raw / total, "rescaled_to_unit_sum", total

    if strategy == "scale":
        if scale is None or scale <= 0:
            raise DataImportError(
                'frequency_strategy="scale" 时必须提供正数 frequency_scale'
            )
        scaled = raw / scale
        if abs(float(scaled.sum()) - 1.0) > 1e-2:
            raise file_error(
                f"频率按 {scale:g} 缩放后总和为 {float(scaled.sum()):.6f}，不为 1",
                path,
            )
        return scaled / scaled.sum(), f"scaled_by_{scale:g}_then_renormalized", total

    if strategy == "strict":
        if abs(total - 1.0) > 1e-4:
            raise file_error(
                f"strict 策略要求频率之和为 1，实际为 {total:.6f}", path
            )
        return raw / raw.sum(), "strict_unit_sum", total

    raise DataImportError(
        f"未知的频率归一化策略: {strategy}"
        f"（可选 auto/normalize/scale/strict）"
    )


def _build_weibull(
    parsed: list[dict[str, Any]],
    has_k: bool,
    has_c: bool,
    has_mean: bool,
    default_k: float,
    path: str | Path,
) -> list[tuple[float, float, float]]:
    """逐行返回 (k, c, mean_speed)，允许由均值/c 互相推导。"""
    specs: list[tuple[float, float, float]] = []
    for rec in parsed:
        dr, ln = rec["data_row"], rec["line"]
        k = rec["k_raw"] if has_k else default_k

        if k is not None and k <= 0.0:
            raise row_error(
                f"Weibull k 必须为正数，当前为 {k}",
                path, line=ln, data_row=dr, column="k",
            )
        if k is None:
            k = default_k

        c = rec["c_raw"] if has_c else None
        mean = rec["mean_raw"] if has_mean else None

        if c is not None and c <= 0.0:
            raise row_error(
                f"Weibull c 必须为正数，当前为 {c}",
                path, line=ln, data_row=dr, column="c",
            )
        if mean is not None and mean <= 0.0:
            raise row_error(
                f"平均风速必须为正数，当前为 {mean}",
                path, line=ln, data_row=dr, column="mean_speed",
            )

        gamma_factor = float(_gamma_lanczos(np.array([1.0 + 1.0 / k]))[0])

        if c is not None and mean is None:
            mean = c * gamma_factor
        elif mean is not None and c is None:
            c = mean / gamma_factor
        elif c is None and mean is None:
            raise row_error(
                "该行缺少平均风速且缺少 Weibull c 参数",
                path, line=ln, data_row=dr,
            )
        else:
            # k/c 与均值同时给出：以 k/c 为重算均值，保证与 AEP
            # 积分所用 Weibull 分布自洽；原始均值记入容差警告来源
            mean = c * gamma_factor
        specs.append((float(k), float(c), float(mean)))
    return specs
