"""自定义机组功率曲线与推力系数曲线 CSV 导入器。

加载随风速变化的推力系数 Ct 曲线（厂商推力表通常按风速给点）与功率曲线，
自动识别 kW / MW 功率单位，并校验风速单调性、取值范围与点数。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.turbine import Turbine
from .csv_reader import parse_float, read_csv_rows, resolve_relative_path
from .errors import DataImportError, file_error, row_error
from .fingerprint import SourceFingerprint, make_fingerprint

# MW 自动判别的上限：机组额定功率 <= 该值视为 MW（适用于 MW 级风电机组表）
_AUTO_MW_MAX = 50.0

_SPEED_COLUMNS: dict[str, tuple[str, ...]] = {
    "wind_speed": (
        "wind_speed", "windspeed", "speed", "v", "ws", "wind_velocity",
        "u", "v_wind",
        "风速", "来流风速", "风速度",
    ),
}

_POWER_COLUMNS: dict[str, tuple[str, ...]] = {
    "power_kw": (
        "power_kw", "p_kw", "powerkw", "kw",
        "功率kw", "功率千瓦",
    ),
    "power_mw": (
        "power_mw", "p_mw", "powermw", "mw",
        "功率mw", "功率兆瓦",
    ),
    "power_generic": (
        "power", "p", "power_output", "output", "active_power",
        "功率", "输出功率", "机组功率",
    ),
}

_CT_COLUMNS: dict[str, tuple[str, ...]] = {
    "ct": (
        "ct", "thrust_coefficient", "thrust_coeff", "c_t", "ct_curve",
        "thrust",
        "推力系数", "推力系数ct", "ct值",
    ),
}


@dataclass
class TurbineImportResult:
    """机组导入结果。

    Attributes
    ----------
    turbine : Turbine
        规范化后的机组（功率统一为 kW，Ct 可随风速变化）
    fingerprint : SourceFingerprint
        来源指纹与规范化摘要
    """

    turbine: Turbine
    fingerprint: SourceFingerprint


def load_turbine_csv(
    path: str | Path,
    *,
    name: str,
    hub_height: float,
    rotor_diameter: float,
    referenced_as: Optional[str] = None,
    base_dir: Optional[str | Path] = None,
    power_unit: str = "auto",
    default_thrust_coefficient: Optional[float] = None,
) -> TurbineImportResult:
    """从厂商曲线 CSV 加载自定义机组。

    Parameters
    ----------
    path : str | Path
        CSV 路径；相对路径按 ``base_dir`` 解析
    name : str
        机组型号名称（写入 Turbine.name）
    hub_height : float
        轮毂高度 (m)，由配置提供
    rotor_diameter : float
        风轮直径 (m)，由配置提供
    power_unit : str
        功率单位：``"auto"``（默认，按表头与数值量级判别）、``"kw"``、``"mw"``
    default_thrust_coefficient : Optional[float]
        CSV 无 Ct 列时使用的常数推力系数；有 Ct 列时忽略
    """
    if base_dir is not None and not Path(path).is_absolute():
        path = resolve_relative_path(base_dir, str(path))
    path = Path(path)

    if hub_height <= 0:
        raise DataImportError(f"轮毂高度必须为正数，当前为 {hub_height}")
    if rotor_diameter <= 0:
        raise DataImportError(f"风轮直径必须为正数，当前为 {rotor_diameter}")
    if power_unit.lower() not in ("auto", "kw", "mw"):
        raise DataImportError(
            f"未知的功率单位策略: {power_unit}（可选 auto/kw/mw）"
        )

    rows, col_index, raw_headers = read_csv_rows(
        path,
        required_columns=_SPEED_COLUMNS,
        optional_columns={**_POWER_COLUMNS, **_CT_COLUMNS},
    )

    power_present = [c for c in ("power_kw", "power_mw", "power_generic")
                     if col_index[c] >= 0]
    if not power_present:
        raise file_error(
            "机组曲线表必须包含功率列（power / power_kw / power_mw / 功率）",
            path,
        )
    if len(power_present) > 1:
        raise file_error(
            f"功率列存在歧义，同时匹配到: {', '.join(power_present)}，"
            f"请只保留一种功率列",
            path,
        )
    power_col = power_present[0]
    has_ct = col_index["ct"] >= 0
    if not has_ct and default_thrust_coefficient is None:
        raise file_error(
            "CSV 中没有推力系数列，且配置未提供默认推力系数 "
            "thrust_coefficient",
            path,
        )

    # ---- 逐行解析 ----
    points: list[tuple[float, float, Optional[float]]] = []
    prev_speed: Optional[float] = None
    for row in rows:
        speed = parse_float(row, "wind_speed", path)
        if speed is None or speed < 0.0:
            raise row_error(
                f"风速必须为非负数，当前为 {speed}",
                path, line=row.line, data_row=row.data_row, column="wind_speed",
            )
        if prev_speed is not None and speed <= prev_speed:
            raise row_error(
                f"风速必须严格单调递增：当前 {speed:g} m/s，"
                f"上一行为 {prev_speed:g} m/s",
                path, line=row.line, data_row=row.data_row,
                column="wind_speed",
            )
        prev_speed = speed

        power = parse_float(row, power_col, path)
        if power is None or power < 0.0:
            raise row_error(
                f"功率必须为非负数，当前为 {power}",
                path, line=row.line, data_row=row.data_row, column=power_col,
            )

        ct = parse_float(row, "ct", path, allow_blank=True) if has_ct else None
        if ct is not None and not (0.0 <= ct < 1.0):
            raise row_error(
                f"推力系数必须在 [0, 1) 范围内，当前为 {ct}",
                path, line=row.line, data_row=row.data_row, column="ct",
            )

        points.append((speed, power, ct))

    if len(points) < 2:
        raise file_error(
            f"功率曲线至少需要 2 个数据点，当前仅 {len(points)} 个", path
        )

    speeds = np.array([p[0] for p in points], dtype=np.float64)
    powers_raw = np.array([p[1] for p in points], dtype=np.float64)

    # ---- 单位识别与换算（内部统一 kW） ----
    unit, unit_source = _detect_power_unit(
        power_col, powers_raw, power_unit, raw_headers
    )
    powers_kw = powers_raw * 1e3 if unit == "mw" else powers_raw

    # ---- Ct 曲线 ----
    thrust_curve: Optional[np.ndarray] = None
    scalar_ct: Optional[float] = None
    ct_action = "constant_from_config"
    if has_ct:
        ct_values = [p[2] for p in points]
        if any(v is None for v in ct_values):
            first_blank = next(
                r.data_row for r, p in zip(rows, points) if p[2] is None
            )
            raise row_error(
                "推力系数列不允许出现空值（要么整列提供，要么整列省略）",
                path, data_row=first_blank, column="ct",
            )
        ct_arr = np.array(ct_values, dtype=np.float64)
        if not np.any(ct_arr > 0.0):
            raise file_error("推力系数曲线全为 0，无法用于尾流计算", path)
        thrust_curve = np.column_stack([speeds, ct_arr])
        scalar_ct = float(np.max(ct_arr))
        ct_action = "curve_from_csv"
    else:
        scalar_ct = float(default_thrust_coefficient)  # type: ignore[arg-type]
        if not (0.0 < scalar_ct <= 1.0):
            raise DataImportError(
                f"默认推力系数必须在 (0, 1] 范围内，当前为 {scalar_ct}"
            )

    power_curve = np.column_stack([speeds, powers_kw])

    turbine = Turbine(
        name=name,
        hub_height=float(hub_height),
        rotor_diameter=float(rotor_diameter),
        thrust_coefficient=scalar_ct,
        power_curve=power_curve,
        thrust_curve=thrust_curve,
    )

    details: dict[str, Any] = {
        "columns": raw_headers,
        "num_points": len(points),
        "speed_range_ms": [float(speeds[0]), float(speeds[-1])],
        "power": {
            "detected_unit": unit,
            "unit_source": unit_source,
            "rated_in_file": float(np.max(powers_raw)),
            "rated_kw": float(np.max(powers_kw)),
        },
        "thrust_coefficient": {
            "source": ct_action,
            "constant": scalar_ct if thrust_curve is None else None,
            "curve_max": float(np.max(thrust_curve[:, 1]))
            if thrust_curve is not None else None,
        },
        "geometry": {
            "hub_height_m": float(hub_height),
            "rotor_diameter_m": float(rotor_diameter),
        },
    }

    fp = make_fingerprint(
        kind="turbine",
        path=path,
        referenced_as=referenced_as,
        rows=len(points),
        details=details,
    )
    turbine.provenance = fp
    return TurbineImportResult(turbine=turbine, fingerprint=fp)


def _detect_power_unit(
    power_col: str,
    powers: np.ndarray,
    configured: str,
    raw_headers: list[str],
) -> tuple[str, str]:
    """判别功率单位，返回 (kw|mw, 判别依据)。"""
    configured = configured.lower()
    if configured in ("kw", "mw"):
        return configured, "config"

    if power_col == "power_kw":
        return "kw", "column_name"
    if power_col == "power_mw":
        return "mw", "column_name"

    # 通用 power 列：按量级判别
    max_power = float(np.max(powers))
    if max_power <= 0.0:
        raise DataImportError(
            f"功率列最大值为 0，无法判别单位（表头: {raw_headers}）"
        )
    if max_power <= _AUTO_MW_MAX:
        return "mw", f"magnitude(max={max_power:g}<={_AUTO_MW_MAX:g})"
    return "kw", f"magnitude(max={max_power:g}>{_AUTO_MW_MAX:g})"
