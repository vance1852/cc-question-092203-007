"""风机模型定义。"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class Turbine:
    """风机参数类。

    Parameters
    ----------
    name : str
        风机名称/型号
    hub_height : float
        轮毂高度 (m)
    rotor_diameter : float
        转子直径 (m)
    thrust_coefficient : float
        代表性推力系数 Ct (0-1]，取曲线最大值或额定点值；
        未提供 ``thrust_curve`` 时同时作为常数 Ct 使用
    power_curve : np.ndarray
        功率曲线，形状为 (N, 2)，第一列为风速 (m/s)，第二列为功率 (kW)
    cut_in_speed : float
        切入风速 (m/s)，从功率曲线自动推断
    rated_speed : float
        额定风速 (m/s)，从功率曲线自动推断
    cut_out_speed : float
        切出风速 (m/s)，从功率曲线自动推断
    rated_power : float
        额定功率 (kW)，从功率曲线自动推断
    position : Optional[Tuple[float, float]]
        风机位置 (x, y) (m)，可选
    thrust_curve : Optional[np.ndarray]
        随风速变化的推力系数曲线，形状为 (M, 2)，
        第一列风速 (m/s) 须严格单调递增，第二列 Ct ∈ [0, 1)；
        为 None 时全程使用常数 ``thrust_coefficient``
    provenance : Optional[Any]
        外部数据来源指纹（``io.SourceFingerprint``），内置机组为 None
    """

    name: str
    hub_height: float
    rotor_diameter: float
    thrust_coefficient: float
    power_curve: np.ndarray
    position: Optional[Tuple[float, float]] = None
    thrust_curve: Optional[np.ndarray] = None
    provenance: Optional[Any] = field(default=None, repr=False, compare=False)

    cut_in_speed: float = field(init=False)
    rated_speed: float = field(init=False)
    cut_out_speed: float = field(init=False)
    rated_power: float = field(init=False)

    def __post_init__(self) -> None:
        self.power_curve = np.asarray(self.power_curve, dtype=np.float64)
        if self.power_curve.ndim != 2 or self.power_curve.shape[1] != 2:
            raise ValueError("功率曲线必须是形状为 (N, 2) 的数组")

        if self.thrust_curve is not None:
            self.thrust_curve = np.asarray(self.thrust_curve, dtype=np.float64)
            if self.thrust_curve.ndim != 2 or self.thrust_curve.shape[1] != 2:
                raise ValueError("推力系数曲线必须是形状为 (N, 2) 的数组")
            ct_speeds = self.thrust_curve[:, 0]
            if len(ct_speeds) < 2 or np.any(np.diff(ct_speeds) <= 0):
                raise ValueError("推力系数曲线的风速列必须严格单调递增且至少 2 个点")
            ct_values = self.thrust_curve[:, 1]
            if np.any(ct_values < 0.0) or np.any(ct_values >= 1.0):
                raise ValueError("推力系数曲线的 Ct 值必须在 [0, 1) 范围内")

        self._derive_parameters()

        if not (0.0 < self.thrust_coefficient <= 1.0):
            raise ValueError(f"推力系数必须在 (0, 1] 范围内，当前为 {self.thrust_coefficient}")

    def _derive_parameters(self) -> None:
        """从功率曲线推导出切入、额定、切出风速和额定功率。"""
        ws = self.power_curve[:, 0]
        pw = self.power_curve[:, 1]

        positive_idx = np.where(pw > 0.0)[0]
        if len(positive_idx) == 0:
            raise ValueError("功率曲线中没有正功率点")

        self.cut_in_speed = float(ws[positive_idx[0]])
        self.rated_power = float(np.max(pw))

        rated_idx = np.where(pw >= self.rated_power * 0.999)[0]
        if len(rated_idx) > 0:
            self.rated_speed = float(ws[rated_idx[0]])
        else:
            self.rated_speed = float(ws[np.argmax(pw)])

        self.cut_out_speed = float(ws[-1])

    def power(self, wind_speed: float | np.ndarray) -> float | np.ndarray:
        """根据风速计算功率。

        Parameters
        ----------
        wind_speed : float | np.ndarray
            风速 (m/s)

        Returns
        -------
        float | np.ndarray
            功率 (kW)
        """
        ws = np.asarray(wind_speed, dtype=np.float64)
        result = np.interp(ws, self.power_curve[:, 0], self.power_curve[:, 1],
                          left=0.0, right=0.0)
        return result if ws.ndim > 0 else float(result)

    def thrust_at(self, wind_speed: float | np.ndarray) -> float | np.ndarray:
        """根据来流风速计算推力系数 Ct。

        有推力曲线时在曲线上线性插值（曲线范围之外取端点值）；
        无曲线时返回常数 ``thrust_coefficient``。

        Parameters
        ----------
        wind_speed : float | np.ndarray
            来流风速 (m/s)

        Returns
        -------
        float | np.ndarray
            推力系数，与输入形状一致
        """
        if self.thrust_curve is None:
            ws = np.asarray(wind_speed, dtype=np.float64)
            result = np.full_like(ws, self.thrust_coefficient)
            return result if ws.ndim > 0 else float(self.thrust_coefficient)
        ws = np.asarray(wind_speed, dtype=np.float64)
        result = np.interp(
            ws,
            self.thrust_curve[:, 0],
            self.thrust_curve[:, 1],
            left=float(self.thrust_curve[0, 1]),
            right=float(self.thrust_curve[-1, 1]),
        )
        return result if ws.ndim > 0 else float(result)

    @property
    def rotor_area(self) -> float:
        """风轮扫掠面积 (m^2)。"""
        return np.pi * (self.rotor_diameter / 2.0) ** 2


def create_default_turbine(model: str = "V164-9.5MW") -> Turbine:
    """创建默认风机模型。

    Parameters
    ----------
    model : str
        风机型号，可选 "V164-9.5MW" 或 "V126-3.45MW"

    Returns
    -------
    Turbine
        风机实例
    """
    if model == "V164-9.5MW":
        speeds = np.arange(0.0, 31.0, 1.0)
        powers = np.zeros_like(speeds)
        cut_in = 4.0
        rated = 11.5
        cut_out = 25.0

        for i, ws in enumerate(speeds):
            if ws < cut_in or ws > cut_out:
                powers[i] = 0.0
            elif ws <= rated:
                powers[i] = 9500.0 * ((ws - cut_in) / (rated - cut_in)) ** 3
            else:
                powers[i] = 9500.0

        power_curve = np.column_stack([speeds, powers])

        return Turbine(
            name="V164-9.5MW",
            hub_height=105.0,
            rotor_diameter=164.0,
            thrust_coefficient=0.80,
            power_curve=power_curve,
        )

    elif model == "V126-3.45MW":
        speeds = np.arange(0.0, 26.0, 1.0)
        powers = np.zeros_like(speeds)
        cut_in = 3.5
        rated = 12.0
        cut_out = 25.0

        for i, ws in enumerate(speeds):
            if ws < cut_in or ws > cut_out:
                powers[i] = 0.0
            elif ws <= rated:
                powers[i] = 3450.0 * ((ws - cut_in) / (rated - cut_in)) ** 3
            else:
                powers[i] = 3450.0

        power_curve = np.column_stack([speeds, powers])

        return Turbine(
            name="V126-3.45MW",
            hub_height=87.0,
            rotor_diameter=126.0,
            thrust_coefficient=0.82,
            power_curve=power_curve,
        )

    else:
        raise ValueError(f"未知的风机型号: {model}")
