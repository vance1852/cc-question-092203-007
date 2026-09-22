"""风机模型定义。"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

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
    thrust_coefficient : float, optional
        恒定推力系数 Ct (0, 1)。当提供 ``thrust_curve`` 时可为 None。
    power_curve : np.ndarray
        功率曲线，形状为 (N, 2)，第一列为风速 (m/s)，第二列为功率 (kW)
    thrust_curve : np.ndarray, optional
        随风速变化的推力系数曲线，形状为 (M, 2)，第一列为风速 (m/s)，
        第二列为推力系数 Ct。风速范围之外按首值/0 外延。
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
    """

    name: str
    hub_height: float
    rotor_diameter: float
    thrust_coefficient: Optional[float] = None
    power_curve: Optional[np.ndarray] = None
    position: Optional[Tuple[float, float]] = None
    thrust_curve: Optional[np.ndarray] = None
    source: Optional[dict] = None

    cut_in_speed: float = field(init=False)
    rated_speed: float = field(init=False)
    cut_out_speed: float = field(init=False)
    rated_power: float = field(init=False)

    def __post_init__(self) -> None:
        self.power_curve = np.asarray(self.power_curve, dtype=np.float64)
        if self.power_curve.ndim != 2 or self.power_curve.shape[1] != 2:
            raise ValueError("功率曲线必须是形状为 (N, 2) 的数组")

        self._validate_monotonic_speed(self.power_curve, "功率曲线")
        if np.any(self.power_curve[:, 1] < 0.0):
            raise ValueError("功率曲线中的功率不能为负值")

        self._derive_parameters()

        if self.thrust_curve is not None:
            self.thrust_curve = np.asarray(self.thrust_curve, dtype=np.float64)
            if self.thrust_curve.ndim != 2 or self.thrust_curve.shape[1] != 2:
                raise ValueError("推力系数曲线必须是形状为 (M, 2) 的数组")
            self._validate_monotonic_speed(self.thrust_curve, "推力系数曲线")
            ct = self.thrust_curve[:, 1]
            if np.any(ct < 0.0) or np.any(ct >= 1.0):
                raise ValueError("推力系数曲线中的 Ct 必须在 [0, 1) 范围内")
        elif self.thrust_coefficient is None:
            raise ValueError("必须提供恒定推力系数 thrust_coefficient 或推力系数曲线 thrust_curve")

        if self.thrust_coefficient is not None and not (0.0 < self.thrust_coefficient < 1.0):
            raise ValueError(
                f"推力系数必须在 (0, 1) 范围内，当前为 {self.thrust_coefficient}"
            )

    @staticmethod
    def _validate_monotonic_speed(curve: np.ndarray, label: str) -> None:
        """校验曲线风速列严格单调递增。"""
        ws = curve[:, 0]
        if len(ws) < 2:
            raise ValueError(f"{label}至少需要两个数据点")
        diff = np.diff(ws)
        if np.any(diff <= 0.0):
            bad = int(np.argmax(diff <= 0.0)) + 1
            raise ValueError(
                f"{label}的风速必须严格单调递增：第 {bad + 1} 个点 "
                f"({ws[bad]:g} m/s) 不大于前一点 ({ws[bad - 1]:g} m/s)"
            )

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

    @property
    def has_thrust_curve(self) -> bool:
        """是否使用随风速变化的推力系数曲线。"""
        return self.thrust_curve is not None

    def thrust_coefficient_at(self, wind_speed: float | np.ndarray) -> float | np.ndarray:
        """查询给定风速下的推力系数。

        有推力曲线时线性插值；低于曲线起点取首值，高于终点取 0（切出停机）。
        无曲线时返回恒定推力系数。

        Parameters
        ----------
        wind_speed : float | np.ndarray
            风速 (m/s)

        Returns
        -------
        float | np.ndarray
            推力系数，标量输入返回 float，数组输入返回数组
        """
        ws = np.asarray(wind_speed, dtype=np.float64)
        if self.thrust_curve is not None:
            result = np.interp(
                ws,
                self.thrust_curve[:, 0],
                self.thrust_curve[:, 1],
                left=float(self.thrust_curve[0, 1]),
                right=0.0,
            )
        else:
            result = np.full_like(ws, self.thrust_coefficient)

        return float(result) if ws.ndim == 0 else result

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
