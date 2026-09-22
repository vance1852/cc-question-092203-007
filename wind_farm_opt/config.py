"""配置管理模块。

用于从JSON/YAML文件加载配置，或通过命令行参数构建配置。

外部 CSV 数据
-------------
- ``wind_resource_type: "csv"`` 时，从 ``wind_resource_params.path`` 加载
  测风塔扇区表（不等宽扇区、频率、Weibull 参数/平均风速）。
- ``turbine_model: "custom"`` 时，从 ``turbine_params.path`` 加载
  厂商功率曲线与（可选的）随风速变化的推力系数曲线。
- CSV 的相对路径一律按**配置文件所在目录**解析。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .core.turbine import Turbine, create_default_turbine
from .core.wind_resource import WindResource, create_default_wind_resource
from .core.wake import JensenWake, GaussianWake, WakeModel
from .constraints.boundary import (
    SiteBoundary,
    create_rectangular_boundary,
    create_hexagonal_boundary,
    create_irregular_boundary,
)
from .io.turbine_loader import load_turbine_csv
from .io.wind_resource_loader import load_wind_resource_csv


@dataclass
class OptimizationConfig:
    """优化算法配置。"""
    algorithm: str = "ga"
    population_size: int = 40
    max_iterations: int = 80
    min_spacing_multiple: float = 5.0
    seed: Optional[int] = 42


@dataclass
class VisualizationConfig:
    """可视化配置。"""
    save_dir: str = "output"
    save_plots: bool = True
    show_plots: bool = False
    plot_wake_heatmap: bool = True


@dataclass
class EconomicConfig:
    """经济性分析配置。"""
    electricity_price: float = 0.45
    discount_rate: float = 0.06
    enable_analysis: bool = True


@dataclass
class WindFarmConfig:
    """完整的风电场分析配置。"""
    n_turbines: int = 15
    turbine_model: str = "V126-3.45MW"
    turbine_params: dict = field(default_factory=dict)
    wake_model: str = "jensen"
    wake_decay: float = 0.07
    superposition_method: str = "sum_of_squares"

    boundary_type: str = "rectangular"
    boundary_params: dict = field(default_factory=lambda: {
        "width": 4000,
        "height": 4000,
        "center_x": 0,
        "center_y": 0,
    })

    wind_resource_type: str = "default"
    wind_resource_params: dict = field(default_factory=lambda: {
        "num_sectors": 12,
        "dominant_direction": 270.0,
        "mean_speed": 8.5,
    })

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)

    # 配置文件所在目录（运行时设置，非序列化字段），用于解析 CSV 相对路径
    config_dir: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_json(cls, filepath: str) -> "WindFarmConfig":
        """从JSON文件加载配置。

        外部 CSV 的相对路径按该配置文件所在目录解析。
        """
        filepath = str(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        opt_config = OptimizationConfig(**data.get("optimization", {}))
        vis_config = VisualizationConfig(**data.get("visualization", {}))
        econ_config = EconomicConfig(**data.get("economic", {}))

        return cls(
            n_turbines=data.get("n_turbines", 15),
            turbine_model=data.get("turbine_model", "V126-3.45MW"),
            turbine_params=data.get("turbine_params", {}),
            wake_model=data.get("wake_model", "jensen"),
            wake_decay=data.get("wake_decay", 0.07),
            superposition_method=data.get("superposition_method", "sum_of_squares"),
            boundary_type=data.get("boundary_type", "rectangular"),
            boundary_params=data.get("boundary_params", {}),
            wind_resource_type=data.get("wind_resource_type", "default"),
            wind_resource_params=data.get("wind_resource_params", {}),
            optimization=opt_config,
            visualization=vis_config,
            economic=econ_config,
            config_dir=str(Path(filepath).resolve().parent),
        )

    def to_json(self, filepath: str) -> None:
        """保存配置到JSON文件。"""
        data = {
            "n_turbines": self.n_turbines,
            "turbine_model": self.turbine_model,
            "turbine_params": self.turbine_params,
            "wake_model": self.wake_model,
            "wake_decay": self.wake_decay,
            "superposition_method": self.superposition_method,
            "boundary_type": self.boundary_type,
            "boundary_params": self.boundary_params,
            "wind_resource_type": self.wind_resource_type,
            "wind_resource_params": self.wind_resource_params,
            "optimization": self.optimization.__dict__,
            "visualization": self.visualization.__dict__,
            "economic": self.economic.__dict__,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_turbines(self) -> list[Turbine]:
        """根据配置创建风机列表。

        ``turbine_model`` 为内置型号（V126-3.45MW / V164-9.5MW）时使用内置曲线；
        为 ``"custom"`` 时从 ``turbine_params.path`` 指定的 CSV 加载
        自定义功率曲线与推力系数曲线。
        """
        turbine = self._create_single_turbine()
        return [turbine for _ in range(self.n_turbines)]

    def _create_single_turbine(self) -> Turbine:
        """创建单台风机（内置型号或外部 CSV）。"""
        model = self.turbine_model
        if model.lower() in ("custom", "csv", "external"):
            tp = self.turbine_params
            if "path" not in tp:
                raise ValueError(
                    'turbine_model 为 "custom" 时必须在 turbine_params.path '
                    "中指定机组曲线 CSV 路径"
                )
            result = load_turbine_csv(
                tp["path"],
                name=tp.get("name", "custom"),
                hub_height=float(tp.get("hub_height", 80.0)),
                rotor_diameter=float(tp.get("rotor_diameter", 100.0)),
                referenced_as=tp["path"],
                base_dir=self.config_dir,
                power_unit=tp.get("power_unit", "auto"),
                default_thrust_coefficient=tp.get("thrust_coefficient"),
            )
            return result.turbine
        return create_default_turbine(model)

    def create_wake_model(self) -> WakeModel:
        """根据配置创建尾流模型。"""
        if self.wake_model.lower() == "jensen":
            return JensenWake(wake_decay=self.wake_decay)
        elif self.wake_model.lower() == "gaussian":
            return GaussianWake(wake_decay=0.035)
        else:
            raise ValueError(f"未知的尾流模型: {self.wake_model}")

    def create_boundary(self) -> SiteBoundary:
        """根据配置创建场地边界。"""
        bp = self.boundary_params
        if self.boundary_type.lower() == "rectangular":
            return create_rectangular_boundary(
                width=bp.get("width", 4000),
                height=bp.get("height", 4000),
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif self.boundary_type.lower() == "hexagonal":
            return create_hexagonal_boundary(
                radius=bp.get("radius", 2500),
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif self.boundary_type.lower() == "irregular":
            return create_irregular_boundary()
        elif self.boundary_type.lower() == "custom":
            vertices = np.array(bp["vertices"], dtype=np.float64)
            return SiteBoundary(vertices)
        else:
            raise ValueError(f"未知的边界类型: {self.boundary_type}")

    def create_wind_resource(self) -> WindResource:
        """根据配置创建风资源。

        ``wind_resource_type`` 为 ``"csv"`` 时从
        ``wind_resource_params.path`` 加载外部测风塔扇区表；
        ``"default"`` / ``"uniform"`` 保持内置兼容性。
        """
        wrp = self.wind_resource_params
        wtype = self.wind_resource_type.lower()
        if wtype == "default":
            return create_default_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                dominant_direction=wrp.get("dominant_direction", 270.0),
                mean_speed=wrp.get("mean_speed", 8.5),
            )
        elif wtype == "uniform":
            from .core.wind_resource import create_simple_wind_resource
            return create_simple_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                uniform=True,
                mean_speed=wrp.get("mean_speed", 8.0),
            )
        elif wtype in ("csv", "external", "sectors"):
            if "path" not in wrp:
                raise ValueError(
                    'wind_resource_type 为 "csv" 时必须在 '
                    "wind_resource_params.path 中指定扇区表 CSV 路径"
                )
            result = load_wind_resource_csv(
                wrp["path"],
                referenced_as=wrp["path"],
                base_dir=self.config_dir,
                frequency_strategy=wrp.get("frequency_strategy", "auto"),
                frequency_scale=wrp.get("frequency_scale"),
                default_k=float(wrp.get("default_weibull_k", 2.0)),
                allow_partial_coverage=bool(
                    wrp.get("allow_partial_coverage", False)
                ),
            )
            return result.wind_resource
        else:
            raise ValueError(f"未知的风资源类型: {self.wind_resource_type}")


def create_sample_config() -> WindFarmConfig:
    """创建示例配置。"""
    return WindFarmConfig(
        n_turbines=12,
        turbine_model="V126-3.45MW",
        wake_model="jensen",
        wake_decay=0.07,
        boundary_type="rectangular",
        boundary_params={"width": 3500, "height": 3500, "center_x": 0, "center_y": 0},
        wind_resource_type="default",
        wind_resource_params={"num_sectors": 12, "dominant_direction": 270.0, "mean_speed": 8.5},
        optimization=OptimizationConfig(
            algorithm="ga",
            population_size=30,
            max_iterations=50,
            min_spacing_multiple=5.0,
            seed=42,
        ),
    )
