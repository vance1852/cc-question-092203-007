"""配置管理模块。

用于从JSON/YAML文件加载配置，或通过命令行参数构建配置。

外部 CSV 数据通过配置引用，相对路径按配置文件所在目录解析：

- ``wind_resource_type: "external"`` + ``wind_resource_params.csv_path``
- ``turbine_model: "custom"`` + ``custom_turbine``（含曲线 CSV 路径）
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional, List

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
from .io.csv_import import (
    load_custom_turbine,
    load_wind_resource_csv,
)


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

    # 自定义机组配置（turbine_model 为 "custom" 时使用），曲线 CSV 的相对
    # 路径按配置文件所在目录解析
    custom_turbine: Optional[dict] = None

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)

    def __post_init__(self) -> None:
        # 配置文件所在目录，外部 CSV 相对路径的解析基准
        self._config_dir: Optional[str] = None
        self._custom_turbine_instance: Optional[Turbine] = None

    @classmethod
    def from_json(cls, filepath: str) -> "WindFarmConfig":
        """从JSON文件加载配置。

        外部 CSV 的相对路径以该配置文件所在目录为基准解析。
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        opt_config = OptimizationConfig(**data.get("optimization", {}))
        vis_config = VisualizationConfig(**data.get("visualization", {}))
        econ_config = EconomicConfig(**data.get("economic", {}))

        config = cls(
            n_turbines=data.get("n_turbines", 15),
            turbine_model=data.get("turbine_model", "V126-3.45MW"),
            wake_model=data.get("wake_model", "jensen"),
            wake_decay=data.get("wake_decay", 0.07),
            superposition_method=data.get("superposition_method", "sum_of_squares"),
            boundary_type=data.get("boundary_type", "rectangular"),
            boundary_params=data.get("boundary_params", {}),
            wind_resource_type=data.get("wind_resource_type", "default"),
            wind_resource_params=data.get("wind_resource_params", {}),
            custom_turbine=data.get("custom_turbine"),
            optimization=opt_config,
            visualization=vis_config,
            economic=econ_config,
        )
        config._config_dir = os.path.dirname(os.path.abspath(filepath))
        return config

    def to_json(self, filepath: str) -> None:
        """保存配置到JSON文件。"""
        data = {
            "n_turbines": self.n_turbines,
            "turbine_model": self.turbine_model,
            "wake_model": self.wake_model,
            "wake_decay": self.wake_decay,
            "superposition_method": self.superposition_method,
            "boundary_type": self.boundary_type,
            "boundary_params": self.boundary_params,
            "wind_resource_type": self.wind_resource_type,
            "wind_resource_params": self.wind_resource_params,
            "custom_turbine": self.custom_turbine,
            "optimization": self.optimization.__dict__,
            "visualization": self.visualization.__dict__,
            "economic": self.economic.__dict__,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_turbines(self) -> list[Turbine]:
        """根据配置创建风机列表。"""
        if self.turbine_model.lower() == "custom":
            if not self.custom_turbine:
                raise ValueError(
                    "turbine_model 为 'custom' 时必须提供 custom_turbine 配置"
                )
            turbine = self._create_custom_turbine()
        else:
            turbine = create_default_turbine(self.turbine_model)
        return [turbine for _ in range(self.n_turbines)]

    def _create_custom_turbine(self) -> Turbine:
        """加载（并缓存）自定义机组。"""
        if self._custom_turbine_instance is None:
            self._custom_turbine_instance = load_custom_turbine(
                self.custom_turbine, base_dir=self._config_dir
            )
        return self._custom_turbine_instance

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
        """根据配置创建风资源。"""
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
        elif wtype == "external":
            csv_path = wrp.get("csv_path")
            if not csv_path:
                raise ValueError(
                    "external 风资源需要在 wind_resource_params.csv_path 中指定扇区表文件"
                )
            return load_wind_resource_csv(
                csv_path,
                frequency_normalization=wrp.get("frequency_normalization", "strict"),
                base_dir=self._config_dir,
            )
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
