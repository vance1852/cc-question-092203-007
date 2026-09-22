"""外部 CSV 数据导入模块。

支持从测风塔扇区表与厂商机组曲线 CSV 文件导入：

- 不等宽风向扇区（显式起止角度，允许跨越 0°）
- 频率（自动识别百分比/小数、可配置归一化策略）
- Weibull k/c 参数，或仅给均值时按默认 k 推导（k+均值亦可推导 c）
- 自定义机组功率曲线（自动识别 kW/MW）与随风速变化的推力系数曲线

所有数据级错误都定位到文件与数据行号（表头为第 1 行）。
"""

from .csv_import import (
    DataImportError,
    ImportProblem,
    load_custom_turbine,
    load_wind_resource_csv,
    normalize_relative_path,
)

__all__ = [
    "DataImportError",
    "ImportProblem",
    "load_custom_turbine",
    "load_wind_resource_csv",
    "normalize_relative_path",
]
