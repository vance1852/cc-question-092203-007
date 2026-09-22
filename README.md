# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 引用外部 CSV 数据

除内置风况与两种固定机型外，可以直接引用测风塔扇区表与厂商机组曲线。配置中的相对路径**按配置文件所在目录解析**（与运行时的工作目录无关），绝对路径原样使用。示例见 [`examples/`](examples/)。

### 外部风资源（不等宽扇区）

```json
{
  "wind_resource_type": "external",
  "wind_resource_params": {
    "csv_path": "wind_sectors.csv",
    "frequency_normalization": "strict"
  }
}
```

扇区表 CSV 要求：

- 分隔符自动识别（逗号、分号、Tab、竖线），支持 UTF-8 BOM、空行与 `#` 注释；表头支持中英文，如 `起始角/结束角/频率(%)/平均风速/k` 或 `start,end,frequency,mean_speed,k`。
- 角度约定 0°=正北、顺时针，可写以下任一种：`start`+`end`、`start`+`width`、`end`+`width`、`center`+`width`。起始角 ≥ 结束角即识别为**跨越 0°** 的扇区（如 345°→15°）；`end=360` 表示顺时针到达正北，不算跨越。
- 系统校验扇区**两两不重叠且无缺口**地覆盖整圈，允许不等宽。
- 频率列支持小数（合计 1）或百分比（表头含 `%` 或单元格带 `%` 后缀）。
  - `strict`（默认）：合计必须为 1（容差 ±0.002），微小闭合差自动校正到最大频率扇区；
  - `normalize`：按比例缩放到 1。
- Weibull 列可给 `k`+`c`、`mean_speed`+`k`（推导 c）、`k`+`c`（推导均值）、仅 `mean_speed`（取行业默认 k=2.0 推导 c）、或 `mean_speed`+`c`（反推 k）；三者同给时做一致性校验。

### 自定义机组（功率曲线 + 随风速变化的推力系数）

```json
{
  "turbine_model": "custom",
  "custom_turbine": {
    "name": "Demo-3.0MW-D130",
    "hub_height": 90.0,
    "rotor_diameter": 130.0,
    "curve_csv": "turbine_curve.csv",
    "power_unit": "auto"
  }
}
```

- `curve_csv` 为单文件（列：风速、功率、可选 Ct）；也可用 `power_curve_csv` + `thrust_curve_csv` 分文件提供。
- 功率单位按表头（`功率(MW)`/`power_kw`）或单元格后缀（`3.0 MW`）自动识别 kW/MW，统一规范化为 kW；也可用 `power_unit` 强制指定。
- 推力系数支持**随风速变化的 Ct 曲线**（线性插值，高于曲线末端按 0 处理切出停机）；无曲线时可用 `thrust_coefficient` 给恒定 Ct。
- 风速列必须**严格单调递增**，否则报错并指出具体数据行。

### 错误定位与结果溯源

数据问题（不可解析数值、角度重叠/缺口、频率不闭合、非单调风速、Ct 越界、未知单位等）一次性聚合报告，并定位到 `文件:行号`，例如：

```
外部数据导入失败：
  - /path/wind.csv: 风向角度 [180°, 200°) 未被任何扇区覆盖（存在 20° 缺口）
```

成功运行后，`results.json` 中新增 `data_sources` 段，记录每个外部文件的规范化来源摘要（扇区宽度、跨零行、频率输入与归一化策略、Weibull 推导行、功率单位、切入/额定/切出风速等）以及对文件原始字节计算的 **SHA-256 指纹**，保证可重复识别。内置风况与固定机型的配置方式保持完全兼容。
