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

除内置风况与两种固定机型外，可以在配置中引用资源分析师手头的测风塔扇区表与厂商机组曲线表（CSV）。**CSV 相对路径一律按配置文件所在目录解析**，因此可以把配置与数据放在同一目录整体搬迁。

### 测风塔扇区表（不等宽扇区）

配置中设置 `wind_resource_type: "csv"`：

```json
"wind_resource_type": "csv",
"wind_resource_params": {
  "path": "wind_sectors.csv",
  "frequency_strategy": "auto",
  "default_weibull_k": 2.0
}
```

扇区表支持两种角度写法，二选一：

- 起始/终止角度列：`start,end`（中文表头 `起始角度,终止角度`），区间为左闭右开 `[start, end)`，允许**跨越零度**（如 `350 → 10`）；
- 中心/宽度列：`direction,width`（`风向,扇区宽度`），支持**不等宽**扇区。

其余列：

- 频率：`frequency/freq/频率`。`frequency_strategy` 支持 `auto`（自动识别比例、百分数、任意权重并归一化）、`normalize`、`scale`（配合 `frequency_scale`，如 8760 小时）、`strict`（要求和严格为 1）；
- 风速分布：给 `mean_speed`（平均风速）可由 `default_weibull_k` 推导 Weibull `c`；给 `weibull_k,weibull_c` 可反推平均风速；也可以三者都给（以 k/c 为准保证自洽）。

校验包括：跨零度识别、扇区**重叠**与**缺口**（缺口可用 `allow_partial_coverage: true` 放行）、频率归一化、非数值/负值等，错误均定位到**文件、物理行、数据行、列**。

### 自定义机组（功率曲线 + 随风速变化的推力系数）

配置中设置 `turbine_model: "custom"`：

```json
"turbine_model": "custom",
"turbine_params": {
  "path": "turbine_curve.csv",
  "name": "示例3.45MW机组",
  "hub_height": 87.0,
  "rotor_diameter": 126.0,
  "power_unit": "auto"
}
```

机组曲线表列为：`wind_speed`（风速，须**严格单调递增**）、`power/power_kw/power_mw/功率`（功率，自动识别 **kW/MW**，按表头或数值量级判别）、`ct/推力系数`（可选，随风速变化的推力系数曲线；缺省该列时用 `thrust_coefficient` 常数）。功率在内部统一换算为 kW。

完整示例见 `examples/custom_data_config.json`、`examples/wind_sectors.csv`、`examples/turbine_curve.csv`：

```bash
python -m wind_farm_opt --config examples/custom_data_config.json
```

### 来源摘要与指纹

成功运行后，`results.json` 中写入 `data_sources`：每个外部文件的引用路径、解析后的绝对路径、原始字节 **SHA-256 指纹**、行数，以及规范化摘要（识别的列、频率归一化动作、功率单位与换算、Ct 来源、跨零度扇区、覆盖角度等）。输出目录中的 `config.json` 快照会把相对路径转为绝对路径，保证结果可独立复现。AEP 与尾流计算直接使用导入的扇区、Weibull 参数、功率曲线与 Ct 曲线；不引用 CSV 时内置风况与 V126/V164 机型行为保持不变。

## 测试

```bash
python tests/test_io_import.py     # 外部 CSV 导入与错误定位测试
```

