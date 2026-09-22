"""外部 CSV 导入功能测试。

运行：python -m pytest tests/ -v
（无 pytest 时也可直接执行：python tests/test_io_import.py）
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wind_farm_opt.config import WindFarmConfig
from wind_farm_opt.io import (
    DataImportError,
    load_turbine_csv,
    load_wind_resource_csv,
)
from wind_farm_opt.io.csv_reader import resolve_relative_path
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.farm.aep import AEPCalculator


# ---------------------------------------------------------------------------
# 测风塔扇区表
# ---------------------------------------------------------------------------

class TestWindResourceLoader(unittest.TestCase):
    def _write(self, content: str, name: str = "sectors.csv") -> str:
        fd, path = tempfile.mkstemp(suffix=f"_{name}", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        self.addCleanup(os.remove, path)
        return path

    EQUAL_SECTORS = (
        "start,end,frequency,mean_speed,k,c\n"
        "0,90,0.25,8.0,2.0,9.05\n"
        "90,180,0.25,8.0,2.0,9.05\n"
        "180,270,0.25,8.0,2.0,9.05\n"
        "270,360,0.25,8.0,2.0,9.05\n"
    )

    def test_equal_sectors_basic(self):
        path = self._write(self.EQUAL_SECTORS)
        result = load_wind_resource_csv(path)
        wr = result.wind_resource
        self.assertEqual(wr.num_sectors, 4)
        np.testing.assert_allclose(wr.frequencies, 0.25)
        np.testing.assert_allclose(wr.sector_widths, 90.0)
        # c 给定时，平均风速由 Weibull 推导
        self.assertAlmostEqual(wr.sectors[0].mean_speed, 9.05 * 0.8862269, places=4)

    def test_unequal_widths_and_zero_crossing(self):
        content = (
            "start,end,frequency,mean_speed\n"
            "350,10,0.10,8.0\n"
            "10,350,0.90,9.0\n"
        )
        path = self._write(content)
        result = load_wind_resource_csv(path)
        wr = result.wind_resource
        widths = sorted(wr.sector_widths)
        np.testing.assert_allclose(widths, [20.0, 340.0])
        centers = sorted(wr.directions)
        self.assertAlmostEqual(centers[0], 0.0)      # 跨零度扇区中心
        self.assertAlmostEqual(centers[1], 180.0)
        self.assertEqual(result.fingerprint.details["zero_crossing_sectors"], [1])
        self.assertAlmostEqual(wr.frequencies.sum(), 1.0)

    def test_zero_crossing_via_360_end(self):
        # 330 -> 380（即 330 -> 20）也应识别为跨零度
        content = (
            "start,end,frequency,mean_speed\n"
            "330,380,0.1,8\n20,330,0.9,8\n"
        )
        path = self._write(content)
        result = load_wind_resource_csv(path)
        self.assertEqual(result.fingerprint.details["zero_crossing_sectors"], [1])

    def test_center_width_format(self):
        # [0,30) [30,90) [90,360)，宽度 30/60/270
        content = (
            "direction,width,freq,mean_speed,k\n"
            "15,30,0.1,8,2.1\n60,60,0.2,9,2.1\n225,270,0.7,8,2.1\n"
        )
        path = self._write(content)
        wr = load_wind_resource_csv(path).wind_resource
        np.testing.assert_allclose(sorted(wr.sector_widths), [30.0, 60.0, 270.0])

    def test_overlap_detected_with_rows(self):
        content = (
            "start,end,frequency,mean_speed\n"
            "0,100,0.3,8\n90,200,0.3,8\n200,360,0.4,8\n"
        )
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_wind_resource_csv(path)
        self.assertIn("重叠", str(cm.exception))
        self.assertEqual(cm.exception.data_row, 1)
        self.assertIn(path, str(cm.exception))

    def test_gap_detected(self):
        content = (
            "start,end,frequency,mean_speed\n"
            "0,90,0.4,8\n100,360,0.6,8\n"
        )
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_wind_resource_csv(path)
        self.assertIn("缺口", str(cm.exception))

    def test_gap_allowed_with_flag(self):
        content = (
            "start,end,frequency,mean_speed\n"
            "0,90,0.4,8\n100,360,0.6,8\n"
        )
        path = self._write(content)
        result = load_wind_resource_csv(path, allow_partial_coverage=True)
        self.assertLess(result.fingerprint.details["coverage_degrees"], 360.0)

    def test_frequency_percent_auto(self):
        content = (
            "start,end,frequency,mean_speed\n"
            + "\n".join(f"{a},{a+90},25,8" for a in range(0, 360, 90))
        )
        path = self._write(content)
        result = load_wind_resource_csv(path)
        np.testing.assert_allclose(result.wind_resource.frequencies, 0.25)
        self.assertEqual(
            result.fingerprint.details["frequency"]["action"],
            "percent_divided_by_100",
        )

    def test_frequency_arbitrary_weights_normalized(self):
        content = (
            "start,end,frequency,mean_speed\n"
            + "\n".join(f"{a},{a+90},{w},8"
                        for a, w in zip(range(0, 360, 90), [3, 1, 1, 1]))
        )
        path = self._write(content)
        result = load_wind_resource_csv(path)
        np.testing.assert_allclose(
            result.wind_resource.frequencies, [0.5, 1/6, 1/6, 1/6]
        )

    def test_frequency_strict_failure(self):
        content = (
            "start,end,frequency,mean_speed\n"
            + "\n".join(f"{a},{a+90},0.3,8" for a in range(0, 360, 90))
        )
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_wind_resource_csv(path, frequency_strategy="strict")
        self.assertIn("strict", str(cm.exception))

    def test_frequency_scale_hours(self):
        # 8760 小时分布
        content = (
            "start,end,frequency,mean_speed\n"
            + "\n".join(f"{a},{a+90},{h},8"
                        for a, h in zip(range(0, 360, 90),
                                        [2190, 2190, 2190, 2190]))
        )
        path = self._write(content)
        result = load_wind_resource_csv(
            path, frequency_strategy="scale", frequency_scale=8760.0
        )
        np.testing.assert_allclose(result.wind_resource.frequencies, 0.25)

    def test_mean_speed_derives_c(self):
        content = (
            "start,end,frequency,mean_speed\n"
            "0,180,0.5,9.0\n180,360,0.5,7.0\n"
        )
        path = self._write(content)
        wr = load_wind_resource_csv(path, default_k=2.0).wind_resource
        self.assertGreater(wr.sectors[0].weibull_c, wr.sectors[1].weibull_c)
        self.assertAlmostEqual(wr.sectors[0].mean_speed, 9.0)

    def test_missing_speed_columns(self):
        path = self._write("start,end,frequency\n0,90,0.5\n90,360,0.5\n")
        with self.assertRaises(DataImportError) as cm:
            load_wind_resource_csv(path)
        self.assertIn("平均风速", str(cm.exception))

    def test_negative_frequency_row_located(self):
        content = (
            "start,end,frequency,mean_speed\n"
            "0,90,0.5,8\n90,180,-0.2,8\n180,360,0.7,8\n"
        )
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_wind_resource_csv(path)
        self.assertEqual(cm.exception.data_row, 2)
        self.assertEqual(cm.exception.line, 3)

    def test_non_numeric_value_located(self):
        content = (
            "start,end,frequency,mean_speed\n0,90,0.5,abc\n90,360,0.5,8\n"
        )
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_wind_resource_csv(path)
        self.assertEqual(cm.exception.data_row, 1)
        self.assertEqual(cm.exception.column, "mean_speed")

    def test_missing_file(self):
        with self.assertRaises(DataImportError) as cm:
            load_wind_resource_csv("/nonexistent/sectors.csv")
        self.assertIn("不存在", str(cm.exception))

    def test_comments_and_blank_lines(self):
        content = (
            "# 测风塔导出\n\nstart,end,frequency,mean_speed  # 行内说明\n"
            "0,180,0.5,8\n\n; 另一种注释\n180,360,0.5,9\n"
        )
        path = self._write(content)
        wr = load_wind_resource_csv(path).wind_resource
        self.assertEqual(wr.num_sectors, 2)

    def test_chinese_headers(self):
        content = (
            "起始角度,终止角度,频率,平均风速\n"
            "0,180,0.5,8\n180,360,0.5,9\n"
        )
        path = self._write(content)
        wr = load_wind_resource_csv(path).wind_resource
        self.assertEqual(wr.num_sectors, 2)
        np.testing.assert_allclose(wr.mean_speeds, [8.0, 9.0])

    def test_fingerprint_stable(self):
        path = self._write(self.EQUAL_SECTORS)
        fp1 = load_wind_resource_csv(path).fingerprint.sha256
        fp2 = load_wind_resource_csv(path).fingerprint.sha256
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)


# ---------------------------------------------------------------------------
# 机组曲线表
# ---------------------------------------------------------------------------

class TestTurbineLoader(unittest.TestCase):
    def _write(self, content: str, name: str = "curve.csv") -> str:
        fd, path = tempfile.mkstemp(suffix=f"_{name}", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        self.addCleanup(os.remove, path)
        return path

    CURVE_MW = (
        "wind_speed,power_mw,ct\n"
        "0,0,0.0\n3,0,0.82\n4,0.15,0.84\n8,1.28,0.76\n"
        "12,3.45,0.31\n20,3.45,0.13\n25,3.45,0.08\n26,0,0\n"
    )

    def test_basic_mw_import(self):
        path = self._write(self.CURVE_MW)
        result = load_turbine_csv(
            path, name="T", hub_height=87, rotor_diameter=126
        )
        t = result.turbine
        # MW -> kW
        self.assertAlmostEqual(t.rated_power, 3450.0)
        self.assertAlmostEqual(t.cut_in_speed, 4.0)
        # Ct 曲线插值
        self.assertAlmostEqual(t.thrust_at(6.0), 0.80, places=5)
        self.assertAlmostEqual(t.thrust_at(30.0), 0.0)  # 右端点外延
        # 代表性 Ct = 曲线最大值
        self.assertAlmostEqual(t.thrust_coefficient, 0.84)

    def test_kw_import(self):
        content = (
            "speed,power_kw\n0,0\n4,150\n12,3450\n25,3450\n"
        )
        path = self._write(content)
        result = load_turbine_csv(
            path, name="T", hub_height=80, rotor_diameter=100,
            default_thrust_coefficient=0.8,
        )
        self.assertAlmostEqual(result.turbine.rated_power, 3450.0)
        self.assertIsNone(result.turbine.thrust_curve)

    def test_auto_unit_by_magnitude(self):
        content = "speed,power\n0,0\n12,3.45\n25,3.45\n"
        path = self._write(content)
        result = load_turbine_csv(
            path, name="T", hub_height=80, rotor_diameter=100,
            default_thrust_coefficient=0.8,
        )
        self.assertEqual(result.fingerprint.details["power"]["detected_unit"], "mw")
        self.assertAlmostEqual(result.turbine.rated_power, 3450.0)

    def test_auto_unit_kw_by_magnitude(self):
        content = "speed,power\n0,0\n12,2000\n25,2000\n"
        path = self._write(content)
        result = load_turbine_csv(
            path, name="T", hub_height=80, rotor_diameter=100,
            default_thrust_coefficient=0.8,
        )
        self.assertEqual(result.fingerprint.details["power"]["detected_unit"], "kw")

    def test_non_monotonic_speed_located(self):
        content = (
            "speed,power_mw,ct\n0,0,0\n5,1,0.8\n4,0.5,0.8\n10,2,0.8\n"
        )
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_turbine_csv(path, name="T", hub_height=80, rotor_diameter=100)
        self.assertEqual(cm.exception.data_row, 3)
        self.assertIn("单调", str(cm.exception))

    def test_ct_out_of_range(self):
        content = "speed,power_mw,ct\n0,0,0\n5,1,1.2\n10,2,0.8\n"
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_turbine_csv(path, name="T", hub_height=80, rotor_diameter=100)
        self.assertEqual(cm.exception.data_row, 2)
        self.assertEqual(cm.exception.column, "ct")

    def test_ct_blank_in_column(self):
        content = "speed,power_mw,ct\n0,0,\n5,1,0.8\n10,2,0.8\n"
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_turbine_csv(path, name="T", hub_height=80, rotor_diameter=100)
        self.assertIn("空值", str(cm.exception))

    def test_missing_ct_without_default(self):
        content = "speed,power_mw\n0,0\n12,3.45\n"
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_turbine_csv(path, name="T", hub_height=80, rotor_diameter=100)
        self.assertIn("推力系数", str(cm.exception))

    def test_too_few_points(self):
        content = "speed,power_mw,ct\n10,2.0,0.8\n"
        path = self._write(content)
        with self.assertRaises(DataImportError):
            load_turbine_csv(path, name="T", hub_height=80, rotor_diameter=100)

    def test_ambiguous_power_columns(self):
        content = "speed,power_kw,power_mw\n0,0,0\n12,3450,3.45\n"
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_turbine_csv(path, name="T", hub_height=80, rotor_diameter=100)
        self.assertIn("歧义", str(cm.exception))

    def test_negative_power(self):
        content = "speed,power_mw,ct\n0,0,0\n5,-1,0.8\n10,2,0.8\n"
        path = self._write(content)
        with self.assertRaises(DataImportError) as cm:
            load_turbine_csv(path, name="T", hub_height=80, rotor_diameter=100)
        self.assertEqual(cm.exception.data_row, 2)


# ---------------------------------------------------------------------------
# 相对路径解析与配置接入
# ---------------------------------------------------------------------------

class TestConfigIntegration(unittest.TestCase):
    def test_resolve_relative_path(self):
        p = resolve_relative_path("/cfg/dir", "data/x.csv")
        self.assertEqual(str(p), os.path.join("/cfg/dir", "data", "x.csv"))
        self.assertEqual(
            str(resolve_relative_path("/cfg/dir", "/abs/x.csv")),
            os.path.abspath("/abs/x.csv"),
        )

    def _make_project(self):
        d = tempfile.mkdtemp()
        data_dir = os.path.join(d, "data")
        os.makedirs(data_dir)
        sectors = os.path.join(data_dir, "sectors.csv")
        with open(sectors, "w", encoding="utf-8") as f:
            # 扇区中心为 0/90/180/270（首扇区跨越零度）
            f.write(
                "start,end,frequency,mean_speed\n"
                "315,45,0.25,8.5\n45,135,0.25,8.5\n"
                "135,225,0.25,8.5\n225,315,0.25,8.5\n"
            )
        curve = os.path.join(data_dir, "curve.csv")
        with open(curve, "w", encoding="utf-8") as f:
            f.write(
                "wind_speed,power_kw,ct\n"
                "0,0,0\n4,150,0.82\n12,3450,0.3\n25,3450,0.08\n26,0,0\n"
            )
        cfg_path = os.path.join(d, "project.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "n_turbines": 4,
                "turbine_model": "custom",
                "turbine_params": {
                    "path": "data/curve.csv",
                    "name": "自定义",
                    "hub_height": 80,
                    "rotor_diameter": 100,
                },
                "wind_resource_type": "csv",
                "wind_resource_params": {"path": "data/sectors.csv"},
                "optimization": {"population_size": 6, "max_iterations": 2},
                "visualization": {"save_plots": False},
            }, f)
        return d, cfg_path

    def test_config_relative_paths_and_provenance(self):
        _d, cfg_path = self._make_project()
        config = WindFarmConfig.from_json(cfg_path)
        wr = config.create_wind_resource()
        turbines = config.create_turbines()
        self.assertEqual(wr.num_sectors, 4)
        self.assertIsNotNone(wr.provenance)
        self.assertTrue(
            wr.provenance.resolved_path.endswith(
                os.path.join("data", "sectors.csv")
            )
        )
        self.assertIsNotNone(turbines[0].provenance)
        self.assertAlmostEqual(turbines[0].rated_power, 3450.0)
        self.assertIsNotNone(turbines[0].thrust_curve)

    def test_imported_data_used_in_aep(self):
        _d, cfg_path = self._make_project()
        config = WindFarmConfig.from_json(cfg_path)
        wr = config.create_wind_resource()
        turbines = config.create_turbines()
        calc = AEPCalculator(turbines, wr, JensenWake(0.07), speed_step=1.0)
        positions = np.array([[0, 0], [0, 700], [0, 1400], [0, 2100]], dtype=float)
        result = calc.compute_farm_aep(positions)
        self.assertGreater(result.gross_aep, 0.0)
        self.assertGreater(result.total_wake_loss, 0.0)

    def test_builtin_config_unchanged(self):
        # 无 CSV 引用时 provenance 为 None
        config = WindFarmConfig()
        wr = config.create_wind_resource()
        turbines = config.create_turbines()
        self.assertIsNone(wr.provenance)
        self.assertIsNone(turbines[0].provenance)
        self.assertIsNone(turbines[0].thrust_curve)


if __name__ == "__main__":
    unittest.main(verbosity=2)
