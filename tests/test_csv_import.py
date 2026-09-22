"""外部 CSV 导入与 AEP 集成测试（使用标准库 unittest，无需额外依赖）。"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wind_farm_opt.core.turbine import Turbine, create_default_turbine
from wind_farm_opt.core.wind_resource import create_default_wind_resource
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.io.csv_import import (
    DataImportError,
    load_custom_turbine,
    load_wind_resource_csv,
    normalize_relative_path,
)
from wind_farm_opt.config import WindFarmConfig


def _write(tmp: str, name: str, content: str) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# 12 个等宽 30° 扇区、小数频率、合计 1.0
EQUAL_FRACTIONS = "\n".join(
    f"{start},{start + 30},{1 / 12:.10f},8.5,2.1"
    for start in range(0, 360, 30)
)
EQUAL_HEADER = "start,end,frequency,mean_speed,k"


class WindSectorImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _load(self, content, name="sectors.csv", **kw):
        path = _write(self.tmp, name, content)
        return load_wind_resource_csv(path, **kw)

    def test_equal_sectors_english_headers(self):
        wr = self._load(f"{EQUAL_HEADER}\n{EQUAL_FRACTIONS}\n")
        self.assertEqual(wr.num_sectors, 12)
        self.assertAlmostEqual(wr.frequencies.sum(), 1.0, places=6)
        self.assertFalse(wr.source["unequal_widths"])
        self.assertEqual(wr.source["crosses_zero_sectors"], [])
        self.assertTrue(wr.is_external)
        self.assertEqual(len(wr.source["fingerprint"]), 64)

    def test_cross_zero_and_unequal_widths(self):
        # 跨零扇区 345->15（30°）+ 15->345（330°），不等宽
        content = (
            "start,end,frequency,mean_speed,k\n"
            "345,15,0.2,8.0,2.1\n"
            "15,345,0.8,8.5,2.0\n"
        )
        wr = self._load(content)
        self.assertEqual(wr.num_sectors, 2)
        self.assertTrue(wr.source["unequal_widths"])
        self.assertEqual(wr.source["crosses_zero_sectors"], [2])
        cross = wr.sectors[0]
        self.assertTrue(cross.crosses_zero)
        self.assertAlmostEqual(cross.direction_width, 30.0)
        self.assertAlmostEqual(cross.direction_center, 0.0)

    def test_overlap_reported_with_rows(self):
        # 15->45 与 30->60 在 [30,45) 重叠
        content = (
            "start,end,frequency,mean_speed,k\n"
            "0,15,0.25,8,2\n"
            "15,45,0.25,8,2\n"
            "30,60,0.25,8,2\n"
            "60,360,0.25,8,2\n"
        )
        with self.assertRaises(DataImportError) as ctx:
            self._load(content)
        messages = [str(p) for p in ctx.exception.problems]
        self.assertTrue(any("重叠" in m and ":3:" in m for m in messages), messages)

    def test_gap_reported(self):
        # 缺少 30->60
        content = (
            "start,end,frequency,mean_speed,k\n"
            "0,30,0.5,8,2\n"
            "60,360,0.5,8,2\n"
        )
        with self.assertRaises(DataImportError) as ctx:
            self._load(content)
        self.assertTrue(
            any("缺口" in str(p) for p in ctx.exception.problems)
        )

    def test_percent_detection_and_strict_failure(self):
        # 合计 99%，百分比识别后为 0.99，超出 strict 容差
        content = (
            "起始角,结束角,频率(%),平均风速\n"
            "0,180,49,8\n"
            "180,360,50,8\n"
        )
        with self.assertRaises(DataImportError) as ctx:
            self._load(content)
        self.assertTrue(
            any("strict" in str(p) for p in ctx.exception.problems)
        )
        # normalize 策略应成功
        wr = self._load(content, frequency_normalization="normalize")
        self.assertAlmostEqual(wr.frequencies.sum(), 1.0)
        self.assertAlmostEqual(wr.frequencies[0], 49 / 99)
        self.assertEqual(wr.source["frequency_input"], "percent")

    def test_closure_correction_strict(self):
        # 合计 1.001，闭合差计入最大频率扇区
        body = "\n".join(
            (f"{s},{s + 30},{0.09 if s < 330 else 0.011:g},8,2")
            for s in range(0, 360, 30)
        )
        wr = self._load(f"start,end,frequency,mean_speed,k\n{body}\n")
        self.assertAlmostEqual(wr.frequencies.sum(), 1.0)
        self.assertNotEqual(wr.source["closure_correction"], 0.0)

    def test_default_k_and_derived_c(self):
        content = (
            "start,end,frequency,mean_speed\n"
            "0,180,0.5,8.0\n"
            "180,360,0.5,9.0\n"
        )
        wr = self._load(content)
        self.assertEqual(wr.source["default_k_rows"], [2, 3])
        for sec in wr.sectors:
            self.assertAlmostEqual(sec.weibull_k, 2.0)
            self.assertGreater(sec.weibull_c, 0.0)
            # 均值可由 k,c 还原
            mean = sec.weibull_c * float(
                __import__(
                    "wind_farm_opt.core.wind_resource",
                    fromlist=["_gamma_lanczos"],
                )._gamma_lanczos(np.asarray(1.0 + 1.0 / sec.weibull_k))
            )
            self.assertAlmostEqual(mean, sec.mean_speed, places=5)

    def test_kc_derive_mean(self):
        content = (
            "start,end,frequency,k,c\n"
            "0,180,0.5,2.0,9.0\n"
            "180,360,0.5,2.5,10.0\n"
        )
        wr = self._load(content)
        self.assertEqual(wr.source["derived_mean_rows"], [2, 3])
        self.assertGreater(wr.sectors[0].mean_speed, 0.0)

    def test_mean_c_solve_k(self):
        # k=2.0 时 1+1/k=1.5 > 1.4616，位于 Gamma 上分支（唯一解）
        from wind_farm_opt.core.wind_resource import _gamma_lanczos

        mean = float(10.0 * _gamma_lanczos(np.asarray(1.0 + 1.0 / 2.0)))
        content = (
            "start,end,frequency,mean_speed,c\n"
            f"0,180,0.5,{mean:.6f},10.0\n"
            f"180,360,0.5,{mean:.6f},10.0\n"
        )
        wr = self._load(content)
        self.assertAlmostEqual(wr.sectors[0].weibull_k, 2.0, places=3)

    def test_mean_kc_inconsistency_error(self):
        content = (
            "start,end,frequency,mean_speed,k,c\n"
            "0,180,0.5,5.0,2.0,12.0\n"
            "180,360,0.5,5.0,2.0,12.0\n"
        )
        with self.assertRaises(DataImportError) as ctx:
            self._load(content)
        self.assertTrue(any("不一致" in str(p) for p in ctx.exception.problems))

    def test_center_width_forms(self):
        # 四个扇区宽度 30/60/90/180 连续铺满整圈，首扇区跨零
        content = (
            "center,width,frequency,mean_speed,k\n"
            "0,30,0.08333333,8,2\n"
            "45,60,0.16666667,8,2\n"
            "120,90,0.25,8,2\n"
            "255,180,0.5,8,2\n"
        )
        wr = self._load(content)
        self.assertEqual(wr.num_sectors, 4)
        self.assertAlmostEqual(wr.sectors[0].direction_start, 345.0)
        self.assertAlmostEqual(wr.sectors[0].direction_end, 15.0)
        self.assertTrue(wr.sectors[0].crosses_zero)
        self.assertEqual(wr.source["crosses_zero_sectors"], [2])

    def test_row_located_parse_error(self):
        content = (
            "start,end,frequency,mean_speed,k\n"
            "0,30,0.5,8,2\n"
            "30,60,oops,8,2\n"
            + "\n".join(
                f"{s},{s + 30},{1 / 10:.4f},8,2"
                for s in range(60, 360, 30)
            )
        )
        with self.assertRaises(DataImportError) as ctx:
            self._load(content)
        self.assertTrue(
            any(p.line == 3 and "无法解析" in str(p) for p in ctx.exception.problems)
        )

    def test_missing_file(self):
        with self.assertRaises(DataImportError):
            load_wind_resource_csv(os.path.join(self.tmp, "nope.csv"))

    def test_relative_path_resolved_from_base_dir(self):
        sub = os.path.join(self.tmp, "sub")
        os.makedirs(sub)
        _write(self.tmp, "sectors.csv", f"{EQUAL_HEADER}\n{EQUAL_FRACTIONS}\n")
        cfg = {
            "wind_resource_type": "external",
            "wind_resource_params": {"csv_path": "../sectors.csv"},
        }
        # 直接验证相对路径基准
        resolved = normalize_relative_path("../sectors.csv", sub)
        self.assertTrue(os.path.isfile(resolved))

    def test_end_angle_360_not_cross_zero(self):
        content = (
            "start,end,frequency,mean_speed,k\n"
            "0,180,0.5,8,2\n"
            "180,360,0.5,9,2\n"
        )
        wr = self._load(content)
        self.assertEqual(wr.source["crosses_zero_sectors"], [])
        self.assertFalse(wr.sectors[1].crosses_zero)

    def test_full_circle_single_sector(self):
        content = "start,end,frequency,mean_speed\n0,360,1,8\n"
        wr = self._load(content)
        self.assertEqual(wr.num_sectors, 1)
        self.assertAlmostEqual(wr.sectors[0].direction_width, 360.0)

    def test_semicolon_delimiter_cell_percent(self):
        content = (
            "起始角;结束角;频率;平均风速\n"
            "0;180;45%;8\n"
            "180;360;55%;8\n"
        )
        wr = self._load(content)
        self.assertEqual(wr.source["frequency_input"], "percent")
        self.assertAlmostEqual(wr.frequencies.sum(), 1.0)

    def test_multiple_errors_aggregated(self):
        content = (
            "start,end,frequency,mean_speed,k\n"
            "0,180,0.4,8,2\n"
            "180,200,0.6,abc,2\n"
        )
        with self.assertRaises(DataImportError) as ctx:
            self._load(content)
        self.assertGreaterEqual(len(ctx.exception.problems), 2)
        self.assertTrue(any(p.line == 3 for p in ctx.exception.problems))
        self.assertTrue(any("缺口" in str(p) for p in ctx.exception.problems))

    def test_unknown_frequency_strategy(self):
        path = _write(
            self.tmp,
            "s.csv",
            f"{EQUAL_HEADER}\n{EQUAL_FRACTIONS}\n",
        )
        with self.assertRaises(DataImportError):
            load_wind_resource_csv(path, frequency_normalization="bogus")


class TurbineImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _curve(self, unit_header="功率(MW)", rated=3.0):
        rows = ["风速(m/s)," + unit_header + ",Ct", "0,0,0"]
        for v in range(3, 14):
            p = rated * min(1.0, ((v - 3) / 10) ** 3)
            rows.append(f"{v},{p:.4f},{0.8 if v < 12 else 0.3:.2f}")
        for v in range(14, 26):
            rows.append(f"{v},{rated:.2f},0.25")
        rows.append("26,0,0")
        return "\n".join(rows) + "\n"

    def test_mw_header_and_ct_curve(self):
        path = _write(self.tmp, "curve.csv", self._curve())
        turb = load_custom_turbine(
            {
                "name": "T1",
                "hub_height": 90,
                "rotor_diameter": 130,
                "curve_csv": path,
            }
        )
        self.assertAlmostEqual(turb.rated_power, 3000.0)
        self.assertTrue(turb.has_thrust_curve)
        self.assertAlmostEqual(turb.thrust_coefficient_at(10.0), 0.8, places=6)
        self.assertAlmostEqual(turb.thrust_coefficient_at(20.0), 0.25, places=6)
        # 高于曲线末端按 0（切出停机）
        self.assertAlmostEqual(turb.thrust_coefficient_at(30.0), 0.0)
        self.assertEqual(turb.source["power_unit"], "mw")

    def test_kw_unit(self):
        content = self._curve(unit_header="power_kw", rated=3000.0)
        path = _write(self.tmp, "curve_kw.csv", content)
        turb = load_custom_turbine(
            {
                "name": "T2",
                "hub_height": 80,
                "rotor_diameter": 120,
                "curve_csv": path,
            }
        )
        self.assertAlmostEqual(turb.rated_power, 3000.0)
        self.assertEqual(turb.source["power_unit"], "kw")

    def test_cell_level_mw_suffix(self):
        content = (
            "speed,power,ct\n"
            "0,0 kW,0\n"
            "4,100 kW,0.8\n"
            "12,3000 kW,0.3\n"
            "25,3000 kW,0.2\n"
            "26,0,0\n"
        )
        path = _write(self.tmp, "suffix.csv", content)
        turb = load_custom_turbine(
            {"name": "T", "hub_height": 80, "rotor_diameter": 120, "curve_csv": path}
        )
        self.assertAlmostEqual(turb.rated_power, 3000.0)

    def test_non_monotonic_speed_error(self):
        content = (
            "speed,power,ct\n"
            "4,100,0.8\n"
            "6,300,0.8\n"
            "6,320,0.8\n"
            "12,3000,0.3\n"
        )
        path = _write(self.tmp, "bad.csv", content)
        with self.assertRaises(DataImportError) as ctx:
            load_custom_turbine(
                {"name": "T", "hub_height": 80, "rotor_diameter": 120, "curve_csv": path}
            )
        self.assertTrue(
            any(p.line == 4 and "单调递增" in p.message for p in ctx.exception.problems)
        )

    def test_ct_out_of_range(self):
        content = (
            "speed,power,ct\n"
            "4,100,1.2\n"
            "12,3000,0.3\n"
        )
        path = _write(self.tmp, "ctbad.csv", content)
        with self.assertRaises(DataImportError) as ctx:
            load_custom_turbine(
                {"name": "T", "hub_height": 80, "rotor_diameter": 120, "curve_csv": path}
            )
        self.assertTrue(any("Ct" in str(p) and ":2:" in str(p)
                            for p in ctx.exception.problems))

    def test_constant_ct_fallback(self):
        content = (
            "speed,power\n"
            "0,0\n"
            "4,100\n"
            "12,3000\n"
            "25,3000\n"
            "26,0\n"
        )
        path = _write(self.tmp, "noct.csv", content)
        turb = load_custom_turbine(
            {
                "name": "T",
                "hub_height": 80,
                "rotor_diameter": 120,
                "curve_csv": path,
                "thrust_coefficient": 0.81,
            }
        )
        self.assertFalse(turb.has_thrust_curve)
        self.assertAlmostEqual(turb.thrust_coefficient_at(9.0), 0.81)

    def test_split_power_and_thrust_files(self):
        power = "speed,power_MW\n0,0\n4,0.1\n12,3.0\n25,3.0\n26,0\n"
        thrust = "v,ct\n3,0.82\n10,0.8\n14,0.3\n25,0.2\n"
        pp = _write(self.tmp, "p.csv", power)
        tp = _write(self.tmp, "ct.csv", thrust)
        turb = load_custom_turbine(
            {
                "name": "T",
                "hub_height": 90,
                "rotor_diameter": 130,
                "power_curve_csv": pp,
                "thrust_curve_csv": tp,
            }
        )
        self.assertTrue(turb.has_thrust_curve)
        self.assertEqual(len(turb.source["files"]), 2)

    def test_missing_required_fields(self):
        with self.assertRaises(DataImportError):
            load_custom_turbine({"hub_height": 90})


class AEPIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _external_setup(self, ct_at_10=0.82):
        sectors = (
            "start,end,frequency(%),mean_speed,k\n"
            "345,15,10,7.5,2.1\n"
            "15,45,5,6.8,2.0\n"
            "45,135,10,7.2,2.0\n"
            "135,225,15,8.6,2.1\n"
            "225,315,40,9.5,2.2\n"
            "315,345,20,8.8,2.0\n"
        )
        sp = _write(self.tmp, "sectors.csv", sectors)
        rows = ["speed,power_kW,ct", "0,0,0"]
        for v in range(3, 13):
            rows.append(f"{v},{3000 * ((v - 3) / 9) ** 3:.2f},{ct_at_10 if v < 12 else 0.3}")
        for v in range(13, 26):
            rows.append(f"{v},3000,0.25")
        rows.append("26,0,0")
        tp = _write(self.tmp, "turbine.csv", "\n".join(rows) + "\n")
        wr = load_wind_resource_csv(sp)
        turb = load_custom_turbine(
            {"name": "T", "hub_height": 90, "rotor_diameter": 130, "curve_csv": tp}
        )
        return wr, turb

    def test_aep_runs_with_imported_data(self):
        wr, turb = self._external_setup()
        calc = AEPCalculator(
            turbines=[turb, turb],
            wind_resource=wr,
            wake_model=JensenWake(0.07),
            speed_step=1.0,
        )
        positions = np.array([[0.0, 0.0], [0.0, 800.0]])
        net = calc.evaluate_layout(positions)
        self.assertGreater(net, 0.0)
        result = calc.compute_farm_aep(positions)
        self.assertGreater(result.gross_aep, result.net_aep)
        self.assertAlmostEqual(result.total_installed_capacity, 6.0)

    def test_ct_curve_affects_wake_loss(self):
        """高 Ct 曲线的尾流损失应显著大于低 Ct 曲线。"""
        positions = np.array([[0.0, 0.0], [0.0, 650.0]])
        losses = {}
        for label, ct in (("high", 0.85), ("low", 0.2)):
            wr, turb = self._external_setup(ct_at_10=ct)
            calc = AEPCalculator(
                turbines=[turb, turb],
                wind_resource=wr,
                wake_model=JensenWake(0.07),
                speed_step=1.0,
            )
            r = calc.compute_farm_aep(positions)
            losses[label] = r.wake_loss_pct
        self.assertGreater(losses["high"], losses["low"])

    def test_config_external_relative_paths(self):
        sub = os.path.join(self.tmp, "run")
        os.makedirs(sub)
        sectors = (
            "起始角,结束角,频率,平均风速,k\n"
            + "\n".join(
                f"{s},{s + 30},{1 / 12:.10f},8.5,2.1"
                for s in range(0, 360, 30)
            )
        )
        _write(self.tmp, "sectors.csv", sectors)
        rows = "风速,功率(kW),Ct\n0,0,0\n4,100,0.8\n12,3000,0.3\n25,3000,0.2\n26,0,0\n"
        _write(self.tmp, "turbine.csv", rows)
        cfg_data = {
            "n_turbines": 2,
            "turbine_model": "custom",
            "custom_turbine": {
                "name": "T",
                "hub_height": 90,
                "rotor_diameter": 130,
                "curve_csv": "../turbine.csv",
            },
            "wind_resource_type": "external",
            "wind_resource_params": {"csv_path": "../sectors.csv"},
            "optimization": {"max_iterations": 1, "population_size": 2},
            "visualization": {"save_dir": self.tmp, "save_plots": False},
            "economic": {"enable_analysis": False},
        }
        cfg_path = os.path.join(sub, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f)
        cfg = WindFarmConfig.from_json(cfg_path)
        wr = cfg.create_wind_resource()
        turbines = cfg.create_turbines()
        self.assertEqual(wr.num_sectors, 12)
        self.assertEqual(len(turbines), 2)
        self.assertTrue(turbines[0].has_thrust_curve)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_builtin_resource_has_no_source(self):
        wr = create_default_wind_resource()
        self.assertFalse(wr.is_external)
        self.assertIsNone(wr.source_fingerprint)

    def test_builtin_turbines_still_work(self):
        for model in ("V126-3.45MW", "V164-9.5MW"):
            t = create_default_turbine(model)
            self.assertIsNone(t.thrust_curve)
            self.assertGreater(t.thrust_coefficient_at(10.0), 0)
            self.assertGreater(t.rated_power, 0)

    def test_default_config_unchanged(self):
        cfg = WindFarmConfig()
        wr = cfg.create_wind_resource()
        turbines = cfg.create_turbines()
        self.assertEqual(wr.num_sectors, 12)
        self.assertEqual(len(turbines), 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
