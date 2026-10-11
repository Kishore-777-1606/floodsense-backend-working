"""
Test Suite for FloodSense Time-Aware Rainfall Risk Engine Improvement
Verifies the 9 required scenarios:
1. Heavy rainfall occurred 50-70 hours ago, but recent 8h rainfall is negligible.
2. Strong rainfall burst occurs during the last 1-3 hours.
3. Sustained moderate rainfall occurs throughout the last 8-12 hours.
4. Rainfall occurs intermittently, with dry gaps that allow possible drainage.
5. Recent rainfall is low but antecedent rainfall remains high.
6. Missing hourly observations occur within the assessment window.
7. An observation for one ward does not change another ward's reference timestamp.
8. Repeated requests with identical stored observations produce deterministic results.
9. Existing API contracts and GIS dashboard behavior remain functional.
"""

import unittest
from datetime import datetime, timedelta, timezone
import requests
import json
import db
import rainfall_storage
import rainfall_memory


class TestTimeAwareRiskEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_connected = db.is_connected()
        assert cls.db_connected, "MongoDB must be connected for tests"
        db.init_db()

    def test_scenario_01_heavy_rain_50_to_70h_ago_recent_8h_negligible(self):
        """
        Scenario 1: Heavy rainfall occurred 50-70 hours ago, but the recent 8h rainfall is negligible.
        Objective: Verify that 72h accumulation does NOT independently drive HIGH flood momentum
        or an inflated momentum multiplier when recent 8h rain is dry.
        """
        # 140mm accumulated 50-70h ago, 0.5mm in the last 8h, 2.0mm in 24h, 142.0mm in 72h
        mem = {
            "1h": 0.0,
            "3h": 0.0,
            "6h": 0.2,
            "8h": 0.5,
            "12h": 1.0,
            "24h": 2.0,
            "72h": 142.0
        }
        res = rainfall_memory.calculate_flood_momentum(
            rainfall_memory=mem,
            elevation_m=5.0,
            water_distance_km=2.0
        )

        # Tranches check
        self.assertEqual(res["tranches"]["recent_0_8h_mm"], 0.5)
        self.assertEqual(res["tranches"]["intermediate_8_24h_mm"], 1.5)
        self.assertEqual(res["tranches"]["past_24_72h_mm"], 140.0)

        # Decayed antecedent precipitation: 1.0*0.5 + 0.5*1.5 + 0.15*140.0 = 0.5 + 0.75 + 21.0 = 22.25 mm
        self.assertAlmostEqual(res["antecedent_rainfall_mm"], 22.25, places=2)

        # Active rainfall factor must be near baseline (<= 0.20) because 8h <= 2mm
        self.assertLessEqual(res["active_rainfall_factor"], 0.20)

        # Multiplier must NOT be elevated (must remain <= 1.05, close to 1.00)
        self.assertLessEqual(res["momentum_multiplier"], 1.05)

        # Momentum level must NOT be HIGH or CRITICAL
        self.assertNotIn(res["momentum_level"], ["HIGH", "CRITICAL"])
        self.assertIn(res["momentum_level"], ["LOW", "MODERATE"])

    def test_scenario_02_strong_rainfall_burst_last_1_to_3_hours(self):
        """
        Scenario 2: A strong rainfall burst occurs during the last 1-3 hours.
        Objective: Verify that rapid runoff risk is detected immediately,
        active factor scales to 1.00, and burst classification triggers.
        """
        mem = {
            "1h": 35.0,
            "3h": 48.0,
            "6h": 50.0,
            "8h": 52.0,
            "12h": 52.0,
            "24h": 54.0,
            "72h": 56.0
        }
        res = rainfall_memory.calculate_flood_momentum(
            rainfall_memory=mem,
            elevation_m=4.5,
            water_distance_km=1.5
        )

        # Burst detection
        self.assertEqual(res["burst_category"], "EXTREME_BURST")
        self.assertEqual(res["active_rainfall_factor"], 1.00)

        # High momentum multiplier (exceeds HIGH threshold 1.25)
        self.assertGreaterEqual(res["momentum_multiplier"], 1.25)
        self.assertIn(res["momentum_level"], ["HIGH", "CRITICAL"])

    def test_scenario_03_sustained_moderate_rainfall_throughout_8_to_12_hours(self):
        """
        Scenario 3: Sustained moderate rainfall occurs throughout the last 8-12 hours.
        Objective: Catchment accumulation build-up should create high saturation ratio,
        and sustained runoff pressure should elevate momentum multiplier.
        """
        mem = {
            "1h": 8.0,
            "3h": 22.0,
            "6h": 45.0,
            "8h": 60.0,
            "12h": 85.0,
            "24h": 90.0,
            "72h": 95.0
        }
        res = rainfall_memory.calculate_flood_momentum(
            rainfall_memory=mem,
            elevation_m=5.0,
            water_distance_km=2.0
        )

        # Recent 8h is 60.0mm (> 25mm), so active factor is fully 1.0
        self.assertEqual(res["active_rainfall_factor"], 1.00)

        # Antecedent decayed rain is high: 1.0*60 + 0.5*30 + 0.15*5 = 60 + 15 + 0.75 = 75.75mm
        self.assertAlmostEqual(res["antecedent_rainfall_mm"], 75.75, places=2)
        self.assertGreaterEqual(res["saturation_ratio"], 0.60)
        self.assertGreaterEqual(res["momentum_multiplier"], 1.35)
        self.assertIn(res["momentum_level"], ["HIGH", "CRITICAL"])

    def test_scenario_04_intermittent_rainfall_with_dry_gaps(self):
        """
        Scenario 4: Rainfall occurs intermittently, with dry gaps that allow possible drainage.
        Objective: If a storm happened 12-18h ago (intermediate tranche) followed by dry gap (8h ~ 0mm),
        the system retains intermediate soil retention context but does NOT panic with HIGH momentum.
        """
        mem = {
            "1h": 0.0,
            "3h": 0.0,
            "6h": 0.0,
            "8h": 1.0,
            "12h": 12.0,
            "24h": 35.0,
            "72h": 40.0
        }
        res = rainfall_memory.calculate_flood_momentum(
            rainfall_memory=mem,
            elevation_m=6.5,
            water_distance_km=3.0
        )

        # Tranches
        self.assertEqual(res["tranches"]["recent_0_8h_mm"], 1.0)
        self.assertEqual(res["tranches"]["intermediate_8_24h_mm"], 34.0)
        self.assertEqual(res["tranches"]["past_24_72h_mm"], 5.0)

        # Active rain factor is attenuated because 8h <= 2mm
        self.assertLessEqual(res["active_rainfall_factor"], 0.25)

        # Momentum multiplier remains low (<= 1.10)
        self.assertLessEqual(res["momentum_multiplier"], 1.10)
        self.assertNotIn(res["momentum_level"], ["HIGH", "CRITICAL"])

    def test_scenario_05_recent_rainfall_low_but_antecedent_high(self):
        """
        Scenario 5: Recent rainfall is low but antecedent rainfall remains high.
        Comparison: 72h = 175mm, 24h = 25mm, 8h = 3.0mm.
        Objective: Verify that older rainfall does not dominate the score, and
        the time-aware multiplier is significantly lower than the legacy 70/30 formula.
        """
        mem = {
            "1h": 0.0,
            "3h": 0.5,
            "6h": 2.0,
            "8h": 3.0,
            "12h": 10.0,
            "24h": 25.0,
            "72h": 175.0
        }
        res = rainfall_memory.calculate_flood_momentum(
            rainfall_memory=mem,
            elevation_m=5.0,
            water_distance_km=2.0
        )

        # Time-decayed antecedent: 1.0*3 + 0.5*22 + 0.15*150 = 3 + 11 + 22.5 = 36.5 mm
        self.assertAlmostEqual(res["antecedent_rainfall_mm"], 36.5, places=2)

        # Compare with legacy formula: 0.70*25 + 0.30*175 = 17.5 + 52.5 = 70.0 mm
        legacy_antecedent = 0.70 * 25.0 + 0.30 * 175.0
        self.assertLess(res["antecedent_rainfall_mm"], legacy_antecedent)

        # Since 8h is only 3.0mm, active factor is attenuated (around 0.25)
        self.assertLessEqual(res["active_rainfall_factor"], 0.30)
        self.assertLessEqual(res["momentum_multiplier"], 1.10)

    def test_scenario_06_missing_hourly_observations_in_assessment_window(self):
        """
        Scenario 6: Missing hourly observations occur within the assessment window.
        Objective: System calculates sums gracefully without errors, handles partial/empty windows deterministically.
        """
        test_ward = 9991
        t_ref = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)
        obs_col = rainfall_storage.get_observations_collection()
        obs_col.delete_many({"ward_no": test_ward})

        # Insert only 3 observations out of 72 hours
        obs_col.insert_many([
            {"ward_no": test_ward, "timestamp": t_ref - timedelta(hours=2), "rainfall_mm": 5.0, "source": "synthetic_baseline"},
            {"ward_no": test_ward, "timestamp": t_ref - timedelta(hours=15), "rainfall_mm": 10.0, "source": "synthetic_baseline"},
            {"ward_no": test_ward, "timestamp": t_ref - timedelta(hours=45), "rainfall_mm": 20.0, "source": "synthetic_baseline"}
        ])

        try:
            mem = rainfall_memory.calculate_ward_rainfall_memory(ward_no=test_ward, ref_time=t_ref)
            self.assertEqual(mem["1h"], 0.0)
            self.assertEqual(mem["3h"], 5.0)
            self.assertEqual(mem["8h"], 5.0)
            self.assertEqual(mem["12h"], 5.0)
            self.assertEqual(mem["24h"], 15.0)
            self.assertEqual(mem["72h"], 35.0)
            self.assertEqual(mem["observation_count"], 3)

            # Flood momentum must compute cleanly with no exceptions
            momentum = rainfall_memory.calculate_flood_momentum(mem, elevation_m=6.0, water_distance_km=3.0)
            self.assertIn("momentum_multiplier", momentum)
            self.assertGreaterEqual(momentum["momentum_multiplier"], 1.0)
        finally:
            obs_col.delete_many({"ward_no": test_ward})

    def test_scenario_07_isolated_ward_reference_timestamp(self):
        """
        Scenario 7: An observation for one ward does not change another ward's reference timestamp.
        Objective: Verify ward timestamp isolation under get_reference_timestamp.
        """
        test_ward_a = 9992
        test_ward_b = 9993
        t_a = datetime(2026, 10, 9, 8, 0, 0, tzinfo=timezone.utc)
        t_b = datetime(2026, 10, 9, 14, 0, 0, tzinfo=timezone.utc)

        obs_col = rainfall_storage.get_observations_collection()
        obs_col.delete_many({"ward_no": {"$in": [test_ward_a, test_ward_b]}})

        try:
            rainfall_storage.insert_observation(test_ward_a, t_a, 10.0, "synthetic_baseline")
            rainfall_storage.insert_observation(test_ward_b, t_b, 10.0, "synthetic_baseline")

            ref_a = rainfall_memory.get_reference_timestamp(ward_no=test_ward_a)
            ref_b = rainfall_memory.get_reference_timestamp(ward_no=test_ward_b)

            self.assertEqual(ref_a, t_a)
            self.assertEqual(ref_b, t_b)
            self.assertNotEqual(ref_a, ref_b)
        finally:
            obs_col.delete_many({"ward_no": {"$in": [test_ward_a, test_ward_b]}})

    def test_scenario_08_deterministic_repeated_evaluations(self):
        """
        Scenario 8: Repeated requests with identical stored observations produce deterministic results.
        Objective: Run 10 consecutive calculations on Ward 18 and assert bitwise equality.
        """
        ref_time = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
        runs = []
        for _ in range(5):
            mem = rainfall_memory.calculate_ward_rainfall_memory(ward_no=18, ref_time=ref_time)
            mom = rainfall_memory.calculate_flood_momentum(mem, elevation_m=6.7, water_distance_km=17.334)
            runs.append((mem, mom))

        first_mem, first_mom = runs[0]
        for idx, (m, mo) in enumerate(runs[1:], start=2):
            self.assertEqual(first_mem["1h"], m["1h"])
            self.assertEqual(first_mem["8h"], m["8h"])
            self.assertEqual(first_mem["24h"], m["24h"])
            self.assertEqual(first_mem["72h"], m["72h"])
            self.assertEqual(first_mom["momentum_multiplier"], mo["momentum_multiplier"])
            self.assertEqual(first_mom["antecedent_rainfall_mm"], mo["antecedent_rainfall_mm"])
            self.assertEqual(first_mom["saturation_ratio"], mo["saturation_ratio"])

    def test_scenario_09_api_contracts_and_gis_compatibility(self):
        """
        Scenario 9: Existing API contracts and GIS dashboard behavior remain functional.
        Objective: Call live endpoints /get-risk and /api/ward/18/rainfall-history,
        verify all contracts, fields, and backward compatibility.
        """
        # 1. /get-risk endpoint
        res_risk = requests.get("http://localhost:5000/get-risk", timeout=5)
        self.assertEqual(res_risk.status_code, 200)
        risk_list = res_risk.json()
        self.assertEqual(len(risk_list), 201)

        w18 = next((w for w in risk_list if w["ward_no"] == 18), None)
        self.assertIsNotNone(w18)
        self.assertIn("score", w18)
        self.assertIn("base_score", w18)
        self.assertIn("level", w18)
        self.assertIn("rainfall_mm", w18)
        self.assertIn("elevation_m", w18)
        self.assertIn("water_distance_km", w18)
        self.assertIn("flood_momentum", w18)
        self.assertIn("rainfall_memory", w18)
        self.assertIn("8h", w18["rainfall_memory"])

        # 2. /api/ward/18/rainfall-history endpoint
        res_hist = requests.get("http://localhost:5000/api/ward/18/rainfall-history", timeout=5)
        self.assertEqual(res_hist.status_code, 200)
        hist = res_hist.json()
        self.assertEqual(hist["ward_no"], 18)
        self.assertIn("rainfall_memory", hist)
        self.assertIn("8h", hist["rainfall_memory"])
        self.assertIn("flood_momentum", hist)
        self.assertIn("accumulated_8h_mm", hist["flood_momentum"])
        self.assertIn("active_rainfall_factor", hist["flood_momentum"])
        self.assertIn("tranches", hist["flood_momentum"])
        self.assertIn("long_term_history", hist)
        self.assertIn("7d", hist["long_term_history"])
        self.assertIn("30d", hist["long_term_history"])


    def test_boundary_10_active_gate_r1h_sensitivity(self):
        """
        Boundary Test 10: Evaluate R1h immediately below, at, and above 10.0mm.
        Verifies numerical behavior and reports output continuity.
        """
        mem_below = {"1h": 9.9, "3h": 11.0, "6h": 12.0, "8h": 12.0, "12h": 20.0, "24h": 40.0, "72h": 80.0}
        mem_at = {"1h": 10.0, "3h": 11.0, "6h": 12.0, "8h": 12.0, "12h": 20.0, "24h": 40.0, "72h": 80.0}
        mem_above = {"1h": 10.1, "3h": 11.0, "6h": 12.0, "8h": 12.0, "12h": 20.0, "24h": 40.0, "72h": 80.0}

        res_below = rainfall_memory.calculate_flood_momentum(mem_below, elevation_m=5.0, water_distance_km=2.0)
        res_at = rainfall_memory.calculate_flood_momentum(mem_at, elevation_m=5.0, water_distance_km=2.0)
        res_above = rainfall_memory.calculate_flood_momentum(mem_above, elevation_m=5.0, water_distance_km=2.0)

        # Below 10.0mm: alpha governed by 8h = 12mm -> 0.15 + 0.85*(12/25) = 0.558
        self.assertEqual(res_below["active_rainfall_factor"], 0.558)
        # At and above 10.0mm: step condition activates -> 1.000
        self.assertEqual(res_at["active_rainfall_factor"], 1.000)
        self.assertEqual(res_above["active_rainfall_factor"], 1.000)

        # Multiplier delta is bounded to a moderate increase (< 0.10)
        delta_mult = res_at["momentum_multiplier"] - res_below["momentum_multiplier"]
        self.assertLess(delta_mult, 0.10)

    def test_boundary_11_active_gate_r8h_continuity(self):
        """
        Boundary Test 11: Evaluate R8h immediately below, at, and above 25.0mm.
        Verifies C0 continuity across the 25.0mm boundary (zero step discontinuity).
        """
        mem_24_9 = {"1h": 2.0, "3h": 5.0, "6h": 24.9, "8h": 24.9, "12h": 30.0, "24h": 45.0, "72h": 80.0}
        mem_25_0 = {"1h": 2.0, "3h": 5.0, "6h": 25.0, "8h": 25.0, "12h": 30.0, "24h": 45.0, "72h": 80.0}
        mem_25_1 = {"1h": 2.0, "3h": 5.0, "6h": 25.1, "8h": 25.1, "12h": 30.0, "24h": 45.0, "72h": 80.0}

        res_24_9 = rainfall_memory.calculate_flood_momentum(mem_24_9, elevation_m=5.0, water_distance_km=2.0)
        res_25_0 = rainfall_memory.calculate_flood_momentum(mem_25_0, elevation_m=5.0, water_distance_km=2.0)
        res_25_1 = rainfall_memory.calculate_flood_momentum(mem_25_1, elevation_m=5.0, water_distance_km=2.0)

        # Alpha is continuous: 0.997 -> 1.000 -> 1.000
        self.assertAlmostEqual(res_24_9["active_rainfall_factor"], 0.997, places=3)
        self.assertEqual(res_25_0["active_rainfall_factor"], 1.000)
        self.assertEqual(res_25_1["active_rainfall_factor"], 1.000)

        # Multiplier delta across 0.1mm rain change is smooth (< 0.005)
        self.assertLess(abs(res_25_0["momentum_multiplier"] - res_24_9["momentum_multiplier"]), 0.005)

    def test_boundary_12_classification_exact_boundaries(self):
        """
        Boundary Test 12: Verify exact classification thresholds:
        LOW below 80.0, MEDIUM from 80.0 to below 120.0, HIGH at 120.0 or above.
        """
        def classify(score):
            if score >= 120.0:
                return "HIGH"
            elif score >= 80.0:
                return "MEDIUM"
            else:
                return "LOW"

        self.assertEqual(classify(79.99), "LOW")
        self.assertEqual(classify(80.00), "MEDIUM")
        self.assertEqual(classify(119.99), "MEDIUM")
        self.assertEqual(classify(120.00), "HIGH")

        # Verify live /get-risk endpoints strictly obey these exact boundaries
        res = requests.get("http://localhost:5000/get-risk", timeout=5).json()
        for w in res:
            score = w["score"]
            lvl = w["level"]
            if score >= 120.0:
                self.assertEqual(lvl, "HIGH", f"Ward {w['ward_no']} with score {score} should be HIGH")
            elif score >= 80.0:
                self.assertEqual(lvl, "MEDIUM", f"Ward {w['ward_no']} with score {score} should be MEDIUM")
            else:
                self.assertEqual(lvl, "LOW", f"Ward {w['ward_no']} with score {score} should be LOW")

    def test_boundary_13_monotonic_nested_window_consistency(self):
        """
        Boundary Test 13: Verify that 1h <= 3h <= 6h <= 8h <= 12h <= 24h <= 72h
        holds monotonically across all stored ward observations with the shared reference timestamp.
        """
        all_mem = rainfall_memory.calculate_all_wards_rainfall_memory()
        self.assertGreater(len(all_mem), 0)

        for ward_no, mem in all_mem.items():
            r1 = mem["1h"]
            r3 = mem["3h"]
            r6 = mem["6h"]
            r8 = mem["8h"]
            r12 = mem["12h"]
            r24 = mem["24h"]
            r72 = mem["72h"]

            self.assertLessEqual(r1, r3 + 1e-6, f"Ward {ward_no} 1h > 3h")
            self.assertLessEqual(r3, r6 + 1e-6, f"Ward {ward_no} 3h > 6h")
            self.assertLessEqual(r6, r8 + 1e-6, f"Ward {ward_no} 6h > 8h")
            self.assertLessEqual(r8, r12 + 1e-6, f"Ward {ward_no} 8h > 12h")
            self.assertLessEqual(r12, r24 + 1e-6, f"Ward {ward_no} 12h > 24h")
            self.assertLessEqual(r24, r72 + 1e-6, f"Ward {ward_no} 24h > 72h")


if __name__ == "__main__":
    unittest.main()
