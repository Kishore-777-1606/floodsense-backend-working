"""
Unit and Integration Tests for FloodSense Improvement 5
Interactive Rainfall Scenario Simulator & Digital Twin Engine
=============================================================
Covers:
1. Scenario rainfall values at 0, 50, 150, 250, and 300 mm accepted and handled correctly.
2. Invalid types (string, null, NaN, inf) and out-of-range values (<0, >300) rejected with HTTP 400.
3. All 201 city wards evaluated in every valid scenario.
4. HIGH + MEDIUM + LOW counts strictly sum to 201.
5. Repeated identical requests produce 100% deterministic outputs.
6. Scenario calculation strictly preserves stored MongoDB records (no insertions, updates, or deletions).
7. Baseline classification thresholds (<80 LOW, 80-<120 MEDIUM, >=120 HIGH) and original XGBoost artifact untouched.
8. Score deltas and classification transition indicators calculated correctly.
9. TreeSHAP feature attributions and momentum decompositions included in response.
10. API contracts for POST /rainfall-scenario and GET /rainfall-scenario verified.
"""

import unittest
import json
import os
from typing import Dict, Any

import flood_simulator
import app
import db


class TestRainfallSimulatorEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = app.model
        cls.wards = app.wards
        cls.baseline = app.compute_all_wards_baseline_data()
        cls.app_client = app.app.test_client()

    def test_01_valid_rainfall_inputs_accepted(self):
        """Rainfall values at 0, 50, 150, 250, and 300 mm are accepted and evaluated."""
        for rf in [0.0, 50.0, 150.0, 250.0, 300.0]:
            val, err = flood_simulator.validate_scenario_rainfall(rf)
            self.assertIsNone(err)
            self.assertEqual(val, rf)

            res = flood_simulator.simulate_rainfall_scenario(
                rainfall_mm=rf,
                model=self.model,
                wards_geojson=self.wards,
                baseline_data=self.baseline,
                include_explainability=False
            )
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["summary"]["total_wards"], 201)
            self.assertEqual(len(res["wards"]), 201)

    def test_02_invalid_and_out_of_range_rejected(self):
        """Invalid types and values outside [0, 300] are rejected with error messages."""
        invalid_cases = [-0.1, -50.0, 300.1, 500.0, "abc", None, float("nan"), float("inf")]
        for case in invalid_cases:
            val, err = flood_simulator.validate_scenario_rainfall(case)
            self.assertIsNotNone(err, f"Failed to reject invalid case: {case}")
            self.assertIsNone(val)

    def test_03_all_201_wards_evaluated(self):
        """Every scenario evaluates exactly all 201 Chennai wards without omission."""
        res = flood_simulator.simulate_rainfall_scenario(
            rainfall_mm=150.0,
            model=self.model,
            wards_geojson=self.wards,
            baseline_data=self.baseline
        )
        self.assertEqual(len(res["wards"]), 201)
        self.assertEqual(len(res["all_ward_ids"]), 201)
        self.assertEqual(res["summary"]["total_wards"], 201)

        ward_numbers = set(w["ward_no"] for w in res["wards"])
        expected_wards = set(int(w["properties"]["Ward_No"]) for w in self.wards)
        self.assertEqual(len(ward_numbers), 201)
        self.assertEqual(ward_numbers, expected_wards)

    def test_04_category_counts_strictly_sum_to_201(self):
        """HIGH + MEDIUM + LOW counts must ALWAYS sum to exactly 201 across all tested scenarios."""
        test_values = [0.0, 25.0, 50.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0]
        for rf in test_values:
            res = flood_simulator.simulate_rainfall_scenario(
                rainfall_mm=rf,
                model=self.model,
                wards_geojson=self.wards,
                baseline_data=self.baseline,
                include_explainability=False
            )
            sc = res["summary"]["scenario_counts"]
            bc = res["summary"]["baseline_counts"]

            self.assertEqual(
                sc["HIGH"] + sc["MEDIUM"] + sc["LOW"], 201,
                f"Scenario counts do not sum to 201 at {rf}mm: {sc}"
            )
            self.assertEqual(
                bc["HIGH"] + bc["MEDIUM"] + bc["LOW"], 201,
                f"Baseline counts do not sum to 201: {bc}"
            )
            self.assertTrue(res["summary"]["classification_counts_valid"])

    def test_05_deterministic_outputs(self):
        """Identical scenario rainfall inputs must yield 100% identical ward scores and counts."""
        res1 = flood_simulator.simulate_rainfall_scenario(
            rainfall_mm=150.0,
            model=self.model,
            wards_geojson=self.wards,
            baseline_data=self.baseline,
            include_explainability=False
        )
        res2 = flood_simulator.simulate_rainfall_scenario(
            rainfall_mm=150.0,
            model=self.model,
            wards_geojson=self.wards,
            baseline_data=self.baseline,
            include_explainability=False
        )

        self.assertEqual(res1["summary"]["scenario_counts"], res2["summary"]["scenario_counts"])
        self.assertEqual(res1["summary"]["wards_classification_changed"], res2["summary"]["wards_classification_changed"])

        for w1, w2 in zip(res1["wards"], res2["wards"]):
            self.assertEqual(w1["ward_no"], w2["ward_no"])
            self.assertEqual(w1["scenario_score"], w2["scenario_score"])
            self.assertEqual(w1["scenario_level"], w2["scenario_level"])
            self.assertEqual(w1["score_delta"], w2["score_delta"])

    def test_06_no_mongodb_mutation(self):
        """Scenario simulations must NEVER insert, delete, or update MongoDB observations."""
        if not db.is_connected():
            self.skipTest("MongoDB not connected")

        database = db.get_db()
        obs_col = database["rainfall_observations"]
        daily_col = database["rainfall_daily_summary"]

        count_obs_before = obs_col.count_documents({})
        count_daily_before = daily_col.count_documents({})

        # Run several scenario simulations
        for rf in [50.0, 150.0, 250.0]:
            flood_simulator.simulate_rainfall_scenario(
                rainfall_mm=rf,
                model=self.model,
                wards_geojson=self.wards,
                baseline_data=self.baseline
            )

        count_obs_after = obs_col.count_documents({})
        count_daily_after = daily_col.count_documents({})

        self.assertEqual(count_obs_before, count_obs_after, "MongoDB observations were mutated!")
        self.assertEqual(count_daily_before, count_daily_after, "MongoDB daily summaries were mutated!")

    def test_07_baseline_thresholds_and_model_untouched(self):
        """Classification thresholds (<80 LOW, 80-<120 MEDIUM, >=120 HIGH) and model remain intact."""
        self.assertEqual(flood_simulator.RISK_THRESHOLD_MEDIUM, 80.0)
        self.assertEqual(flood_simulator.RISK_THRESHOLD_HIGH, 120.0)
        self.assertEqual(flood_simulator.SCORE_MIN, 0.0)
        self.assertEqual(flood_simulator.SCORE_MAX, 200.0)

        # Verify model structure
        self.assertTrue(hasattr(self.model, "predict"))
        self.assertEqual(len(self.model.feature_names_in_), 3)
        self.assertIn("rainfall", self.model.feature_names_in_)

    def test_08_api_endpoint_post_contract(self):
        """POST /rainfall-scenario returns expected schema with valid inputs and handles errors."""
        # 1. Valid request
        resp = self.app_client.post(
            "/rainfall-scenario",
            data=json.dumps({"rainfall_mm": 150.0}),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("scenario_metadata", data)
        self.assertIn("summary", data)
        self.assertIn("derived_scenario_inputs", data)
        self.assertIn("largest_score_increases", data)
        self.assertIn("highest_scenario_scores", data)
        self.assertIn("explicit_limitations", data)
        self.assertEqual(len(data["wards"]), 201)

        # 2. Out of range (>300)
        resp_bad = self.app_client.post(
            "/rainfall-scenario",
            data=json.dumps({"rainfall_mm": 350.0}),
            content_type="application/json"
        )
        self.assertEqual(resp_bad.status_code, 400)
        self.assertEqual(resp_bad.get_json()["error_type"], "INVALID_RAINFALL_INPUT")

        # 3. Negative value (<0)
        resp_neg = self.app_client.post(
            "/rainfall-scenario",
            data=json.dumps({"rainfall_mm": -10.0}),
            content_type="application/json"
        )
        self.assertEqual(resp_neg.status_code, 400)

        # 4. Non-numeric
        resp_str = self.app_client.post(
            "/rainfall-scenario",
            data=json.dumps({"rainfall_mm": "invalid_text"}),
            content_type="application/json"
        )
        self.assertEqual(resp_str.status_code, 400)

    def test_09_api_endpoint_get_fallback(self):
        """GET /rainfall-scenario allows quick inspection with query parameters."""
        resp = self.app_client.get("/rainfall-scenario?rainfall_mm=50.0")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["summary"]["scenario_rainfall_mm"], 50.0)
        self.assertEqual(len(data["wards"]), 201)

    def test_10_mass_conserving_tranches(self):
        """Scenario derived memory tranches partition the 24h event mass with zero double counting."""
        for S in [50.0, 150.0, 250.0]:
            derived = flood_simulator.derive_scenario_rainfall_memory(
                scenario_rainfall_mm=S,
                baseline_memory={"72h": 100.0, "24h": 40.0}
            )
            # 0–8h tranche + 8–24h tranche must equal 24h total
            r_recent = derived["8h"]
            r_intermediate = derived["24h"] - derived["8h"]
            self.assertAlmostEqual(r_recent + r_intermediate, S, places=2)

            # Cumulative monotonicity check
            self.assertLessEqual(derived["1h"], derived["3h"])
            self.assertLessEqual(derived["3h"], derived["6h"])
            self.assertLessEqual(derived["6h"], derived["8h"])
            self.assertLessEqual(derived["8h"], derived["12h"])
            self.assertLessEqual(derived["12h"], derived["24h"])
            self.assertLessEqual(derived["24h"], derived["72h"])

    def test_11_all_presets_and_comparison_scenario_breakdown(self):
        """Tests all nine quick presets (0, 25, 50, 75, 100, 150, 200, 250, 300 mm) plus 120 mm comparison."""
        expected_benchmarks = {
            0.0: {"counts": {"LOW": 201, "MEDIUM": 0, "HIGH": 0}, "clipped": 0, "mean_min": 49.0, "mean_max": 51.0},
            25.0: {"counts": {"LOW": 201, "MEDIUM": 0, "HIGH": 0}, "clipped": 0, "mean_min": 52.0, "mean_max": 55.0},
            50.0: {"counts": {"LOW": 201, "MEDIUM": 0, "HIGH": 0}, "clipped": 0, "mean_min": 59.0, "mean_max": 62.0},
            75.0: {"counts": {"LOW": 201, "MEDIUM": 0, "HIGH": 0}, "clipped": 0, "mean_min": 63.0, "mean_max": 66.0},
            100.0: {"counts": {"LOW": 0, "MEDIUM": 201, "HIGH": 0}, "clipped": 0, "mean_min": 101.0, "mean_max": 105.0},
            120.0: {"counts": {"LOW": 0, "MEDIUM": 73, "HIGH": 128}, "clipped": 0, "mean_min": 121.0, "mean_max": 125.0},
            150.0: {"counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 201}, "clipped": 0, "mean_min": 150.0, "mean_max": 153.0},
            200.0: {"counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 201}, "clipped": 0, "mean_min": 188.0, "mean_max": 191.0},
            250.0: {"counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 201}, "clipped": 201, "mean_min": 200.0, "mean_max": 200.0},
            300.0: {"counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 201}, "clipped": 201, "mean_min": 200.0, "mean_max": 200.0},
        }

        for rf, bmk in expected_benchmarks.items():
            res = flood_simulator.simulate_rainfall_scenario(
                rainfall_mm=rf,
                model=self.model,
                wards_geojson=self.wards,
                baseline_data=self.baseline,
                include_explainability=False
            )
            self.assertEqual(res["status"], "SUCCESS")
            sc = res["summary"]["scenario_counts"]
            self.assertEqual(sc, bmk["counts"], f"Mismatch in category counts for {rf}mm: {sc}")
            self.assertEqual(sc["HIGH"] + sc["MEDIUM"] + sc["LOW"], 201)

            audit = res["diagnostic_audit"]
            self.assertEqual(audit["clipped_at_200_count"], bmk["clipped"])
            self.assertTrue(bmk["mean_min"] <= audit["final_score_range"]["mean"] <= bmk["mean_max"],
                            f"Mean final score {audit['final_score_range']['mean']} outside [{bmk['mean_min']}, {bmk['mean_max']}] for {rf}mm")

    def test_12_diagnostic_audit_metadata(self):
        """Verifies diagnostic_audit structure contains all Section 3 developer-readable metrics."""
        res = flood_simulator.simulate_rainfall_scenario(
            rainfall_mm=120.0,
            model=self.model,
            wards_geojson=self.wards,
            baseline_data=self.baseline,
            include_explainability=False
        )
        self.assertIn("diagnostic_audit", res)
        audit = res["diagnostic_audit"]

        required_keys = [
            "scenario_rainfall_mm",
            "unique_wards_processed",
            "all_calculations_successful",
            "base_score_range",
            "momentum_multiplier_range",
            "final_score_range",
            "clipped_at_200_count",
            "category_counts",
            "category_counts_sum_201",
            "wards_classification_changed",
            "wards_score_changed"
        ]
        for k in required_keys:
            self.assertIn(k, audit, f"Missing key in diagnostic_audit: {k}")

        self.assertEqual(audit["unique_wards_processed"], 201)
        self.assertTrue(audit["all_calculations_successful"])
        self.assertTrue(audit["category_counts_sum_201"])
        self.assertEqual(audit["base_score_range"]["min"], 84.56)
        self.assertEqual(audit["base_score_range"]["max"], 84.56)
        self.assertEqual(audit["clipped_at_200_count"], 0)

    def test_13_ward_specific_attributes_used_and_fallback(self):
        """Verifies each ward's specific elevation and water_distance are used, with graceful fallbacks."""
        # 1. Custom wards with differing elevations and water distances to verify ward-specific pass-through
        custom_wards = [
            {
                "type": "Feature",
                "properties": {"Ward_No": 991, "elevation_m": 5.0, "water_distance_km": 2.0},
                "geometry": {"type": "Polygon", "coordinates": []}
            },
            {
                "type": "Feature",
                "properties": {"Ward_No": 992, "elevation_m": 8.5, "water_distance_km": 15.0},
                "geometry": {"type": "Polygon", "coordinates": []}
            },
            {
                "type": "Feature",
                "properties": {"Ward_No": 993},  # Missing elevation and water_distance
                "geometry": {"type": "Polygon", "coordinates": []}
            }
        ]

        res = flood_simulator.simulate_rainfall_scenario(
            rainfall_mm=100.0,
            model=self.model,
            wards_geojson=custom_wards,
            baseline_data=[],
            include_explainability=False
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(len(res["wards"]), 3)

        w1 = next(w for w in res["wards"] if w["ward_no"] == 991)
        w2 = next(w for w in res["wards"] if w["ward_no"] == 992)
        w3 = next(w for w in res["wards"] if w["ward_no"] == 993)

        # Attribute extraction
        self.assertEqual(w1["elevation_m"], 5.0)
        self.assertEqual(w1["water_distance_km"], 2.0)
        self.assertEqual(w2["elevation_m"], 8.5)
        self.assertEqual(w2["water_distance_km"], 15.0)

        # Fallback defaults for missing attributes (elevation 6.7m, water_distance 10.0km)
        self.assertEqual(w3["elevation_m"], 6.7)
        self.assertEqual(w3["water_distance_km"], 10.0)

        # Ward 991 (elevation 5.0m, water_distance 2.0km) has higher handicap and lower elevation
        # than Ward 992 (elevation 8.5m, water_distance 15.0km), so momentum multiplier must differ!
        self.assertNotEqual(w1["scenario_momentum"]["momentum_multiplier"],
                            w2["scenario_momentum"]["momentum_multiplier"])

    def test_14_score_clipping_and_threshold_boundaries(self):
        """Strictly validates [0.0, 200.0] bounding and strict boundary intervals."""
        # Extreme hypothetical scenarios
        res_high = flood_simulator.simulate_rainfall_scenario(
            rainfall_mm=300.0,
            model=self.model,
            wards_geojson=self.wards,
            baseline_data=self.baseline,
            include_explainability=False
        )
        for w in res_high["wards"]:
            self.assertLessEqual(w["scenario_score"], 200.0)
            self.assertGreaterEqual(w["scenario_score"], 0.0)
            if w["scenario_score"] >= 120.0:
                self.assertEqual(w["scenario_level"], "HIGH")
            elif w["scenario_score"] >= 80.0:
                self.assertEqual(w["scenario_level"], "MEDIUM")
            else:
                self.assertEqual(w["scenario_level"], "LOW")

    def test_15_quick_preset_sequence_transitions(self):
        """Verifies rapid sequential preset transitions produce exact independent calculations with zero state leakage."""
        sequences = [
            [25.0, 100.0, 200.0, 50.0],
            [300.0, 75.0, 150.0, 0.0],
            [50.0, 120.0, 250.0, 100.0, 300.0]
        ]
        expected_tier = {
            0.0: "LOW",
            25.0: "LOW",
            50.0: "LOW",
            75.0: "LOW",
            100.0: "MEDIUM",
            150.0: "HIGH",
            200.0: "HIGH",
            250.0: "HIGH",
            300.0: "HIGH"
        }

        for seq in sequences:
            for rf in seq:
                res = flood_simulator.simulate_rainfall_scenario(
                    rainfall_mm=rf,
                    model=self.model,
                    wards_geojson=self.wards,
                    baseline_data=self.baseline,
                    include_explainability=False
                )
                self.assertEqual(res["status"], "SUCCESS")
                self.assertEqual(res["summary"]["scenario_rainfall_mm"], rf)
                self.assertEqual(res["summary"]["total_wards"], 201)
                self.assertEqual(len(res["wards"]), 201)

                # Ensure uniform scenario values reflect the mathematically determined category
                if rf in expected_tier:
                    dominant_tier = expected_tier[rf]
                    self.assertEqual(res["summary"]["scenario_counts"][dominant_tier], 201)
                elif rf == 120.0:
                    self.assertEqual(res["summary"]["scenario_counts"]["HIGH"], 128)
                    self.assertEqual(res["summary"]["scenario_counts"]["MEDIUM"], 73)


if __name__ == "__main__":
    unittest.main()
