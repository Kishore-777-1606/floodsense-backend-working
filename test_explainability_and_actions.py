"""
Unit Test Suite for FloodSense Improvement 3 (Stage 1)
- flood_explainability.py
- flood_actions.py
"""

import unittest
import os
import joblib
import numpy as np
import xgboost as xgb

from flood_explainability import (
    explain_xgboost_base_prediction,
    explain_momentum_adjustment,
    generate_ward_risk_explanation,
    FEATURE_NAMES,
    EXPECTED_BASE_VALUE,
    NUMERICAL_TOLERANCE
)
from flood_actions import (
    generate_ward_advisory_actions,
    get_available_action_catalog,
    CAT_BURST_RESPONSE,
    CAT_RECENT_RAINFALL,
    CAT_ANTECEDENT_MONITORING,
    CAT_DRAINAGE_INSPECTION,
    CAT_FIELD_VERIFICATION,
    CAT_ROUTINE_MAINTENANCE,
    CAT_DATA_VERIFICATION
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "flood_model.pkl")


class TestFloodExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assertTrue(os.path.exists(MODEL_PATH), f"Model artifact not found at {MODEL_PATH}")
        cls.model = joblib.load(MODEL_PATH)

    def test_feature_names_and_ordering(self):
        """Feature list must match exactly ['rainfall', 'elevation', 'water_distance']."""
        self.assertEqual(FEATURE_NAMES, ["rainfall", "elevation", "water_distance"])

    def test_treeshap_additivity_and_tolerance(self):
        """TreeSHAP base + contributions must equal predict() within NUMERICAL_TOLERANCE (1e-4)."""
        test_cases = [
            (0.0, 6.7, 5.0),
            (12.5, 6.7, 2.1),
            (45.0, 4.2, 0.8),
            (110.0, 8.5, 7.3),
            (250.0, 3.1, 0.4),
        ]
        for rain, elev, wdist in test_cases:
            res = explain_xgboost_base_prediction(self.model, rain, elev, wdist)
            self.assertEqual(res["status"], "SUCCESS")
            self.assertEqual(res["method"], "NATIVE_XGBOOST_TREESHAP")
            self.assertTrue(res["tolerance_verified"])
            self.assertLessEqual(res["reconstruction_diff"], NUMERICAL_TOLERANCE)

            # Reconstruct explicitly in test
            contribs = res["feature_contributions"]
            manual_sum = (
                res["expected_value"]
                + contribs["rainfall_points"]
                + contribs["elevation_points"]
                + contribs["water_distance_points"]
            )
            # 4 rounded terms can have up to 2e-4 discrepancy relative to rounded actual_base_score
            self.assertAlmostEqual(manual_sum, res["actual_base_score"], places=3)

    def test_multiple_ward_profiles(self):
        """Test varied rainfall profiles to confirm attribution direction and sensitivity."""
        # Note: Model trees have first split on rainfall above 75mm (0-75mm has flat -45.52 attribution)
        res_dry = explain_xgboost_base_prediction(self.model, 10.0, 6.7, 5.0)
        res_moderate = explain_xgboost_base_prediction(self.model, 95.0, 6.7, 5.0)
        res_heavy = explain_xgboost_base_prediction(self.model, 160.0, 6.7, 5.0)

        self.assertEqual(res_dry["status"], "SUCCESS")
        self.assertEqual(res_moderate["status"], "SUCCESS")
        self.assertEqual(res_heavy["status"], "SUCCESS")

        dry_rain_pts = res_dry["feature_contributions"]["rainfall_points"]
        mod_rain_pts = res_moderate["feature_contributions"]["rainfall_points"]
        heavy_rain_pts = res_heavy["feature_contributions"]["rainfall_points"]

        # Rainfall contribution strictly increases above model's split threshold
        self.assertLess(dry_rain_pts, mod_rain_pts)
        self.assertLess(mod_rain_pts, heavy_rain_pts)

        # Expected value is consistent
        self.assertAlmostEqual(res_dry["expected_value"], res_heavy["expected_value"], places=4)

    def test_separation_of_attribution_and_momentum(self):
        """Verify that TreeSHAP output points are distinct from momentum multiplier and disclaimed."""
        res = explain_xgboost_base_prediction(self.model, 30.0, 6.7, 2.0)
        self.assertEqual(res["units"], "model_score_points")
        self.assertIn("not probabilities", res["disclaimer"])
        self.assertNotIn("%", str(res["feature_contributions"]))

    def test_invalid_and_edge_case_inputs(self):
        """Invalid inputs must be safely caught without crashing."""
        # Negative rainfall
        r1 = explain_xgboost_base_prediction(self.model, -10.0, 6.7, 2.0)
        self.assertEqual(r1["status"], "ERROR")
        self.assertEqual(r1["error_type"], "INVALID_INPUT")

        # Negative elevation
        r2 = explain_xgboost_base_prediction(self.model, 10.0, -5.0, 2.0)
        self.assertEqual(r2["status"], "ERROR")
        self.assertEqual(r2["error_type"], "INVALID_INPUT")

        # Zero or negative water distance
        r3 = explain_xgboost_base_prediction(self.model, 10.0, 6.7, 0.0)
        self.assertEqual(r3["status"], "ERROR")
        self.assertEqual(r3["error_type"], "INVALID_INPUT")

        # NaN / Inf
        r4 = explain_xgboost_base_prediction(self.model, float('nan'), 6.7, 2.0)
        self.assertEqual(r4["status"], "ERROR")
        self.assertEqual(r4["error_type"], "INVALID_INPUT")

        # Non-numeric
        r5 = explain_xgboost_base_prediction(self.model, "invalid", 6.7, 2.0)
        self.assertEqual(r5["status"], "ERROR")
        self.assertEqual(r5["error_type"], "INVALID_INPUT")

    def test_missing_or_incompatible_model(self):
        """Passing None or an incompatible model must fail safely without fabricating importance."""
        r_none = explain_xgboost_base_prediction(None, 10.0, 6.7, 2.0)
        self.assertEqual(r_none["status"], "ERROR")
        self.assertEqual(r_none["error_type"], "INCOMPATIBLE_OR_MISSING_MODEL")

        r_fake = explain_xgboost_base_prediction({"not": "a model"}, 10.0, 6.7, 2.0)
        self.assertEqual(r_fake["status"], "ERROR")
        self.assertEqual(r_fake["error_type"], "INCOMPATIBLE_OR_MISSING_MODEL")

    def test_momentum_adjustment_decomposition(self):
        """Momentum decomposition must transparently isolate tranches, saturation proxy, and handicap."""
        rainfall_memory = {
            "1h": 5.0,
            "3h": 12.0,
            "8h": 22.0,
            "12h": 30.0,
            "24h": 45.0,
            "72h": 90.0
        }
        flood_momentum = {
            "momentum_multiplier": 1.25,
            "burst_category": "MODERATE_BURST",
            "active_rainfall_factor": 0.85,
            "antecedent_rainfall_mm": 40.2,
            "saturation_ratio": 0.335,
            "tranches": {
                "recent_0_8h_mm": 22.0,
                "intermediate_8_24h_mm": 23.0,
                "past_24_72h_mm": 45.0
            }
        }
        res = explain_momentum_adjustment(
            base_score=75.0,
            rainfall_memory=rainfall_memory,
            flood_momentum=flood_momentum,
            elevation_m=6.7,
            water_distance_km=3.2
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["base_score"], 75.0)
        self.assertEqual(res["momentum_multiplier"], 1.25)
        self.assertEqual(res["points_added_by_momentum"], 18.75)
        self.assertEqual(res["final_bounded_score"], 93.75)
        self.assertEqual(res["final_risk_level"], "MEDIUM")
        self.assertEqual(res["tranche_weights"]["recent_weight"], 1.0)
        self.assertEqual(res["tranche_weights"]["intermediate_weight"], 0.5)
        self.assertEqual(res["tranche_weights"]["past_weight"], 0.15)
        self.assertIn("Antecedent Precipitation Index (API) proxy", res["saturation_proxy"]["proxy_note"])

    def test_generate_ward_risk_explanation(self):
        """End-to-end ward explanation must integrate TreeSHAP, momentum, narrative, and unmeasured factors."""
        ward_props = {"rainfall_mm": 35.0, "elevation_m": 6.7, "water_distance_km": 1.5}
        rainfall_memory = {"1h": 15.0, "3h": 25.0, "8h": 35.0, "24h": 50.0, "72h": 80.0}
        flood_momentum = {
            "momentum_multiplier": 1.30,
            "burst_category": "HEAVY_BURST",
            "active_rainfall_factor": 1.15,
            "antecedent_rainfall_mm": 45.0,
            "saturation_ratio": 0.375,
            "tranches": {
                "recent_0_8h_mm": 35.0,
                "intermediate_8_24h_mm": 15.0,
                "past_24_72h_mm": 30.0
            }
        }

        full_exp = generate_ward_risk_explanation(
            ward_no=42,
            model=self.model,
            ward_props=ward_props,
            rainfall_memory=rainfall_memory,
            flood_momentum=flood_momentum
        )
        self.assertEqual(full_exp["status"], "SUCCESS")
        self.assertEqual(full_exp["ward_no"], 42)
        self.assertIn("risk_assessment", full_exp)
        self.assertIn("narrative_explanation", full_exp)
        self.assertIn("unmeasured_factors", full_exp)
        self.assertGreater(len(full_exp["unmeasured_factors"]), 0)
        self.assertIn("prototype_disclaimer", full_exp)


class TestFloodActions(unittest.TestCase):
    def setUp(self):
        self.base_rf_memory = {
            "1h": 0.0, "3h": 0.0, "8h": 0.0, "12h": 0.0, "24h": 0.0, "72h": 0.0
        }
        self.base_momentum = {
            "momentum_multiplier": 1.0,
            "burst_category": "LOW",
            "active_rainfall_factor": 0.35,
            "antecedent_rainfall_mm": 0.0,
            "saturation_ratio": 0.0,
            "tranches": {
                "recent_0_8h_mm": 0.0,
                "intermediate_8_24h_mm": 0.0,
                "past_24_72h_mm": 0.0
            }
        }

    def test_acute_burst_readiness_trigger(self):
        """Acute 1h burst (>= 25mm) must trigger BURST_RESPONSE_READINESS."""
        rf_mem = dict(self.base_rf_memory)
        rf_mem["1h"] = 35.0
        rf_mem["8h"] = 38.0
        mom = dict(self.base_momentum)
        mom["burst_category"] = "EXTREME_BURST"
        mom["active_rainfall_factor"] = 1.35

        res = generate_ward_advisory_actions(
            ward_no=18,
            risk_level="HIGH",
            final_score=135.0,
            base_score=100.0,
            rainfall_memory=rf_mem,
            flood_momentum=mom,
            elevation_m=6.7,
            water_distance_km=3.5
        )
        self.assertEqual(res["status"], "SUCCESS")
        action_ids = [a["action_id"] for a in res["actions"]]
        self.assertIn("ACT_BURST_RESPONSE_READINESS", action_ids)
        burst_action = next(a for a in res["actions"] if a["action_id"] == "ACT_BURST_RESPONSE_READINESS")
        self.assertEqual(burst_action["category"], CAT_BURST_RESPONSE)
        self.assertEqual(burst_action["priority"], "HIGH")
        self.assertIn("rainfall_1h_mm", burst_action["triggering_evidence"])

    def test_heavy_burst_1h_threshold_and_window_precision(self):
        """HEAVY_BURST triggered by 1h rainfall (e.g. 20.3mm) must report 15.0mm threshold and 1-hour window."""
        rf_mem = dict(self.base_rf_memory)
        rf_mem["1h"] = 20.3
        rf_mem["3h"] = 22.0
        rf_mem["8h"] = 68.8
        mom = dict(self.base_momentum)
        mom["burst_category"] = "HEAVY_BURST"
        mom["active_rainfall_factor"] = 1.0

        res = generate_ward_advisory_actions(
            ward_no=174,
            risk_level="HIGH",
            final_score=151.7,
            base_score=99.4,
            rainfall_memory=rf_mem,
            flood_momentum=mom,
            elevation_m=6.7,
            water_distance_km=12.06
        )
        self.assertEqual(res["status"], "SUCCESS")
        burst_act = next(a for a in res["actions"] if a["action_id"] == "ACT_BURST_RESPONSE_READINESS")
        ev = burst_act["triggering_evidence"]

        # Exact threshold and window
        self.assertEqual(ev["burst_threshold_mm"], 15.0)
        self.assertEqual(ev["trigger_window"], "1-hour")
        self.assertEqual(ev["trigger_type"], "HEAVY_1H_BURST")
        self.assertEqual(ev["burst_category"], "HEAVY_BURST")
        self.assertEqual(ev["rainfall_1h_mm"], 20.3)

        # Separation from active rain gate
        self.assertEqual(ev["active_rain_gate_threshold_mm"], 25.0)
        self.assertNotEqual(ev["burst_threshold_mm"], ev["active_rain_gate_threshold_mm"])
        self.assertEqual(ev["threshold_mm"], 15.0)  # Must NOT be 25.0

    def test_heavy_burst_3h_and_extreme_burst_thresholds(self):
        """Different burst categories must report their own specific triggering rules and thresholds."""
        # 1. Heavy burst triggered by 3h window (1h is low: 8mm, 3h is 32mm >= 30mm)
        rf_mem_3h = dict(self.base_rf_memory)
        rf_mem_3h["1h"] = 8.0
        rf_mem_3h["3h"] = 32.0
        mom_heavy_3h = dict(self.base_momentum)
        mom_heavy_3h["burst_category"] = "HEAVY_BURST"

        res_heavy_3h = generate_ward_advisory_actions(
            ward_no=50,
            risk_level="MEDIUM",
            final_score=95.0,
            base_score=80.0,
            rainfall_memory=rf_mem_3h,
            flood_momentum=mom_heavy_3h,
            elevation_m=6.7,
            water_distance_km=5.0
        )
        act_heavy_3h = next(a for a in res_heavy_3h["actions"] if a["action_id"] == "ACT_BURST_RESPONSE_READINESS")
        ev_3h = act_heavy_3h["triggering_evidence"]
        self.assertEqual(ev_3h["burst_threshold_mm"], 30.0)
        self.assertEqual(ev_3h["trigger_window"], "3-hour")
        self.assertEqual(ev_3h["trigger_type"], "HEAVY_3H_BURST")

        # 2. Extreme burst triggered by 1h window (r_1h = 32.0mm >= 30mm)
        rf_mem_ext = dict(self.base_rf_memory)
        rf_mem_ext["1h"] = 32.0
        mom_ext = dict(self.base_momentum)
        mom_ext["burst_category"] = "EXTREME_BURST"

        res_ext = generate_ward_advisory_actions(
            ward_no=51,
            risk_level="HIGH",
            final_score=140.0,
            base_score=95.0,
            rainfall_memory=rf_mem_ext,
            flood_momentum=mom_ext,
            elevation_m=6.7,
            water_distance_km=5.0
        )
        act_ext = next(a for a in res_ext["actions"] if a["action_id"] == "ACT_BURST_RESPONSE_READINESS")
        ev_ext = act_ext["triggering_evidence"]
        self.assertEqual(ev_ext["burst_threshold_mm"], 30.0)
        self.assertEqual(ev_ext["trigger_window"], "1-hour")
        self.assertEqual(ev_ext["trigger_type"], "EXTREME_1H_BURST")

    def test_dry_recent_window_with_elevated_historical_rainfall(self):
        """Dry recent 8h window with high 72h accumulation must NOT produce emergency escalation."""
        rf_mem = dict(self.base_rf_memory)
        rf_mem["1h"] = 0.0
        rf_mem["3h"] = 0.0
        rf_mem["8h"] = 1.0  # Dry recent window
        rf_mem["24h"] = 10.0
        rf_mem["72h"] = 75.0  # High historical accumulation
        mom = dict(self.base_momentum)
        mom["burst_category"] = "LOW"
        mom["active_rainfall_factor"] = 0.35  # Attenuated
        mom["saturation_ratio"] = 0.45
        mom["tranches"] = {
            "recent_0_8h_mm": 1.0,
            "intermediate_8_24h_mm": 9.0,
            "past_24_72h_mm": 65.0
        }

        res = generate_ward_advisory_actions(
            ward_no=88,
            risk_level="LOW",
            final_score=68.0,
            base_score=65.0,
            rainfall_memory=rf_mem,
            flood_momentum=mom,
            elevation_m=6.7,
            water_distance_km=4.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        action_ids = [a["action_id"] for a in res["actions"]]

        # Must trigger recovery monitoring, NOT burst readiness or emergency surveillance
        self.assertIn("ACT_ANTECEDENT_MONITORING_RECOVERY", action_ids)
        self.assertNotIn("ACT_BURST_RESPONSE_READINESS", action_ids)
        self.assertNotIn("ACT_HIGH_RISK_FIELD_SURVEILLANCE", action_ids)

        recovery_action = next(a for a in res["actions"] if a["action_id"] == "ACT_ANTECEDENT_MONITORING_RECOVERY")
        self.assertEqual(recovery_action["priority"], "LOW")
        self.assertIn("Emergency deployment is not warranted", recovery_action["rationale"])

    def test_high_risk_field_surveillance(self):
        """Wards with HIGH risk (score >= 120) must trigger physical field verification."""
        res = generate_ward_advisory_actions(
            ward_no=119,
            risk_level="HIGH",
            final_score=142.5,
            base_score=110.0,
            rainfall_memory={"1h": 10.0, "8h": 30.0, "24h": 60.0, "72h": 90.0},
            flood_momentum=self.base_momentum,
            elevation_m=6.7,
            water_distance_km=3.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        action_ids = [a["action_id"] for a in res["actions"]]
        self.assertIn("ACT_HIGH_RISK_FIELD_SURVEILLANCE", action_ids)
        surv_action = next(a for a in res["actions"] if a["action_id"] == "ACT_HIGH_RISK_FIELD_SURVEILLANCE")
        self.assertEqual(surv_action["category"], CAT_FIELD_VERIFICATION)
        self.assertEqual(surv_action["priority"], "HIGH")

    def test_canal_proximity_action(self):
        """Wards near open water body / canal (<= 2.5 km) with moderate load must trigger drainage inspection."""
        res = generate_ward_advisory_actions(
            ward_no=5,
            risk_level="MEDIUM",
            final_score=95.0,
            base_score=85.0,
            rainfall_memory={"1h": 5.0, "8h": 20.0, "24h": 35.0, "72h": 40.0},
            flood_momentum=self.base_momentum,
            elevation_m=6.7,
            water_distance_km=1.2  # Close to canal
        )
        self.assertEqual(res["status"], "SUCCESS")
        action_ids = [a["action_id"] for a in res["actions"]]
        self.assertIn("ACT_CANAL_OUTFALL_INSPECTION", action_ids)
        canal_action = next(a for a in res["actions"] if a["action_id"] == "ACT_CANAL_OUTFALL_INSPECTION")
        self.assertEqual(canal_action["category"], CAT_DRAINAGE_INSPECTION)
        self.assertIn("1.20 km", canal_action["action_description"])
        self.assertIn("geographic attribute", canal_action["verification_note"])

    def test_routine_preventive_maintenance_for_low_risk(self):
        """LOW risk with dry conditions must trigger routine preventive maintenance."""
        res = generate_ward_advisory_actions(
            ward_no=2,
            risk_level="LOW",
            final_score=45.0,
            base_score=45.0,
            rainfall_memory=self.base_rf_memory,
            flood_momentum=self.base_momentum,
            elevation_m=6.7,
            water_distance_km=5.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        action_ids = [a["action_id"] for a in res["actions"]]
        self.assertIn("ACT_ROUTINE_DRAIN_MAINTENANCE", action_ids)
        maint_action = next(a for a in res["actions"] if a["action_id"] == "ACT_ROUTINE_DRAIN_MAINTENANCE")
        self.assertEqual(maint_action["category"], CAT_ROUTINE_MAINTENANCE)
        self.assertEqual(maint_action["priority"], "ROUTINE")

    def test_missing_observation_gap_action(self):
        """Observation count < 72 must trigger DATA_GAP_VERIFICATION."""
        res = generate_ward_advisory_actions(
            ward_no=14,
            risk_level="LOW",
            final_score=50.0,
            base_score=50.0,
            rainfall_memory=self.base_rf_memory,
            flood_momentum=self.base_momentum,
            elevation_m=6.7,
            water_distance_km=6.0,
            observation_count=48  # Incomplete data series
        )
        self.assertEqual(res["status"], "SUCCESS")
        action_ids = [a["action_id"] for a in res["actions"]]
        self.assertIn("ACT_DATA_OBSERVATION_AUDIT", action_ids)
        gap_action = next(a for a in res["actions"] if a["action_id"] == "ACT_DATA_OBSERVATION_AUDIT")
        self.assertEqual(gap_action["category"], CAT_DATA_VERIFICATION)
        self.assertEqual(gap_action["triggering_evidence"]["missing_hours"], 24)

    def test_determinism_and_no_duplication(self):
        """Identical inputs must yield identical actions without duplicated action IDs."""
        res1 = generate_ward_advisory_actions(
            ward_no=50,
            risk_level="HIGH",
            final_score=130.0,
            base_score=105.0,
            rainfall_memory={"1h": 30.0, "8h": 40.0, "24h": 60.0, "72h": 90.0},
            flood_momentum={"burst_category": "HEAVY_BURST", "active_rainfall_factor": 1.2, "saturation_ratio": 0.6},
            elevation_m=6.7,
            water_distance_km=1.8
        )
        res2 = generate_ward_advisory_actions(
            ward_no=50,
            risk_level="HIGH",
            final_score=130.0,
            base_score=105.0,
            rainfall_memory={"1h": 30.0, "8h": 40.0, "24h": 60.0, "72h": 90.0},
            flood_momentum={"burst_category": "HEAVY_BURST", "active_rainfall_factor": 1.2, "saturation_ratio": 0.6},
            elevation_m=6.7,
            water_distance_km=1.8
        )
        self.assertEqual(res1, res2)
        action_ids = [a["action_id"] for a in res1["actions"]]
        self.assertEqual(len(action_ids), len(set(action_ids)))

    def test_no_evacuation_or_fabricated_claims(self):
        """No action must command evacuation or assert unverified physical claims."""
        catalog = get_available_action_catalog()
        for item in catalog:
            text = f"{item['description']} {item.get('trigger_condition', '')}".lower()
            self.assertNotIn("evacuate", text)
            self.assertNotIn("evacuation", text)
            self.assertNotIn("mandatory displacement", text)



class TestAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_get_risk_schema_and_backward_compatibility(self):
        """Verify /get-risk returns 201 wards with legacy and new explainability fields."""
        response = self.client.get("/get-risk")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data), 201)

        # Inspect first ward
        w = data[0]
        # Legacy fields
        self.assertIn("ward_no", w)
        self.assertIn("rainfall", w)
        self.assertIn("elevation", w)
        self.assertIn("water_distance", w)
        self.assertIn("base_score", w)
        self.assertIn("score", w)
        self.assertIn("level", w)
        self.assertIn("reason", w)
        self.assertIn("rainfall_memory", w)
        self.assertIn("flood_momentum", w)

        # New Improvement 3 fields
        self.assertIn("explainability", w)
        self.assertIn("evidence_summary", w)
        self.assertIn("action_support", w)
        self.assertIn("data_limitations", w)

        # Deep validation of explainability
        exp = w["explainability"]
        self.assertEqual(exp["status"], "SUCCESS")
        self.assertIn("model_attribution", exp)
        self.assertIn("momentum_decomposition", exp)
        self.assertTrue(exp["reconstruction_verified"])

        # Deep validation of action support
        act = w["action_support"]
        self.assertEqual(act["status"], "SUCCESS")
        self.assertGreater(act["total_actions"], 0)
        self.assertIn("operational_disclaimer", act)

    def test_ward_history_endpoint(self):
        """Verify /ward-history/<ward_no> returns complete individual explanation and actions."""
        response = self.client.get("/ward-history/18")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["ward_no"], 18)
        self.assertEqual(data["explainability"]["status"], "SUCCESS")
        self.assertEqual(data["action_support"]["status"], "SUCCESS")
        self.assertGreater(len(data["action_support"]["actions"]), 0)

    def test_ward_18_cross_endpoint_consistency(self):
        """Verify Ward 18 has identical scores and levels across /get-risk and /ward-history/18."""
        res_get_risk = self.client.get("/get-risk")
        all_data = res_get_risk.get_json()
        w18_bulk = next(w for w in all_data if w["ward_no"] == 18)

        res_single = self.client.get("/ward-history/18")
        w18_single = res_single.get_json()

        self.assertEqual(w18_bulk["base_score"], w18_single["base_score"])
        self.assertEqual(w18_bulk["score"], w18_single["score"])
        self.assertEqual(w18_bulk["level"], w18_single["level"])
        self.assertEqual(w18_bulk["rainfall_memory"]["1h"], w18_single["rainfall_memory"]["1h"])
        self.assertEqual(
            w18_bulk["flood_momentum"]["momentum_multiplier"],
            w18_single["flood_momentum"]["momentum_multiplier"]
        )

    def test_ward_174_cross_endpoint_consistency_and_treeshap(self):
        """Verify Ward 174 features, TreeSHAP values, and actions match across endpoints."""
        res_get_risk = self.client.get("/get-risk")
        all_data = res_get_risk.get_json()
        w174_bulk = next(w for w in all_data if w["ward_no"] == 174)

        res_single = self.client.get("/ward-history/174")
        w174_single = res_single.get_json()

        # Score consistency
        self.assertEqual(w174_bulk["base_score"], 99.39)
        self.assertEqual(w174_bulk["score"], 151.67)
        self.assertEqual(w174_bulk["level"], "HIGH")
        self.assertEqual(w174_bulk["base_score"], w174_single["base_score"])
        self.assertEqual(w174_bulk["score"], w174_single["score"])

        # TreeSHAP verification
        attr = w174_bulk["explainability"]["model_attribution"]
        contribs = attr["feature_contributions"]
        self.assertAlmostEqual(contribs["rainfall_points"], 4.27, places=2)
        self.assertAlmostEqual(contribs["elevation_points"], -10.14, places=2)
        self.assertAlmostEqual(contribs["water_distance_points"], -4.00, places=2)
        self.assertAlmostEqual(attr["expected_value"], 109.26, places=2)
        self.assertLessEqual(attr["reconstruction_diff"], 1e-4)
        self.assertTrue(attr["tolerance_verified"])

        # Burst readiness action trigger clarity
        actions = w174_bulk["action_support"]["actions"]
        burst_act = next(a for a in actions if a["action_id"] == "ACT_BURST_RESPONSE_READINESS")
        ev = burst_act["triggering_evidence"]
        self.assertIn("trigger_criterion", ev)
        self.assertIn("HEAVY_BURST", ev["trigger_criterion"])
        self.assertEqual(ev["burst_threshold_mm"], 15.0)
        self.assertEqual(ev["trigger_window"], "1-hour")
        self.assertEqual(ev["trigger_type"], "HEAVY_1H_BURST")
        self.assertEqual(ev["active_rain_gate_threshold_mm"], 25.0)
        self.assertEqual(ev["threshold_mm"], 15.0)


if __name__ == "__main__":
    unittest.main()

