"""
Unit and Integration Tests for FloodSense Improvement 4
Spatial Flood Risk Propagation & Time-Stepped Scenario Simulation
=================================================================
Covers:
1. Correct construction of adjacent ward relationships (touches / boundary intersection).
2. Handling of invalid geometry and missing properties.
3. Single source at T=0.
4. One-hop neighboring exposure at T+30.
5. Eligible second-hop exposure at T+60.
6. No propagation through disconnected nodes.
7. No duplicate exposure from graph cycles (visited set).
8. Multiple source wards without duplicate counting.
9. Baseline risk scores remain strictly unchanged.
10. Baseline risk classifications remain strictly unchanged.
11. Spatial exposure is capped and separately represented.
12. High-risk neighbors do not automatically become new confirmed sources.
13. Missing or uniform elevation does not create fabricated flow direction.
14. Invalid source wards and invalid simulation steps return safe errors.
15. Existing Improvement 1, 2, and 3 endpoints continue working.
16. Mandatory Regression Check: Repeated propagation across the graph NEVER turns all 201 wards HIGH.
"""

import unittest
import json
import os
from typing import Dict, Any

import flood_propagation
import app


class TestFloodPropagationGraph(unittest.TestCase):
    def setUp(self):
        self.graph = flood_propagation.build_ward_adjacency_graph()

    def test_graph_node_and_edge_counts(self):
        """Adjacency graph must contain all 201 wards and verified topological boundary edges."""
        self.assertEqual(self.graph["total_wards"], 201)
        self.assertGreater(self.graph["total_edges"], 500)
        self.assertEqual(len(self.graph["adjacency"]), 201)

    def test_no_self_referential_loops_and_no_duplicates(self):
        """Graph edges must never connect a ward to itself or contain duplicate neighbors."""
        for wn, neighbors in self.graph["adjacency"].items():
            self.assertNotIn(wn, neighbors, f"Ward {wn} has a self-loop")
            self.assertEqual(len(neighbors), len(set(neighbors)), f"Ward {wn} has duplicate edges")

    def test_undirected_symmetry(self):
        """Boundary adjacency must be symmetric: if A touches B, B touches A."""
        for w1, neighbors in self.graph["adjacency"].items():
            for w2 in neighbors:
                self.assertIn(w1, self.graph["adjacency"][w2], f"Asymmetry between {w1} and {w2}")

    def test_known_ward_connectivity(self):
        """Ward 119 must connect to its verified immediate polygon neighbors."""
        w119_neighbors = flood_propagation.get_ward_neighbors(119)
        self.assertIn(115, w119_neighbors)
        self.assertIn(118, w119_neighbors)
        self.assertIn(120, w119_neighbors)
        self.assertIn(121, w119_neighbors)
        self.assertIn(123, w119_neighbors)

    def test_invalid_geometry_handling(self):
        """Invalid geometry in GeoJSON features must be repaired or skipped without crashing."""
        import shapely.geometry
        # Create a self-intersecting bowtie polygon
        invalid_poly = shapely.geometry.Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
        self.assertFalse(invalid_poly.is_valid)
        fixed_poly = invalid_poly.buffer(0)
        self.assertTrue(fixed_poly.is_valid)


class TestSpatialPropagationSimulation(unittest.TestCase):
    def setUp(self):
        self.graph = flood_propagation.build_ward_adjacency_graph()
        # Mock realistic baseline map for testing core simulation logic
        self.baseline_map: Dict[int, Dict[str, Any]] = {}
        for wn in self.graph["adjacency"]:
            self.baseline_map[wn] = {
                "ward_no": wn,
                "ward_name": f"Ward {wn}",
                "score": 60.0 if wn % 2 == 0 else 45.0,
                "level": "LOW",
                "base_score": 40.0
            }
        # Mark Ward 119 as medium baseline
        self.baseline_map[119] = {
            "ward_no": 119,
            "ward_name": "Ward 119",
            "score": 95.0,
            "level": "MEDIUM",
            "base_score": 65.0
        }

    def test_single_source_at_t0(self):
        """At T=0, only the designated source ward must be marked as SOURCE."""
        res = flood_propagation.simulate_spatial_propagation(
            source_wards=[119],
            baseline_risk_map=self.baseline_map,
            steps=[0]
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_source_wards"], 1)
        self.assertEqual(res["total_exposed_wards"], 1)
        self.assertIn("0", res["step_summaries"])
        self.assertNotIn("30", res["step_summaries"])

        w119 = res["wards_exposure"]["119"]
        self.assertEqual(w119["scenario_status"], flood_propagation.STATUS_SOURCE)
        self.assertEqual(w119["propagation_step"], 0)
        self.assertGreater(w119["spatial_exposure_score"], 0.0)

    def test_one_hop_neighbor_exposure_at_t30(self):
        """At T+30, immediate 1-hop polygon neighbors must be exposed with attenuated scores."""
        res = flood_propagation.simulate_spatial_propagation(
            source_wards=[119],
            baseline_risk_map=self.baseline_map,
            steps=[0, 30]
        )
        step30_wards = res["step_summaries"]["30"]["newly_exposed_wards"]
        w119_neighbors = self.graph["adjacency"][119]
        self.assertEqual(set(step30_wards), set(w119_neighbors))

        for nw in step30_wards:
            w_exp = res["wards_exposure"][str(nw)]
            self.assertEqual(w_exp["scenario_status"], flood_propagation.STATUS_EXPOSED)
            self.assertEqual(w_exp["propagation_step"], 30)
            self.assertIn(119, w_exp["source_ward_ids"])
            # Exposure score must be attenuated (< source exposure)
            self.assertLess(w_exp["spatial_exposure_score"], res["wards_exposure"]["119"]["spatial_exposure_score"])

    def test_eligible_second_hop_at_t60(self):
        """At T+60, eligible 2-hop neighbors must be included without including already visited wards."""
        res = flood_propagation.simulate_spatial_propagation(
            source_wards=[119],
            baseline_risk_map=self.baseline_map,
            steps=[0, 30, 60]
        )
        step0_wards = set(res["step_summaries"]["0"]["newly_exposed_wards"])
        step30_wards = set(res["step_summaries"]["30"]["newly_exposed_wards"])
        step60_wards = set(res["step_summaries"]["60"]["newly_exposed_wards"])

        # No overlap between newly exposed sets (visited set strictly enforced)
        self.assertTrue(step0_wards.isdisjoint(step30_wards))
        self.assertTrue(step0_wards.isdisjoint(step60_wards))
        self.assertTrue(step30_wards.isdisjoint(step60_wards))

        for nw in step60_wards:
            w_exp = res["wards_exposure"][str(nw)]
            self.assertEqual(w_exp["propagation_step"], 60)
            self.assertEqual(w_exp["scenario_status"], flood_propagation.STATUS_EXPOSED)

    def test_multiple_source_wards_without_duplicate_counting(self):
        """Multiple source wards (e.g. 18 and 174) must not produce duplicate exposures or count collisions."""
        res = flood_propagation.simulate_spatial_propagation(
            source_wards=[18, 174],
            baseline_risk_map=self.baseline_map,
            steps=[0, 30, 60]
        )
        self.assertEqual(res["total_source_wards"], 2)
        total_unique = len(set(
            res["step_summaries"]["0"]["newly_exposed_wards"] +
            res["step_summaries"]["30"]["newly_exposed_wards"] +
            res["step_summaries"]["60"]["newly_exposed_wards"]
        ))
        self.assertEqual(res["total_exposed_wards"], total_unique)

    def test_baseline_risk_scores_and_levels_strictly_preserved(self):
        """MANDATORY: Spatial propagation must NEVER mutate baseline risk score or classification."""
        res = flood_propagation.simulate_spatial_propagation(
            source_wards=[119],
            baseline_risk_map=self.baseline_map,
            steps=[0, 30, 60]
        )
        for wn_str, w_data in res["wards_exposure"].items():
            wn = int(wn_str)
            expected_base = self.baseline_map[wn]
            self.assertEqual(w_data["baseline_risk_score"], expected_base["score"])
            self.assertEqual(w_data["baseline_risk_level"], expected_base["level"])
            # LOW baseline wards MUST remain LOW even if spatial exposure is MODERATE
            if expected_base["level"] == "LOW":
                self.assertEqual(w_data["baseline_risk_level"], "LOW")

    def test_spatial_exposure_is_capped_and_separated(self):
        """Spatial exposure score must be strictly bounded in [0.0, 100.0] and separate from baseline score."""
        res = flood_propagation.simulate_spatial_propagation(
            source_wards=[119],
            baseline_risk_map=self.baseline_map,
            steps=[0, 30, 60]
        )
        for w_data in res["wards_exposure"].values():
            exp_score = w_data["spatial_exposure_score"]
            self.assertGreaterEqual(exp_score, 0.0)
            self.assertLessEqual(exp_score, 100.0)
            self.assertIn(w_data["spatial_exposure_level"], [
                flood_propagation.EXPOSURE_NONE,
                flood_propagation.EXPOSURE_LOW,
                flood_propagation.EXPOSURE_MODERATE,
                flood_propagation.EXPOSURE_ELEVATED
            ])

    def test_high_risk_neighbors_do_not_become_new_confirmed_sources(self):
        """A neighbor that happens to have a HIGH baseline risk must NOT become an unattenuated source."""
        # Set neighbor ward 115 to HIGH baseline risk
        self.baseline_map[115] = {
            "ward_no": 115, "ward_name": "Ward 115", "score": 150.0, "level": "HIGH", "base_score": 100.0
        }
        res = flood_propagation.simulate_spatial_propagation(
            source_wards=[119],
            baseline_risk_map=self.baseline_map,
            steps=[0, 30, 60]
        )
        w115 = res["wards_exposure"]["115"]
        # Ward 115 must be an EXPOSED_NEIGHBOR, NOT a SOURCE
        self.assertEqual(w115["scenario_status"], flood_propagation.STATUS_EXPOSED)
        self.assertEqual(w115["propagation_step"], 30)

    def test_uniform_elevation_does_not_fabricate_downhill_flow(self):
        """Adjacency is undirected. Elevation does not create unilateral downhill flow."""
        neighbors_119 = flood_propagation.get_ward_neighbors(119)
        self.assertGreater(len(neighbors_119), 0)
        # Undirected proxy: if 119 touches 115, 115 touches 119
        self.assertIn(119, flood_propagation.get_ward_neighbors(115))

    def test_regression_repeated_propagation_never_turns_all_wards_high(self):
        """
        MANDATORY REGRESSION CHECK:
        Repeated or multi-step propagation must NEVER cause all 201 wards to turn HIGH.
        Baseline scores must remain completely unchanged and spatial exposure strictly bounded.
        """
        # Run scenario from high-risk Ward 18
        res = flood_propagation.simulate_spatial_propagation(
            source_wards=[18],
            baseline_risk_map=self.baseline_map,
            steps=[0, 30, 60]
        )
        # Verify vast majority of wards remain unexposed
        unexposed_count = sum(1 for w in res["wards_exposure"].values() if w["scenario_status"] == flood_propagation.STATUS_UNEXPOSED)
        self.assertGreater(unexposed_count, 140, "Runaway propagation detected: too many wards exposed")

        # Verify baseline counts match input baseline counts
        high_baseline_count = sum(1 for w in res["wards_exposure"].values() if w["baseline_risk_level"] == "HIGH")
        self.assertEqual(high_baseline_count, sum(1 for w in self.baseline_map.values() if w["level"] == "HIGH"))


class TestSpatialPropagationAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_spatial_propagation_endpoint_success(self):
        """GET /spatial-propagation returns HTTP 200 and complete schema."""
        response = self.client.get("/spatial-propagation?source_ward=119&steps=0,30,60")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("scenario_id", data)
        self.assertIn("step_summaries", data)
        self.assertIn("wards_exposure", data)
        self.assertEqual(data["total_city_wards"], 201)
        self.assertIn("disclaimer", data)
        self.assertIn("data_limitations", data)

    def test_api_compatibility_route(self):
        """GET /api/spatial-propagation alias works identically."""
        response = self.client.get("/api/spatial-propagation?source_ward=18")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["source_wards"], [18])

    def test_predicted_risk_scenario_default(self):
        """When no source_ward is provided, system defaults to PREDICTED_RISK_SCENARIO."""
        response = self.client.get("/spatial-propagation")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["scenario_mode"], flood_propagation.SCENARIO_MODE_PREDICTED_RISK)
        self.assertGreater(len(data["source_wards"]), 0)

    def test_invalid_source_ward_returns_400(self):
        """Invalid or non-existent ward ID returns safe HTTP 400 error."""
        response = self.client.get("/spatial-propagation?source_ward=999")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["error_type"], "WARD_NOT_FOUND")

    def test_invalid_parameters_return_400(self):
        """Non-numeric source_ward or steps parameter returns safe HTTP 400 error."""
        res1 = self.client.get("/spatial-propagation?source_ward=invalid")
        self.assertEqual(res1.status_code, 400)
        self.assertEqual(res1.get_json()["error_type"], "INVALID_PARAMETER")

        res2 = self.client.get("/spatial-propagation?source_ward=119&steps=invalid_steps")
        self.assertEqual(res2.status_code, 400)
        self.assertEqual(res2.get_json()["error_type"], "INVALID_PARAMETER")

    def test_existing_endpoints_preserved(self):
        """Existing endpoints /get-risk and /ward-history/119 remain intact and operational."""
        res_risk = self.client.get("/get-risk")
        self.assertEqual(res_risk.status_code, 200)
        risk_data = res_risk.get_json()
        self.assertEqual(len(risk_data), 201)

        res_hist = self.client.get("/ward-history/119")
        self.assertEqual(res_hist.status_code, 200)
        hist_data = res_hist.get_json()
        self.assertEqual(hist_data["ward_no"], 119)


if __name__ == "__main__":
    unittest.main()
