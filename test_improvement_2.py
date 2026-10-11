"""
Test Suite for FloodSense Improvement 2.0 — Rainfall Memory & Flood Momentum
Verifies:
1. MongoDB connection & indexing
2. Timestamped rainfall storage (ISODate format, field types)
3. Controlled window calculations (1h, 3h, 6h, 12h, 24h, 72h)
4. Ward isolation (Ward 119 vs Ward 120)
5. Determinism across repeated executions
6. Flood momentum differentiation (Ward A vs Ward B with identical current rain)
7. Full 201-ward API regression & backward compatibility
"""

import unittest
from datetime import datetime, timedelta, timezone
import requests
import db
import rainfall_storage
import rainfall_memory


class TestImprovement2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure DB connected and initialized
        cls.db_connected = db.is_connected()
        assert cls.db_connected, "MongoDB must be connected for tests"
        db.init_db()

    def test_01_mongodb_connection_and_indexes(self):
        """Test MongoDB connectivity and index presence."""
        self.assertTrue(db.is_connected())
        database = db.get_db()
        obs_indexes = database["rainfall_observations"].index_information()
        daily_indexes = database["rainfall_daily_summary"].index_information()

        self.assertIn("idx_ward_timestamp", obs_indexes)
        self.assertIn("idx_ward_date_unique", daily_indexes)

    def test_02_timestamped_observation_insertion_and_retrieval(self):
        """Test that observations are stored with native datetime types and correct fields."""
        test_ward = 9999
        t0 = datetime(2026, 10, 9, 10, 0, 0, tzinfo=timezone.utc)
        
        # Clean up any leftover test docs
        obs_col = rainfall_storage.get_observations_collection()
        obs_col.delete_many({"ward_no": test_ward})

        doc_id = rainfall_storage.insert_observation(
            ward_no=test_ward,
            timestamp=t0,
            rainfall_mm=17.5,
            source="controlled_test_sensor"
        )
        self.assertIsNotNone(doc_id)

        # Retrieve and verify types
        doc = obs_col.find_one({"ward_no": test_ward})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["ward_no"], test_ward)
        self.assertEqual(doc["rainfall_mm"], 17.5)
        self.assertEqual(doc["source"], "controlled_test_sensor")
        self.assertIsInstance(doc["timestamp"], datetime)
        self.assertIsInstance(doc["created_at"], datetime)

        # Clean up
        obs_col.delete_many({"ward_no": test_ward})

    def test_03_controlled_window_calculations(self):
        """
        Inject controlled observations at exact timestamps:
        t0 - 30m:  5.0 mm (in 1h, 3h, 6h, 12h, 24h, 72h)
        t0 - 2h:   6.0 mm (in 3h, 6h, 12h, 24h, 72h)
        t0 - 5h:   7.0 mm (in 6h, 12h, 24h, 72h)
        t0 - 10h:  8.0 mm (in 12h, 24h, 72h)
        t0 - 20h:  9.0 mm (in 24h, 72h)
        t0 - 50h: 10.0 mm (in 72h)
        t0 - 80h: 99.0 mm (OUTSIDE 72h - must NOT be included)
        """
        test_ward = 9998
        t0 = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)
        obs_col = rainfall_storage.get_observations_collection()
        obs_col.delete_many({"ward_no": test_ward})

        controlled_data = [
            (timedelta(minutes=30), 5.0),
            (timedelta(hours=2), 6.0),
            (timedelta(hours=5), 7.0),
            (timedelta(hours=10), 8.0),
            (timedelta(hours=20), 9.0),
            (timedelta(hours=50), 10.0),
            (timedelta(hours=80), 99.0),
        ]
        for delta, mm in controlled_data:
            rainfall_storage.insert_observation(test_ward, t0 - delta, mm, "controlled")

        mem = rainfall_memory.calculate_ward_rainfall_memory(test_ward, ref_time=t0)

        # Expected:
        # 1h: 5.0
        # 3h: 5.0 + 6.0 = 11.0
        # 6h: 11.0 + 7.0 = 18.0
        # 12h: 18.0 + 8.0 = 26.0
        # 24h: 26.0 + 9.0 = 35.0
        # 72h: 35.0 + 10.0 = 45.0 (excluding 99.0)
        self.assertEqual(mem["1h"], 5.0)
        self.assertEqual(mem["3h"], 11.0)
        self.assertEqual(mem["6h"], 18.0)
        self.assertEqual(mem["12h"], 26.0)
        self.assertEqual(mem["24h"], 35.0)
        self.assertEqual(mem["72h"], 45.0)

        # Clean up
        obs_col.delete_many({"ward_no": test_ward})

    def test_04_ward_isolation(self):
        """Rainfall injected into Ward 9991 must not affect Ward 9992."""
        ward_a = 9991
        ward_b = 9992
        t0 = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)
        obs_col = rainfall_storage.get_observations_collection()
        obs_col.delete_many({"ward_no": {"$in": [ward_a, ward_b]}})

        # Insert 100mm into Ward A
        rainfall_storage.insert_observation(ward_a, t0 - timedelta(hours=2), 100.0, "ward_a_sensor")

        mem_a = rainfall_memory.calculate_ward_rainfall_memory(ward_a, ref_time=t0)
        mem_b = rainfall_memory.calculate_ward_rainfall_memory(ward_b, ref_time=t0)

        self.assertEqual(mem_a["3h"], 100.0)
        self.assertEqual(mem_b["3h"], 0.0)
        self.assertEqual(mem_b["72h"], 0.0)

        obs_col.delete_many({"ward_no": {"$in": [ward_a, ward_b]}})

    def test_05_determinism(self):
        """Repeated queries with identical stored data must return identical float values."""
        mem1 = rainfall_memory.calculate_ward_rainfall_memory(119)
        mem2 = rainfall_memory.calculate_ward_rainfall_memory(119)
        self.assertEqual(mem1, mem2)

        mom1 = rainfall_memory.calculate_flood_momentum(mem1, 6.7, 10.0)
        mom2 = rainfall_memory.calculate_flood_momentum(mem2, 6.7, 10.0)
        self.assertEqual(mom1, mom2)

    def test_06_momentum_differentiation_identical_current_rain(self):
        """
        Verify Requirement 6:
        Ward A (low accumulation) vs Ward B (high accumulation) with IDENTICAL current rainfall (12 mm),
        identical elevation (6.7m), and identical water distance (10.0 km).
        Ward B's flood momentum and risk score must be significantly higher.
        """
        mem_low = {"24h": 5.0, "72h": 10.0}
        mem_high = {"24h": 110.0, "72h": 190.0}

        mom_a = rainfall_memory.calculate_flood_momentum(mem_low, elevation_m=6.7, water_distance_km=10.0)
        mom_b = rainfall_memory.calculate_flood_momentum(mem_high, elevation_m=6.7, water_distance_km=10.0)

        self.assertLess(mom_a["momentum_multiplier"], mom_b["momentum_multiplier"])
        self.assertEqual(mom_a["momentum_level"], "LOW")
        self.assertIn(mom_b["momentum_level"], ["HIGH", "CRITICAL"])

        base_score = 45.0
        score_a = round(base_score * mom_a["momentum_multiplier"], 2)
        score_b = round(base_score * mom_b["momentum_multiplier"], 2)
        self.assertGreater(score_b, score_a)

    def test_07_api_regression_all_201_wards(self):
        """Verify that /get-risk returns all 201 wards with correct schema and no nulls."""
        resp = requests.get("http://127.0.0.1:5000/get-risk", timeout=10)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 201)

        ward_nos = set()
        for ward in data:
            self.assertIn("ward_no", ward)
            ward_nos.add(ward["ward_no"])
            self.assertIn("rainfall", ward)
            self.assertIn("rainfall_mm", ward)
            self.assertIn("elevation", ward)
            self.assertIn("water_distance", ward)
            self.assertIn("base_score", ward)
            self.assertIn("score", ward)
            self.assertIn("level", ward)
            self.assertIn(ward["level"], ["HIGH", "MEDIUM", "LOW"])
            self.assertIn("reason", ward)

            # Improvement 2.0 fields
            self.assertIn("rainfall_memory", ward)
            rmem = ward["rainfall_memory"]
            for win in ["1h", "3h", "6h", "12h", "24h", "72h"]:
                self.assertIn(win, rmem)
                self.assertIsInstance(rmem[win], (int, float))

            self.assertIn("flood_momentum", ward)
            fmom = ward["flood_momentum"]
            self.assertIn("momentum_multiplier", fmom)
            self.assertIn("momentum_level", fmom)
            self.assertIn("saturation_ratio", fmom)

        self.assertEqual(len(ward_nos), 201)


if __name__ == "__main__":
    unittest.main()
