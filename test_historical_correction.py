"""
Comprehensive Test Suite for FloodSense Historical Rainfall Data Correction (Post-Audit)

Covers all 12 verification criteria specified in Post-Audit Section 10:
1. Deterministic synthetic baseline generation.
2. Correct synthetic source metadata.
3. Daily summaries matching the raw hourly observations used to generate them.
4. Correct total_rainfall_mm, max_hourly_mm, and observation_count.
5. Correct UTC calendar-day boundaries.
6. Exactly 7, 30, 90, 180, and 270 calendar-day windows when the corresponding dates are available.
7. Correct reporting when observations or daily summaries are missing.
8. Ward-specific reference timestamps.
9. A test observation in Ward 119 does not shift Ward 18's reference time.
10. Existing rainfall-memory and flood-momentum behavior.
11. Existing risk classification and GIS functionality.
12. API backward compatibility & response contract.
"""

import unittest
from datetime import datetime, timedelta, timezone
import requests
import json
import db
import rainfall_storage
import rainfall_memory


class TestHistoricalCorrection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_connected = db.is_connected()
        assert cls.db_connected, "MongoDB must be connected for tests"
        db.init_db()

    def test_01_deterministic_synthetic_baseline_generation(self):
        """1. Deterministic synthetic baseline generation produces identical, non-negative results."""
        sample_feature = [{
            "properties": {
                "Ward_No": 18,
                "rainfall_mm": 120.0,
                "elevation_m": 6.7,
                "water_distance_km": 2.5
            }
        }]
        t_ref = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)

        obs1, daily1 = rainfall_storage.generate_synthetic_baseline(sample_feature, base_ref_time=t_ref)
        obs2, daily2 = rainfall_storage.generate_synthetic_baseline(sample_feature, base_ref_time=t_ref)

        self.assertEqual(len(obs1), len(obs2))
        self.assertEqual(len(daily1), len(daily2))
        self.assertEqual(len(daily1), 270, "Should generate exactly 270 historical daily summaries")

        # Verify exact equality across all generated records
        for o1, o2 in zip(obs1, obs2):
            self.assertEqual(o1["timestamp"], o2["timestamp"])
            self.assertEqual(o1["rainfall_mm"], o2["rainfall_mm"])
            self.assertGreaterEqual(o1["rainfall_mm"], 0.0, "Rainfall must be non-negative")
            self.assertEqual(o1["source"], rainfall_storage.SYNTHETIC_SOURCE)

        for d1, d2 in zip(daily1, daily2):
            self.assertEqual(d1["date"], d2["date"])
            self.assertEqual(d1["total_rainfall_mm"], d2["total_rainfall_mm"])
            self.assertEqual(d1["max_hourly_mm"], d2["max_hourly_mm"])
            self.assertGreaterEqual(d1["total_rainfall_mm"], 0.0)
            self.assertEqual(d1["source"], rainfall_storage.SYNTHETIC_SOURCE)

    def test_02_correct_synthetic_source_metadata(self):
        """2. Correct synthetic source metadata (synthetic_baseline, no unverified tags)."""
        obs_col = rainfall_storage.get_observations_collection()
        daily_col = rainfall_storage.get_daily_summary_collection()

        # Check Ward 18 records
        sample_obs = obs_col.find_one({"ward_no": 18, "source": rainfall_storage.SYNTHETIC_SOURCE})
        self.assertIsNotNone(sample_obs)
        self.assertEqual(sample_obs["source"], "synthetic_baseline")

        sample_daily = daily_col.find_one({"ward_no": 18, "source": rainfall_storage.SYNTHETIC_SOURCE})
        self.assertIsNotNone(sample_daily)
        self.assertEqual(sample_daily["source"], "synthetic_baseline")

        # Zero unverified legacy tags
        self.assertEqual(obs_col.count_documents({"source": "tnsdma_imd_gauge"}), 0)
        self.assertEqual(daily_col.count_documents({"source": "tnsdma_imd_gauge"}), 0)
        self.assertEqual(daily_col.count_documents({"source": "daily_rollup_service"}), 0)

    def test_03_daily_summaries_matching_raw_hourly_observations(self):
        """3. Daily summaries strictly match the raw hourly observations used to generate them."""
        daily_col = rainfall_storage.get_daily_summary_collection()
        obs_col = rainfall_storage.get_observations_collection()

        test_wards = [18, 42, 105, 119]
        for ward in test_wards:
            sample_docs = list(daily_col.aggregate([
                {"$match": {"ward_no": ward, "source": "synthetic_baseline"}},
                {"$sample": {"size": 3}}
            ]))
            for d_doc in sample_docs:
                dt_start = d_doc["date"]
                if dt_start.tzinfo is None:
                    dt_start = dt_start.replace(tzinfo=timezone.utc)
                dt_end = dt_start + timedelta(days=1)

                hourly_obs = list(obs_col.find({
                    "ward_no": ward,
                    "timestamp": {"$gte": dt_start, "$lt": dt_end}
                }))
                raw_sum = round(sum(o.get("rainfall_mm", 0.0) for o in hourly_obs), 2)
                self.assertAlmostEqual(
                    d_doc["total_rainfall_mm"],
                    raw_sum,
                    places=2,
                    msg=f"Mismatch for Ward {ward} on {dt_start}: daily={d_doc['total_rainfall_mm']} vs raw={raw_sum}"
                )

    def test_04_correct_totals_max_hourly_and_observation_count(self):
        """4. Correct total_rainfall_mm, max_hourly_mm, and observation_count."""
        obs_col = rainfall_storage.get_observations_collection()
        daily_col = rainfall_storage.get_daily_summary_collection()

        dt_start = datetime(2026, 10, 5, 0, 0, 0, tzinfo=timezone.utc)
        dt_end = dt_start + timedelta(days=1)

        hourly_obs = list(obs_col.find({
            "ward_no": 18,
            "timestamp": {"$gte": dt_start, "$lt": dt_end}
        }))
        self.assertEqual(len(hourly_obs), 24, "Full day must have exactly 24 observations")

        daily_doc = daily_col.find_one({"ward_no": 18, "date": dt_start})
        self.assertIsNotNone(daily_doc)
        self.assertEqual(daily_doc["observation_count"], 24)

        expected_max = round(max(o["rainfall_mm"] for o in hourly_obs), 2)
        expected_total = round(sum(o["rainfall_mm"] for o in hourly_obs), 2)

        self.assertAlmostEqual(daily_doc["max_hourly_mm"], expected_max, places=2)
        self.assertAlmostEqual(daily_doc["total_rainfall_mm"], expected_total, places=2)

    def test_05_correct_utc_calendar_day_boundaries(self):
        """5. Correct UTC calendar-day boundaries in update_daily_summary_for_date."""
        test_ward = 9993
        obs_col = rainfall_storage.get_observations_collection()
        daily_col = rainfall_storage.get_daily_summary_collection()

        obs_col.delete_many({"ward_no": test_ward})
        daily_col.delete_many({"ward_no": test_ward})

        d_target = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

        # Insert 1 observation just before day start (23:59 previous day)
        rainfall_storage.insert_observation(test_ward, d_target - timedelta(minutes=1), 50.0, "edge_test")
        # Insert 1 observation inside day at 00:00:00
        rainfall_storage.insert_observation(test_ward, d_target, 12.0, "edge_test")
        # Insert 1 observation inside day at 23:59:00
        rainfall_storage.insert_observation(test_ward, d_target + timedelta(hours=23, minutes=59), 18.0, "edge_test")
        # Insert 1 observation just after day end (00:01 next day)
        rainfall_storage.insert_observation(test_ward, d_target + timedelta(days=1, minutes=1), 99.0, "edge_test")

        summary = rainfall_storage.update_daily_summary_for_date(test_ward, d_target)
        self.assertIsNotNone(summary)
        # Should only sum 12.0 + 18.0 = 30.0
        self.assertEqual(summary["total_rainfall_mm"], 30.0)
        self.assertEqual(summary["observation_count"], 2)
        self.assertEqual(summary["max_hourly_mm"], 18.0)

        # Cleanup
        obs_col.delete_many({"ward_no": test_ward})
        daily_col.delete_many({"ward_no": test_ward})

    def test_06_exact_calendar_day_windows(self):
        """6. Exactly 7, 30, 90, 180, and 270 calendar-day windows when available."""
        ref_time = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
        history = rainfall_memory.get_ward_long_term_history(18, ref_time=ref_time)

        windows_spec = [
            ("7d", 7, "2026-10-01", "2026-10-07"),
            ("30d", 30, "2026-09-08", "2026-10-07"),
            ("3m", 90, "2026-07-10", "2026-10-07"),
            ("6m", 180, "2026-04-11", "2026-10-07"),
            ("9m", 270, "2026-01-11", "2026-10-07"),
        ]

        for w_key, expected_days, exp_start, exp_end in windows_spec:
            w_data = history[w_key]
            self.assertEqual(w_data["expected_days"], expected_days)
            self.assertEqual(w_data["available_days"], expected_days)
            self.assertEqual(w_data["missing_days"], 0)
            self.assertEqual(w_data["incomplete_days"], 0)
            self.assertEqual(w_data["start_date"], exp_start)
            self.assertEqual(w_data["end_date"], exp_end)

            # Validate date arithmetic
            d_start = datetime.strptime(w_data["start_date"], "%Y-%m-%d").date()
            d_end = datetime.strptime(w_data["end_date"], "%Y-%m-%d").date()
            span = (d_end - d_start).days + 1
            self.assertEqual(span, expected_days)

    def test_07_correct_reporting_missing_or_incomplete_days(self):
        """7. Correct reporting when observations or daily summaries are missing or incomplete."""
        test_ward = 9994
        daily_col = rainfall_storage.get_daily_summary_collection()
        obs_col = rainfall_storage.get_observations_collection()

        daily_col.delete_many({"ward_no": test_ward})
        obs_col.delete_many({"ward_no": test_ward})

        # Insert summaries for only 3 days out of the 7-day window (ending on 2026-10-07)
        # One complete day (24 obs, 10.0mm)
        daily_col.insert_one({
            "ward_no": test_ward,
            "date": datetime(2026, 10, 7, 0, 0, 0, tzinfo=timezone.utc),
            "total_rainfall_mm": 10.0,
            "max_hourly_mm": 5.0,
            "observation_count": 24,
            "is_complete": True,
            "source": "missing_test"
        })
        # One complete day (24 obs, 20.0mm)
        daily_col.insert_one({
            "ward_no": test_ward,
            "date": datetime(2026, 10, 6, 0, 0, 0, tzinfo=timezone.utc),
            "total_rainfall_mm": 20.0,
            "max_hourly_mm": 10.0,
            "observation_count": 24,
            "is_complete": True,
            "source": "missing_test"
        })
        # One incomplete day (only 12 obs recorded, 6.0mm)
        daily_col.insert_one({
            "ward_no": test_ward,
            "date": datetime(2026, 10, 5, 0, 0, 0, tzinfo=timezone.utc),
            "total_rainfall_mm": 6.0,
            "max_hourly_mm": 2.0,
            "observation_count": 12,
            "is_complete": False,
            "source": "missing_test"
        })

        ref_t = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
        history = rainfall_memory.get_ward_long_term_history(test_ward, ref_time=ref_t)
        w7d = history["7d"]

        self.assertEqual(w7d["expected_days"], 7)
        self.assertEqual(w7d["available_days"], 3)
        self.assertEqual(w7d["missing_days"], 4)
        self.assertEqual(w7d["incomplete_days"], 1)
        self.assertEqual(w7d["total_mm"], 36.0)
        # Crucial: average must be over available observed days (36.0 / 3 = 12.0), NOT divided by 7 (5.14)!
        self.assertEqual(w7d["avg_daily_mm"], 12.0)
        self.assertEqual(w7d["denominator_basis"], "available_observed_days")
        self.assertEqual(w7d["max_daily_mm"], 20.0)
        self.assertEqual(w7d["max_daily_date"], "2026-10-06")

        # Cleanup
        daily_col.delete_many({"ward_no": test_ward})

    def test_08_ward_specific_reference_timestamps(self):
        """8. Ward-specific reference timestamps prevent cross-ward clock leakage."""
        ref_119 = rainfall_memory.get_reference_timestamp(ward_no=119)
        ref_18 = rainfall_memory.get_reference_timestamp(ward_no=18)

        self.assertIsNotNone(ref_119)
        self.assertIsNotNone(ref_18)
        self.assertIsInstance(ref_119, datetime)
        self.assertIsInstance(ref_18, datetime)

    def test_09_test_observation_ward119_does_not_shift_ward18(self):
        """9. A test observation in Ward 119 does not shift Ward 18's reference time."""
        # Ward 119 has an observation at 13:00 UTC (from manual_test_station)
        # Ward 18 only has observations up to 12:00 UTC
        ref_119 = rainfall_memory.get_reference_timestamp(ward_no=119)
        ref_18 = rainfall_memory.get_reference_timestamp(ward_no=18)

        self.assertEqual(ref_119.hour, 13)
        self.assertEqual(ref_18.hour, 12)
        self.assertNotEqual(ref_119, ref_18)

    def test_10_existing_rainfall_memory_and_flood_momentum(self):
        """10. Time-aware rainfall-memory and flood-momentum behavior."""
        r24h = 40.0
        r72h = 60.0
        elevation = 6.0
        water_dist = 2.0

        res = rainfall_memory.calculate_flood_momentum(
            rainfall_memory={"24h": r24h, "72h": r72h},
            elevation_m=elevation,
            water_distance_km=water_dist
        )

        # In time-aware formula with fallback 8h = 0.40 * 24h:
        # recent = 16.0, intermediate = 24.0, past = 20.0
        expected_antecedent = round((1.00 * 16.0) + (0.50 * 24.0) + (0.15 * 20.0), 2)
        expected_saturation = round(min(1.0, max(0.0, expected_antecedent / 120.0)), 3)

        self.assertEqual(res["antecedent_rainfall_mm"], expected_antecedent)
        self.assertEqual(res["saturation_ratio"], expected_saturation)
        self.assertGreaterEqual(res["momentum_multiplier"], 1.0)
        self.assertLessEqual(res["momentum_multiplier"], 1.75)
        self.assertIn(res["momentum_level"], ["LOW", "MODERATE", "HIGH", "CRITICAL"])

    def test_11_existing_risk_classification_and_gis_functionality(self):
        """11. Existing risk classification and GIS functionality preserved."""
        from app import model, wards
        self.assertEqual(len(wards), 201)

        # Sample ward feature validation
        feat = wards[0]
        self.assertIn("properties", feat)
        props = feat["properties"]
        self.assertIn("Ward_No", props)
        self.assertIn("rainfall_mm", props)
        self.assertIn("elevation_m", props)
        self.assertIn("water_distance_km", props)

        # Verify XGBoost evaluation using exactly 3 features
        test_inputs = [[float(props["rainfall_mm"]), float(props["elevation_m"]), float(props["water_distance_km"])]]
        base_score = float(model.predict(test_inputs)[0])
        self.assertGreaterEqual(base_score, 0.0)

    def test_12_api_backward_compatibility(self):
        """12. API backward compatibility: GET /ward-history/<ward_no> returns complete schema."""
        try:
            resp = requests.get("http://127.0.0.1:5000/ward-history/18", timeout=2)
            data = resp.json()
        except Exception:
            # Fall back to Flask in-process test client
            from app import app
            client = app.test_client()
            resp = client.get("/ward-history/18")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()

        expected_top_keys = [
            "ward_no", "ward_name", "elevation_m", "water_distance_km",
            "rainfall_memory", "flood_momentum", "recent_observations", "long_term_history"
        ]
        for k in expected_top_keys:
            self.assertIn(k, data, f"Missing top-level key {k}")

        for win in ["1h", "3h", "6h", "12h", "24h", "72h"]:
            self.assertIn(win, data["rainfall_memory"])

        for period in ["7d", "30d", "3m", "6m", "9m"]:
            self.assertIn(period, data["long_term_history"])
            p_data = data["long_term_history"][period]
            self.assertIn("total_mm", p_data)
            self.assertIn("avg_daily_mm", p_data)
            self.assertIn("max_daily_mm", p_data)
            self.assertIn("start_date", p_data)
            self.assertIn("end_date", p_data)
            self.assertIn("expected_days", p_data)
            self.assertIn("available_days", p_data)
            self.assertIn("missing_days", p_data)
            self.assertIn("denominator_basis", p_data)

        self.assertGreater(len(data["recent_observations"]), 0)
        first_obs = data["recent_observations"][0]
        self.assertIn("source", first_obs)
        self.assertEqual(first_obs["source"], "synthetic_baseline")


if __name__ == "__main__":
    unittest.main()
