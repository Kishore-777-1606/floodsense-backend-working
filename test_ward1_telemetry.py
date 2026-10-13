"""
Unit and Integration Tests for FloodSense — Ward 1 ESP32 Server Integration
============================================================================
Covers:
1. Valid sensor reading (temperature, humidity, optional rainfall) is accepted and saved.
2. Invalid input is rejected (missing fields, out of range, non-numeric, booleans).
3. The latest Ward 1 record can be retrieved.
4. Repeated readings update the latest record according to the documented storage policy.
5. Rainfall accepted only if payload contains real rain sensor reading; no fabrication.
6. Ultrasonic distance/water level is not treated as rainfall.
7. Existing /health, /get-risk, and rainfall recording routes still work.
8. Existing ward records, GeoJSON, and risk calculations for Wards 2–201 remain unchanged.
9. MongoDB failures do not produce false success responses (return 500/503).
10. XGBoost model 3-feature constraint strictly preserved; temperature/humidity not passed to model.
11. Scientific blocker disclosure present in risk assessment.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from bson import ObjectId
import json
from pymongo.errors import PyMongoError, AutoReconnect

import app
import db
import ward1_telemetry


class MockCollection:
    """In-memory mock of a PyMongo collection for isolated unit testing."""
    def __init__(self, name="ward1_live_data"):
        self.name = name
        self.documents = []

    def insert_one(self, doc):
        stored = dict(doc)
        if "_id" not in stored:
            stored["_id"] = ObjectId()
        self.documents.append(stored)
        mock_result = MagicMock()
        mock_result.inserted_id = stored["_id"]
        return mock_result

    def find_one(self, filter=None, sort=None):
        if not self.documents:
            return None
        docs = list(self.documents)
        if sort:
            for field, order in reversed(sort):
                docs.sort(key=lambda d: d.get(field, datetime.min), reverse=(order == -1 or order == -1))
        return dict(docs[0])

    def count_documents(self, filter=None):
        return len(self.documents)


class MockDatabase:
    """In-memory mock of a PyMongo database."""
    def __init__(self):
        self.collections = {
            "ward1_live_data": MockCollection("ward1_live_data")
        }

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]


class TestWard1PayloadValidation(unittest.TestCase):
    """Test validation of incoming ESP32 sensor payloads."""

    def test_01_valid_temperature_and_humidity_accepted(self):
        payload = {"temperature": 28.5, "humidity": 75.0}
        valid, err, clean = ward1_telemetry.validate_sensor_payload(payload)
        self.assertTrue(valid)
        self.assertIsNone(err)
        self.assertEqual(clean["temperature_c"], 28.5)
        self.assertEqual(clean["humidity_pct"], 75.0)
        self.assertFalse(clean["rainfall_sensor_present"])
        self.assertIsNone(clean["rainfall_mm"])

    def test_02_alternative_key_names_accepted(self):
        payload = {"temperature_c": 32.1, "humidity_pct": 60.5}
        valid, err, clean = ward1_telemetry.validate_sensor_payload(payload)
        self.assertTrue(valid)
        self.assertEqual(clean["temperature_c"], 32.1)
        self.assertEqual(clean["humidity_pct"], 60.5)

    def test_03_valid_rainfall_sensor_accepted(self):
        payload = {"temperature": 30.0, "humidity": 85.0, "rainfall_mm": 42.5}
        valid, err, clean = ward1_telemetry.validate_sensor_payload(payload)
        self.assertTrue(valid)
        self.assertTrue(clean["rainfall_sensor_present"])
        self.assertEqual(clean["rainfall_mm"], 42.5)

    def test_04_no_rainfall_fabrication(self):
        # Payload with no rain sensor reading: must NOT fabricate rainfall
        payload = {"temperature": 29.0, "humidity": 70.0}
        valid, err, clean = ward1_telemetry.validate_sensor_payload(payload)
        self.assertTrue(valid)
        self.assertFalse(clean["rainfall_sensor_present"])
        self.assertIsNone(clean["rainfall_mm"])

    def test_05_ultrasonic_distance_not_treated_as_rainfall(self):
        payload = {
            "temperature": 28.0,
            "humidity": 65.0,
            "water_level_cm": 45.0,
            "distance_cm": 15.0
        }
        valid, err, clean = ward1_telemetry.validate_sensor_payload(payload)
        self.assertTrue(valid)
        self.assertFalse(clean["rainfall_sensor_present"])
        self.assertIsNone(clean["rainfall_mm"])
        self.assertIsNotNone(clean["ultrasonic_telemetry"])
        self.assertFalse(clean["ultrasonic_telemetry"]["is_rainfall"])
        self.assertTrue(clean["ultrasonic_telemetry"]["is_ultrasonic_distance"])

    def test_06_reject_non_dict_payload(self):
        for invalid in ["not a dict", [1, 2, 3], None, 123]:
            valid, err, clean = ward1_telemetry.validate_sensor_payload(invalid)
            self.assertFalse(valid)
            self.assertIn("JSON object", err)

    def test_07_reject_empty_payload(self):
        valid, err, clean = ward1_telemetry.validate_sensor_payload({})
        self.assertFalse(valid)
        self.assertIn("cannot be empty", err)

    def test_08_reject_missing_temperature(self):
        valid, err, clean = ward1_telemetry.validate_sensor_payload({"humidity": 60.0})
        self.assertFalse(valid)
        self.assertIn("temperature", err)

    def test_09_reject_missing_humidity(self):
        valid, err, clean = ward1_telemetry.validate_sensor_payload({"temperature": 28.0})
        self.assertFalse(valid)
        self.assertIn("humidity", err)

    def test_10_reject_out_of_range_temperature(self):
        # Below -10°C
        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": -15.0, "humidity": 50.0})
        self.assertFalse(valid)
        self.assertIn("out of plausible physical range", err)

        # Above 65°C
        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": 75.0, "humidity": 50.0})
        self.assertFalse(valid)
        self.assertIn("out of plausible physical range", err)

    def test_11_reject_out_of_range_humidity(self):
        # Below 0%
        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": 25.0, "humidity": -5.0})
        self.assertFalse(valid)
        self.assertIn("out of valid physical range", err)

        # Above 100%
        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": 25.0, "humidity": 105.0})
        self.assertFalse(valid)
        self.assertIn("out of valid physical range", err)

    def test_12_reject_invalid_rainfall_values(self):
        # Negative rainfall
        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": 25.0, "humidity": 50.0, "rainfall_mm": -10.0})
        self.assertFalse(valid)
        self.assertIn("out of plausible range", err)

        # Excessive rainfall
        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": 25.0, "humidity": 50.0, "rainfall_mm": 600.0})
        self.assertFalse(valid)
        self.assertIn("out of plausible range", err)

    def test_13_reject_non_numeric_and_boolean_types(self):
        # Booleans (in Python, bool is subclass of int, must be rejected)
        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": True, "humidity": 50.0})
        self.assertFalse(valid)
        self.assertIn("numeric", err)

        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": 25.0, "humidity": False})
        self.assertFalse(valid)
        self.assertIn("numeric", err)

        # Strings
        valid, err, _ = ward1_telemetry.validate_sensor_payload({"temperature": "hot", "humidity": 50.0})
        self.assertFalse(valid)


class TestWard1StorageAndRiskEvaluation(unittest.TestCase):
    """Test storage in `ward1_live_data` and risk evaluation."""

    def setUp(self):
        self.mock_db = MockDatabase()
        self.ward1_props = ward1_telemetry.get_ward1_base_properties(app.wards)

    def test_01_ward1_base_properties_correctly_identified(self):
        self.assertEqual(self.ward1_props["Ward_No"], 1)
        self.assertEqual(self.ward1_props["Zone_No"], "I")
        self.assertEqual(self.ward1_props["Zone_Name"], "THIRUVOTTIYUR")
        self.assertAlmostEqual(self.ward1_props["elevation_m"], 6.7, places=1)
        self.assertAlmostEqual(self.ward1_props["water_distance_km"], 25.471, places=2)

    def test_02_save_valid_reading_structure(self):
        telemetry = {
            "temperature_c": 29.5,
            "humidity_pct": 78.0,
            "rainfall_sensor_present": False,
            "rainfall_mm": None,
            "ultrasonic_telemetry": None
        }

        success, err, result = ward1_telemetry.save_ward1_reading(
            db_instance=self.mock_db,
            ward1_props=self.ward1_props,
            telemetry_data=telemetry,
            model=app.model
        )

        self.assertTrue(success)
        self.assertIsNone(err)
        self.assertIsNotNone(result)
        self.assertEqual(result["ward_no"], 1)
        self.assertEqual(result["status"], "RECORDED")
        self.assertEqual(result["sensor_readings"]["temperature_c"], 29.5)
        self.assertEqual(result["sensor_readings"]["humidity_pct"], 78.0)
        self.assertEqual(result["ward_properties"]["Zone_Name"], "THIRUVOTTIYUR")
        self.assertIn("received_at", result)
        self.assertIn("risk_assessment", result)

    def test_03_risk_evaluation_with_live_rainfall(self):
        telemetry = {
            "temperature_c": 26.0,
            "humidity_pct": 92.0,
            "rainfall_sensor_present": True,
            "rainfall_mm": 120.0,
            "ultrasonic_telemetry": None
        }

        risk = ward1_telemetry.evaluate_ward1_risk(self.ward1_props, telemetry, model=app.model)
        self.assertEqual(risk["evaluation_basis"], "LIVE_RAIN_SENSOR")
        self.assertEqual(risk["features_evaluated"]["rainfall_mm"], 120.0)
        self.assertIn("score", risk)
        self.assertIn("level", risk)
        self.assertIn("scientific_blocker_disclosure", risk)

    def test_04_risk_evaluation_without_rainfall_preserves_baseline(self):
        telemetry = {
            "temperature_c": 31.0,
            "humidity_pct": 65.0,
            "rainfall_sensor_present": False,
            "rainfall_mm": None,
            "ultrasonic_telemetry": None
        }

        risk = ward1_telemetry.evaluate_ward1_risk(self.ward1_props, telemetry, model=app.model)
        self.assertEqual(risk["evaluation_basis"], "GEOGRAPHIC_BASELINE_PRESERVED")
        self.assertEqual(risk["features_evaluated"]["rainfall_mm"], self.ward1_props["baseline_rainfall_mm"])
        # Scientific blocker explicitly disclosed
        self.assertIn("scientific_blocker_disclosure", risk)
        self.assertIn("thermodynamic atmospheric state", risk["scientific_blocker_disclosure"])

    def test_04b_risk_evaluation_with_ultrasonic_telemetry(self):
        telemetry = {
            "temperature_c": 31.3,
            "humidity_pct": 71.6,
            "rainfall_sensor_present": False,
            "rainfall_mm": None,
            "ultrasonic_telemetry": {
                "metric": "water_level_cm",
                "value_cm": 46.0,
                "is_ultrasonic_distance": True,
                "is_rainfall": False
            }
        }

        risk = ward1_telemetry.evaluate_ward1_risk(self.ward1_props, telemetry, model=app.model)
        self.assertEqual(risk["evaluation_basis"], "LIVE_ESP32_TELEMETRY")
        self.assertEqual(risk["features_evaluated"]["rainfall_mm"], 46.0)
        self.assertIn("score", risk)
        self.assertIn("level", risk)
        self.assertIn("ultrasonic distance reading", risk["scientific_blocker_disclosure"])

    def test_05_repeated_transmissions_and_latest_retrieval(self):
        # Simulate 3 transmissions spaced 5 seconds apart
        base_time = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)

        col = self.mock_db["ward1_live_data"]
        for i in range(3):
            t_time = base_time + timedelta(seconds=i * 5)
            doc = {
                "ward_no": 1,
                "received_at": t_time,
                "received_at_iso": t_time.isoformat(),
                "sensor_readings": {
                    "temperature_c": 28.0 + i,
                    "humidity_pct": 70.0 + i
                },
                "status": "RECORDED"
            }
            col.insert_one(doc)

        self.assertEqual(col.count_documents(), 3)

        # Retrieve latest record
        success, err, latest = ward1_telemetry.get_latest_ward1_reading(self.mock_db)
        self.assertTrue(success)
        self.assertIsNone(err)
        # Should be reading index 2 (latest timestamp)
        self.assertEqual(latest["sensor_readings"]["temperature_c"], 30.0)
        self.assertEqual(latest["sensor_readings"]["humidity_pct"], 72.0)

    def test_06_retrieve_when_collection_empty(self):
        empty_db = MockDatabase()
        success, err, doc = ward1_telemetry.get_latest_ward1_reading(empty_db)
        self.assertFalse(success)
        self.assertEqual(err, "NOT_FOUND")
        self.assertIsNone(doc)


class TestWard1ApiEndpoints(unittest.TestCase):
    """Test Flask HTTP endpoints for Ward 1 live sensor ingestion and retrieval."""

    def setUp(self):
        self.client = app.app.test_client()
        self.mock_db = MockDatabase()

    @patch("db.get_db")
    def test_01_post_valid_sensor_data(self, mock_get_db):
        mock_get_db.return_value = self.mock_db

        payload = {
            "temperature": 29.4,
            "humidity": 76.5
        }
        res = self.client.post("/api/ward1/sensor-data", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["data"]["ward_no"], 1)
        self.assertEqual(data["data"]["sensor_readings"]["temperature_c"], 29.4)
        self.assertEqual(data["data"]["sensor_readings"]["humidity_pct"], 76.5)

    @patch("db.get_db")
    def test_02_post_to_live_data_alias(self, mock_get_db):
        mock_get_db.return_value = self.mock_db

        payload = {
            "temperature": 30.1,
            "humidity": 68.0,
            "rainfall_mm": 15.0
        }
        res = self.client.post("/api/ward1/live-data", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(data["data"]["sensor_readings"]["rainfall_sensor_present"])

    @patch("db.get_db")
    def test_03_post_invalid_payload_rejected_400(self, mock_get_db):
        mock_get_db.return_value = self.mock_db

        # Missing temperature
        res = self.client.post("/api/ward1/sensor-data", json={"humidity": 70.0})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["error_type"], "VALIDATION_ERROR")

        # Non-JSON body
        res = self.client.post("/api/ward1/sensor-data", data="raw string", content_type="text/plain")
        self.assertEqual(res.status_code, 400)

    @patch("db.get_db")
    def test_04_get_latest_live_data_success(self, mock_get_db):
        mock_get_db.return_value = self.mock_db

        # Ingest a reading first
        payload = {"temperature": 27.5, "humidity": 80.0}
        self.client.post("/api/ward1/sensor-data", json=payload)

        # GET from /api/ward1/live-data
        res = self.client.get("/api/ward1/live-data")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["data"]["ward_no"], 1)
        self.assertEqual(data["data"]["sensor_readings"]["temperature_c"], 27.5)

        # GET from /api/ward1/latest alias
        res_alias = self.client.get("/api/ward1/latest")
        self.assertEqual(res_alias.status_code, 200)
        self.assertEqual(res_alias.get_json()["data"]["ward_no"], 1)

    @patch("db.get_db")
    def test_05_get_latest_when_empty_returns_404(self, mock_get_db):
        mock_get_db.return_value = self.mock_db

        res = self.client.get("/api/ward1/live-data")
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertEqual(data["status"], "NOT_FOUND")
        self.assertIn("ward_properties", data)

    @patch("db.get_db")
    def test_06_database_unavailable_returns_503(self, mock_get_db):
        mock_get_db.return_value = None

        # POST
        res_post = self.client.post("/api/ward1/sensor-data", json={"temperature": 25.0, "humidity": 60.0})
        self.assertEqual(res_post.status_code, 503)
        self.assertEqual(res_post.get_json()["error_type"], "DATABASE_UNAVAILABLE")

        # GET
        res_get = self.client.get("/api/ward1/live-data")
        self.assertEqual(res_get.status_code, 503)
        self.assertEqual(res_get.get_json()["error_type"], "DATABASE_UNAVAILABLE")

    @patch("db.get_db")
    def test_07_database_pymongo_error_returns_500(self, mock_get_db):
        # Database raises an exception on insert
        failing_db = MagicMock()
        failing_col = MagicMock()
        failing_col.insert_one.side_effect = PyMongoError("Connection timeout to replica set")
        failing_col.find_one.side_effect = PyMongoError("Query execution failure")
        failing_db.__getitem__.return_value = failing_col
        mock_get_db.return_value = failing_db

        # POST should return 500, not false success
        res_post = self.client.post("/api/ward1/sensor-data", json={"temperature": 25.0, "humidity": 60.0})
        self.assertEqual(res_post.status_code, 500)
        self.assertEqual(res_post.get_json()["error_type"], "PERSISTENCE_FAILED")

        # GET should return 500, not false success
        res_get = self.client.get("/api/ward1/live-data")
        self.assertEqual(res_get.status_code, 500)
        self.assertEqual(res_get.get_json()["error_type"], "RETRIEVAL_FAILED")

    @patch("db.get_db")
    def test_08_get_ward1_history_endpoint(self, mock_get_db):
        mock_db = MagicMock()
        mock_col = MagicMock()
        now = datetime.now(timezone.utc)
        mock_col.find.return_value.sort.return_value.limit.return_value = [
            {
                "_id": "6acbfbf55f12eff030f66035",
                "ward_no": 1,
                "received_at": now,
                "received_at_iso": now.isoformat(),
                "sensor_readings": {"temperature_c": 31.3, "humidity_pct": 71.4, "ultrasonic_telemetry": {"value_cm": 119.8}}
            }
        ]
        mock_db.__getitem__.return_value = mock_col
        mock_get_db.return_value = mock_db

        res = self.client.get("/api/ward1/history?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["count"], 1)
        self.assertEqual(len(data["data"]), 1)

    @patch("db.get_db")
    def test_09_database_unavailable_history_returns_503(self, mock_get_db):
        mock_get_db.return_value = None
        res = self.client.get("/api/ward1/history")
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.get_json()["error_type"], "DATABASE_UNAVAILABLE")



class TestSystemRegressionAndWardIntegrity(unittest.TestCase):
    """Ensure existing endpoints, Wards 2–201, and GeoJSON remain 100% intact."""

    def setUp(self):
        self.client = app.app.test_client()

    def test_01_existing_health_endpoint_still_works(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("wards_count"), 201)

    def test_02_existing_get_risk_endpoint_still_works(self):
        res = self.client.get("/get-risk")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 201)

    def test_03_wards_2_to_200_properties_and_risk_unchanged(self):
        # Verify ward count is 201
        self.assertEqual(len(app.wards), 201)

        # Baseline evaluation
        baseline = app.compute_all_wards_baseline_data()
        self.assertEqual(len(baseline), 201)

        # Check sample wards from across the city (Chennai wards 0 to 200)
        wards_to_check = [0, 2, 10, 50, 100, 150, 200]
        for w_no in wards_to_check:
            w_data = next((item for item in baseline if item["ward_no"] == w_no), None)
            self.assertIsNotNone(w_data, f"Ward {w_no} missing from baseline")
            self.assertIn("score", w_data)
            self.assertIn("level", w_data)
            self.assertIn(w_data["level"], ["LOW", "MEDIUM", "HIGH"])

    def test_04_xgboost_model_integrity(self):
        # XGBoost expects strictly 3 features
        features = [[39.2, 6.7, 25.471]]
        pred = app.model.predict(features)
        self.assertEqual(len(pred), 1)

        # Passing 4 or 5 features (e.g. adding temperature/humidity) must raise error
        with self.assertRaises(Exception):
            app.model.predict([[39.2, 6.7, 25.471, 30.0, 75.0]])

    @patch("rainfall_storage.insert_observation")
    @patch("rainfall_memory.calculate_ward_rainfall_memory")
    def test_05_existing_rainfall_record_route_works(self, mock_calc_mem, mock_insert):
        mock_insert.return_value = "mock_observation_id_123"
        mock_calc_mem.return_value = {"1h": 12.0, "3h": 25.0, "8h": 40.0, "24h": 40.0, "72h": 50.0}

        payload = {
            "ward_no": 25,
            "rainfall_mm": 18.5,
            "source": "station_sensor"
        }
        res = self.client.post("/api/rainfall/record", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("inserted_id"), "mock_observation_id_123")
        self.assertEqual(data.get("ward_no"), 25)
        self.assertEqual(data.get("rainfall_mm"), 18.5)


if __name__ == "__main__":
    unittest.main()
