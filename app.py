from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timezone
import json
import joblib
import os

import db
import rainfall_storage
import rainfall_memory
import flood_explainability
import flood_actions
import flood_propagation
import flood_simulator
import ward1_telemetry

app = Flask(__name__)
CORS(app)

# ✅ BASE DIRECTORY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ LOAD PRE-TRAINED XGBOOST MODEL (Expects 3 features: [rainfall, elevation, water_distance])
model_path = os.path.join(BASE_DIR, "flood_model.pkl")
model = joblib.load(model_path)

# ✅ LOAD GEOJSON
geojson_path = os.path.join(BASE_DIR, "Wards.geojson")
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)
    wards = geojson_data["features"]

# Extract authoritative Ward 1 baseline properties for edge integration
ward1_properties = ward1_telemetry.get_ward1_base_properties(wards)

# ✅ INITIALIZE DATABASE AND PERSISTENT MEMORY
db.init_db()
rainfall_storage.seed_baseline_if_empty(wards)


def compute_all_wards_baseline_data():
    """
    Computes authoritative baseline risk data for all 201 wards using the approved
    XGBoost model, time-aware rainfall momentum, and Improvement 3 explainability.
    Shared across /get-risk and /spatial-propagation for 100% data parity.
    """
    all_memory = rainfall_memory.calculate_all_wards_rainfall_memory()
    results = []

    for ward in wards:
        properties = ward["properties"]
        ward_no = int(properties["Ward_No"])

        # Hyperlocal environmental features established in Improvement 1
        rainfall_mm = float(properties.get("rainfall_mm", 0.0))
        elevation_m = float(properties.get("elevation_m", 0.0))
        water_distance_km = float(properties.get("water_distance_km", 0.0))

        # Scope Protection: Ward 1 Live IoT Validation Integration
        live_telemetry_payload = None
        live_received_at_iso = None
        if ward_no == 1:
            try:
                database = db.get_db()
                if database is not None:
                    success, _, latest_doc = ward1_telemetry.get_latest_ward1_reading(database)
                    if success and latest_doc:
                        readings = latest_doc.get("sensor_readings", {})
                        live_telemetry_payload = readings
                        live_received_at_iso = latest_doc.get("received_at_iso")
                        # Ward 1 prediction: Replace static GeoJSON baseline rainfall (39.2 mm)
                        # with latest ESP32 telemetry numeric value as requested.
                        # Note: This ESP32 value is an ultrasonic reading, not measured rainfall.
                        u_data = readings.get("ultrasonic_telemetry") or {}
                        if readings.get("rainfall_sensor_present") and readings.get("rainfall_mm") is not None:
                            rainfall_mm = float(readings["rainfall_mm"])
                        elif u_data.get("value_cm") is not None:
                            rainfall_mm = float(u_data["value_cm"])
            except Exception:
                pass

        # Model Input (EXACTLY 3 features preserved for XGBoost compatibility)
        features = [[rainfall_mm, elevation_m, water_distance_km]]
        base_score = float(model.predict(features)[0])

        # Ward-specific timestamped rainfall memory from MongoDB
        ward_mem = all_memory.get(ward_no)
        if ward_mem is None:
            ward_mem = rainfall_memory.calculate_ward_rainfall_memory(ward_no)

        # Flood Momentum & Soil Saturation calculation
        momentum = rainfall_memory.calculate_flood_momentum(
            rainfall_memory=ward_mem,
            elevation_m=elevation_m,
            water_distance_km=water_distance_km
        )

        # Momentum-augmented score (deterministic)
        multiplier = momentum["momentum_multiplier"]
        final_score = round(min(200.0, max(0.0, base_score * multiplier)), 2)

        # Risk level classification
        # Exact boundaries: LOW (< 80.0), MEDIUM (80.0 to < 120.0), HIGH (>= 120.0)
        if final_score >= 120.0:
            level = "HIGH"
        elif final_score >= 80.0:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Contextual hydrological reasoning integrating time-aware momentum & environment
        r_1h = float(ward_mem.get("1h", 0.0))
        r_8h = float(ward_mem.get("8h", 0.0))
        r_72h = float(momentum.get("accumulated_72h_mm", 0.0))
        mom_level = momentum["momentum_level"]
        burst = momentum.get("burst_category", "LOW")

        if burst in ["EXTREME_BURST", "HEAVY_BURST"]:
            reason = f"Intense short-term rain burst ({r_1h}mm in 1h) with rapid runoff risk on {elevation_m:.1f}m terrain"
        elif mom_level in ["CRITICAL", "HIGH"] and r_8h >= 15.0:
            reason = f"High short-term flood momentum ({r_8h}mm in 8h) saturating {elevation_m:.1f}m elevation terrain"
        elif r_72h >= 50.0 and r_8h < 5.0:
            reason = f"Past rainfall ({r_72h:.1f}mm in 72h) attenuated by dry recent window ({r_8h:.1f}mm in 8h; residual ponding unmeasured)"
        elif mom_level in ["MODERATE", "MEDIUM"]:
            reason = f"Moderate rainfall accumulation ({r_8h}mm in 8h) across {elevation_m:.1f}m terrain"
        elif elevation_m < 5.0 and water_distance_km < 3.0:
            reason = "Low elevation terrain in close proximity to drainage canals"
        else:
            reason = "Lower estimated runoff pressure with low antecedent soil saturation (standing water not measured)"

        # Improvement 3: Explainability and Action Support Integration
        try:
            eval_props = properties
            if ward_no == 1:
                eval_props = dict(properties)
                eval_props["rainfall_mm"] = rainfall_mm

            exp_res = flood_explainability.generate_ward_risk_explanation(
                ward_no=ward_no,
                model=model,
                ward_props=eval_props,
                rainfall_memory=ward_mem,
                flood_momentum=momentum,
                base_score=base_score
            )
            act_res = flood_actions.generate_ward_advisory_actions(
                ward_no=ward_no,
                risk_level=level,
                final_score=final_score,
                base_score=base_score,
                rainfall_memory=ward_mem,
                flood_momentum=momentum,
                elevation_m=elevation_m,
                water_distance_km=water_distance_km,
                observation_count=ward_mem.get("observation_count", 72)
            )
            explainability_payload = {
                "status": exp_res.get("status", "SUCCESS"),
                "primary_driver": exp_res.get("primary_driver", "BASELINE_TOPOGRAPHICAL_PROFILE"),
                "model_attribution": exp_res.get("model_attribution", {}),
                "momentum_decomposition": exp_res.get("momentum_explanation", {}),
                "reconstruction_verified": exp_res.get("model_attribution", {}).get("tolerance_verified", True)
            }
            evidence_summary_payload = exp_res.get("narrative_explanation", reason)
            action_support_payload = {
                "status": act_res.get("status", "SUCCESS"),
                "total_actions": act_res.get("total_actions", len(act_res.get("actions", []))),
                "actions": act_res.get("actions", []),
                "operational_disclaimer": act_res.get("operational_disclaimer", "")
            }
            data_limitations_payload = {
                "unmeasured_factors": exp_res.get("unmeasured_factors", []),
                "synthetic_baseline_disclaimer": (
                    "Historical baseline rainfall was generated using synthetic climatological distributions "
                    "for prototype validation and does not originate from verified physical rain gauges."
                ),
                "model_disclaimer": exp_res.get("prototype_disclaimer", "")
            }
        except Exception:
            explainability_payload = {
                "status": "UNAVAILABLE",
                "message": "Model explainability attribution unavailable for this ward."
            }
            evidence_summary_payload = reason
            action_support_payload = {
                "status": "UNAVAILABLE",
                "total_actions": 0,
                "actions": [],
                "operational_disclaimer": "Advisories temporarily unavailable."
            }
            data_limitations_payload = {
                "unmeasured_factors": [],
                "synthetic_baseline_disclaimer": "Historical rainfall baseline is synthetic.",
                "model_disclaimer": "Experimental risk engine."
            }

        results.append({
            "ward_no": ward_no,
            "ward_name": f"Ward {ward_no}",

            # Backward-compatible fields
            "rainfall": round(rainfall_mm, 2),
            "elevation": round(elevation_m, 2),
            "water_distance": round(water_distance_km, 3),

            # Updated schema fields
            "rainfall_mm": round(rainfall_mm, 2),
            "elevation_m": round(elevation_m, 2),
            "water_distance_km": round(water_distance_km, 3),

            # Ward 1 Live Edge Telemetry (None for Wards 2-201)
            "live_telemetry": live_telemetry_payload,
            "live_received_at": live_received_at_iso,

            # Model outputs
            "base_score": round(base_score, 2),
            "score": final_score,
            "level": level,
            "reason": reason,

            # Time-Aware Rainfall Memory (Operational Windows: 1h, 3h, 6h, 8h, 12h, 24h, 72h)
            "rainfall_memory": {
                "1h": ward_mem.get("1h", 0.0),
                "3h": ward_mem.get("3h", 0.0),
                "6h": ward_mem.get("6h", 0.0),
                "8h": ward_mem.get("8h", 0.0),
                "12h": ward_mem.get("12h", 0.0),
                "24h": ward_mem.get("24h", 0.0),
                "72h": ward_mem.get("72h", 0.0),
                "observation_count": ward_mem.get("observation_count", 0),
                "latest_observation_time": ward_mem.get("latest_observation_time")
            },

            # Flood Momentum / Soil Saturation Metrics
            "flood_momentum": momentum,

            # Improvement 3: Explainable Flood Intelligence & Action Engine
            "explainability": explainability_payload,
            "evidence_summary": evidence_summary_payload,
            "action_support": action_support_payload,
            "data_limitations": data_limitations_payload
        })

    return results


@app.route("/get-risk")
def get_risk():
    """
    Authoritative risk endpoint for all 201 wards.
    Evaluates XGBoost model, timestamped rainfall memory, and Improvement 3 explainability.
    """
    return jsonify(compute_all_wards_baseline_data())


@app.route("/spatial-propagation", methods=["GET"])
@app.route("/api/spatial-propagation", methods=["GET"])
def get_spatial_propagation():
    """
    Improvement 4: Spatial Flood Risk Propagation & Time-Stepped Scenario Simulation.
    Query parameters:
    - source_ward: comma-separated list of ward IDs (e.g. '119' or '18,174')
    - scenario_mode: PREDICTED_RISK_SCENARIO (default if source not given) | USER_DEFINED_SCENARIO | OBSERVED_EVENT
    - steps: comma-separated list of simulation intervals (default '0,30,60')
    """
    try:
        baseline_results = compute_all_wards_baseline_data()
        baseline_map = {int(w["ward_no"]): w for w in baseline_results}

        source_param = request.args.get("source_ward")
        scenario_mode_param = request.args.get("scenario_mode")

        if source_param:
            try:
                source_wards = [int(s.strip()) for s in source_param.split(",") if s.strip()]
            except ValueError:
                return jsonify({
                    "status": "ERROR",
                    "error_type": "INVALID_PARAMETER",
                    "message": f"Invalid source_ward parameter '{source_param}'. Must be comma-separated integers."
                }), 400

            scenario_mode = scenario_mode_param or flood_propagation.SCENARIO_MODE_USER_DEFINED
        else:
            sorted_by_score = sorted(baseline_results, key=lambda x: x.get("score", 0.0), reverse=True)
            if sorted_by_score:
                source_wards = [int(sorted_by_score[0]["ward_no"])]
            else:
                source_wards = [119]
            scenario_mode = scenario_mode_param or flood_propagation.SCENARIO_MODE_PREDICTED_RISK

        # Validate source wards against existing wards
        invalid_sources = [w for w in source_wards if w not in baseline_map]
        if invalid_sources:
            return jsonify({
                "status": "ERROR",
                "error_type": "WARD_NOT_FOUND",
                "message": f"Wards {invalid_sources} not found among the 201 city wards (valid range 1-201)."
            }), 400

        # Parse steps param
        steps_param = request.args.get("steps", "0,30,60")
        try:
            steps = [int(s.strip()) for s in steps_param.split(",") if s.strip()]
        except ValueError:
            return jsonify({
                "status": "ERROR",
                "error_type": "INVALID_PARAMETER",
                "message": f"Invalid steps parameter '{steps_param}'. Must be comma-separated integers like '0,30,60'."
            }), 400

        scenario_result = flood_propagation.simulate_spatial_propagation(
            source_wards=source_wards,
            baseline_risk_map=baseline_map,
            scenario_mode=scenario_mode,
            steps=steps
        )

        return jsonify(scenario_result)

    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error_type": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.route("/ward-history/<int:ward_no>")
@app.route("/api/ward/<int:ward_no>/rainfall-history")
def get_ward_history(ward_no):
    """
    Returns full rainfall memory, long-term history (7d, 30d, 3m, 6m, 9m),
    and recent raw observations for an individual ward, augmented with
    Improvement 3 explainability and action support.
    """
    ref_t = rainfall_memory.get_reference_timestamp(None)
    mem = rainfall_memory.calculate_ward_rainfall_memory(ward_no, ref_time=ref_t)
    long_term = rainfall_memory.get_ward_long_term_history(ward_no, ref_time=ref_t)

    # Find ward properties
    ward_props = next((f["properties"] for f in wards if int(f["properties"]["Ward_No"]) == ward_no), None)
    elev = float(ward_props.get("elevation_m", 6.7)) if ward_props else 6.7
    w_dist = float(ward_props.get("water_distance_km", 10.0)) if ward_props else 10.0
    r_curr = float(ward_props.get("rainfall_mm", mem.get("1h", 0.0))) if ward_props else float(mem.get("1h", 0.0))

    # Ward 1 Scope Protection: Live IoT Validation Integration
    live_telemetry_payload = None
    if ward_no == 1:
        try:
            database = db.get_db()
            if database is not None:
                success, _, latest_doc = ward1_telemetry.get_latest_ward1_reading(database)
                if success and latest_doc:
                    readings = latest_doc.get("sensor_readings", {})
                    live_telemetry_payload = readings
                    u_data = readings.get("ultrasonic_telemetry") or {}
                    if readings.get("rainfall_sensor_present") and readings.get("rainfall_mm") is not None:
                        r_curr = float(readings["rainfall_mm"])
                    elif u_data.get("value_cm") is not None:
                        r_curr = float(u_data["value_cm"])
        except Exception:
            pass

    mom = rainfall_memory.calculate_flood_momentum(mem, elev, w_dist)

    # Compute base and final score
    base_s = float(model.predict([[r_curr, elev, w_dist]])[0])
    mult = mom.get("momentum_multiplier", 1.0)
    f_score = round(min(200.0, max(0.0, base_s * mult)), 2)
    lvl = "HIGH" if f_score >= 120.0 else ("MEDIUM" if f_score >= 80.0 else "LOW")

    ward_props_dict = {"rainfall_mm": r_curr, "elevation_m": elev, "water_distance_km": w_dist}
    try:
        exp_res = flood_explainability.generate_ward_risk_explanation(
            ward_no=ward_no,
            model=model,
            ward_props=ward_props_dict,
            rainfall_memory=mem,
            flood_momentum=mom,
            base_score=base_s
        )
        act_res = flood_actions.generate_ward_advisory_actions(
            ward_no=ward_no,
            risk_level=lvl,
            final_score=f_score,
            base_score=base_s,
            rainfall_memory=mem,
            flood_momentum=mom,
            elevation_m=elev,
            water_distance_km=w_dist,
            observation_count=mem.get("observation_count", 72)
        )
        exp_payload = {
            "status": exp_res.get("status", "SUCCESS"),
            "primary_driver": exp_res.get("primary_driver", "BASELINE_TOPOGRAPHICAL_PROFILE"),
            "model_attribution": exp_res.get("model_attribution", {}),
            "momentum_decomposition": exp_res.get("momentum_explanation", {}),
            "reconstruction_verified": exp_res.get("model_attribution", {}).get("tolerance_verified", True)
        }
        evidence_summary_val = exp_res.get("narrative_explanation", "")
        act_payload = {
            "status": act_res.get("status", "SUCCESS"),
            "total_actions": act_res.get("total_actions", len(act_res.get("actions", []))),
            "actions": act_res.get("actions", []),
            "operational_disclaimer": act_res.get("operational_disclaimer", "")
        }
        lim_payload = {
            "unmeasured_factors": exp_res.get("unmeasured_factors", []),
            "synthetic_baseline_disclaimer": (
                "Historical baseline rainfall was generated using synthetic climatological distributions "
                "for prototype validation and does not originate from verified physical rain gauges."
            ),
            "model_disclaimer": exp_res.get("prototype_disclaimer", "")
        }
    except Exception:
        exp_payload = {"status": "UNAVAILABLE"}
        evidence_summary_val = ""
        act_payload = {"status": "UNAVAILABLE", "actions": []}
        lim_payload = {"unmeasured_factors": []}

    # Fetch recent 24 raw observations
    ref_t = rainfall_memory.get_reference_timestamp(ward_no=ward_no)
    from datetime import timedelta
    start_t = ref_t - timedelta(hours=24)
    raw_obs = rainfall_storage.get_observations_for_ward(ward_no, start_time=start_t, end_time=ref_t)

    # Format raw observations for JSON
    obs_formatted = [
        {
            "timestamp": o["timestamp"].isoformat(),
            "rainfall_mm": o["rainfall_mm"],
            "source": o.get("source", "sensor")
        }
        for o in raw_obs
    ]

    return jsonify({
        "ward_no": ward_no,
        "ward_name": f"Ward {ward_no}",
        "elevation_m": elev,
        "water_distance_km": w_dist,
        "rainfall": round(r_curr, 2),
        "rainfall_mm": round(r_curr, 2),
        "live_telemetry": live_telemetry_payload,
        "base_score": round(base_s, 2),
        "score": f_score,
        "level": lvl,
        "rainfall_memory": mem,
        "flood_momentum": mom,
        "long_term_history": long_term,
        "recent_observations": obs_formatted,
        "explainability": exp_payload,
        "evidence_summary": evidence_summary_val,
        "action_support": act_payload,
        "data_limitations": lim_payload
    })


@app.route("/api/rainfall/record", methods=["POST"])
def record_rainfall():
    """
    Ingests a new timestamped rainfall observation for a ward into MongoDB.
    Body:
    {
        "ward_no": 119,
        "timestamp": "2026-10-09T01:00:00Z",  (optional, defaults to UTC now)
        "rainfall_mm": 14.5,
        "source": "station_sensor"
    }
    """
    data = request.get_json(force=True)
    if not data or "ward_no" not in data or "rainfall_mm" not in data:
        return jsonify({"error": "Missing required fields: ward_no, rainfall_mm"}), 400

    ward_no = int(data["ward_no"])
    rainfall_mm = float(data["rainfall_mm"])
    source = str(data.get("source", "api_ingest"))

    ts_str = data.get("timestamp")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception as e:
            return jsonify({"error": f"Invalid timestamp format: {e}"}), 400
    else:
        ts = datetime.now(timezone.utc)

    doc_id = rainfall_storage.insert_observation(ward_no, ts, rainfall_mm, source)

    # Recalculate updated memory for this ward
    updated_mem = rainfall_memory.calculate_ward_rainfall_memory(ward_no, ref_time=ts)

    return jsonify({
        "status": "success",
        "inserted_id": doc_id,
        "ward_no": ward_no,
        "timestamp": ts.isoformat(),
        "rainfall_mm": rainfall_mm,
        "updated_rainfall_memory": updated_mem
    }), 201


@app.route("/health")
def health():
    db_ok = db.is_connected()
    obs_count = db.get_db()["rainfall_observations"].count_documents({}) if db_ok else 0
    daily_count = db.get_db()["rainfall_daily_summary"].count_documents({}) if db_ok else 0
    return jsonify({
        "status": "healthy" if db_ok else "degraded",
        "database": {
            "connected": db_ok,
            "observations_count": obs_count,
            "daily_summaries_count": daily_count
        },
        "wards_count": len(wards)
    })


@app.route("/")
def home():
    return jsonify({
        "message": "FloodSense Backend Running",
        "version": "2.0",
        "database": "MongoDB Connected" if db.is_connected() else "Disconnected"
    })


@app.route("/rainfall-scenario", methods=["POST", "GET"])
@app.route("/api/rainfall-scenario", methods=["POST", "GET"])
def rainfall_scenario_endpoint():
    """
    Improvement 5: Interactive Rainfall Scenario Simulator & Digital Twin Prototype.
    Query / Body parameters:
    - rainfall_mm: float between 0.0 and 300.0 (required for POST, default 50.0 for GET)
    """
    try:
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            rainfall_val = data.get("rainfall_mm") if "rainfall_mm" in data else data.get("rainfall")
            if rainfall_val is None:
                rainfall_val = request.form.get("rainfall_mm") or request.form.get("rainfall")
            if rainfall_val is None:
                rainfall_val = request.args.get("rainfall_mm") or request.args.get("rainfall")
        else:
            rainfall_val = request.args.get("rainfall_mm") or request.args.get("rainfall", 50.0)

        valid_rf, err = flood_simulator.validate_scenario_rainfall(rainfall_val)
        if err:
            return jsonify({
                "status": "ERROR",
                "error_type": "INVALID_RAINFALL_INPUT",
                "message": err
            }), 400

        # Authoritative baseline parity
        baseline_results = compute_all_wards_baseline_data()
        scenario_result = flood_simulator.simulate_rainfall_scenario(
            rainfall_mm=valid_rf,
            model=model,
            wards_geojson=wards,
            baseline_data=baseline_results,
            include_explainability=True
        )

        return jsonify(scenario_result)

    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error_type": "INTERNAL_SIMULATOR_ERROR",
            "message": str(e)
        }), 500


# ============================================================================
# WARD 1 ESP32 LIVE SENSOR INTEGRATION
# ============================================================================

@app.route("/api/ward1/sensor-data", methods=["POST"])
@app.route("/ward1/sensor-data", methods=["POST"])
def ingest_ward1_sensor_data():
    """
    Ingests live telemetry from the dedicated ESP32 edge device deployed in Ward 1.
    Accepts validated temperature and humidity. Accepts rainfall only if a real
    rainfall sensor reading is present. Never fabricates rainfall or maps ultrasonic
    distance to rainfall.
    Persists combined record to `ward1_live_data` collection in MongoDB.
    """
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({
                "status": "ERROR",
                "error_type": "INVALID_JSON",
                "message": "Request body must be a valid JSON object"
            }), 400

        is_valid, err_msg, clean_data = ward1_telemetry.validate_sensor_payload(payload)
        if not is_valid:
            return jsonify({
                "status": "ERROR",
                "error_type": "VALIDATION_ERROR",
                "message": err_msg
            }), 400

        database = db.get_db()
        if database is None:
            return jsonify({
                "status": "ERROR",
                "error_type": "DATABASE_UNAVAILABLE",
                "message": "Database connection is unavailable"
            }), 503

        success, save_err, saved_doc = ward1_telemetry.save_ward1_reading(
            db_instance=database,
            ward1_props=ward1_properties,
            telemetry_data=clean_data,
            model=model
        )

        if not success:
            return jsonify({
                "status": "ERROR",
                "error_type": "PERSISTENCE_FAILED",
                "message": save_err or "Failed to record Ward 1 sensor data"
            }), 500

        return jsonify({
            "status": "SUCCESS",
            "message": "Ward 1 sensor reading accepted and recorded",
            "data": saved_doc
        }), 201

    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error_type": "INTERNAL_INGESTION_ERROR",
            "message": str(e)
        }), 500


@app.route("/api/ward1/latest", methods=["GET"])
@app.route("/ward1/latest", methods=["GET"])
def get_latest_ward1_live_data():
    """
    Returns the latest combined live telemetry record for Ward 1 (Thiruvottiyur)
    from the dedicated `ward1_live_data` collection.
    """
    try:
        database = db.get_db()
        if database is None:
            return jsonify({
                "status": "ERROR",
                "error_type": "DATABASE_UNAVAILABLE",
                "message": "Database connection is unavailable"
            }), 503

        success, err_msg, doc = ward1_telemetry.get_latest_ward1_reading(database)
        if not success:
            if err_msg == "NOT_FOUND":
                return jsonify({
                    "status": "NOT_FOUND",
                    "message": "No live sensor readings found for Ward 1 yet",
                    "ward_properties": ward1_properties
                }), 404
            return jsonify({
                "status": "ERROR",
                "error_type": "RETRIEVAL_FAILED",
                "message": err_msg or "Failed to retrieve latest Ward 1 record"
            }), 500

        return jsonify({
            "status": "SUCCESS",
            "data": doc
        }), 200

    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error_type": "INTERNAL_RETRIEVAL_ERROR",
            "message": str(e)
        }), 500


@app.route("/api/ward1/live-data", methods=["GET", "POST"])
@app.route("/ward1/live-data", methods=["GET", "POST"])
def ward1_live_data_unified():
    """
    Unified route supporting both GET (latest reading) and POST (sensor ingestion)
    for Ward 1 live telemetry.
    """
    if request.method == "POST":
        return ingest_ward1_sensor_data()
    return get_latest_ward1_live_data()


@app.route("/api/ward1/history", methods=["GET"])
@app.route("/ward1/history", methods=["GET"])
def get_ward1_telemetry_history():
    """
    Returns bounded chronological history of live Ward 1 telemetry records
    for time-series charting.
    """
    try:
        database = db.get_db()
        if database is None:
            return jsonify({
                "status": "ERROR",
                "error_type": "DATABASE_UNAVAILABLE",
                "message": "Database connection is unavailable"
            }), 503

        limit = request.args.get("limit", default=30, type=int)
        success, err_msg, docs = ward1_telemetry.get_ward1_history(database, limit=limit)
        if not success:
            return jsonify({
                "status": "ERROR",
                "error_type": "RETRIEVAL_FAILED",
                "message": err_msg or "Failed to retrieve Ward 1 history"
            }), 500

        return jsonify({
            "status": "SUCCESS",
            "count": len(docs),
            "data": docs
        }), 200

    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error_type": "INTERNAL_RETRIEVAL_ERROR",
            "message": str(e)
        }), 500



if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "false").strip().lower() in ("true", "1", "t")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode
    )