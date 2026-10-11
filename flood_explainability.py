"""
FloodSense Improvement 3 — Explainability Engine (Backend Module)

Provides model explainability for the FloodSense flood risk pipeline:
1. Native TreeSHAP attribution for the XGBoost base regression model
   (rainfall, elevation, water_distance).
2. Explicit, transparent decomposition of the time-aware hydrological momentum adjustment
   (time-decayed tranches, saturation proxy, active-rain gating factor, terrain handicap).
3. Evidence-based natural-language narrative synthesis.
4. Explicit disclosure of unmeasured physical factors (standing water, canal flow, etc.).

Strict Constraints:
- Zero external explanation dependencies (uses native XGBoost TreeSHAP C++ runtime via pred_contribs).
- TreeSHAP contributions are additive model score points, NOT probabilities or percentages.
- Never invents contributions or silently substitutes global feature importance when local attribution fails.
- Preserves existing model, risk formulas, thresholds, and MongoDB collections.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np


# Expected feature schema and order enforced by trained XGBoost model
FEATURE_NAMES: List[str] = ["rainfall", "elevation", "water_distance"]
EXPECTED_BASE_VALUE: float = 109.26364  # Trained model expected/bias value E[Y]
NUMERICAL_TOLERANCE: float = 1e-4       # Maximum allowed difference between sum(SHAP) and predict()


def explain_xgboost_base_prediction(
    model: Any,
    rainfall_mm: float,
    elevation_m: float,
    water_distance_km: float
) -> Dict[str, Any]:
    """
    Computes per-instance TreeSHAP feature attributions for the XGBoost base model
    using XGBoost's native pred_contribs capability.

    Parameters:
        model: Trained XGBoost model (XGBRegressor or Booster)
        rainfall_mm: Instantaneous rainfall input in mm (Feature 0)
        elevation_m: Terrain elevation in meters (Feature 1)
        water_distance_km: Distance to nearest water body/canal in km (Feature 2)

    Returns:
        Structured explanation dictionary with:
        - Expected value E[Y]
        - Per-feature attributions in model score points
        - Reconstructed prediction
        - Actual prediction
        - Numerical tolerance verification
    """
    # 1. Input Validation
    validation_err = _validate_model_inputs(rainfall_mm, elevation_m, water_distance_km)
    if validation_err:
        return {
            "status": "ERROR",
            "error_type": "INVALID_INPUT",
            "message": validation_err,
            "inputs_provided": {
                "rainfall_mm": rainfall_mm,
                "elevation_m": elevation_m,
                "water_distance_km": water_distance_km
            }
        }

    # 2. Model Compatibility Validation
    booster, model_err = _extract_and_validate_booster(model)
    if model_err:
        return {
            "status": "ERROR",
            "error_type": "INCOMPATIBLE_OR_MISSING_MODEL",
            "message": model_err
        }

    # 3. Construct input vector adhering strictly to FEATURE_NAMES
    input_vector = [float(rainfall_mm), float(elevation_m), float(water_distance_km)]

    try:
        import xgboost as xgb
        dmat = xgb.DMatrix([input_vector], feature_names=FEATURE_NAMES)
        contribs = booster.predict(dmat, pred_contribs=True)

        if contribs is None or len(contribs) == 0 or len(contribs[0]) != 4:
            return {
                "status": "ERROR",
                "error_type": "TREESHAP_OUTPUT_SHAPE_MISMATCH",
                "message": f"Expected pred_contribs shape (1, 4), got {getattr(contribs, 'shape', None)}"
            }

        c_row = contribs[0]
        rain_pts = float(c_row[0])
        elev_pts = float(c_row[1])
        wdist_pts = float(c_row[2])
        base_val = float(c_row[3])

        reconstructed_score = float(rain_pts + elev_pts + wdist_pts + base_val)

        # Actual prediction via model or booster
        if hasattr(model, "predict"):
            actual_score = float(model.predict([input_vector])[0])
        else:
            actual_score = float(booster.predict(dmat)[0])

        diff = abs(reconstructed_score - actual_score)
        tolerance_passed = bool(diff <= NUMERICAL_TOLERANCE)

        return {
            "status": "SUCCESS",
            "method": "NATIVE_XGBOOST_TREESHAP",
            "feature_order": FEATURE_NAMES,
            "inputs_used": {
                "rainfall_mm": round(float(rainfall_mm), 2),
                "elevation_m": round(float(elevation_m), 2),
                "water_distance_km": round(float(water_distance_km), 3)
            },
            "expected_value": round(base_val, 4),
            "feature_contributions": {
                "rainfall_points": round(rain_pts, 4),
                "elevation_points": round(elev_pts, 4),
                "water_distance_points": round(wdist_pts, 4)
            },
            "reconstructed_base_score": round(reconstructed_score, 4),
            "actual_base_score": round(actual_score, 4),
            "reconstruction_diff": round(diff, 8),
            "tolerance_verified": tolerance_passed,
            "units": "model_score_points",
            "disclaimer": (
                "TreeSHAP contributions represent additive model score points relative to the base value "
                f"({base_val:.2f}). They are not probabilities, percentages of flood risk, or causal proof."
            )
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "error_type": "TREESHAP_EXECUTION_FAILURE",
            "message": f"Native TreeSHAP execution failed: {str(e)}"
        }


def explain_momentum_adjustment(
    base_score: float,
    rainfall_memory: Dict[str, Any],
    flood_momentum: Dict[str, Any],
    elevation_m: float,
    water_distance_km: float
) -> Dict[str, Any]:
    """
    Decomposes the time-aware rainfall-momentum adjustment into explicit, transparent
    mathematical factors, separating model attribution from hydrological heuristics.

    Parameters:
        base_score: XGBoost base score (0 to ~200)
        rainfall_memory: Dict containing 1h, 3h, 6h, 8h, 12h, 24h, 72h totals
        flood_momentum: Dict returned by calculate_flood_momentum()
        elevation_m: Ward terrain elevation in meters
        water_distance_km: Nearest water body distance in km
    """
    try:
        base_s = float(base_score)
        mult = float(flood_momentum.get("momentum_multiplier", 1.0))
        raw_score = base_s * mult
        pts_added = round(raw_score - base_s, 2)
        bounded_score = round(min(200.0, max(0.0, raw_score)), 2)

        if bounded_score >= 120.0:
            level = "HIGH"
        elif bounded_score >= 80.0:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Explicitly recalculate the topography handicap factors for transparency
        elev_handicap = max(0.0, (8.0 - float(elevation_m)) / 25.0)
        water_handicap = max(0.0, (5.0 - min(float(water_distance_km), 5.0)) / 25.0)
        topography_handicap = 1.0 + elev_handicap + water_handicap

        tranches = flood_momentum.get("tranches", {})
        r_recent = float(tranches.get("recent_0_8h_mm", rainfall_memory.get("8h", 0.0)))
        r_intermediate = float(tranches.get("intermediate_8_24h_mm", 0.0))
        r_past = float(tranches.get("past_24_72h_mm", 0.0))
        decayed_ant = float(flood_momentum.get("antecedent_rainfall_mm", 0.0))
        sat_ratio = float(flood_momentum.get("saturation_ratio", 0.0))
        alpha_active = float(flood_momentum.get("active_rainfall_factor", 1.0))

        return {
            "status": "SUCCESS",
            "method": "TIME_AWARE_MOMENTUM_DECOMPOSITION",
            "base_score": round(base_s, 2),
            "momentum_multiplier": round(mult, 3),
            "points_added_by_momentum": pts_added,
            "final_bounded_score": bounded_score,
            "final_risk_level": level,
            "recent_rainfall_windows": {
                "1h_mm": round(float(rainfall_memory.get("1h", 0.0)), 2),
                "3h_mm": round(float(rainfall_memory.get("3h", 0.0)), 2),
                "8h_mm": round(float(rainfall_memory.get("8h", 0.0)), 2)
            },
            "time_decayed_tranches": {
                "recent_0_8h_mm": round(r_recent, 2),
                "intermediate_8_24h_mm": round(r_intermediate, 2),
                "past_24_72h_mm": round(r_past, 2),
                "decayed_antecedent_total_mm": round(decayed_ant, 2)
            },
            "tranche_weights": {
                "recent_weight": 1.00,
                "intermediate_weight": 0.50,
                "past_weight": 0.15
            },
            "active_rainfall_factor": round(alpha_active, 3),
            "burst_category": flood_momentum.get("burst_category", "LOW"),
            "saturation_proxy": {
                "saturation_ratio": round(sat_ratio, 3),
                "saturation_cap_mm": 120.0,
                "proxy_note": "Calculated Antecedent Precipitation Index (API) proxy, not measured soil column moisture."
            },
            "topography_handicap": {
                "elevation_handicap": round(elev_handicap, 4),
                "water_distance_handicap": round(water_handicap, 4),
                "total_handicap": round(topography_handicap, 4)
            },
            "prototype_notes": [
                "Multiplicative momentum model augments XGBoost base score based on soil saturation proxy and terrain handicap.",
                "Active rainfall factor attenuates the multiplier when the recent 8h window has dried out.",
                "Does not measure standing surface water or drainage discharge."
            ]
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "error_type": "MOMENTUM_EXPLANATION_FAILURE",
            "message": f"Momentum decomposition failed: {str(e)}"
        }


def generate_ward_risk_explanation(
    ward_no: int,
    model: Any,
    ward_props: Dict[str, Any],
    rainfall_memory: Dict[str, Any],
    flood_momentum: Dict[str, Any],
    base_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Comprehensive entry-point generating a unified, evidence-based explainability package
    for an individual ward.

    Combines:
    1. Local XGBoost TreeSHAP attributions
    2. Time-aware momentum decomposition
    3. Natural-language narrative summary
    4. Explicit statement of unmeasured physical factors
    """
    try:
        rain_mm = float(ward_props.get("rainfall_mm", 0.0))
        elev_m = float(ward_props.get("elevation_m", 6.7))
        wdist_km = float(ward_props.get("water_distance_km", 10.0))

        # 1. Base model TreeSHAP attribution
        shap_res = explain_xgboost_base_prediction(model, rain_mm, elev_m, wdist_km)
        if shap_res.get("status") == "SUCCESS":
            calc_base_score = shap_res["actual_base_score"]
        elif base_score is not None:
            calc_base_score = float(base_score)
        else:
            calc_base_score = 0.0

        # 2. Momentum decomposition
        mom_res = explain_momentum_adjustment(
            base_score=calc_base_score,
            rainfall_memory=rainfall_memory,
            flood_momentum=flood_momentum,
            elevation_m=elev_m,
            water_distance_km=wdist_km
        )

        final_score = mom_res.get("final_bounded_score", round(calc_base_score, 2))
        risk_lvl = mom_res.get("final_risk_level", "LOW")

        # 3. Deterministic natural-language synthesis
        narrative = _synthesize_evidence_narrative(
            ward_no=ward_no,
            risk_level=risk_lvl,
            final_score=final_score,
            base_score=calc_base_score,
            shap_res=shap_res,
            mom_res=mom_res,
            rainfall_mm=rain_mm,
            elevation_m=elev_m,
            water_distance_km=wdist_km
        )

        # 4. Primary driver identification
        primary_driver = _identify_primary_driver(shap_res, mom_res, rain_mm)

        return {
            "status": "SUCCESS",
            "ward_no": int(ward_no),
            "ward_name": f"Ward {ward_no}",
            "risk_assessment": {
                "level": risk_lvl,
                "final_score": final_score,
                "base_score": round(calc_base_score, 2),
                "multiplier": mom_res.get("momentum_multiplier", 1.0)
            },
            "primary_driver": primary_driver,
            "narrative_explanation": narrative,
            "model_attribution": shap_res,
            "momentum_explanation": mom_res,
            "unmeasured_factors": [
                "Standing floodwater depth (no physical depth sensors or telemetry)",
                "Stormwater canal discharge velocity and pump throughput (no flow meters)",
                "In-situ soil moisture (saturation is a calculated rainfall proxy only)",
                "Localized drain blockages or road depression micro-topography"
            ],
            "prototype_disclaimer": (
                "FloodSense models precipitation pressure and antecedent soil saturation memory. "
                "It does not physically observe standing floodwater or guarantee that a ward is flooded or drained."
            )
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "error_type": "WARD_EXPLANATION_FAILURE",
            "ward_no": ward_no,
            "message": f"Failed to generate explanation: {str(e)}"
        }


# ---------------------------------------------------------------------------
# Internal Helper Functions
# ---------------------------------------------------------------------------

def _validate_model_inputs(rainfall: float, elevation: float, water_dist: float) -> Optional[str]:
    """Validates numeric ranges and types for the 3 model features."""
    for name, val in [("rainfall_mm", rainfall), ("elevation_m", elevation), ("water_distance_km", water_dist)]:
        if val is None or not isinstance(val, (int, float)):
            return f"Feature '{name}' must be a numeric value, got {type(val)}"
        if math.isnan(val) or math.isinf(val):
            return f"Feature '{name}' contains invalid float (NaN or Inf)"
    if rainfall < 0.0:
        return f"rainfall_mm cannot be negative ({rainfall})"
    if elevation < 0.0:
        return f"elevation_m cannot be negative ({elevation})"
    if water_dist <= 0.0:
        return f"water_distance_km must be positive ({water_dist})"
    return None


def _extract_and_validate_booster(model: Any) -> Tuple[Optional[Any], Optional[str]]:
    """Extracts native booster and verifies feature alignment."""
    if model is None:
        return None, "Model artifact is None"

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif type(model).__name__ == "Booster":
        booster = model
    else:
        return None, f"Unsupported model object type '{type(model)}'; requires XGBoost model or Booster."

    # Validate feature count and names if present
    f_names = getattr(booster, "feature_names", None)
    if f_names is not None and list(f_names) != FEATURE_NAMES:
        return None, f"Model feature names mismatch: expected {FEATURE_NAMES}, got {f_names}"

    return booster, None


def _identify_primary_driver(shap_res: Dict[str, Any], mom_res: Dict[str, Any], rainfall_mm: float) -> str:
    """Identifies the dominant factor driving the score."""
    if mom_res.get("burst_category") in ["EXTREME_BURST", "HEAVY_BURST"]:
        return "ACUTE_PRECIPITATION_BURST"
    if mom_res.get("points_added_by_momentum", 0.0) >= 25.0:
        return "HIGH_ANTECEDENT_SOIL_SATURATION"
    if shap_res.get("status") == "SUCCESS":
        contribs = shap_res.get("feature_contributions", {})
        rain_pts = contribs.get("rainfall_points", 0.0)
        if rain_pts >= 15.0:
            return "INSTANTANEOUS_RAINFALL_LOAD"
        elif rain_pts <= -25.0:
            return "LOW_PRECIPITATION_ATTENUATION"
    if rainfall_mm >= 150.0:
        return "HEAVY_MONSOONAL_PRECIPITATION"
    return "BASELINE_TOPOGRAPHICAL_PROFILE"


def _synthesize_evidence_narrative(
    ward_no: int,
    risk_level: str,
    final_score: float,
    base_score: float,
    shap_res: Dict[str, Any],
    mom_res: Dict[str, Any],
    rainfall_mm: float,
    elevation_m: float,
    water_distance_km: float
) -> str:
    """Synthesizes a clear, evidence-based multi-sentence explanation."""
    sentences = []

    # 1. Headline classification
    sentences.append(
        f"Ward {ward_no} is estimated at {risk_level} flood risk with a score of "
        f"{final_score:.1f} / 200 (Base XGBoost: {base_score:.1f}, Multiplier: {mom_res.get('momentum_multiplier', 1.0):.3f}x)."
    )

    # 2. XGBoost Feature Attribution
    if shap_res.get("status") == "SUCCESS":
        c = shap_res["feature_contributions"]
        base_v = shap_res["expected_value"]
        sentences.append(
            f"Relative to the model base expected value of {base_v:.1f} points, "
            f"instant rainfall ({rainfall_mm:.1f} mm) contributed {c['rainfall_points']:+.1f} points, "
            f"centroid elevation ({elevation_m:.1f} m) contributed {c['elevation_points']:+.1f} points, "
            f"and distance to canal ({water_distance_km:.2f} km) contributed {c['water_distance_points']:+.1f} points."
        )

    # 3. Hydrological Momentum & Antecedent Context
    r_1h = mom_res.get("recent_rainfall_windows", {}).get("1h_mm", 0.0)
    r_8h = mom_res.get("recent_rainfall_windows", {}).get("8h_mm", 0.0)
    burst = mom_res.get("burst_category", "LOW")
    alpha = mom_res.get("active_rainfall_factor", 1.0)
    pts_mom = mom_res.get("points_added_by_momentum", 0.0)

    if burst in ["EXTREME_BURST", "HEAVY_BURST"]:
        sentences.append(
            f"An acute short-term burst ({r_1h:.1f} mm in 1h, category: {burst}) elevated the active rainfall factor "
            f"to {alpha:.2f}x, adding {pts_mom:+.1f} points to the final risk score."
        )
    elif r_8h <= 2.0 and mom_res.get("time_decayed_tranches", {}).get("past_24_72h_mm", 0.0) >= 40.0:
        sentences.append(
            f"Although older antecedent rainfall remains in the soil memory, a dry recent 8-hour window ({r_8h:.1f} mm) "
            f"gated the momentum multiplier (active factor: {alpha:.2f}x), preventing past accumulation from escalating current risk."
        )
    elif pts_mom > 0.0:
        sentences.append(
            f"Soil column saturation proxy and terrain drainage handicap added {pts_mom:+.1f} points via the hydrological multiplier."
        )

    return " ".join(sentences)
