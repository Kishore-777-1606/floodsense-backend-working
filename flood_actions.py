"""
FloodSense Improvement 3 — Action Engine (Backend Module)

Generates deterministic, transparent, evidence-based advisory actions
for ward flood management based on existing risk-engine outputs,
hydrological momentum, and verified spatial attributes.

Strict Architectural Constraints:
- 100% deterministic rule evaluation (no LLM, no external service).
- No automated evacuation commands or assertions of confirmed flooding.
- No fabricated measurements (water level, drain blockage, soil moisture, flow rate).
- Waterbody proximity treated strictly as spatial distance, not proof of inadequate drainage.
- Elevation (currently uniform across wards) is not used as a differentiating trigger.
- Missing observations or data gaps are explicitly reported as data verification actions.
- Action IDs are unique and deduplicated per response.
"""

from typing import Dict, Any, List, Optional


# Standard Advisory Categories
CAT_BURST_RESPONSE = "BURST_RESPONSE_READINESS"
CAT_RECENT_RAINFALL = "RECENT_RAINFALL_REVIEW"
CAT_ANTECEDENT_MONITORING = "ANTECEDENT_RAINFALL_MONITORING"
CAT_DRAINAGE_INSPECTION = "DRAINAGE_INSPECTION"
CAT_FIELD_VERIFICATION = "FIELD_VERIFICATION"
CAT_ROUTINE_MAINTENANCE = "ROUTINE_PREVENTIVE_MAINTENANCE"
CAT_DATA_VERIFICATION = "DATA_GAP_VERIFICATION"

VALID_CATEGORIES = [
    CAT_BURST_RESPONSE,
    CAT_RECENT_RAINFALL,
    CAT_ANTECEDENT_MONITORING,
    CAT_DRAINAGE_INSPECTION,
    CAT_FIELD_VERIFICATION,
    CAT_ROUTINE_MAINTENANCE,
    CAT_DATA_VERIFICATION,
]

VALID_PRIORITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "ROUTINE"]


def generate_ward_advisory_actions(
    ward_no: int,
    risk_level: str,
    final_score: float,
    base_score: float,
    rainfall_memory: Dict[str, Any],
    flood_momentum: Dict[str, Any],
    elevation_m: float,
    water_distance_km: float,
    observation_count: Optional[int] = 72
) -> Dict[str, Any]:
    """
    Generates deterministic, evidence-backed advisory action items for an individual ward.

    Parameters:
        ward_no: Integer ward identifier (1-200)
        risk_level: 'LOW', 'MEDIUM', or 'HIGH'
        final_score: Bounded risk score (0.0 - 200.0)
        base_score: XGBoost base score
        rainfall_memory: Dict with keys '1h', '3h', '6h', '8h', '12h', '24h', '72h'
        flood_momentum: Dict returned by calculate_flood_momentum()
        elevation_m: Centroid elevation in meters (uniform across wards in prototype)
        water_distance_km: Distance to nearest water body / canal in km
        observation_count: Number of hourly observations available in 72h window (expected: 72)

    Returns:
        Structured dictionary containing:
        - status: 'SUCCESS' or 'ERROR'
        - ward_no: int
        - risk_level: str
        - final_score: float
        - total_actions: int
        - actions: List of action items with stable ID, category, priority, description,
                   triggering evidence, rationale, and verification note.
    """
    # 1. Input Validation
    validation_err = _validate_action_inputs(
        ward_no, risk_level, final_score, base_score,
        rainfall_memory, flood_momentum, water_distance_km
    )
    if validation_err:
        return {
            "status": "ERROR",
            "error_type": "INVALID_INPUT",
            "ward_no": ward_no,
            "message": validation_err,
            "actions": []
        }

    # Normalize inputs
    r_1h = float(rainfall_memory.get("1h", 0.0))
    r_3h = float(rainfall_memory.get("3h", 0.0))
    r_8h = float(rainfall_memory.get("8h", 0.0))
    r_24h = float(rainfall_memory.get("24h", 0.0))
    r_72h = float(rainfall_memory.get("72h", 0.0))

    tranches = flood_momentum.get("tranches", {})
    r_recent = float(tranches.get("recent_0_8h_mm", r_8h))
    r_inter = float(tranches.get("intermediate_8_24h_mm", 0.0))
    r_past = float(tranches.get("past_24_72h_mm", 0.0))
    decayed_ant = float(flood_momentum.get("antecedent_rainfall_mm", 0.0))
    sat_ratio = float(flood_momentum.get("saturation_ratio", 0.0))
    alpha_active = float(flood_momentum.get("active_rainfall_factor", 1.0))
    burst_cat = str(flood_momentum.get("burst_category", "LOW"))
    w_dist = float(water_distance_km)
    norm_risk = str(risk_level).upper()

    actions: List[Dict[str, Any]] = []
    seen_action_ids = set()

    def add_action(action_dict: Dict[str, Any]):
        aid = action_dict["action_id"]
        if aid not in seen_action_ids:
            seen_action_ids.add(aid)
            actions.append(action_dict)

    # -------------------------------------------------------------------------
    # RULE 1: Acute Short-Term Burst Readiness (BURST_RESPONSE_READINESS)
    # -------------------------------------------------------------------------
    # Trigger: Severe 1h rain (>= 25mm) or burst_category is HEAVY or EXTREME
    if r_1h >= 25.0 or burst_cat in ["EXTREME_BURST", "HEAVY_BURST"]:
        is_extreme = (r_1h >= 35.0 or burst_cat == "EXTREME_BURST")

        # Expose exact trigger rule, threshold, and observation window
        if burst_cat == "EXTREME_BURST":
            if r_1h >= 30.0:
                trig_type = "EXTREME_1H_BURST"
                trig_win = "1-hour"
                burst_thresh = 30.0
                trig_crit = "1-hour rainfall >= 30.0 mm (EXTREME_BURST)"
            else:
                trig_type = "EXTREME_3H_BURST"
                trig_win = "3-hour"
                burst_thresh = 50.0
                trig_crit = "3-hour rainfall >= 50.0 mm (EXTREME_BURST)"
        elif burst_cat == "HEAVY_BURST":
            if r_1h >= 15.0:
                trig_type = "HEAVY_1H_BURST"
                trig_win = "1-hour"
                burst_thresh = 15.0
                trig_crit = "1-hour rainfall >= 15.0 mm (HEAVY_BURST)"
            else:
                trig_type = "HEAVY_3H_BURST"
                trig_win = "3-hour"
                burst_thresh = 30.0
                trig_crit = "3-hour rainfall >= 30.0 mm (HEAVY_BURST)"
        else:
            trig_type = "ACUTE_1H_PRECIPITATION"
            trig_win = "1-hour"
            burst_thresh = 25.0
            trig_crit = "1-hour acute rainfall >= 25.0 mm"

        add_action({
            "action_id": "ACT_BURST_RESPONSE_READINESS",
            "category": CAT_BURST_RESPONSE,
            "priority": "HIGH" if is_extreme else "MEDIUM",
            "action_description": (
                "Inspect critical road intersections and low-lying underpasses for localized surface runoff ponding; "
                "verify pump station operational readiness."
            ),
            "triggering_evidence": {
                "rainfall_1h_mm": round(r_1h, 2),
                "rainfall_3h_mm": round(r_3h, 2),
                "burst_category": burst_cat,
                "trigger_type": trig_type,
                "burst_threshold_mm": burst_thresh,
                "trigger_window": trig_win,
                "trigger_criterion": trig_crit,
                "active_rainfall_factor": round(alpha_active, 3),
                "active_rain_gate_threshold_mm": 25.0,
                "threshold_mm": burst_thresh
            },
            "rationale": (
                f"Short-term precipitation intensity of {r_1h:.1f} mm in {trig_win} ({burst_cat}) can temporarily exceed "
                "local road-surface drainage intake capacity before steady gravity flow is established."
            ),
            "verification_note": (
                "Advisory indicates acute precipitation load. It does not confirm standing flood depth or "
                "mandate public evacuation. Field verification required."
            )
        })

    # -------------------------------------------------------------------------
    # RULE 2: Recent Rainfall Review (RECENT_RAINFALL_REVIEW)
    # -------------------------------------------------------------------------
    # Trigger: 8h rainfall >= 25mm (and not extreme 1h burst) OR 24h rainfall >= 50mm
    if (r_8h >= 25.0 and r_1h < 35.0) or (r_24h >= 50.0):
        is_high_load = (r_8h >= 45.0 or r_24h >= 80.0 or norm_risk == "HIGH")
        if r_8h >= 25.0 and r_24h >= 50.0:
            r2_trig_type = "SUSTAINED_8H_AND_24H_LOAD"
            r2_win = "8-hour & 24-hour"
            r2_thresh = 25.0
            r2_crit = "8-hour active rainfall >= 25.0 mm and 24-hour rainfall >= 50.0 mm"
        elif r_8h >= 25.0:
            r2_trig_type = "SUSTAINED_8H_ACTIVE_LOAD"
            r2_win = "8-hour"
            r2_thresh = 25.0
            r2_crit = "8-hour active rainfall >= 25.0 mm"
        else:
            r2_trig_type = "ACCUMULATED_24H_LOAD"
            r2_win = "24-hour"
            r2_thresh = 50.0
            r2_crit = "24-hour accumulated rainfall >= 50.0 mm"

        add_action({
            "action_id": "ACT_RECENT_RAINFALL_REVIEW",
            "category": CAT_RECENT_RAINFALL,
            "priority": "HIGH" if is_high_load else "MEDIUM",
            "action_description": (
                "Conduct field surveillance along primary stormwater collector mains and monitor local drainage outfalls for tailwater rise."
            ),
            "triggering_evidence": {
                "rainfall_8h_mm": round(r_8h, 2),
                "rainfall_24h_mm": round(r_24h, 2),
                "recent_0_8h_tranche_mm": round(r_recent, 2),
                "threshold_8h_mm": r2_thresh,
                "trigger_type": r2_trig_type,
                "trigger_window": r2_win,
                "trigger_criterion": r2_crit
            },
            "rationale": (
                f"Sustained precipitation ({r_8h:.1f} mm in 8h, {r_24h:.1f} mm in 24h) creates sustained hydrodynamic load "
                "on secondary and primary stormwater conduit networks."
            ),
            "verification_note": (
                "Gravity discharge rate depends on canal receiving levels; verify receiving water elevation manually."
            )
        })

    # -------------------------------------------------------------------------
    # RULE 3: Antecedent Soil Saturation & Historical Rainfall Monitoring
    # -------------------------------------------------------------------------
    # Scenario A: Dry recent window (r_8h <= 5.0) but high historical accumulation (past_24_72h >= 30.0 or r_72h >= 40.0)
    # Explicitly prevents emergency escalation for dry recent periods!
    if r_8h <= 5.0 and (r_past >= 30.0 or r_72h >= 40.0):
        add_action({
            "action_id": "ACT_ANTECEDENT_MONITORING_RECOVERY",
            "category": CAT_ANTECEDENT_MONITORING,
            "priority": "LOW",
            "action_description": (
                "Maintain passive ground monitoring. Note that while recent 8h rainfall has subsided, elevated antecedent moisture proxy "
                "reduces infiltration buffer for future rainfall."
            ),
            "triggering_evidence": {
                "rainfall_8h_mm": round(r_8h, 2),
                "past_24_72h_mm": round(r_past, 2),
                "total_72h_mm": round(r_72h, 2),
                "saturation_ratio": round(sat_ratio, 3),
                "active_rainfall_factor": round(alpha_active, 3)
            },
            "rationale": (
                f"Historical 72-hour accumulation ({r_72h:.1f} mm) keeps soil saturation proxy elevated ({sat_ratio:.2f}), "
                f"but recent dry conditions ({r_8h:.1f} mm in 8h, active factor {alpha_active:.2f}x) indicate immediate runoff generation is low. "
                "Emergency deployment is not warranted."
            ),
            "verification_note": (
                "Saturation ratio is a mathematical Antecedent Precipitation Index (API) proxy, not measured soil moisture. "
                "No active inundation is implied."
            )
        })

    # Scenario B: High antecedent saturation with ongoing rain (sat_ratio >= 0.65 and r_8h > 5.0)
    elif sat_ratio >= 0.65 and r_8h > 5.0:
        add_action({
            "action_id": "ACT_HIGH_ANTECEDENT_SATURATION_ALERT",
            "category": CAT_ANTECEDENT_MONITORING,
            "priority": "HIGH" if norm_risk == "HIGH" else "MEDIUM",
            "action_description": (
                "Alert ward maintenance crew to rapid runoff risk; high cumulative antecedent rainfall implies reduced ground absorption capacity."
            ),
            "triggering_evidence": {
                "saturation_ratio": round(sat_ratio, 3),
                "decayed_antecedent_mm": round(decayed_ant, 2),
                "rainfall_8h_mm": round(r_8h, 2)
            },
            "rationale": (
                f"Calculated saturation proxy is {sat_ratio:.2f} with decayed antecedent accumulation of {decayed_ant:.1f} mm. "
                "Ground infiltration is likely restricted, increasing surface runoff fraction from ongoing precipitation."
            ),
            "verification_note": (
                "Antecedent precipitation proxy does not measure groundwater table depth or localized soil hydraulic conductivity."
            )
        })

    # -------------------------------------------------------------------------
    # RULE 4: Drainage & Canal Outfall Inspection (DRAINAGE_INSPECTION)
    # -------------------------------------------------------------------------
    # Spatial proximity to open waterbody/canal (<= 2.5 km) with moderate or high hydrological load
    if w_dist <= 2.5 and (norm_risk in ["MEDIUM", "HIGH"] or r_24h >= 20.0 or r_8h >= 15.0):
        add_action({
            "action_id": "ACT_CANAL_OUTFALL_INSPECTION",
            "category": CAT_DRAINAGE_INSPECTION,
            "priority": "HIGH" if norm_risk == "HIGH" else "MEDIUM",
            "action_description": (
                f"Inspect flap gates, sluices, and outfall drains connecting ward catchments to nearby open water bodies or canals "
                f"({w_dist:.2f} km) to ensure unrestricted gravity discharge."
            ),
            "triggering_evidence": {
                "water_distance_km": round(w_dist, 3),
                "distance_threshold_km": 2.5,
                "ward_risk_level": norm_risk,
                "rainfall_24h_mm": round(r_24h, 2)
            },
            "rationale": (
                f"Ward centroid is {w_dist:.2f} km from a major water body or canal. During rainfall events, "
                "gravity discharge depends on unobstructed outfalls and favorable receiving water levels."
            ),
            "verification_note": (
                "Water distance is a GIS geographic attribute. It is not a measurement of canal stage, "
                "siltation level, or drain blockage."
            )
        })

    # -------------------------------------------------------------------------
    # RULE 5: High-Risk Field Verification (FIELD_VERIFICATION)
    # -------------------------------------------------------------------------
    # Model indicates HIGH risk (final_score >= 120.0)
    if norm_risk == "HIGH" or final_score >= 120.0:
        add_action({
            "action_id": "ACT_HIGH_RISK_FIELD_SURVEILLANCE",
            "category": CAT_FIELD_VERIFICATION,
            "priority": "HIGH",
            "action_description": (
                "Dispatch zonal rapid response squad to conduct physical drive-by verification of known chronic inundation hotspots "
                "and underpasses in the ward."
            ),
            "triggering_evidence": {
                "final_risk_score": round(final_score, 2),
                "base_score": round(base_score, 2),
                "risk_level": norm_risk,
                "high_risk_threshold": 120.0
            },
            "rationale": (
                f"Statistical risk engine and momentum multiplier indicate elevated hydrological pressure ({final_score:.1f} / 200). "
                "Physical ground observation is necessary to confirm actual surface water status before deploying heavy equipment."
            ),
            "verification_note": (
                "Predictive decision support only. Does not assert confirmed flooding or mandate emergency evacuation."
            )
        })

    # -------------------------------------------------------------------------
    # RULE 6: Routine Preventive Maintenance (ROUTINE_PREVENTIVE_MAINTENANCE)
    # -------------------------------------------------------------------------
    # Baseline condition: LOW risk and no acute triggers
    if norm_risk == "LOW" and final_score < 80.0 and r_1h < 15.0 and r_8h < 20.0:
        add_action({
            "action_id": "ACT_ROUTINE_DRAIN_MAINTENANCE",
            "category": CAT_ROUTINE_MAINTENANCE,
            "priority": "ROUTINE",
            "action_description": (
                "Continue scheduled pre-monsoon catch-pit cleaning, silt removal, and routine solid waste desilting along roadside drains."
            ),
            "triggering_evidence": {
                "final_risk_score": round(final_score, 2),
                "risk_level": norm_risk,
                "rainfall_8h_mm": round(r_8h, 2),
                "low_risk_ceiling": 80.0
            },
            "rationale": (
                f"Ward displays low modeled risk ({final_score:.1f} / 200) and minimal acute rainfall ({r_8h:.1f} mm in 8h). "
                "Favorable conditions exist for routine preventive maintenance."
            ),
            "verification_note": (
                "Standard municipal maintenance protocol during baseline non-flood conditions."
            )
        })

    # -------------------------------------------------------------------------
    # RULE 7: Data Gap / Telemetry Verification (DATA_GAP_VERIFICATION)
    # -------------------------------------------------------------------------
    # Trigger: Observation count is incomplete (< 72 observations in 72-hour window)
    if observation_count is not None and observation_count < 72:
        add_action({
            "action_id": "ACT_DATA_OBSERVATION_AUDIT",
            "category": CAT_DATA_VERIFICATION,
            "priority": "MEDIUM",
            "action_description": (
                f"Verify telemetry feed continuity and data station synchronization; only {observation_count} / 72 hourly "
                "observations were available for this ward."
            ),
            "triggering_evidence": {
                "available_observations": int(observation_count),
                "expected_observations": 72,
                "missing_hours": 72 - int(observation_count)
            },
            "rationale": (
                f"Hydrological momentum calculation relies on a complete 72-hour sliding window. "
                f"An observation count of {observation_count} may distort antecedent soil saturation or tranche decay values."
            ),
            "verification_note": (
                "Sensor communication or database ingest audit recommended. Missing observations should not be assumed zero without verification."
            )
        })

    # Fallback safety: If for any reason no actions were matched, provide baseline review
    if not actions:
        actions.append({
            "action_id": "ACT_BASELINE_WARD_MONITORING",
            "category": CAT_ROUTINE_MAINTENANCE,
            "priority": "ROUTINE",
            "action_description": "Maintain standard ward drainage watch and check weather radar forecasts.",
            "triggering_evidence": {"final_score": round(final_score, 2), "risk_level": norm_risk},
            "rationale": "No acute thresholds exceeded; standard monitoring applies.",
            "verification_note": "Passive operational watch."
        })

    return {
        "status": "SUCCESS",
        "ward_no": int(ward_no),
        "risk_level": norm_risk,
        "final_score": round(final_score, 2),
        "total_actions": len(actions),
        "actions": actions,
        "operational_disclaimer": (
            "Advisories are generated deterministically from predictive model scores, rainfall momentum, and spatial coordinates. "
            "They represent operational decision-support recommendations and do not command emergency evacuations or assert observed inundation."
        )
    }


def get_available_action_catalog() -> List[Dict[str, Any]]:
    """
    Returns the static catalog of all documented advisory actions, categories,
    priorities, and default rule descriptions.
    """
    return [
        {
            "action_id": "ACT_BURST_RESPONSE_READINESS",
            "category": CAT_BURST_RESPONSE,
            "default_priority": "HIGH",
            "trigger_condition": "r_1h >= 25.0 mm or burst_category in ['EXTREME_BURST', 'HEAVY_BURST']",
            "description": "Inspect critical road intersections and underpasses for acute surface runoff ponding; verify pump readiness."
        },
        {
            "action_id": "ACT_RECENT_RAINFALL_REVIEW",
            "category": CAT_RECENT_RAINFALL,
            "default_priority": "MEDIUM",
            "trigger_condition": "r_8h >= 25.0 mm or r_24h >= 50.0 mm",
            "description": "Conduct field surveillance along primary stormwater collector mains and monitor drainage outfalls."
        },
        {
            "action_id": "ACT_ANTECEDENT_MONITORING_RECOVERY",
            "category": CAT_ANTECEDENT_MONITORING,
            "default_priority": "LOW",
            "trigger_condition": "r_8h <= 5.0 mm and past_24_72h_mm >= 30.0 mm (dry recent window with historical accumulation)",
            "description": "Maintain passive ground monitoring; avoid emergency escalation during dry recent conditions."
        },
        {
            "action_id": "ACT_HIGH_ANTECEDENT_SATURATION_ALERT",
            "category": CAT_ANTECEDENT_MONITORING,
            "default_priority": "MEDIUM",
            "trigger_condition": "saturation_ratio >= 0.65 and r_8h > 5.0 mm (wet ongoing conditions on saturated ground)",
            "description": "Alert maintenance crew to rapid runoff risk due to reduced soil infiltration capacity."
        },
        {
            "action_id": "ACT_CANAL_OUTFALL_INSPECTION",
            "category": CAT_DRAINAGE_INSPECTION,
            "default_priority": "MEDIUM",
            "trigger_condition": "water_distance_km <= 2.5 km and risk_level in ['MEDIUM', 'HIGH']",
            "description": "Inspect flap gates and sluices connecting ward catchments to nearby open water bodies or canals."
        },
        {
            "action_id": "ACT_HIGH_RISK_FIELD_SURVEILLANCE",
            "category": CAT_FIELD_VERIFICATION,
            "default_priority": "HIGH",
            "trigger_condition": "risk_level == 'HIGH' or final_score >= 120.0",
            "description": "Dispatch zonal rapid response squad for physical field verification of known chronic inundation hotspots."
        },
        {
            "action_id": "ACT_ROUTINE_DRAIN_MAINTENANCE",
            "category": CAT_ROUTINE_MAINTENANCE,
            "default_priority": "ROUTINE",
            "trigger_condition": "risk_level == 'LOW' and final_score < 80.0 and dry conditions",
            "description": "Continue scheduled pre-monsoon catch-pit cleaning, silt removal, and routine solid waste desilting."
        },
        {
            "action_id": "ACT_DATA_OBSERVATION_AUDIT",
            "category": CAT_DATA_VERIFICATION,
            "default_priority": "MEDIUM",
            "trigger_condition": "observation_count < 72 (incomplete 72-hour sliding window)",
            "description": "Audit station telemetry and database ingest; fewer than 72 hourly observations available."
        }
    ]


def _validate_action_inputs(
    ward_no: int,
    risk_level: str,
    final_score: float,
    base_score: float,
    rainfall_memory: Dict[str, Any],
    flood_momentum: Dict[str, Any],
    water_distance_km: float
) -> Optional[str]:
    """Validates types and ranges for action engine inputs."""
    if not isinstance(ward_no, int) or ward_no <= 0:
        return f"ward_no must be a positive integer, got {ward_no}"

    if not isinstance(risk_level, str) or risk_level.upper() not in ["LOW", "MEDIUM", "HIGH"]:
        return f"risk_level must be 'LOW', 'MEDIUM', or 'HIGH', got {risk_level}"

    for name, val in [("final_score", final_score), ("base_score", base_score), ("water_distance_km", water_distance_km)]:
        if val is None or not isinstance(val, (int, float)):
            return f"Parameter '{name}' must be numeric, got {type(val)}"

    if not isinstance(rainfall_memory, dict):
        return f"rainfall_memory must be a dictionary, got {type(rainfall_memory)}"

    if not isinstance(flood_momentum, dict):
        return f"flood_momentum must be a dictionary, got {type(flood_momentum)}"

    return None
