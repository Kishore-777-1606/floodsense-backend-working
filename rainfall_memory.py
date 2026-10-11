"""
FloodSense Rainfall Memory & Flood Momentum Engine
Calculates operational short-term rainfall memory (1h, 3h, 6h, 12h, 24h, 72h),
aggregates long-term historical context (7d, 30d, 3m, 6m, 9m),
and derives Antecedent Precipitation / Flood Momentum metrics.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from db import get_db, is_connected
from rainfall_storage import get_latest_observation_time


def get_reference_timestamp(
    ref_time: Optional[datetime] = None,
    ward_no: Optional[int] = None
) -> datetime:
    """
    Returns an anchored reference timestamp for deterministic window calculations.
    If ref_time is provided, uses that timestamp.
    If ward_no is provided, uses the latest observation timestamp for that specific ward
    to prevent cross-ward reference clock leakage.
    Falls back to global latest observation or UTC now.
    """
    if ref_time is not None:
        if ref_time.tzinfo is None:
            return ref_time.replace(tzinfo=timezone.utc)
        return ref_time.astimezone(timezone.utc)

    # Ward-specific isolation: fetch latest observation for this specific ward
    latest = get_latest_observation_time(ward_no=ward_no)
    if latest is None and ward_no is not None:
        # Fall back to global latest if this specific ward has no records
        latest = get_latest_observation_time(ward_no=None)

    if latest is not None:
        if latest.tzinfo is None:
            return latest.replace(tzinfo=timezone.utc)
        return latest.astimezone(timezone.utc)

    return datetime.now(timezone.utc)


def calculate_ward_rainfall_memory(
    ward_no: int,
    ref_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Calculates operational short-term rainfall memory for a specific ward.
    Windows: 1h, 3h, 6h, 12h, 24h, 72h.
    Derived dynamically from timestamped observations in MongoDB.
    """
    if not is_connected():
        return {
            "ward_no": int(ward_no),
            "1h": 0.0, "3h": 0.0, "6h": 0.0, "12h": 0.0, "24h": 0.0, "72h": 0.0,
            "observation_count": 0,
            "ref_timestamp": None,
            "status": "db_not_connected"
        }

    t_ref = get_reference_timestamp(ref_time, ward_no=ward_no)
    t_1h = t_ref - timedelta(hours=1)
    t_3h = t_ref - timedelta(hours=3)
    t_6h = t_ref - timedelta(hours=6)
    t_8h = t_ref - timedelta(hours=8)
    t_12h = t_ref - timedelta(hours=12)
    t_24h = t_ref - timedelta(hours=24)
    t_72h = t_ref - timedelta(hours=72)

    db = get_db()
    pipeline = [
        {
            "$match": {
                "ward_no": int(ward_no),
                "timestamp": {"$gte": t_72h, "$lte": t_ref}
            }
        },
        {
            "$group": {
                "_id": "$ward_no",
                "rain_1h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_1h]}, "$rainfall_mm", 0]}
                },
                "rain_3h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_3h]}, "$rainfall_mm", 0]}
                },
                "rain_6h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_6h]}, "$rainfall_mm", 0]}
                },
                "rain_8h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_8h]}, "$rainfall_mm", 0]}
                },
                "rain_12h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_12h]}, "$rainfall_mm", 0]}
                },
                "rain_24h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_24h]}, "$rainfall_mm", 0]}
                },
                "rain_72h": {
                    "$sum": "$rainfall_mm"
                },
                "count": {"$sum": 1},
                "latest_obs_time": {"$max": "$timestamp"}
            }
        }
    ]

    results = list(db["rainfall_observations"].aggregate(pipeline))
    if not results:
        return {
            "ward_no": int(ward_no),
            "1h": 0.0,
            "3h": 0.0,
            "6h": 0.0,
            "8h": 0.0,
            "12h": 0.0,
            "24h": 0.0,
            "72h": 0.0,
            "observation_count": 0,
            "ref_timestamp": t_ref.isoformat(),
            "latest_observation_time": None
        }

    row = results[0]
    return {
        "ward_no": int(ward_no),
        "1h": round(float(row.get("rain_1h", 0.0)), 2),
        "3h": round(float(row.get("rain_3h", 0.0)), 2),
        "6h": round(float(row.get("rain_6h", 0.0)), 2),
        "8h": round(float(row.get("rain_8h", 0.0)), 2),
        "12h": round(float(row.get("rain_12h", 0.0)), 2),
        "24h": round(float(row.get("rain_24h", 0.0)), 2),
        "72h": round(float(row.get("rain_72h", 0.0)), 2),
        "observation_count": int(row.get("count", 0)),
        "ref_timestamp": t_ref.isoformat(),
        "latest_observation_time": row.get("latest_obs_time").isoformat() if row.get("latest_obs_time") else None
    }


def calculate_all_wards_rainfall_memory(
    ref_time: Optional[datetime] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Computes 1h, 3h, 6h, 12h, 24h, 72h rainfall memory for all wards in a single
    efficient MongoDB aggregation pass using the shared/city-wide snapshot timestamp.
    """
    if not is_connected():
        return {}

    t_ref = get_reference_timestamp(ref_time, ward_no=None)
    t_1h = t_ref - timedelta(hours=1)
    t_3h = t_ref - timedelta(hours=3)
    t_6h = t_ref - timedelta(hours=6)
    t_8h = t_ref - timedelta(hours=8)
    t_12h = t_ref - timedelta(hours=12)
    t_24h = t_ref - timedelta(hours=24)
    t_72h = t_ref - timedelta(hours=72)

    db = get_db()
    pipeline = [
        {
            "$match": {
                "timestamp": {"$gte": t_72h, "$lte": t_ref}
            }
        },
        {
            "$group": {
                "_id": "$ward_no",
                "rain_1h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_1h]}, "$rainfall_mm", 0]}
                },
                "rain_3h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_3h]}, "$rainfall_mm", 0]}
                },
                "rain_6h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_6h]}, "$rainfall_mm", 0]}
                },
                "rain_8h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_8h]}, "$rainfall_mm", 0]}
                },
                "rain_12h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_12h]}, "$rainfall_mm", 0]}
                },
                "rain_24h": {
                    "$sum": {"$cond": [{"$gte": ["$timestamp", t_24h]}, "$rainfall_mm", 0]}
                },
                "rain_72h": {
                    "$sum": "$rainfall_mm"
                },
                "count": {"$sum": 1},
                "latest_obs_time": {"$max": "$timestamp"}
            }
        }
    ]

    records = db["rainfall_observations"].aggregate(pipeline)
    ward_memory_map = {}

    for row in records:
        w_no = int(row["_id"])
        ward_memory_map[w_no] = {
            "ward_no": w_no,
            "1h": round(float(row.get("rain_1h", 0.0)), 2),
            "3h": round(float(row.get("rain_3h", 0.0)), 2),
            "6h": round(float(row.get("rain_6h", 0.0)), 2),
            "8h": round(float(row.get("rain_8h", 0.0)), 2),
            "12h": round(float(row.get("rain_12h", 0.0)), 2),
            "24h": round(float(row.get("rain_24h", 0.0)), 2),
            "72h": round(float(row.get("rain_72h", 0.0)), 2),
            "observation_count": int(row.get("count", 0)),
            "ref_timestamp": t_ref.isoformat(),
            "latest_observation_time": row.get("latest_obs_time").isoformat() if row.get("latest_obs_time") else None
        }

    return ward_memory_map


def get_ward_long_term_history(
    ward_no: int,
    ref_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Retrieves aggregated longer-term calendar history (7d, 30d, 3m, 6m, 9m)
    from `rainfall_daily_summary`.
    Windows use explicit calendar-day boundaries and report available days,
    missing days, date ranges, and verified sums.
    """
    if not is_connected():
        empty_window = {
            "total_mm": 0.0, "avg_daily_mm": 0.0, "max_daily_mm": 0.0,
            "expected_days": 0, "available_days": 0, "missing_days": 0, "incomplete_days": 0
        }
        return {
            "ward_no": int(ward_no),
            "7d": empty_window,
            "30d": empty_window,
            "3m": empty_window,
            "6m": empty_window,
            "9m": empty_window,
            "status": "db_not_connected"
        }

    t_ref = get_reference_timestamp(ref_time, ward_no=ward_no)

    db = get_db()
    daily_col = db["rainfall_daily_summary"]

    # Explicit calendar-day boundary: historical rollups strictly cover completed
    # calendar days prior to the current reference day (t_ref).
    ref_day_start = t_ref.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = ref_day_start - timedelta(days=1)

    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # Windows configuration: (key, expected_calendar_days)
    windows_config = [
        ("7d", 7),
        ("30d", 30),
        ("3m", 90),
        ("6m", 180),
        ("9m", 270),
    ]

    # Earliest boundary needed for the 270-day window
    earliest_start = end_date - timedelta(days=269)

    # Fetch all daily summary records in range for this ward in a single query
    daily_records = list(daily_col.find(
        {
            "ward_no": int(ward_no),
            "date": {"$gte": earliest_start, "$lte": end_date}
        }
    ).sort("date", 1))

    # Helper function to compute window statistics
    def _compute_calendar_window(expected_days: int) -> Dict[str, Any]:
        start_date = end_date - timedelta(days=expected_days - 1)

        # Filter matching records in memory
        matching_docs = []
        for d in daily_records:
            doc_date = d["date"].replace(tzinfo=timezone.utc) if d["date"].tzinfo is None else d["date"]
            if start_date <= doc_date <= end_date:
                matching_docs.append(d)

        avail_cnt = len(matching_docs)
        missing_cnt = max(0, expected_days - avail_cnt)
        incomplete_cnt = sum(1 for d in matching_docs if int(d.get("observation_count", 24)) < 24)

        tot = round(sum(float(d.get("total_rainfall_mm", 0.0)) for d in matching_docs), 2)
        avg = round(tot / avail_cnt, 2) if avail_cnt > 0 else 0.0

        if matching_docs:
            peak_doc = max(matching_docs, key=lambda d: float(d.get("total_rainfall_mm", 0.0)))
            mx = round(float(peak_doc.get("total_rainfall_mm", 0.0)), 2)
            pk_date = peak_doc["date"].strftime("%Y-%m-%d") if isinstance(peak_doc.get("date"), datetime) else str(peak_doc.get("date"))
        else:
            mx = 0.0
            pk_date = None

        return {
            # Backward-compatible fields
            "total_mm": tot,
            "avg_daily_mm": avg,
            "max_daily_mm": mx,

            # Corrected window audit metadata
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "expected_days": expected_days,
            "available_days": avail_cnt,
            "missing_days": missing_cnt,
            "incomplete_days": incomplete_cnt,
            "max_daily_date": pk_date,
            "denominator_basis": "available_observed_days"
        }

    return {
        "ward_no": int(ward_no),
        "7d": _compute_calendar_window(7),
        "30d": _compute_calendar_window(30),
        "3m": _compute_calendar_window(90),
        "6m": _compute_calendar_window(180),
        "9m": _compute_calendar_window(270),
    }


def calculate_flood_momentum(
    rainfall_memory: Dict[str, Any],
    elevation_m: float,
    water_distance_km: float
) -> Dict[str, Any]:
    """
    Computes Time-Aware Antecedent Precipitation & Flood Momentum.
    
    Hydrological & Physical Design:
    1. Multi-Tranche Time Decay:
       Instead of allowing rainfall from 3 days ago to indefinitely escalate risk,
       the 72-hour period is partitioned into 3 non-overlapping chronological tranches:
       - Active/Recent Tranche (0–8h): Full weight (1.00) driving active surface runoff.
       - Intermediate Tranche (8–24h): Attenuated weight (0.50) reflecting soil moisture
         retention in the upper vadose zone after initial gravity drainage.
       - Past Baseline Tranche (24–72h): Attenuated weight (0.15) reflecting deep percolation
         and baseflow elevation. Old rainfall alone CANNOT trigger HIGH flood momentum.
       
    2. Active Rainfall Gating (alpha_active):
       True flood momentum requires active rainfall pressure to mobilize runoff.
       If recent rainfall is negligible (0–8h <= 2mm), alpha_active attenuates the momentum
       multiplier (down to 0.15 baseline), preventing old storms from inflating current threat.
       If an intense burst occurs (1h >= 10mm or 8h >= 25mm), alpha_active scales up to 1.00.
       
    3. Topographical Drainage Handicap:
       Preserves terrain elevation (<8m) and proximity to water bodies (<5km) factors.
    """
    r_1h = float(rainfall_memory.get("1h", 0.0))
    r_3h = float(rainfall_memory.get("3h", 0.0))
    r_24h = float(rainfall_memory.get("24h", 0.0))
    r_72h = float(rainfall_memory.get("72h", 0.0))

    # 8h window fallback: if 8h is not in rainfall_memory, infer from 6h or 24h for backward compatibility
    r_8h = float(rainfall_memory.get("8h", rainfall_memory.get("6h", round(r_24h * 0.40, 2))))

    # 1. Non-overlapping chronological tranches (all in mm)
    r_recent = max(0.0, r_8h)
    r_intermediate = max(0.0, r_24h - r_8h)
    r_past = max(0.0, r_72h - r_24h)

    # 2. Time-decayed effective antecedent precipitation (mm)
    antecedent_mm = round((1.00 * r_recent) + (0.50 * r_intermediate) + (0.15 * r_past), 2)

    # 3. Soil Saturation Ratio (0.0 to 1.0, saturated at ~120mm effective antecedent precipitation)
    saturation_ratio = min(1.0, max(0.0, antecedent_mm / 120.0))

    # 4. Topographical drainage handicap factor
    elev_handicap = max(0.0, (8.0 - elevation_m) / 25.0)
    water_handicap = max(0.0, (5.0 - min(water_distance_km, 5.0)) / 25.0)
    topography_handicap = 1.0 + elev_handicap + water_handicap

    # 5. Active rainfall factor (alpha_active in [0.15, 1.00])
    # Prevents old rain from driving flood momentum when the recent period has dried out
    if r_1h >= 10.0 or r_8h >= 25.0:
        alpha_active = 1.00
    else:
        alpha_active = round(min(1.00, max(0.15, 0.15 + 0.85 * (r_8h / 25.0))), 3)

    # 6. Momentum multiplier (bounded strictly between 1.00 and 1.75)
    raw_multiplier = 1.0 + (0.50 * saturation_ratio * topography_handicap * alpha_active)
    momentum_multiplier = round(min(1.75, max(1.0, raw_multiplier)), 3)

    # 7. Recent burst categorization
    if r_1h >= 30.0 or r_3h >= 50.0:
        burst_category = "EXTREME_BURST"
    elif r_1h >= 15.0 or r_3h >= 30.0:
        burst_category = "HEAVY_BURST"
    elif r_1h >= 5.0 or r_3h >= 12.0:
        burst_category = "MODERATE"
    else:
        burst_category = "LOW"

    # 8. Momentum Level classification (requires active recent rain for HIGH / CRITICAL)
    if momentum_multiplier >= 1.40 or (antecedent_mm >= 80.0 and r_8h >= 20.0):
        momentum_level = "CRITICAL"
    elif momentum_multiplier >= 1.25 or (antecedent_mm >= 45.0 and r_8h >= 10.0):
        momentum_level = "HIGH"
    elif momentum_multiplier >= 1.08 or antecedent_mm >= 20.0 or r_8h >= 8.0:
        momentum_level = "MODERATE"
    else:
        momentum_level = "LOW"

    return {
        # Backward-compatible fields
        "antecedent_rainfall_mm": antecedent_mm,
        "saturation_ratio": round(saturation_ratio, 3),
        "momentum_multiplier": momentum_multiplier,
        "momentum_level": momentum_level,
        "accumulated_24h_mm": r_24h,
        "accumulated_72h_mm": r_72h,

        # Time-aware enhancement fields
        "accumulated_8h_mm": r_8h,
        "active_rainfall_factor": alpha_active,
        "burst_category": burst_category,
        "tranches": {
            "recent_0_8h_mm": r_recent,
            "intermediate_8_24h_mm": r_intermediate,
            "past_24_72h_mm": r_past,
        },
        "experimental_time_aware": True
    }
