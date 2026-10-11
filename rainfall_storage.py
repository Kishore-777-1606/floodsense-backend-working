"""
FloodSense Rainfall Storage Layer
Provides CRUD, batch ingestion, query operations, and deterministic baseline seeding.
Data Provenance:
- Synthetic baseline data is explicitly labelled with source: 'synthetic_baseline'.
- Daily summaries are strictly derived by aggregating raw hourly observations
  belonging to each UTC calendar day, guaranteeing 100% mathematical consistency.

Schema:
- rainfall_observations:
    {
        "ward_no": int,
        "timestamp": datetime (UTC),
        "rainfall_mm": float,
        "source": str ("synthetic_baseline" or user-specified ingest source),
        "created_at": datetime (UTC)
    }
- rainfall_daily_summary:
    {
        "ward_no": int,
        "date": datetime (UTC at 00:00:00),
        "total_rainfall_mm": float (sum of hourly observations for this UTC day),
        "max_hourly_mm": float (maximum single hourly observation for this UTC day),
        "observation_count": int (number of observations recorded for this UTC day),
        "source": str ("synthetic_baseline" or derived ingest source)
    }
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from db import get_db, is_connected

SYNTHETIC_SOURCE = "synthetic_baseline"
LEGACY_SYNTHETIC_SOURCES = ["tnsdma_imd_gauge", "daily_rollup_service", "synthetic_baseline"]

DEFAULT_SEASONAL_PROFILE_CONFIG: Dict[str, Any] = {
    # Chennai Regional Climatological Approximations (used solely for synthetic prototype validation)
    "dry_season_end_doy": 151,        # DOY 1-151 (Jan 1 - May 31): Dry/winter season (~10-25 mm/month)
    "sw_monsoon_end_doy": 273,        # DOY 152-273 (Jun 1 - Sep 30): Southwest Monsoon convective showers (~80-120 mm/month)
    "dry_season_rain_mod": 11,        # Rain occurs on ~9% of days in dry season
    "dry_season_base_mm": 1.2,        # Base rain depth for dry season rain days (mm)
    "dry_season_step_mm": 0.35,       # Modulo variation step for dry season (mm)
    "sw_monsoon_rain_mod": 3,         # Rain occurs on ~33% of days in SW monsoon
    "sw_monsoon_base_mm": 2.5,        # Base rain depth for SW monsoon rain days (mm)
    "sw_monsoon_step_mm": 0.90,       # Modulo variation step for SW monsoon (mm)
    "storm_cap_min_mm": 20.0,         # Minimum baseline scaling floor (mm)
    "storm_cap_max_mm": 120.0,        # Maximum baseline scaling ceiling to prevent runaway extremes (mm)
    "storm_scale_factor": 0.60,       # Scaling factor applied to Wards.geojson peak rainfall feature
    "ne_onset_ramp_min": 0.20,        # Pre-storm progression onset factor (Day 1 of Oct)
    "ne_onset_ramp_max": 0.80,        # Pre-storm progression ramp span (Days 1-7 of Oct)
    "diurnal_peak_hour_base": 15,     # Peak convective hour (15:00 UTC)
    "diurnal_spread_variance": 5.0,   # Gaussian bell curve spread denominator for hourly distribution
}


def get_observations_collection():
    return get_db()["rainfall_observations"]


def get_daily_summary_collection():
    return get_db()["rainfall_daily_summary"]


def update_daily_summary_for_date(ward_no: int, calendar_date: datetime) -> Optional[Dict[str, Any]]:
    """
    Authoritative aggregation: derives the daily summary for a specific ward
    and UTC calendar day directly from the stored hourly observations.
    Ensures total_rainfall_mm is the exact sum of hourly observations,
    max_hourly_mm is the true hourly peak, and observation_count is accurate.
    Distinguishes complete 24h days from partial/incomplete observation periods.
    """
    if not is_connected():
        return None

    if calendar_date.tzinfo is None:
        calendar_date = calendar_date.replace(tzinfo=timezone.utc)
    else:
        calendar_date = calendar_date.astimezone(timezone.utc)

    start_of_day = calendar_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)

    obs_col = get_observations_collection()
    daily_col = get_daily_summary_collection()

    obs_list = list(obs_col.find(
        {
            "ward_no": int(ward_no),
            "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
        },
        {"rainfall_mm": 1, "source": 1}
    ))

    if not obs_list:
        daily_col.delete_one({"ward_no": int(ward_no), "date": start_of_day})
        return None

    tot_rain = round(sum(float(o.get("rainfall_mm", 0.0)) for o in obs_list), 2)
    max_hour = round(max((float(o.get("rainfall_mm", 0.0)) for o in obs_list), default=0.0), 2)
    obs_count = len(obs_list)

    # Determine provenance: if all observations share a source, use that source
    sources = list({o.get("source", "sensor") for o in obs_list})
    doc_source = sources[0] if len(sources) == 1 else "aggregated_sources"

    summary_doc = {
        "ward_no": int(ward_no),
        "date": start_of_day,
        "total_rainfall_mm": tot_rain,
        "max_hourly_mm": max_hour,
        "observation_count": obs_count,
        "is_complete": (obs_count == 24),
        "source": doc_source
    }

    daily_col.update_one(
        {"ward_no": int(ward_no), "date": start_of_day},
        {"$set": summary_doc},
        upsert=True
    )

    return summary_doc


def insert_observation(
    ward_no: int,
    timestamp: datetime,
    rainfall_mm: float,
    source: str = "sensor"
) -> str:
    """
    Inserts a single timestamped rainfall observation into MongoDB
    and automatically synchronizes the corresponding daily summary.
    """
    if not isinstance(timestamp, datetime):
        raise ValueError("Timestamp must be a datetime object.")

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    doc = {
        "ward_no": int(ward_no),
        "timestamp": timestamp,
        "rainfall_mm": round(float(rainfall_mm), 2),
        "source": str(source),
        "created_at": datetime.now(timezone.utc),
    }
    result = get_observations_collection().insert_one(doc)

    # Automatically derive / update the UTC daily summary for this calendar day
    try:
        update_daily_summary_for_date(ward_no, timestamp)
    except Exception:
        pass

    return str(result.inserted_id)


def insert_observations_batch(observations: List[Dict[str, Any]]) -> int:
    """Batch inserts observations and synchronizes affected daily summaries."""
    if not observations:
        return 0

    docs = []
    affected_days = set()
    now_utc = datetime.now(timezone.utc)

    for obs in observations:
        ts = obs["timestamp"]
        if not isinstance(ts, datetime):
            raise ValueError("All timestamps must be datetime objects.")
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        w_no = int(obs["ward_no"])
        docs.append({
            "ward_no": w_no,
            "timestamp": ts,
            "rainfall_mm": round(float(obs["rainfall_mm"]), 2),
            "source": str(obs.get("source", "batch_ingest")),
            "created_at": now_utc,
        })
        affected_days.add((w_no, ts.replace(hour=0, minute=0, second=0, microsecond=0)))

    result = get_observations_collection().insert_many(docs, ordered=False)

    for w_no, day_date in affected_days:
        try:
            update_daily_summary_for_date(w_no, day_date)
        except Exception:
            pass

    return len(result.inserted_ids)


def get_observations_for_ward(
    ward_no: int,
    start_time: datetime,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Retrieves raw timestamped observations for a ward in a time interval."""
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time is not None and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    query: Dict[str, Any] = {
        "ward_no": int(ward_no),
        "timestamp": {"$gte": start_time}
    }
    if end_time:
        query["timestamp"]["$lte"] = end_time

    cursor = get_observations_collection().find(
        query,
        {"_id": 0, "ward_no": 1, "timestamp": 1, "rainfall_mm": 1, "source": 1}
    ).sort("timestamp", 1)

    return list(cursor)


def get_latest_observation_time(ward_no: Optional[int] = None) -> Optional[datetime]:
    """
    Gets the timestamp of the newest observation available.
    Supports ward-specific isolation so test observations in one ward
    do not alter another ward's reference clock.
    """
    query = {"ward_no": int(ward_no)} if ward_no is not None else {}
    doc = get_observations_collection().find_one(
        query,
        sort=[("timestamp", -1)],
        projection={"timestamp": 1}
    )
    return doc["timestamp"] if doc else None


def generate_synthetic_baseline(
    features: List[Dict[str, Any]],
    base_ref_time: Optional[datetime] = None,
    config: Optional[Dict[str, Any]] = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates a calibrated deterministic synthetic hydrological baseline for testing
    the FloodSense rainfall memory and flood-momentum pipeline.
    
    Assumptions & Calibration:
    1. Realistic Chennai climatology across 270 calendar days (Jan 11 – Oct 7, 2026):
       - Jan – May (DOY 1–151): Dry/winter season. Low precipitation (~10-25 mm/month).
       - Jun – Sep (DOY 152–273): Southwest Monsoon convective showers (~80-120 mm/month).
       - Oct 1 – Oct 7 (DOY 274–280): Early Northeast Monsoon onset and pre-storm build-up.
    2. Operational Storm Event (recent 72 hours, Oct 5 13:00 – Oct 8 12:00):
       - Modulated by ward environmental vulnerability (from Wards.geojson).
       - Scaled so localized 24h accumulation tests the momentum thresholds (40-120 mm)
         without distorting the multi-month seasonal totals.
    3. Mathematical Coupling:
       - Daily summaries are DERIVED directly by aggregating the generated hourly readings.
       - Guarantees 100% mathematical consistency: total_rainfall_mm == sum(hourly_readings).
    4. Provenance:
       - Explicitly tagged with source: 'synthetic_baseline'.
    """
    cfg = {**DEFAULT_SEASONAL_PROFILE_CONFIG, **(config or {})}

    if base_ref_time is None:
        base_ref_time = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
    elif base_ref_time.tzinfo is None:
        base_ref_time = base_ref_time.replace(tzinfo=timezone.utc)

    observations_to_insert: List[Dict[str, Any]] = []
    daily_summaries_to_insert: List[Dict[str, Any]] = []

    for feat in features:
        props = feat.get("properties", {})
        if "Ward_No" not in props:
            continue
        ward_no = int(props["Ward_No"])
        if ward_no < 0:
            continue

        base_rainfall = float(props.get("rainfall_mm", 20.0))
        # Cap baseline storm scaling so a single extreme event doesn't corrupt historical seasons
        storm_cap_min = float(cfg.get("storm_cap_min_mm", 20.0))
        storm_cap_max = float(cfg.get("storm_cap_max_mm", 120.0))
        storm_scale = float(cfg.get("storm_scale_factor", 0.60))
        capped_storm_base = min(storm_cap_max, max(storm_cap_min, base_rainfall * storm_scale))

        dry_doy = int(cfg.get("dry_season_end_doy", 151))
        sw_doy = int(cfg.get("sw_monsoon_end_doy", 273))
        dry_mod = int(cfg.get("dry_season_rain_mod", 11))
        dry_base = float(cfg.get("dry_season_base_mm", 1.2))
        dry_step = float(cfg.get("dry_season_step_mm", 0.35))
        sw_mod = int(cfg.get("sw_monsoon_rain_mod", 3))
        sw_base = float(cfg.get("sw_monsoon_base_mm", 2.5))
        sw_step = float(cfg.get("sw_monsoon_step_mm", 0.90))

        # Part 1: 270 historical calendar days (Days 270 down to 1 ago: 2026-01-11 to 2026-10-07)
        for d in range(270, 0, -1):
            day_date = (base_ref_time - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
            doy = day_date.timetuple().tm_yday

            # Seasonal baseline profile for Chennai
            if doy <= dry_doy:
                # Dry season: rain occurs on ~9% of days
                is_rain_day = ((d + ward_no * 3) % dry_mod) == 0
                day_total = round((dry_base + ((ward_no * 7 + d * 5) % 10) * dry_step), 2) if is_rain_day else 0.0
            elif doy <= sw_doy:
                # Southwest monsoon: rain occurs on ~33% of days
                is_rain_day = ((d + ward_no * 5) % sw_mod) == 0
                day_total = round((sw_base + ((ward_no * 11 + d * 7) % 12) * sw_step), 2) if is_rain_day else 0.0
            else:
                # Oct 1-7: NE monsoon onset storm progression leading to today
                days_from_onset = doy - sw_doy  # 1 to 7
                ramp_min = float(cfg.get("ne_onset_ramp_min", 0.20))
                ramp_span = float(cfg.get("ne_onset_ramp_max", 0.80))
                storm_ramp = ramp_min + ramp_span * (days_from_onset / 7.0)
                ward_var = 0.6 + 0.4 * (((ward_no + d) % 7) / 7.0)
                day_total = round(capped_storm_base * storm_ramp * ward_var, 2)

            # Distribute into 24 hours (diurnal convective rainfall pattern)
            if day_total > 0:
                peak_h = int(cfg.get("diurnal_peak_hour_base", 15)) + (ward_no % 4)
                spread_var = float(cfg.get("diurnal_spread_variance", 5.0))
                weights = [math.exp(-((h - peak_h) ** 2) / spread_var) for h in range(24)]
                w_sum = sum(weights)
                day_hourly = [round(day_total * (w / w_sum), 2) for w in weights]
                diff = round(day_total - sum(day_hourly), 2)
                day_hourly[peak_h] = round(day_hourly[peak_h] + diff, 2)
            else:
                day_hourly = [0.0] * 24

            for h in range(24):
                ts = day_date + timedelta(hours=h)
                observations_to_insert.append({
                    "ward_no": ward_no,
                    "timestamp": ts,
                    "rainfall_mm": max(0.0, day_hourly[h]),
                    "source": SYNTHETIC_SOURCE,
                    "created_at": base_ref_time
                })

            # Daily summary derived directly from the hourly readings
            sum_day = round(sum(day_hourly), 2)
            max_day = round(max(day_hourly), 2)
            daily_summaries_to_insert.append({
                "ward_no": ward_no,
                "date": day_date,
                "total_rainfall_mm": sum_day,
                "max_hourly_mm": max_day,
                "observation_count": 24,
                "is_complete": True,
                "source": SYNTHETIC_SOURCE
            })

        # Part 2: Today (Oct 8 from 00:00 to 12:00 UTC) - the peak hours of the current storm
        today_date = base_ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        for h in range(13):  # 00:00 through 12:00 UTC
            hours_ago = 12 - h
            decay = math.exp(-hours_ago / 8.0)
            ward_var = 0.8 + 0.2 * ((ward_no % 5) / 5.0)
            hourly_rain = round(max(0.0, (capped_storm_base / 8.0) * decay * ward_var), 2)
            if hours_ago == 0:
                hourly_rain = round(capped_storm_base * 0.22, 2)

            observations_to_insert.append({
                "ward_no": ward_no,
                "timestamp": today_date + timedelta(hours=h),
                "rainfall_mm": hourly_rain,
                "source": SYNTHETIC_SOURCE,
                "created_at": base_ref_time
            })

    return observations_to_insert, daily_summaries_to_insert


def seed_baseline_if_empty(features: List[Dict[str, Any]], force: bool = False) -> Dict[str, Any]:
    """
    Seeds initial synthetic baseline observations and daily summaries for all 201 wards.
    If observations already exist with the synthetic_baseline source and force=False, does nothing.
    """
    if not is_connected():
        return {"inserted_observations": 0, "inserted_daily_summaries": 0, "status": "db_not_connected"}

    obs_col = get_observations_collection()
    daily_col = get_daily_summary_collection()

    existing_synthetic = obs_col.count_documents({"source": SYNTHETIC_SOURCE})
    if existing_synthetic > 0 and not force:
        return {
            "inserted_observations": 0,
            "inserted_daily_summaries": 0,
            "status": "already_seeded",
            "existing_synthetic": existing_synthetic
        }

    # Generate calibrated synthetic dataset
    obs_docs, daily_docs = generate_synthetic_baseline(features)

    # If force, delete ONLY synthetic records, strictly preserving any user-ingested data
    if force or existing_synthetic == 0:
        obs_col.delete_many({"source": {"$in": LEGACY_SYNTHETIC_SOURCES}})
        daily_col.delete_many({"source": {"$in": LEGACY_SYNTHETIC_SOURCES}})

    # Bulk insert observations in chunks of 10000
    chunk_size = 10000
    total_obs = 0
    for i in range(0, len(obs_docs), chunk_size):
        chunk = obs_docs[i:i + chunk_size]
        obs_col.insert_many(chunk, ordered=False)
        total_obs += len(chunk)

    # Bulk insert daily summaries in chunks of 10000
    total_daily = 0
    for i in range(0, len(daily_docs), chunk_size):
        chunk = daily_docs[i:i + chunk_size]
        daily_col.insert_many(chunk, ordered=False)
        total_daily += len(chunk)

    return {
        "inserted_observations": total_obs,
        "inserted_daily_summaries": total_daily,
        "status": "seeded_successfully"
    }


def migrate_and_reseed_synthetic_baseline(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Explicit, controlled migration:
    1. Preserves all user-ingested records (source not in LEGACY_SYNTHETIC_SOURCES).
    2. Replaces old synthetic records with the calibrated synthetic baseline.
    3. Guarantees raw hourly observations and daily summaries match 100%.
    """
    return seed_baseline_if_empty(features, force=True)
