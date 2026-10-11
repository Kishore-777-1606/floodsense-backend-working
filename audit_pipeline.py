import json
import joblib
import numpy as np
import rainfall_memory
import app

model = joblib.load('flood_model.pkl')
with open('Wards.geojson', encoding='utf-8') as f:
    geojson = json.load(f)
wards = geojson['features']

print("=== 1. Wards.geojson Properties Audit ===")
elevs = [float(w['properties'].get('elevation_m', 0)) for w in wards]
wdists = [float(w['properties'].get('water_distance_km', 0)) for w in wards]
base_rains = [float(w['properties'].get('rainfall_mm', 0)) for w in wards]

print(f"Elevation: min={min(elevs):.2f}, max={max(elevs):.2f}, unique={len(set(elevs))}")
print(f"Water Distance: min={min(wdists):.2f}, max={max(wdists):.2f}, unique={len(set(wdists))}")
print(f"Baseline Rainfall: min={min(base_rains):.2f}, max={max(base_rains):.2f}, mean={np.mean(base_rains):.2f}, unique={len(set(base_rains))}")

print("\n=== 2. Baseline Calculation Audit ===")
base_data = app.compute_all_wards_baseline_data()
base_scores = [d['score'] for d in base_data]
base_base_scores = [d['base_score'] for d in base_data]
base_mults = [d['flood_momentum']['momentum_multiplier'] for d in base_data]
base_levels = [d['level'] for d in base_data]

print(f"Baseline Base Scores: min={min(base_base_scores):.2f}, max={max(base_base_scores):.2f}, mean={np.mean(base_base_scores):.2f}")
print(f"Baseline Multipliers: min={min(base_mults):.3f}, max={max(base_mults):.3f}, mean={np.mean(base_mults):.3f}")
print(f"Baseline Final Scores: min={min(base_scores):.2f}, max={max(base_scores):.2f}, mean={np.mean(base_scores):.2f}")
print(f"Baseline Levels: LOW={base_levels.count('LOW')}, MED={base_levels.count('MEDIUM')}, HIGH={base_levels.count('HIGH')}")

print("\n=== 3. Model Predictions on Uniform S vs Variable Attributes ===")
for S in [0, 50, 80, 100, 120, 150, 200, 250, 300]:
    scores = []
    for w in wards:
        e = float(w['properties'].get('elevation_m', 6.7))
        wd = float(w['properties'].get('water_distance_km', 10.0))
        p = float(model.predict([[float(S), e, wd]])[0])
        scores.append(p)
    print(f"S={S:3d} mm: BaseScore range=[{min(scores):.2f}, {max(scores):.2f}], spread={max(scores)-min(scores):.2f}, mean={np.mean(scores):.2f}, std={np.std(scores):.3f}")

print("\n=== 4. Momentum Multipliers under Current Scenario Tranches ===")
all_mem = rainfall_memory.calculate_all_wards_rainfall_memory()
for S in [0, 50, 80, 100, 120, 150, 200, 250, 300]:
    mults = []
    final_scores_before_clip = []
    final_scores_after_clip = []
    clipped_count = 0
    
    for w in wards:
        wn = int(w['properties']['Ward_No'])
        e = float(w['properties'].get('elevation_m', 6.7))
        wd = float(w['properties'].get('water_distance_km', 10.0))
        bm = all_mem.get(wn, {})
        r_past = max(0.0, float(bm.get('72h', 0.0)) - float(bm.get('24h', 0.0)))
        
        # Current scenario tranches:
        scen_mem = {
            '1h': round(0.20 * S, 2),
            '3h': round(0.35 * S, 2),
            '6h': round(0.45 * S, 2),
            '8h': round(0.50 * S, 2),
            '12h': round(0.70 * S, 2),
            '24h': round(1.00 * S, 2),
            '72h': round(1.00 * S + r_past, 2)
        }
        mom = rainfall_memory.calculate_flood_momentum(scen_mem, e, wd)
        mult = mom['momentum_multiplier']
        mults.append(mult)
        
        base_s = float(model.predict([[float(S), e, wd]])[0])
        raw_final = base_s * mult
        final_scores_before_clip.append(raw_final)
        clipped = round(min(200.0, max(0.0, raw_final)), 2)
        final_scores_after_clip.append(clipped)
        if raw_final > 200.0:
            clipped_count += 1
            
    print(f"S={S:3d} mm: Mult range=[{min(mults):.3f}, {max(mults):.3f}] | RawScore range=[{min(final_scores_before_clip):.1f}, {max(final_scores_before_clip):.1f}] | ClippedCount={clipped_count}/201 | Final range=[{min(final_scores_after_clip):.1f}, {max(final_scores_after_clip):.1f}]")
