"""FloodLens demonstration data.

All reports, coordinates and historical counts in this module are ILLUSTRATIVE
sample data generated for demonstration purposes. They do not represent real
flood observations.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

AREAS: dict[str, tuple[float, float]] = {
    "Al Quoz": (25.1380, 55.2270),
    "Business Bay": (25.1860, 55.2640),
    "Deira": (25.2700, 55.3150),
    "Al Barsha": (25.1130, 55.2000),
    "Dubai Marina": (25.0800, 55.1400),
    "Downtown Dubai": (25.1950, 55.2740),
    "Mirdif": (25.2210, 55.4190),
    "Jumeirah": (25.2040, 55.2420),
    "Al Garhoud": (25.2420, 55.3500),
}

SEVERITIES = ["High", "Medium", "Low"]
STATUSES = ["Active", "Monitoring", "Resolved"]

DUBAI_CENTER = (25.185, 55.265)

DESCRIPTIONS = {
    "High": [
        "Water covering the full carriageway; several vehicles stalled in the underpass.",
        "Deep standing water blocking both lanes, traffic diverted by patrol.",
        "Road fully submerged at the junction, drainage grates not clearing.",
        "Underpass flooded to roughly knee height, access closed to cars.",
        "Fast-moving surface water entering ground-floor parking entrance.",
        "Major ponding across the main road, drains overflowing.",
    ],
    "Medium": [
        "Standing water across two lanes, vehicles slowing to a crawl.",
        "Large pool at the roundabout exit, partially blocking the left lane.",
        "Water pooling along the kerb for about 100 metres, drains slow to clear.",
        "Service road partly covered in water, pedestrians using the verge.",
        "Ponding at the junction approach causing queueing.",
    ],
    "Low": [
        "Shallow puddles along the kerb, road passable.",
        "Light ponding near a drain, traffic unaffected.",
        "Small area of standing water on the pavement edge.",
        "Damp road with minor pooling at the low point.",
        "Shallow water at the car park entrance, drains clearing.",
    ],
}


def dubai_now() -> datetime:
    """Current Dubai local time as a naive datetime."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Dubai")).replace(tzinfo=None, microsecond=0)
    except Exception:  # pragma: no cover - fallback when tz database missing
        return datetime.utcnow().replace(microsecond=0) + timedelta(hours=4)


def make_demo_reports(now: datetime, n: int = 48, seed: int = 2026) -> pd.DataFrame:
    """Generate illustrative flood reports relative to ``now``."""
    rng = np.random.default_rng(seed)
    names = list(AREAS)
    weights = np.array([0.17, 0.12, 0.13, 0.10, 0.09, 0.10, 0.08, 0.11, 0.10])
    weights = weights / weights.sum()

    rows = []
    for i in range(n):
        area = str(rng.choice(names, p=weights))
        lat0, lon0 = AREAS[area]
        lat = float(lat0 + rng.normal(0, 0.0045))
        lon = float(lon0 + rng.normal(0, 0.0055))

        if i < 7:  # guarantee a handful of fresh high-severity incidents
            severity = "High"
            age_h = float(rng.uniform(0.5, 26))
            status = "Monitoring" if i == 6 else "Active"
        else:
            severity = str(rng.choice(SEVERITIES, p=[0.10, 0.50, 0.40]))
            age_h = float(min(rng.gamma(1.6, 40), 24 * 21))
            if age_h < 30:
                status = str(rng.choice(STATUSES, p=[0.8, 0.2, 0.0]))
            elif age_h < 90:
                status = str(rng.choice(STATUSES, p=[0.25, 0.3, 0.45]))
            else:
                status = str(rng.choice(STATUSES, p=[0.0, 0.2, 0.8]))

        rows.append(
            {
                "area": area,
                "lat": lat,
                "lon": lon,
                "ts": pd.Timestamp(now - timedelta(hours=age_h)).floor("min"),
                "severity": severity,
                "status": status,
                "description": str(rng.choice(DESCRIPTIONS[severity])),
                "image": None,
            }
        )

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df.insert(0, "id", [f"FL-{2001 + i}" for i in range(len(df))])
    df["image"] = df["image"].astype(object)
    return df


# --------------------------------------------------------------------------
# Historical (illustrative) monthly counts
# --------------------------------------------------------------------------
AREA_BASE = {
    "Al Quoz": 5.2,
    "Business Bay": 3.5,
    "Deira": 4.4,
    "Al Barsha": 2.8,
    "Dubai Marina": 2.2,
    "Downtown Dubai": 2.9,
    "Mirdif": 2.4,
    "Jumeirah": 3.1,
    "Al Garhoud": 2.0,
}
HIGH_SHARE = {
    "Al Quoz": 0.34,
    "Business Bay": 0.27,
    "Deira": 0.31,
    "Al Barsha": 0.19,
    "Dubai Marina": 0.16,
    "Downtown Dubai": 0.22,
    "Mirdif": 0.18,
    "Jumeirah": 0.21,
    "Al Garhoud": 0.15,
}
SEASON = {1: 1.5, 2: 1.6, 3: 1.4, 4: 1.1, 5: 0.3, 6: 0.1, 7: 0.15, 8: 0.2, 9: 0.2, 10: 0.4, 11: 1.0, 12: 1.5}


def make_history(now: datetime, seed: int = 11) -> pd.DataFrame:
    """Long-format monthly report counts per area since January 2024."""
    rng = np.random.default_rng(seed)
    last = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
    months = pd.date_range("2024-01-01", last, freq="MS")
    rows = []
    for m in months:
        for area, base in AREA_BASE.items():
            lam = base * SEASON[m.month]
            if m.year == 2024 and m.month == 4:
                lam *= 5.5  # illustrative spike for the April 2024 rainfall event
            rows.append({"month": m, "area": area, "reports": int(rng.poisson(lam))})
    return pd.DataFrame(rows)


def priority_table(hist: pd.DataFrame) -> pd.DataFrame:
    """Rank locations for infrastructure inspection from recurrence + severity."""
    g = hist.groupby("area")
    out = pd.DataFrame(
        {
            "total": g["reports"].sum(),
            "recurring": g["reports"].apply(lambda s: int((s >= 3).sum())),
        }
    )
    out["high_share"] = [HIGH_SHARE[a] for a in out.index]

    def norm(s: pd.Series) -> pd.Series:
        rng_ = s.max() - s.min()
        return (s - s.min()) / rng_ if rng_ else s * 0 + 0.5

    out["score"] = (
        100 * (0.5 * norm(out["total"]) + 0.3 * norm(out["recurring"]) + 0.2 * norm(out["high_share"]))
    ).round().astype(int)
    out = out.sort_values("score", ascending=False).reset_index().rename(columns={"area": "location"})
    out.insert(0, "rank", range(1, len(out) + 1))
    out["action"] = np.where(out["score"] >= 70, "Inspect first", np.where(out["score"] >= 45, "Schedule inspection", "Monitor"))
    return out
