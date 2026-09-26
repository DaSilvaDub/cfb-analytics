"""Situational Weather, Venue, Elevation, and Travel Fatigue Engine.

Models micro-climate conditions at kickoff and throughout game duration:
- Weather attenuation curves (wind, gust, precipitation, extreme cold, extreme heat).
- Pure standard library volume redistribution law: Delta_pass shifts 0.85 * Delta_pass into rush attempts.
- Indoor/dome strict bypass.
- Venue elevation physics (>=1200m cardiovascular fatigue tax and field goal range boost).
- Haversine travel distance, time-zone displacement, and the Late Kickoff Penalty
  (10:30 PM / 11:00 PM ET road favorite fatigue for Grok Rule C).
"""

from __future__ import annotations

import math

from cfb_analytics.reasoning.models import (
    TravelProfile,
    VolumeRedistribution,
    WeatherProfile,
)


def calculate_weather_multipliers(
    *,
    temp_c: float,
    wind_kph: float,
    gust_kph: float,
    precip_mm: float,
    precip_prob: float = 0.0,
    is_dome: bool = False,
    humidity: float | None = None,
) -> WeatherProfile:
    """Compute atmospheric attenuation multipliers and volume redistribution coefficients."""
    if is_dome:
        return WeatherProfile(
            temperature_c=temp_c,
            wind_kph=0.0,
            gust_kph=0.0,
            precip_mm=0.0,
            is_dome=True,
            pass_vol_mult=1.0,
            rush_vol_mult=1.0,
            scoring_mult=1.0,
            effective_wind_kph=0.0,
            excess_wind_kph=0.0,
            cold_deficit_c=0.0,
            heat_excess_c=0.0,
            precip_prob=0.0,
            humidity=humidity,
            comp_prob_mult=1.0,
            ypc_mult=1.0,
            is_extreme_weather=False,
            weather_summary="Indoor/Dome stadium. Climate controlled; neutral atmospheric conditions.",
        )

    # 1. Wind Speed & Turbulent Gusts
    effective_wind = max(wind_kph, 0.75 * gust_kph)
    excess_wind = max(0.0, effective_wind - 15.0)

    m_wind_pass_vol = max(0.65, 1.0 - 0.008 * excess_wind)
    m_wind_comp_prob = max(0.70, 1.0 - 0.006 * excess_wind)
    m_wind_ypc = max(0.70, 1.0 - 0.007 * excess_wind)
    m_wind_pts = max(0.75, 1.0 - 0.006 * excess_wind)

    # 2. Precipitation
    precip_eff = (
        (precip_prob - 50.0) * 0.02
        if (precip_mm == 0.0 and precip_prob >= 60.0)
        else precip_mm
    )
    excess_precip = min(15.0, max(0.0, precip_eff - 0.1))

    m_precip_pass_vol = max(0.75, 1.0 - 0.025 * excess_precip)
    m_precip_comp_prob = max(0.80, 1.0 - 0.030 * excess_precip)
    m_precip_ypc = max(0.85, 1.0 - 0.020 * excess_precip)
    m_precip_pts = max(0.80, 1.0 - 0.025 * excess_precip)

    # 3. Temperature Extremes (Cold Deficit < 5.0C, Heat Excess > 30.0C)
    cold_deficit = min(25.0, max(0.0, 5.0 - temp_c))
    heat_excess = min(15.0, max(0.0, temp_c - 30.0))

    if cold_deficit > 0.0:
        m_temp_pass_vol = max(0.80, 1.0 - 0.008 * cold_deficit)
        m_temp_comp_prob = max(0.75, 1.0 - 0.010 * cold_deficit)
        m_temp_ypc = max(0.80, 1.0 - 0.008 * cold_deficit)
        m_temp_pts = max(0.75, 1.0 - 0.010 * cold_deficit)
    elif heat_excess > 0.0:
        m_temp_pass_vol = max(0.95, 1.0 - 0.003 * heat_excess)
        m_temp_comp_prob = max(0.95, 1.0 - 0.003 * heat_excess)
        m_temp_ypc = 1.0
        m_temp_pts = max(0.92, 1.0 - 0.006 * heat_excess)
    else:
        m_temp_pass_vol = 1.0
        m_temp_comp_prob = 1.0
        m_temp_ypc = 1.0
        m_temp_pts = 1.0

    # 4. Composite Multipliers
    pass_vol_mult = round(
        max(0.50, min(1.0, m_wind_pass_vol * m_precip_pass_vol * m_temp_pass_vol)), 4
    )
    comp_prob_mult = round(
        max(0.50, min(1.0, m_wind_comp_prob * m_precip_comp_prob * m_temp_comp_prob)),
        4,
    )
    ypc_mult = round(
        max(0.50, min(1.0, m_wind_ypc * m_precip_ypc * m_temp_ypc)), 4
    )
    scoring_mult = round(
        max(0.55, min(1.0, m_wind_pts * m_precip_pts * m_temp_pts)), 4
    )

    # 5. Run/Pass volume shift: lost pass volume converted 0.85x into rushing volume
    rush_vol_mult = round(1.0 + (1.0 - pass_vol_mult) * 0.85, 4)

    is_extreme = bool(
        wind_kph >= 35.0
        or gust_kph >= 48.0
        or precip_mm >= 5.0
        or temp_c < 0.0
    )

    summary = (
        f"Outdoor weather: {temp_c:.1f}°C, wind {wind_kph:.1f} kph (gusts {gust_kph:.1f} kph), "
        f"precip {precip_mm:.1f} mm. Multipliers: pass {pass_vol_mult:.2f}x, "
        f"rush {rush_vol_mult:.2f}x, scoring {scoring_mult:.2f}x."
    )

    return WeatherProfile(
        temperature_c=temp_c,
        wind_kph=wind_kph,
        gust_kph=gust_kph,
        precip_mm=precip_mm,
        is_dome=False,
        pass_vol_mult=pass_vol_mult,
        rush_vol_mult=rush_vol_mult,
        scoring_mult=scoring_mult,
        effective_wind_kph=round(effective_wind, 1),
        excess_wind_kph=round(excess_wind, 1),
        cold_deficit_c=round(cold_deficit, 1),
        heat_excess_c=round(heat_excess, 1),
        precip_prob=precip_prob,
        humidity=humidity,
        comp_prob_mult=comp_prob_mult,
        ypc_mult=ypc_mult,
        is_extreme_weather=is_extreme,
        weather_summary=summary,
    )


def redistribute_pass_to_rush(
    base_pass: float, base_rush: float, pass_vol_mult: float
) -> VolumeRedistribution:
    """Applies the 0.85 * Delta_pass volume redistribution law.

    When pass attempts are depressed by adverse weather, 85% of lost attempts
    convert into rushing attempts. The remaining 15% accounts for clock runoff.
    """
    adj_pass = round(max(10.0, min(65.0, base_pass * pass_vol_mult)), 1)
    delta_pass = max(0.0, base_pass - adj_pass)
    rush_boost = round(max(0.0, delta_pass * 0.85), 1)
    adj_rush = round(max(15.0, min(75.0, base_rush + rush_boost)), 1)
    adj_pace = round(adj_pass + adj_rush, 1)

    return VolumeRedistribution(
        orig_pass_attempts=base_pass,
        orig_rush_attempts=base_rush,
        adj_pass_attempts=adj_pass,
        adj_rush_attempts=adj_rush,
        delta_pass=round(delta_pass, 1),
        rush_boost=rush_boost,
        adj_pace=adj_pace,
    )


def calculate_haversine_distance_miles(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Pure stdlib Haversine great-circle distance in miles."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(3958.8 * c, 1)


def calculate_altitude_fatigue_tax(
    venue_elevation_m: float | None,
    visitor_elevation_m: float | None = 0.0,
) -> float:
    """Compute spread point penalty for visiting team playing at high altitude (>=1200m)."""
    h_venue = venue_elevation_m or 0.0
    h_vis = visitor_elevation_m or 0.0
    delta_elev = max(0.0, h_venue - h_vis)

    if h_venue >= 1200.0 and delta_elev >= 800.0:
        tax = min(2.5, max(0.0, ((h_venue - 1000.0) / 500.0) * 0.75))
        return round(tax, 2)
    return 0.0


def calculate_fg_range_boost(venue_elevation_m: float | None) -> float:
    """Calculate field goal range boost in yards due to reduced air density at altitude."""
    h = venue_elevation_m or 0.0
    return round(min(4.5, max(0.0, (h / 500.0) * 1.0)), 1)


def parse_kickoff_et_hour(kickoff_et: str) -> float | None:
    """Parse string kickoff time like '10:30 p.m.', '11:00 PM', '10:30PM ET', or '22:30' into float hours."""
    if not kickoff_et:
        return None
    clean = kickoff_et.strip().lower()
    clean = clean.replace("a.m.", "am").replace("p.m.", "pm")
    for tz in ("et", "est", "edt", "ct", "cst", "cdt", "pt", "pst", "pdt", "mt", "mst", "mdt"):
        clean = clean.replace(tz, "")
    clean = clean.strip()
    try:
        meridiem = ""
        if clean.endswith("am"):
            meridiem = "am"
            clean = clean[:-2].strip()
        elif clean.endswith("pm"):
            meridiem = "pm"
            clean = clean[:-2].strip()
        else:
            parts = clean.split()
            if len(parts) >= 2 and parts[1] in ("am", "pm"):
                meridiem = parts[1]
                clean = parts[0]

        h_str, m_str = clean.split(":")
        h = int(h_str)
        m = int(m_str)
        if meridiem == "pm" and h != 12:
            h += 12
        elif meridiem == "am" and h == 12:
            h = 0
        return round(h + m / 60.0, 2)
    except Exception:
        return None


def evaluate_travel_profile(
    *,
    distance_miles: float,
    timezone_shift_hours: int = 0,
    is_westward: bool = False,
    kickoff_et: str = "",
    venue_elevation_m: float | None = 0.0,
    visitor_elevation_m: float | None = 0.0,
    spread_line: float | None = None,
) -> TravelProfile:
    """Quantifies travel fatigue tax, circadian disruption, and Grok Rule C qualification."""
    kickoff_hour = parse_kickoff_et_hour(kickoff_et) or 15.5
    is_late_kickoff = kickoff_hour >= 22.5  # 10:30 PM ET or later

    # Distance tax: cross-country travel > 750 miles
    dist_tax = (
        round(min(1.5, ((distance_miles - 750.0) / 1000.0) * 0.75), 2)
        if distance_miles >= 750.0
        else 0.0
    )

    # Time zone tax: 2+ hour shifts
    tz_tax = round((timezone_shift_hours - 1) * 0.5, 2) if timezone_shift_hours >= 2 else 0.0

    # Late kickoff tax: circadian fatigue
    if is_late_kickoff and is_westward:
        late_tax = 1.5
    elif is_late_kickoff:
        late_tax = 1.0
    else:
        late_tax = 0.0

    alt_tax = calculate_altitude_fatigue_tax(venue_elevation_m, visitor_elevation_m)
    total_tax = round(min(4.0, dist_tax + tz_tax + late_tax + alt_tax), 2)

    # Grok Rule C: Fat road dogs vs fatigued late road favorites (-20+ at >=10:30 PM ET)
    rule_c_triggered = bool(
        is_late_kickoff and spread_line is not None and spread_line <= -20.0
    )

    return TravelProfile(
        distance_miles=round(distance_miles, 1),
        timezone_shift_hours=timezone_shift_hours,
        is_westward=is_westward,
        kickoff_et_hour=kickoff_hour,
        is_late_kickoff=is_late_kickoff,
        distance_fatigue_tax=dist_tax,
        timezone_fatigue_tax=tz_tax,
        late_kickoff_fatigue_tax=late_tax,
        altitude_fatigue_tax=alt_tax,
        total_travel_fatigue_tax=total_tax,
        rule_c_triggered=rule_c_triggered,
    )
