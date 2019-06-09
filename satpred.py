import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytz import timezone
from tzwhere import tzwhere
from datetime import datetime, timedelta
from itertools import zip_longest
from skyfield import almanac, earthlib
from skyfield.api import Loader, Topos

DIRECTION_NAMES = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW', 'N']
DIRECTION_DEGREES = [0.0,22.5,45.0,67.5,90.0,112.5,135.0,157.5,180.0,202.5,225.0,247.5,270.0,292.5,315.0,337.5,360.0]
EARTH_RADIUS = earthlib.earth_radius_au
SUN_RADIUS = 0.00465 # Sun's average radius in AU
SUN_ALTITUDE_CIVIL_TWILIGHT = -6.
SATELLITE_LIGHTING_CONDITIONS = ['umbral','penumbral','annular','sunlit']
SATELLITE_MINIMUM_OBSERVABLE_ALTITUDE = 10.
SATELLITE_INTRINSIC_MAGNITUDE = -1.3
SATELLITE_PASS_ROUGH_PERIOD = 0.01

def find_maximum(t0, t1, epsilon, f):
    ts, jd0, jd1 = t0.ts, t0.tt, t1.tt
    while jd1 - jd0 > epsilon:
        jd = np.linspace(jd0, jd1, 5)
        t = ts.tt(jd=jd)
        i = np.argmax(f(t))
        jd0, jd1 = jd[np.max([0,i-1])], jd[np.min([i+1,4])]
    return ts.tt(jd=jd0), f(ts.tt(jd=jd0))

def find_minimum(t0, t1, epsilon, f):
    ts, jd0, jd1 = t0.ts, t0.tt, t1.tt
    while jd1 - jd0 > epsilon:
        jd = np.linspace(jd0, jd1, 5)
        t = ts.tt(jd=jd)
        i = np.argmin(f(t))
        jd0, jd1 = jd[np.max([0,i-1])], jd[np.min([i+1,4])]
    return ts.tt(jd=jd0), f(ts.tt(jd=jd0))

def direction(degrees_az):
    degrees = np.asarray(DIRECTION_DEGREES)
    idx = (np.abs(degrees - degrees_az)).argmin()
    return DIRECTION_NAMES[idx]

# based on method from https://www.celestrak.com/columns/v03n01/, "Visually Observing Earth Satellites"
def semidiameter(radius, distance):
    return np.arcsin(radius / distance)

def eclipse_parameters(satellite, earth, sun, t):
    sat = earth + satellite
    barycentric_e = sat.at(t).observe(earth)
    barycentric_s = sat.at(t).observe(sun)
    _, _, distance_to_earth = barycentric_e.radec()
    _, _, distance_to_sun = barycentric_s.radec()
    theta_e = semidiameter(EARTH_RADIUS, distance_to_earth.au)
    theta_s = semidiameter(SUN_RADIUS, distance_to_sun.au)
    theta = barycentric_e.separation_from(barycentric_s).radians
    return theta, theta_e, theta_s

def umbral_eclipse(satellite, earth, sun, t):
    theta, theta_e, theta_s = eclipse_parameters(satellite, earth, sun, t)
    return np.logical_and(theta_e > theta_s, theta < (theta_e - theta_s))

def penumbral_eclipse(satellite, earth, sun, t):
    theta, theta_e, theta_s = eclipse_parameters(satellite, earth, sun, t)
    return np.logical_and(np.abs(theta_e - theta_s) < theta, theta < (theta_e + theta_s))

def annular_eclipse(satellite, earth, sun, t):
    theta, theta_e, theta_s = eclipse_parameters(satellite, earth, sun, t)
    return np.logical_and(theta_s > theta_e, theta < (theta_s - theta_e))

def civil_twilight(topos, earth, sun, t):
    location = earth + topos
    astrocentric = location.at(t).observe(sun).apparent()
    alt, _, _ = astrocentric.altaz('standard')
    return alt.degrees <= SUN_ALTITUDE_CIVIL_TWILIGHT

# based on method from
# https://astronomy.stackexchange.com/questions/28744/calculating-the-apparent-magnitude-of-a-satellite,
# "Calculating the apparent magnitude of a satellite"

def apparent_magnitude(satellite, topos, earth, sun, t):
    sat = earth + satellite
    observer = earth + topos
    barycentric_o = sat.at(t).observe(observer)
    barycentric_s = sat.at(t).observe(sun)
    phase_angle = barycentric_o.separation_from(barycentric_s).radians
    _, _, distance = barycentric_o.radec()
    term_2 = +5.0 * np.log10(distance.km / 1000.)
    arg = np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle)
    term_3 = -2.5 * np.log10(arg)
    return SATELLITE_INTRINSIC_MAGNITUDE + term_2 + term_3

# based on definition of sunrise_sunset in skyfields.almanac

def satellite_passes(topos, satellite, earth, sun, visible=True):
    difference = satellite - topos
    def sat_observable(t):
        topocentric = difference.at(t)
        alt, _, _ = topocentric.altaz('standard')
        if visible:
            in_umbral_eclipse = umbral_eclipse(satellite, earth, sun, t)
            return np.logical_and(np.logical_and(alt.degrees >= SATELLITE_MINIMUM_OBSERVABLE_ALTITUDE, civil_twilight(topos, earth,sun, t)),
                                  np.logical_not(in_umbral_eclipse))
        else:
            return alt.degrees >= SATELLITE_MINIMUM_OBSERVABLE_ALTITUDE
    sat_observable.rough_period = SATELLITE_PASS_ROUGH_PERIOD
    return sat_observable

# Split this into three things: 1) generate list of dicts with julian dates, 2) localize to tz, 3) generate DataFrame

def passes(t0, t1, topos, satellite, earth, sun, visible=True):
    t, y = almanac.find_discrete(t0, t1, satellite_passes(topos, satellite, earth, sun, visible))
    timezone_str = tzwhere.tzwhere().tzNameAt(topos.latitude.degrees, topos.longitude.degrees)
    tz = timezone(timezone_str)
    ts = t0.ts
    passes = []
    difference = satellite - topos

    for i in range(1, len(t), 2):

        start_t, end_t = t[i-1], t[i] # check that y[i-1] and not y[i]

        start_local_datetime = start_t.astimezone(timezone(timezone_str)) # can we get locality from topos?
        start_local_date_str = start_local_datetime.isoformat(' ', timespec='seconds')[:11]
        start_local_time_str = start_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        start_alt, start_az, start_d = difference.at(start_t).altaz('standard')

        culm_t, culm_alt = find_maximum(start_t, end_t, 1.6E-07, lambda t: difference.at(t).altaz('standard')[0].degrees)
        culm_local_datetime = culm_t.astimezone(timezone(timezone_str))
        culm_local_time_str = culm_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        culm_alt, culm_az, culm_d = difference.at(culm_t).altaz('standard')

        end_local_datetime = end_t.astimezone(timezone(timezone_str))
        end_local_time_str = end_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        end_alt, end_az, end_d = difference.at(end_t).altaz('standard')

        f_mag = lambda t: apparent_magnitude(satellite, topos, earth, sun, t)
        f_mag.rough_period = SATELLITE_PASS_ROUGH_PERIOD
        brightest_t, mag = find_minimum(start_t, end_t, 1.6E-07, f_mag)

        if civil_twilight(topos, earth, sun, start_t) and civil_twilight(topos, earth, sun, end_t):
            if umbral_eclipse(satellite, earth, sun, start_t) and umbral_eclipse(satellite, earth, sun, end_t):
                pass_type = 'eclipsed'
            else:
                pass_type = 'visible'
        else:
            pass_type = 'daylight'

        passes.append({
            'date': start_local_date_str,
            'mag': np.round(mag, 1),
            'start': start_local_time_str,
            'start_alt': int(np.round(start_alt.degrees)),
            'start_az': direction(start_az.degrees),
            'start_d': int(np.round(start_d.km)),
            'culm': culm_local_time_str,
            'culm_alt': int(np.round(culm_alt.degrees)),
            'culm_az': direction(culm_az.degrees),
            'culm_d': int(np.round(culm_d.km)),
            'end': end_local_time_str,
            'end_alt': int(np.round(end_alt.degrees)),
            'end_az': direction(end_az.degrees),
            'end_d': int(np.round(end_d.km)),
            'pass_type': pass_type,
            })

    return pd.DataFrame(passes)
