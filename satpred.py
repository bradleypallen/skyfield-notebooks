import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytz import timezone
from datetime import datetime
from itertools import zip_longest
from skyfield import almanac, earthlib
from skyfield.api import Loader, Topos

DIRECTION_NAMES = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
DIRECTION_DEGREES = [0.0,22.5,45.0,67.5,90.0,112.5,135.0,157.5,180.0,202.5,225.0,247.5,270.0,292.5,315.0,337.5,360.0]
EARTH_RADIUS = earthlib.earth_radius_au
SUN_RADIUS = 0.00465 # Sun's average radius in AU
SUN_ALTITUDE_NAUTICAL_TWILIGHT = -6.
SATELLITE_LIGHTING_CONDITIONS = ['umbral','penumbral','annular','sunlit']
SATELLITE_MINIMUM_OBSERVABLE_ALTITUDE = 10.
SATELLITE_INTRINSIC_MAGNITUDE = -1.3
SATELLITE_PASS_ROUGH_PERIOD = 0.01

def find_maximum(t0, t1, epsilon, f):
    jd0, jd1 = t0.tt, t1.tt
    while jd1 - jd0 > epsilon:
        jd = np.linspace(jd0, jd1, 16)
        t = ts.tt(jd=jd)
        i = np.argmax(f(t) > 0.)
        jd0, jd1 = jd[i-1], jd[i]
    return ts.tt(jd=jd0)

def direction(degrees_az):
    degrees = np.asarray(DIRECTION_DEGREES)
    idx = (np.abs(degrees - degrees_az)).argmin()
    return DIRECTION_NAMES[idx]

# based on method from https://www.celestrak.com/columns/v03n01/, "Visually Observing Earth Satellites"
def semidiameter(radius, distance):
    return np.arcsin(radius/distance)

def satellite_lighting(satellite, earth, sun, t):
    position = earth + satellite
    def lighting(t):
        barycentric_e = position.at(t).observe(earth)
        barycentric_s = position.at(t).observe(sun)
        _, _, distance_to_earth = barycentric_e.radec()
        _, _, distance_to_sun = barycentric_s.radec()
        theta_e = semidiameter(EARTH_RADIUS, distance_to_earth.au)
        theta_s = semidiameter(SUN_RADIUS,distance_to_sun.au)
        theta = barycentric_e.separation_from(barycentric_s).radians
        if np.logical_and(theta_e > theta_s, theta < (theta_e - theta_s)): # in umbral eclipse
            return 0
        elif np.logical_and(np.abs(theta_e - theta_s) < theta, theta < (theta_e + theta_s)): # in penumbral eclipse
            return 1
        elif np.logical_and(theta_s > theta_e, theta < (theta_s - theta_e)): # in annular eclipse
            return 2
        else: # in full sunlight
            return 3
    direction.rough_period = SATELLITE_PASS_ROUGH_PERIOD
    return lighting

def nautical_twilight(topos, earth, sun):
    location = earth + topos
    def twilight(t):
        astrocentric = location.at(t).observe(sun).apparent()
        alt, _, _ = astrocentric.altaz('standard')
        return alt.degrees < SUN_ALTITUDE_NAUTICAL_TWILIGHT
    twilight.rough_period = SATELLITE_PASS_ROUGH_PERIOD
    return twilight

# based on method from
# https://astronomy.stackexchange.com/questions/28744/calculating-the-apparent-magnitude-of-a-satellite,
# "Calculating the apparent magnitude of a satellite"
# Need to figure out how to make this use observer location rather than earth center

def apparent_magnitude(satellite, earth, sun):
    sat = earth + satellite
    def sat_magnitude(t):
        barycentric_e = sat.at(t).observe(earth)
        barycentric_s = sat.at(t).observe(sun)
        phase_angle = barycentric_e.separation_from(barycentric_s).radians
        _, _, distance = barycentric_e.radec()
        term_2 = +5.0 * np.log10(distance.km/1000.)
        arg = np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle)
        term_3 = -2.5 * np.log10(arg)
        return SATELLITE_INTRINSIC_MAGNITUDE + term_2 + term_3
    sat_magnitude.rough_period = 0.01
    return sat_magnitude

# based on definition of sunrise_sunset in skyfields.almanac

def satellite_passes(topos, satellite, earth, sun):
    difference = satellite - topos
    def sat_observable(t):
        topocentric = difference.at(t)
        alt, _, _ = topocentric.altaz('standard')
        return alt.degrees >= SATELLITE_MINIMUM_OBSERVABLE_ALTITUDE
    sat_observable.rough_period = 0.01
    return sat_observable

# Flow:
#
# satellite_passes between start_date, end_date => t, y
# for each pair t_start t_end representing a rise/set pair:
#   find_discrete between t_start, t_end for nautical_twilight, satellite_lighting
#   find_max between t_start, t_end for apparent_magnitude, satellite alt
#   find direction, satellite alt for t_start, alt_end
#   find satellite alt for t_culm
#

def passes(t0, t1, topos, satellite, earth, sun):
    t, y = almanac.find_discrete(t0, t1, satellite_passes(topos, satellite, earth, sun))
    passes = []
    difference = satellite - topos
    for pass_times in grouper(t, 2):
        start = pass_times[0]
        end = pass_times[1]
        t_p = ts.utc(pass_datetimes(start, end))
        y_p = satellite_alt(manhattan_beach_ca_usa, iss, t_p)
        culmination_alt = np.amax(y_p)
        culmination = t_p[np.where(y_p == np.amax(y_p))[0][0]]
        topocentric = difference.at(start)
        alt_start, az_start, d_start = topocentric.altaz('standard')
        topocentric = difference.at(culmination)
        alt_culm, az_culm, d_culm = topocentric.altaz('standard')
        topocentric = difference.at(end)
        alt_end, az_end, d_end = topocentric.altaz('standard')
        if nautical_twilight(topos, earth, sun, start) and nautical_twilight(topos, earth, sun, end):
            umbral_eclipse_rise, _, _ = eclipse_condition(satellite, earth, sun, start)
            umbral_eclipse_set, _, _ = eclipse_condition(satellite, earth, sun, end)
            if umbral_eclipse_rise and umbral_eclipse_set:
                pass_type = 'eclipsed'
            else:
                pass_type = 'visible'
        else:
            pass_type = 'daytime'
        passes.append({ "date": start.astimezone(timezone('US/Pacific')).isoformat(' ', timespec='seconds')[:11],
                          "brightness": np.round(apparent_magnitude(SATELLITE_INTRINSIC_MAGNITUDE,
                                                                d_culm.km,
                                                                phase_angle(iss, earth, sun, culmination)),
                                             1),
                          "start": start.astimezone(timezone('US/Pacific')).isoformat(' ', timespec='seconds')[11:19],
                          "start_alt": np.floor(alt_start.degrees),
                          "start_az": direction(az_start.degrees),
                          "start_d": np.floor(d_start.km),
                          "culmination": culmination.astimezone(timezone('US/Pacific')).isoformat(' ',
                                                                                              timespec='seconds')[11:19],
                          "culm_alt": np.floor(alt_culm.degrees),
                          "culm_az": direction(az_culm.degrees),
                          "culm_d": np.floor(d_culm.km),
                          "end": end.astimezone(timezone('US/Pacific')).isoformat(' ', timespec='seconds')[11:19],
                          "end_alt": np.floor(alt_end.degrees),
                          "end_az": direction(az_end.degrees),
                          "end_d": np.floor(d_end.km),
                          "pass_type": pass_type
                         })
    return pd.DataFrame(passes)[["date", "brightness", "start", "start_alt", "start_az", "start_d",
                                     "culmination", "culm_alt", "culm_az", "culm_d",
                                     "end", "end_alt", "end_az", "end_d", "pass_type"]]
