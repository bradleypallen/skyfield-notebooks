import numpy as np
from pytz import timezone
from tzwhere import tzwhere
from skyfield import almanac, earthlib

DIRECTION_NAMES = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW', 'N']
DIRECTION_DEGREES = [0.0,22.5,45.0,67.5,90.0,112.5,135.0,157.5,180.0,202.5,225.0,247.5,270.0,292.5,315.0,337.5,360.0]
EARTH_RADIUS = earthlib.earth_radius_au
SUN_RADIUS = 0.00465 # Sun's average radius in AU
SUN_ALTITUDE_CIVIL_TWILIGHT = -6.
SATELLITE_LIGHTING_CONDITIONS = ['umbral','penumbral','annular','sunlit']
SATELLITE_MINIMUM_OBSERVABLE_ALTITUDE = 10.
SATELLITE_INTRINSIC_MAGNITUDE = -1.3
SATELLITE_PASS_ROUGH_PERIOD = 0.01
EXTREMUM_SEARCH_NUM_POINTS = 5
EXTREMUM_SEARCH_EPSILON = 0.5 / 86400. # A half-second fraction of a Julian day

# Return a global extremum for a function ranging over the time period of a pass.
# Assumes well-behaved functions with a global extremum and no other local extrema.

def find_extremum(t0, t1, extremum, f, epsilon=EXTREMUM_SEARCH_EPSILON, num=EXTREMUM_SEARCH_NUM_POINTS):
    ts, jd0, jd1 = t0.ts, t0.tt, t1.tt
    while jd1 - jd0 > epsilon:
        jd = np.linspace(jd0, jd1, 5)
        t = ts.tt(jd=jd)
        i = extremum(f(t))
        jd0, jd1 = jd[np.max([0,i-1])], jd[np.min([i+1,num-1])]
    return ts.tt(jd=jd0), f(ts.tt(jd=jd0))

def find_minimum(t0, t1, f, epsilon=EXTREMUM_SEARCH_EPSILON, num=EXTREMUM_SEARCH_NUM_POINTS):
    return find_extremum(t0, t1, np.argmin, f, epsilon, num)

def find_maximum(t0, t1, f, epsilon=EXTREMUM_SEARCH_EPSILON, num=EXTREMUM_SEARCH_NUM_POINTS):
    return find_extremum(t0, t1, np.argmax, f, epsilon, num)

# Functions for detemining whether a satellite is in eclipse in a particular time in a pass.
# Based on the method described in https://www.celestrak.com/columns/v03n01/, "Visually Observing Earth Satellites".

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

# Function for estimatating the apparent magnitude of a satellite for an observer during a pass.
# Based on the method described in
# https://astronomy.stackexchange.com/questions/28744/calculating-the-apparent-magnitude-of-a-satellite,
# "Calculating the apparent magnitude of a satellite".

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

# FUnction to be used with skyfield.find_discrete to find satellite passes for an observer within a time period.
# Based on the definition of sunrise_sunset in skyfields.almanac

def satellite_pass(topos, satellite, earth, sun, visible=True):
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

# Function to generate lists of dicts, each dict describing a satellite pass.

# Return the name of a direction given an azimuth in degrees.
# Used to mimic the way azimuth directions are rported in Heavens Above pass prediction tables.

def direction(degrees_az):
    degrees = np.asarray(DIRECTION_DEGREES)
    idx = (np.abs(degrees - degrees_az)).argmin()
    return DIRECTION_NAMES[idx]

def prettify_pass(pass_dict, timezone_str):
    start_local_datetime = pass_dict['start_time'].astimezone(timezone(timezone_str))
    start_alt, start_az, start_d = pass_dict['start_position'].altaz('standard')
    culm_alt, culm_az, culm_d = pass_dict['culmination_position'].altaz('standard')
    end_alt, end_az, end_d = pass_dict['end_position'].altaz('standard')
    pretty_dict = {
        'date': start_local_datetime.isoformat(' ', timespec='seconds')[:11],
        'timezone': timezone_str,
        'start': start_local_datetime.isoformat(' ', timespec='seconds')[11:19],
        'start_alt': int(np.round(start_alt.degrees)),
        'start_az': direction(start_az.degrees),
        'start_d': int(np.round(start_d.km)),
        'culm': pass_dict['culmination_time'].astimezone(timezone(timezone_str)).isoformat(' ', timespec='seconds')[11:19],
        'culm_alt': int(np.round(culm_alt.degrees)),
        'culm_az': direction(culm_az.degrees),
        'culm_d': int(np.round(culm_d.km)),
        'end': pass_dict['end_time'].astimezone(timezone(timezone_str)).isoformat(' ', timespec='seconds')[11:19],
        'end_alt': int(np.round(end_alt.degrees)),
        'end_az': direction(end_az.degrees),
        'end_d': int(np.round(end_d.km))
    }
    pass_dict.update(pretty_dict)

def passes(t0, t1, topos, satellite, earth, sun, visible=True, pretty=False):
    t, y = almanac.find_discrete(t0, t1, satellite_pass(topos, satellite, earth, sun, visible))
    passes = []
    difference = satellite - topos
    if pretty:
            timezone_str = tzwhere.tzwhere().tzNameAt(topos.latitude.degrees, topos.longitude.degrees)
    for i in range(1, len(t), 2):
        start_t, end_t = t[i-1], t[i] # check that y[i-1] and not y[i]
        culm_t, _ = find_maximum(start_t, end_t, lambda t: difference.at(t).altaz('standard')[0].degrees)
        brightest_t, mag = find_minimum(start_t, end_t, lambda t: apparent_magnitude(satellite, topos, earth, sun, t))
        if civil_twilight(topos, earth, sun, start_t) and civil_twilight(topos, earth, sun, end_t):
            if umbral_eclipse(satellite, earth, sun, start_t) and umbral_eclipse(satellite, earth, sun, end_t):
                pass_type = 'eclipsed'
            else:
                pass_type = 'visible'
        else:
            pass_type = 'daylight'
        dict = {
            'satellite': satellite,
            'topos': topos,
            'start_time': start_t,
            'start_position': difference.at(start_t),
            'culmination_time': culm_t,
            'culmination_position': difference.at(culm_t),
            'end_time': end_t,
            'end_position': difference.at(end_t),
            'pass_type': pass_type,
            'peak_magnitude': np.round(mag, 1)
        }
        if pretty:
            prettify_pass(dict, timezone_str)
        passes.append(dict)
    return passes
