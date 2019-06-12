# -*- coding: utf-8 -*-
"""satpred.py

A Python module for computing satellite passes for
a given satellite and observer.

Todo:
    * A more robust version of find_extremeum() that can deal with local extrema.
    * Check that the t_0 and t_1 supplied to passes() do not fall inside a pass.

"""


import numpy as np
import pytz
from tzwhere import tzwhere
from skyfield import almanac, earthlib


DIRECTION_NAMES = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
DIRECTION_DEGREES = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,
                     180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5, 360.0]
EARTH_RADIUS = earthlib.earth_radius_au
SUN_RADIUS = 0.00465 # Sun's average radius in AU
SUN_ALTITUDE_CIVIL_TWILIGHT = -6.
SAT_LIGHTING_CONDITIONS = ['umbral', 'penumbral', 'annular', 'sunlit']
SAT_MINIMUM_OBSERVABLE_ALTITUDE = 10.
SAT_INTRINSIC_MAGNITUDE = -1.3
SAT_PASS_ROUGH_PERIOD = 0.01
EXTREMUM_SEARCH_NUM_POINTS = 5
EXTREMUM_SEARCH_EPSILON = 0.5 / 86400. # A half-second fraction of a Julian day


def find_extremum(t_0, t_1, extremum, function, epsilon=EXTREMUM_SEARCH_EPSILON,
                  num=EXTREMUM_SEARCH_NUM_POINTS):
    """Returns a global extremum for a function ranging over the time period of a pass.

    Assumes well-behaved functions with a global extremum and no other local extrema.

    Args:
        t_0: Start time for period to be searched.
        t_1: End time for period to be searched.
        extremum: function that computes an extremum for a function ranging over time.
        function: a function that takes time as its single argument, returning a numeric value.
        epsilon: a float defining the distance less than which two times will
            be treated as equal.

    Returns:
        Two values, the first being the time of the extremum, and the second
        being the value of the extremum.
    """
    timescale, jd_0, jd_1 = t_0.ts, t_0.tt, t_1.tt
    while jd_1 - jd_0 > epsilon:
        date = np.linspace(jd_0, jd_1, 5)
        time = timescale.tt(jd=date)
        i = extremum(function(time))
        jd_0, jd_1 = date[np.max([0, i-1])], date[np.min([i+1, num-1])]
    return timescale.tt(jd=jd_0), function(timescale.tt(jd=jd_0))


def find_minimum(t_0, t_1, function, epsilon=EXTREMUM_SEARCH_EPSILON,
                 num=EXTREMUM_SEARCH_NUM_POINTS):
    """Return global minimum for a function ranging over a ranging over
    the time period of a pass.
    """
    return find_extremum(t_0, t_1, np.argmin, function, epsilon, num)


def find_maximum(t_0, t_1, function, epsilon=EXTREMUM_SEARCH_EPSILON,
                 num=EXTREMUM_SEARCH_NUM_POINTS):
    """Return global maximum for a function ranging over a ranging over
    the time period of a pass.
    """
    return find_extremum(t_0, t_1, np.argmax, function, epsilon, num)

# Functions for detemining


def semidiameter(radius, distance):
    """Return the angular semidiameter of an object viewed at a distance from
    from an observer.

    Args:
        radius: Radius in AU of the object.
        distance: Distance in AU of the object.

    Returns:
        The semidiameter of the object in radians.
    """
    return np.arcsin(radius / distance)


def eclipse_parameters(sat, earth, sun, time):
    """Return parameters used to determine whether a whether a satellite is
    in eclipse in a particular time in a pass. Based on the method described in
    https://www.celestrak.com/columns/v03n01/, "Visually Observing Earth Satellites".
    """
    position = earth + sat
    barycentric_e = position.at(time).observe(earth)
    barycentric_s = position.at(time).observe(sun)
    _, _, distance_to_earth = barycentric_e.radec()
    _, _, distance_to_sun = barycentric_s.radec()
    theta_e = semidiameter(EARTH_RADIUS, distance_to_earth.au)
    theta_s = semidiameter(SUN_RADIUS, distance_to_sun.au)
    theta = barycentric_e.separation_from(barycentric_s).radians
    return theta, theta_e, theta_s


def umbral_eclipse(sat, earth, sun, time):
    """ Return True if the satellite is in an umbral eclipse, False otherwise.
    """
    theta, theta_e, theta_s = eclipse_parameters(sat, earth, sun, time)
    return np.logical_and(theta_e > theta_s, theta < (theta_e - theta_s))


def penumbral_eclipse(sat, earth, sun, time):
    """ Return True if the satellite is in an penumbral eclipse, False otherwise.
    """
    theta, theta_e, theta_s = eclipse_parameters(sat, earth, sun, time)
    return np.logical_and(np.abs(theta_e - theta_s) < theta, theta < (theta_e + theta_s))


def annular_eclipse(sat, earth, sun, time):
    """Return True if the satellite is in an annular eclipse, False otherwise.
    """
    theta, theta_e, theta_s = eclipse_parameters(sat, earth, sun, time)
    return np.logical_and(theta_s > theta_e, theta < (theta_s - theta_e))


def civil_twilight(topos, earth, sun, time):
    """ Return True if the observer at the given topos is in civil twilight
    or darker, False otherwise. Civil twilight is defined as the Sun being at
    least SUN_ALTITUDE_CIVIL_TWILIGHT below the local horizon.
    """
    location = earth + topos
    astrocentric = location.at(time).observe(sun).apparent()
    alt, _, _ = astrocentric.altaz('standard')
    return alt.degrees <= SUN_ALTITUDE_CIVIL_TWILIGHT


def apparent_magnitude(sat, topos, earth, sun, time):
    """Return the apparent visual magnitude of a satellite for an observer
    during a pass.

    Based on the method described in
    https://astronomy.stackexchange.com/questions/28744/calculating-the-apparent-magnitude-of-a-satellite,
    "Calculating the apparent magnitude of a satellite".
    """
    position = earth + sat
    observer = earth + topos
    barycentric_o = position.at(time).observe(observer)
    barycentric_s = position.at(time).observe(sun)
    phase_angle = barycentric_o.separation_from(barycentric_s).radians
    _, _, distance = barycentric_o.radec()
    term_2 = +5.0 * np.log10(distance.km / 1000.)
    arg = np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle)
    term_3 = -2.5 * np.log10(arg)
    return SAT_INTRINSIC_MAGNITUDE + term_2 + term_3


def satellite_pass(topos, sat, earth, sun, visible=True):
    """Return a function to be used with skyfield.find_discrete to find
    satellite passes for an observer within a time period.

    Based on the definition of sunrise_sunset in skyfields.almanac.
    """
    difference = sat - topos
    def sat_observable(time):
        topocentric = difference.at(time)
        alt, _, _ = topocentric.altaz('standard')
        if visible:
            observable = np.logical_and(
                np.logical_and(alt.degrees >= SAT_MINIMUM_OBSERVABLE_ALTITUDE,
                               civil_twilight(topos, earth, sun, time)),
                np.logical_not(umbral_eclipse(sat, earth, sun, time))
            )
        else:
            observable = alt.degrees >= SAT_MINIMUM_OBSERVABLE_ALTITUDE
        return observable
    sat_observable.rough_period = SAT_PASS_ROUGH_PERIOD
    return sat_observable

# Function to generate lists of dicts, each dict describing a satellite pass.

# Return the name of a direction given an azimuth in degrees.
# Used to mimic the way azimuth directions are rported in Heavens Above pass prediction tables.


def direction(degrees_az):
    """Return the name of a direction given an azimuth in degrees.

    Used to mimic the way azimuth directions are reported in Heavens Above pass
    prediction tables.
    """
    degrees = np.asarray(DIRECTION_DEGREES)
    idx = (np.abs(degrees - degrees_az)).argmin()
    return DIRECTION_NAMES[idx]


def prettify_pass(pass_dict, timezone_str):
    """Add keys and values to a dict describing a satellite pass, where
    these keys and values represent information useful in display pass information
    succinctly to a human reader, in emulation of the sort of tables and charts
    in the Heavens Above web site.
    """
    start_local_datetime = pass_dict['start_time'].astimezone(pytz.timezone(timezone_str))
    start_alt, start_az, start_d = pass_dict['start_position'].altaz('standard')
    culm_local_datetime = pass_dict['culmination_time'].astimezone(pytz.timezone(timezone_str))
    culm_alt, culm_az, culm_d = pass_dict['culmination_position'].altaz('standard')
    end_local_datetime = pass_dict['end_time'].astimezone(pytz.timezone(timezone_str))
    end_alt, end_az, end_d = pass_dict['end_position'].altaz('standard')
    pretty_dict = {
        'date': start_local_datetime.isoformat(' ', timespec='seconds')[:11],
        'timezone': timezone_str,
        'start': start_local_datetime.isoformat(' ', timespec='seconds')[11:19],
        'start_alt': int(np.round(start_alt.degrees)),
        'start_az': direction(start_az.degrees),
        'start_d': int(np.round(start_d.km)),
        'culm': culm_local_datetime.isoformat(' ', timespec='seconds')[11:19],
        'culm_alt': int(np.round(culm_alt.degrees)),
        'culm_az': direction(culm_az.degrees),
        'culm_d': int(np.round(culm_d.km)),
        'end': end_local_datetime.isoformat(' ', timespec='seconds')[11:19],
        'end_alt': int(np.round(end_alt.degrees)),
        'end_az': direction(end_az.degrees),
        'end_d': int(np.round(end_d.km))
    }
    pass_dict.update(pretty_dict)


def passes(t_0, t_1, topos, sat, earth, sun, visible=True, pretty=False):
    """Return a list of dicts describing passes for a given satellite, given an
    observer as a specific topos and a start and end time.
    """
    time, _ = almanac.find_discrete(t_0, t_1, satellite_pass(topos, sat, earth, sun, visible))
    sat_passes = []
    difference = sat - topos
    if pretty:
        timezone_str = tzwhere.tzwhere().tzNameAt(topos.latitude.degrees,
                                                  topos.longitude.degrees)
    for i in range(1, len(time), 2):
        start_t, end_t = time[i-1], time[i] # TODO: check that y[i-1] and not y[i]
        culm_t, _ = find_maximum(start_t, end_t,
                                 lambda t: difference.at(t).altaz('standard')[0].degrees)
        _, mag = find_minimum(start_t, end_t,
                              lambda t: apparent_magnitude(sat, topos, earth, sun, t))
        if civil_twilight(topos, earth, sun, start_t) and civil_twilight(topos, earth, sun, end_t):
            if umbral_eclipse(sat, earth, sun, start_t) and umbral_eclipse(sat, earth, sun, end_t):
                pass_type = 'eclipsed'
            else:
                pass_type = 'visible'
        else:
            pass_type = 'daylight'
        basic_dict = {
            'satellite': sat,
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
            prettify_pass(basic_dict, timezone_str)
        sat_passes.append(basic_dict)
    return sat_passes
