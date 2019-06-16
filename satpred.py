# -*- coding: utf-8 -*-
"""satpred.py

A Python module for computing satellite passes for
a given satellite and observer.

Todo:
    * A more robust version of find_extremum() that can deal with local extrema.
    * Check that the t_0 and t_1 supplied to passes() do not fall inside a pass.

"""


import numpy as np
import pandas as pd
import pytz
from tzwhere import tzwhere
from skyfield import almanac, earthlib
import skychart


DIRECTION_NAMES = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
DIRECTION_DEGREES = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,
                     180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5, 360.0]
EARTH_RADIUS = earthlib.earth_radius_au
SUN_RADIUS = 0.00465 # Sun's average radius in AU
SUN_ALTITUDE_CIVIL_TWILIGHT = -6.
PASS_TYPES = ['eclipsed', 'visible', 'daylight']
SAT_MINIMUM_OBSERVABLE_ALTITUDE = 10.
SAT_INTRINSIC_MAGNITUDE = -1.3
SAT_PASS_ROUGH_PERIOD = 0.01
EXTREMUM_SEARCH_NUM_POINTS = 5
EXTREMUM_SEARCH_EPSILON = 0.5 / 86400. # A half-second fraction of a Julian day


def find_extremum(t_0, t_1, extremum, function, epsilon=EXTREMUM_SEARCH_EPSILON,
                  num=EXTREMUM_SEARCH_NUM_POINTS):
    """Find a global extremum for a function with a domain of skyfield.api.Time
    for values between t_0 and t_1.

    Assumes well-behaved functions with a global extremum and no other
    local extrema.

    Args:
        t_0: Start time for period to be searched.
        t_1: End time for period to be searched.
        extremum: function that computes an extremum.
        function: a function that takes time as its single argument, returning
            a numeric value.
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
    """Find a global minimum for a function with a domain of skyfield.api.Time
    for values between t_0 and t_1.

    Args:
        t_0: Start time for period to be searched.
        t_1: End time for period to be searched.
        function: a function that takes time as its single argument, returning
            a numeric value.
        epsilon: a float defining the distance less than which two times will
            be treated as equal.

    Returns:
        Two values, the first being the time of the minimum, and the second
        being the value of the minimum.
    """

    return find_extremum(t_0, t_1, np.argmin, function, epsilon, num)


def find_maximum(t_0, t_1, function, epsilon=EXTREMUM_SEARCH_EPSILON,
                 num=EXTREMUM_SEARCH_NUM_POINTS):
    """Find a global maximum for a function with a domain of skyfield.api.Time
    for values between t_0 and t_1.

    Args:
        t_0: Start time for period to be searched.
        t_1: End time for period to be searched.
        function: a function that takes time as its single argument, returning
            a numeric value.
        epsilon: a float defining the distance less than which two times will
            be treated as equal.

    Returns:
        Two values, the first being the time of the maximum, and the second
        being the value of the maximum.
    """

    return find_extremum(t_0, t_1, np.argmax, function, epsilon, num)


def semidiameter(radius, distance):
    """Compute the angular semidiameter of an object viewed at a distance from
    from an observer.

    Args:
        radius: Radius in AU of the object.
        distance: Distance in AU of the object.

    Returns:
        The semidiameter of the object in radians.
    """

    return np.arcsin(radius / distance)


def eclipse_parameters(sat, earth, sun, time):
    """Compute the parameters used to determine whether a whether a satellite is
    in eclipse in a particular time in a pass. Based on the method described by
    T.S. Kelso in "Visually Observing Earth Satellites",
    https://www.celestrak.com/columns/v03n01/.

    Args:
        sat: A skyfield.sgp4lib.EarthSatellite object.
        earth: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of Earth relative to
            the Solar System barycenter.
        sun: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of the Sun relative to
            the Solar System barycenter.
        time: A skyfield.api.Time object.

    Returns:
        theta: The phase angle from the satellite to the Earth and the Sun at
            the given time.
        theta_e: the semidiameter of the Earth as observed from the satellite at
            the given time.
        theta_s: the semidiameter of the Sun as observed from the satellite at
            the given time.
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
    """ Determine if the satellite is in an umbral eclipse.

    Args:
        sat: A skyfield.sgp4lib.EarthSatellite object.
        earth: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of Earth relative to
            the Solar System barycenter.
        sun: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of the Sun relative to
            the Solar System barycenter.
        time: A skyfield.api.Time object.

    Returns:
        True if the satellite is in umbral eclipse of the Sun by the Earth at
            the given time, otherwise False.
    """

    theta, theta_e, theta_s = eclipse_parameters(sat, earth, sun, time)
    return np.logical_and(theta_e > theta_s,
                          theta < (theta_e - theta_s))


def penumbral_eclipse(sat, earth, sun, time):
    """ Determine if the satellite is in an penumbral eclipse.

    Args:
        sat: A skyfield.sgp4lib.EarthSatellite object.
        earth: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of Earth relative to
            the Solar System barycenter.
        sun: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of the Sun relative to
            the Solar System barycenter.
        time: A skyfield.api.Time object.

    Returns:
        True if the satellite is in penumbral eclipse of the Sun by the Earth at
            the given time, otherwise False.
    """

    theta, theta_e, theta_s = eclipse_parameters(sat, earth, sun, time)
    return np.logical_and(np.abs(theta_e - theta_s) < theta,
                          theta < (theta_e + theta_s))


def annular_eclipse(sat, earth, sun, time):
    """Determine if the satellite is in an annular eclipse.

    Args:
        sat: A skyfield.sgp4lib.EarthSatellite object.
        earth: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of Earth relative to
            the Solar System barycenter.
        sun: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of the Sun relative to
            the Solar System barycenter.
        time: A skyfield.api.Time object.

    Returns:
        True if the satellite is in annular eclipse of the Sun by the Earth at
            the given time, otherwise False.
    """

    theta, theta_e, theta_s = eclipse_parameters(sat, earth, sun, time)
    return np.logical_and(theta_s > theta_e,
                          theta < (theta_s - theta_e))


def civil_twilight(topos, earth, sun, time):
    """ Determine if the observer at a topos is in civil twilight
    or darker.

    Civil twilight is defined as the Sun being at least
    SUN_ALTITUDE_CIVIL_TWILIGHT below the local horizon.

    Args:
        topos: A skyfield.toposlib.Topos object.
        earth: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of Earth relative to
            the Solar System barycenter.
        sun: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of the Sun relative to
            the Solar System barycenter.
        time: A skyfield.api.Time object.

    Returns:
        True if the observer at the given topos is in civil twilight
            or darker, False otherwise.
    """

    location = earth + topos
    astrocentric = location.at(time).observe(sun).apparent()
    alt, _, _ = astrocentric.altaz('standard')
    return alt.degrees <= SUN_ALTITUDE_CIVIL_TWILIGHT


def apparent_magnitude(sat, topos, earth, sun, time):
    """Find the apparent visual magnitude of a satellite for an observer
    during a pass.

    Based on the method described in "Calculating the apparent magnitude of
    a satellite",
    https://astronomy.stackexchange.com/questions/28744/calculating-the-apparent-magnitude-of-a-satellite.

    Args:
        sat: A skyfield.sgp4lib.EarthSatellite object.
        topos: A skyfield.toposlib.Topos object.
        earth: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of Earth relative to
            the Solar System barycenter.
        sun: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of the Sun relative to
            the Solar System barycenter.
        time: A skyfield.api.Time object.

    Returns:
        The estimated apparent visual magnitude of the satellite at
            the given time, for the given observer.
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


def satellite_pass(sat, topos, earth, sun, visible=True):
    """Generate a function to be used to determine if a satellite is passing
    over an observer at a given time.

    Based on the definition of skyfield.almanac.sunrise_sunset.

    Args:
        sat: A skyfield.sgp4lib.EarthSatellite object.
        topos: A skyfield.toposlib.Topos object.
        earth: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of Earth relative to
            the Solar System barycenter.
        sun: A vector (obtained by loading an ephemeris file) supporting
            the calculation of the position of the Sun relative to
            the Solar System barycenter.
        visible: If True, then only nighttime, uneclipsed passes will be
            returned by sat_observable(), otherwise all passes with culmination
            altitude > 10 degrees will be returned.

    Returns:
        A function that can be passed to skyfield.almanac.find_discrete to
        compute whether the given satellite and observer is observable
        at a given time.
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


def direction(degrees_az):
    """Return the name of a direction given an azimuth in degrees.

    Used to mimic the way azimuth directions are reported in Heavens Above pass
    prediction tables.

    Args:
        degrees_az: The azimuth in degrees to map to a direction.

    Returns:
        A direction in DIRECTION_NAMES cooresponding to the degrees to which
            the azimuth is nearest.
    """

    degrees = np.asarray(DIRECTION_DEGREES)
    idx = (np.abs(degrees - degrees_az)).argmin()
    return DIRECTION_NAMES[idx]


class SatellitePass:

    def __init__(self, sat, topos, timezone_str, start_time, culmination_time,
                 end_time, pass_type, peak_magnitude):

        difference = sat - topos

        # These instance variables contain the main information describing
        # the satellite pass, represented using values of types
        # skyfield.sgp4lib.EarthSatellite, skyfield.toposlib.Topos,
        # skyfield.timelib.Time, and skyfield.positionlib.Geocentric.

        self.satellite = sat
        self.topos = topos
        self.start_time = start_time
        self.start_position = difference.at(start_time)
        self.culmination_time = culmination_time
        self.culmination_position = difference.at(culmination_time)
        self.end_time = end_time
        self.end_position = difference.at(end_time)

        # These additional instance variables represent information useful in
        # displaying pass information succinctly to a human reader, represented
        # using floats and strings, in emulation of those shown in tables and
        # charts on the Heavens Above web site.

        start_local_datetime = self.start_time.astimezone(pytz.timezone(timezone_str))
        start_alt, start_az, start_d = self.start_position.altaz('standard')
        culm_local_datetime = self.culmination_time.astimezone(pytz.timezone(timezone_str))
        culm_alt, culm_az, culm_d = self.culmination_position.altaz('standard')
        end_local_datetime = self.end_time.astimezone(pytz.timezone(timezone_str))
        end_alt, end_az, end_d = self.end_position.altaz('standard')

        self.date = start_local_datetime.isoformat(' ', timespec='seconds')[:10]
        self.timezone = timezone_str
        self.start = start_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        self.start_alt = int(np.round(start_alt.degrees))
        self.start_az = direction(start_az.degrees)
        self.start_d = int(np.round(start_d.km))
        self.culm = culm_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        self.culm_alt = int(np.round(culm_alt.degrees))
        self.culm_az = direction(culm_az.degrees)
        self.culm_d = int(np.round(culm_d.km))
        self.end = end_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        self.end_alt = int(np.round(end_alt.degrees))
        self.end_az = direction(end_az.degrees)
        self.end_d = int(np.round(end_d.km))
        self.pass_type = pass_type
        self.peak_magnitude = np.round(peak_magnitude, 1)


    def to_json(self):
        return {
            'satellite': self.satellite,
            'topos': self.topos,
            'start_time': self.start_time,
            'start_position': self.start_position,
            'culmination_time': self.culmination_time,
            'culmination_position': self.culmination_position,
            'end_time': self.end_time,
            'end_position': self.end_position,
            'date': self.date,
            'timezone': self.timezone,
            'start': self.start,
            'start_alt': self.start_alt,
            'start_az': self.start_az,
            'start_d': self.start_d,
            'culm': self.culm,
            'culm_alt': self.culm_alt,
            'culm_az': self.culm_az,
            'culm_d': self.culm_d,
            'end': self.end,
            'end_alt': self.end_alt,
            'end_az': self.end_az,
            'end_d': self.end_d,
            'pass_type': self.pass_type,
            'peak_magnitude': self.peak_magnitude
        }


class SatelliteEphemeris:

    def __init__(self, t_0, t_1, sat, topos, earth, sun, visible=True):
        """Create a SatelliteEphemeris that describes passes for
        a given satellite, observer at a specific topos, a start and end time,
        data describing the positions of the Earth and Sun, and whether
        only visible passes or all passes should be included.

        Args:
            t_0:  A skyfield.api.Time object that is the start time for the period
                to search for passes.
            t_1: A skyfield.api.Time object that is the end time for the period
                to search for passes.
            sat: A skyfield.sgp4lib.EarthSatellite object.
            topos: A skyfield.toposlib.Topos object.
            earth: A vector (obtained by loading an ephemeris file) supporting
                the calculation of the position of Earth relative to
                the Solar System barycenter.
            sun: A vector (obtained by loading an ephemeris file) supporting
                the calculation of the position of the Sun relative to
                the Solar System barycenter.
            visible: If True, then only nighttime, uneclipsed passes will be
                returned, otherwise all passes with culmination
                altitude > 10 degrees will be returned.

        Returns:
            An instance of a SatelliteEphemeris class.
        """
        time, _ = almanac.find_discrete(t_0, t_1,
                                        satellite_pass(sat, topos, earth,
                                                       sun, visible))
        difference = sat - topos
        self.passes = []
        timezone_str = tzwhere.tzwhere().tzNameAt(topos.latitude.degrees,
                                                  topos.longitude.degrees)
        for i in range(1, len(time), 2):
            start_t, end_t = time[i-1], time[i] # TODO: check that y[i-1] and not y[i]
            culm_t, _ = find_maximum(start_t, end_t,
                                     lambda t: difference.at(t).altaz('standard')[0].degrees)
            _, mag = find_minimum(start_t, end_t,
                                  lambda t: apparent_magnitude(sat, topos,
                                                               earth, sun, t))
            if civil_twilight(topos, earth, sun, start_t) and civil_twilight(topos, earth, sun, end_t):
                if umbral_eclipse(sat, earth, sun, start_t) and umbral_eclipse(sat, earth, sun, end_t):
                    pass_type = PASS_TYPES[0]
                else:
                    pass_type = PASS_TYPES[1]
            else:
                pass_type = PASS_TYPES[2]
            self.passes.append(SatellitePass(sat, topos, timezone_str, start_t,
                                             culm_t, end_t, pass_type,
                                             np.round(mag, 1)))


    def to_dataframe(self):
        return pd.DataFrame([p.to_json() for p in self.passes])
