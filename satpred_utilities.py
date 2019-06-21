# -*- coding: utf-8 -*-
"""satpred_utilities.py

Various functions supporting the calculation of satellite ephemerides.
"""


import numpy as np
from skyfield import earthlib


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
    theta_e = semidiameter(earthlib.earth_radius_au, distance_to_earth.au)
    theta_s = semidiameter(0.00465, distance_to_sun.au) # Sun's average radius in AU = 0.00465
    theta = barycentric_e.separation_from(barycentric_s).radians
    return theta, theta_e, theta_s


def umbral_eclipse(sat, earth, sun, time):
    """Determine if the satellite is in an umbral eclipse.

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
    """Determine if the satellite is in an penumbral eclipse.

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
    """Determine if the observer at a topos is in civil twilight
    or darker.

    Civil twilight is defined as the Sun being at least 6 degrees below 
    the local horizon.

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
    return alt.degrees <= -6.0 # definition of civil twilight


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
    term_1 = -1.3 # standard satellite intrinsic magnitude
    term_2 = +5.0 * np.log10(distance.km / 1000.)
    arg = np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle)
    term_3 = -2.5 * np.log10(arg)
    return term_1 + term_2 + term_3

