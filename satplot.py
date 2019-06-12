# -*- coding: utf-8 -*-
"""satplot.py

A Python module for generating data about satellite passes for
a given satellite and observer.

Todo:
    * RA/dec displays of satellite passes.

"""


import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import Star


SAT_PASS_NUM_PLOT_POINTS = 16


def magnitude_to_marker_size(v_mag):
    """Returns a float to be used as the size of a matplotlib plot marker representing
    an object with a given visual megnitude.

    A third-degree polynomial was fit to a few hand-curated examples of
    marker sizes for magnitudes, using the following code:

    >>> x = np.array([-1.44, -0.5, 0., 1., 2., 3., 4., 5.])
    >>> y = np.array([120., 90., 60., 30., 15., 11., 6., 1.])
    >>> coeffs = np.polyfit(x, y, 3)

    This function is valid over the range -2.0 < v <= 6.0; v < -2.0 returns
    size = 160. and v > 6.0 returns size = 1.
    """
    if v_mag < -2.0:
        size = 160.0
    elif v_mag > 6.0:
        size = 1.0
    else:
        coeffs = [-0.39439046, 6.21313285, -33.09853387, 62.07732768]
        size = coeffs[0] * v_mag**3. + coeffs[1] * v_mag**2. + coeffs[2] * v_mag + coeffs[3]
    return size


def times_for_satellite_pass(sat_pass, num=SAT_PASS_NUM_PLOT_POINTS):
    """
    Returns a sequence of num evenly-spaced times during a satellite pass,
    starting with the start of the pass and ending with the end of the pass.
    """
    timescale = sat_pass['start_time'].ts
    jd_0, jd_1 = sat_pass['start_time'].tt, sat_pass['end_time'].tt
    date = np.linspace(jd_0, jd_1, num)
    return timescale.tt(jd=date)


def altaz_for_satellite_pass(sat_pass, time):
    """Returns the alt/az coordinates for the position of the satellite during
    the pass at a given time
    """
    difference = sat_pass['satellite'] - sat_pass['topos']
    topocentric = difference.at(time)
    return topocentric.altaz()


def altaz_and_mag_for_stars(sat_pass, observer, stars):
    """Returns arrays for the altitudes, azimuths and visual magnitudes of
    bright stars at the start time of a satellite pass.

    NOTE: this is an expensive workaround for the fact that
    Skyfield doesn't handle the one-observer-many-objects case for Apparent.altaz()
    (see https://github.com/skyfielders/python-skyfield/issues/229)
    """
    theta, r_angle, mag = [], [], []
    for i in range(len(stars)):
        star = Star.from_dataframe(stars.iloc[i])
        app = observer.at(sat_pass['start_time']).observe(star).apparent()
        altitude, azimuth, _ = app.altaz()
        theta.append(azimuth.radians)
        r_angle.append(altitude.degrees)
        mag.append(stars.iloc[i]['magnitude'])
    return np.array(r_angle), np.array(theta), np.array(mag)


def altaz_for_ephemeris_objects(observer, obj, time):
    """Returns the alt/az coordinates for an ephemeris object for a given time
    during the satellite pass.
    """
    apparent = observer.at(time).observe(obj).apparent()
    return apparent.altaz()


def satellite_pass_chart(sat_pass, ephemeris, bright_stars):
    """Plots a polar-coordinate chart displaying the satellite pass.
    """
    # Create matplotlib chart with polar projection
    axes = plt.subplot(111, projection='polar')
    axes.set_rlim(0, 90)
    axes.set_theta_zero_location('N')
    axes.set_theta_direction(1)
    axes.set_yticks(np.arange(0, 105, 15))
    axes.set_yticklabels(axes.get_yticks()[::-1])
    # Gather visualization data
    line_1a = f"{sat_pass['satellite'].name} {sat_pass['pass_type']}"
    line_1b = f"pass on {sat_pass['date']} {sat_pass['start']} - {sat_pass['end']}"
    line_2 = f"View from lat. {sat_pass['topos'].latitude}, long. {sat_pass['topos'].longitude}"
    line_3 = f"TZ {sat_pass['timezone']}, peak app. mag. {sat_pass['peak_magnitude']}"
    plt.title(f"{line_1a} {line_1b}\n\n{line_2}\n{line_3}")
    time = times_for_satellite_pass(sat_pass)
    sat_alt, sat_az, _ = altaz_for_satellite_pass(sat_pass, time)
    observer = ephemeris['earth'] + sat_pass['topos']
    sun_alt, sun_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['sun'],
                                                     sat_pass['start_time'])
    moon_alt, moon_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['moon'],
                                                       sat_pass['start_time'])
    if not sat_pass['pass_type'] == 'daylight':
        mercury_alt, mercury_az, _ = altaz_for_ephemeris_objects(observer,
                                                                 ephemeris['mercury'],
                                                                 sat_pass['start_time'])
        venus_alt, venus_az, _ = altaz_for_ephemeris_objects(observer,
                                                             ephemeris['venus'],
                                                             sat_pass['start_time'])
        mars_alt, mars_az, _ = altaz_for_ephemeris_objects(observer,
                                                           ephemeris['mars'],
                                                           sat_pass['start_time'])
        jupiter_alt, jupiter_az, _ = altaz_for_ephemeris_objects(observer,
                                                                 ephemeris['JUPITER BARYCENTER'],
                                                                 sat_pass['start_time'])
        saturn_alt, saturn_az, _ = altaz_for_ephemeris_objects(observer,
                                                               ephemeris['SATURN BARYCENTER'],
                                                               sat_pass['start_time'])
        stars_alt_degrees, stars_az_radians, stars_v = altaz_and_mag_for_stars(sat_pass,
                                                                               observer,
                                                                               bright_stars)
    # Plot visualization data
    # For visible (i.e., nighttime) passes, plot stars
    if not sat_pass['pass_type'] == 'daylight':
        plt.scatter(stars_az_radians,
                    90.-stars_alt_degrees,
                    [magnitude_to_marker_size(v) for v in stars_v],
                    'k')
    # Plot sun and moon on chart
    plt.scatter(sun_az.radians, 90.-sun_alt.degrees, 400, 'y')
    plt.scatter(moon_az.radians, 90.-moon_alt.degrees, 400, 'silver')
    # For visible (i.e., nighttime) passes, plot naked eye planets
    if not sat_pass['pass_type'] == 'daylight':
        plt.scatter(mercury_az.radians, 90.-mercury_alt.degrees, 50, 'y')
        plt.scatter(venus_az.radians, 90.-venus_alt.degrees, 150, 'c')
        plt.scatter(mars_az.radians, 90.-mars_alt.degrees, 100, 'r')
        plt.scatter(jupiter_az.radians, 90.-jupiter_alt.degrees, 150, 'y')
        plt.scatter(saturn_az.radians, 90.-saturn_alt.degrees, 125, 'y')
    # Plot satellite trajectory
    plt.plot(sat_az.radians, 90.-sat_alt.degrees, 'grey')
    # Show plot
    plt.show()
