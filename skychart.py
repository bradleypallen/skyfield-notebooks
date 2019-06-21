# -*- coding: utf-8 -*-
"""skychart.py

A Python module for generating charts displaying the sky for a given observer
and time.

"""


import numpy as np
import matplotlib.pyplot as plt
from skyfield import api


def _magnitude_to_marker_size(v_mag):
    """Calculate the size of a matplotlib plot marker representing an object with
    a given visual megnitude.

    A third-degree polynomial was fit to a few hand-curated examples of
    marker sizes for magnitudes, as follows:

    >>> x = np.array([-1.44, -0.5, 0., 1., 2., 3., 4., 5.])
    >>> y = np.array([120., 90., 60., 30., 15., 11., 6., 1.])
    >>> coeffs = np.polyfit(x, y, 3)

    This function is valid over the range -2.0 < v <= 6.0; v < -2.0 returns
    size = 160. and v > 6.0 returns size = 1.

    Args:
        v_mag: A float representing the visual magnitude of a sky object.

    Returns:
        A float to be used as the size of a marker depicting the sky object in a
            matplotlib.plt scatterplot.
    """

    if v_mag < -2.0:
        size = 160.0
    elif v_mag > 6.0:
        size = 1.0
    else:
        coeffs = [-0.39439046, 6.21313285, -33.09853387, 62.07732768]
        size = coeffs[0] * v_mag**3. + coeffs[1] * v_mag**2. + coeffs[2] * v_mag + coeffs[3]
    return size


class AltAzFullSkyChart:
    """A matplotlib plot of the whole sky in alt/az coordinates.

    Args:
        observer_position: A position of the pass observer at a Topos, relative to the
            barycenter of the Solar System.
        time: A skyfield.timelib.Time object.: 

    Attributes:
        fig: A matplotlib.plt.Figure object.
        ax: A matplotlib.plt.Axes object.
        observer_position: A position of the pass observer at a Topos, relative to the
            barycenter of the Solar System.
        time: A skyfield.timelib.Time object.
    """

    def __init__(self, observer_position, time):
        plt.ioff()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, polar=True)
        self.ax.set_rlim(0, 90)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(1)
        self.ax.set_yticks(np.arange(0, 105, 15))
        self.ax.set_yticklabels(self.ax.get_yticks()[::-1])
        self.observer_position = observer_position
        self.time = time


    def plot_stars(self, stars, color='k'):
        """ Plot a set of stars on the chart.

        Args:
            stars: A pandas.DataFrame containing star records from a catalog.
            color: A string indicating the color in which to plot markers 
                representing stars.

        Note:
            This implements an expensive workaround for the fact that
                Skyfield doesn't handle the one-observer-many-objects case for Apparent.altaz().
                See https://github.com/skyfielders/python-skyfield/issues/229.

        Returns:
            None.
        """
        theta, r_angle, stars_v = [], [], []
        for i in range(len(stars)):
            star = api.Star.from_dataframe(stars.iloc[i])
            app = self.observer_position.at(self.time).observe(star).apparent()
            altitude, azimuth, _ = app.altaz()
            theta.append(azimuth.radians)
            r_angle.append(altitude.degrees)
            stars_v.append(stars.iloc[i]['magnitude'])
        self.ax.scatter(np.array(theta), 90.-np.array(r_angle),
                    [_magnitude_to_marker_size(v) for v in stars_v], color)


    def plot_ephemeris_object(self, obj, size, color='y'):
        """ Plot a solar system object on the chart.

        Args:
            obj: A vector (obtained by loading an ephemeris file) supporting
                the calculation of the position of the given object relative to
                the Solar System barycenter.
            size: A intger indicating the size of the marker to use to display 
                the object.
            color: A string indicating the color in which to plot markers 
                representing stars.

        Returns:
            None
        """
        apparent = self.observer_position.at(self.time).observe(obj).apparent()
        alt, az, _ = apparent.altaz()
        self.ax.scatter(az.radians, 90.-alt.degrees, size, color)


    def plot_satellite_pass(self, satellite_pass, num=32, color='grey'):
        """ Plot a set of stars on the chart.

        Args:
            satellite_pass: A satpred.SatellitePass object. 
            num: A integer indicating how many coordinates to use in plotting
                a line on the chart representing the satellite's path across
                the sky during the pass.
            color: The color to use to plot the satellite's path.

        Returns:
            None
        """
        timescale = satellite_pass.start_time.ts
        jd_0, jd_1 = satellite_pass.start_time.tt, satellite_pass.end_time.tt
        date = np.linspace(jd_0, jd_1, num)
        time = timescale.tt(jd=date)
        difference = satellite_pass.satellite - satellite_pass.topos
        topocentric = difference.at(time)
        sat_alt, sat_az, _ = topocentric.altaz()
        self.ax.plot(sat_az.radians, 90.-sat_alt.degrees, color)


    def display(self):
        """ Show the chart,

        Returns:
            None
        """
        plt.show()


class RADecSkyChart:
    """A matplotlib plot of a section of the sky in right ascension/declination coordinates.

    Args:
        observer_position: A position of the pass observer at a Topos, relative to the
            barycenter of the Solar System.
        time: A skyfield.timelib.Time object.: 
        ra_lim: A pair of right ascensions in decimal hours.
        dec_lim: A pair of declinations in decimal degrees.

    Attributes:
        fig: A matplotlib.plt.Figure object.
        ax: A matplotlib.plt.Axes object.
        observer_position: A position of the pass observer at a Topos, relative to the
            barycenter of the Solar System.
        time: A skyfield.timelib.Time object.
    """


    def __init__(self, observer_position, time, ra_lim, dec_lim):
        plt.ioff()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.grid()
        self.ax.set_xlim(ra_lim[0], ra_lim[1])
        self.ax.set_ylim(dec_lim[0], dec_lim[1])
        self.ax.tick_params(direction='in')
        self.ax.set_xticks(np.linspace(ra_lim[0], ra_lim[1], 5))
        self.ax.set_yticks(np.linspace(dec_lim[0], dec_lim[1], 5))
        self.observer_position = observer_position
        self.time = time


    def plot_stars(self, stars, color='k'):
        """ Plot a set of stars on the chart.

        Args:
            stars: A pandas.DataFrame containing star records from a catalog.
            color: A string indicating the color in which to plot markers 
                representing stars.

        Returns:
            None.
        """
        stars_v = stars['magnitude'].tolist()
        astrometric = self.observer_position.at(self.time).observe(api.Star.from_dataframe(stars))
        ra, dec, _ = astrometric.radec()
        self.ax.scatter(ra.hours, dec.degrees, [_magnitude_to_marker_size(v) for v in stars_v], color)


    def plot_ephemeris_object(self, obj, size, color='y'):
        """ Plot a solar system object on the chart.

        Args:
            obj: A vector (obtained by loading an ephemeris file) supporting
                the calculation of the position of the given object relative to
                the Solar System barycenter.
            size: A intger indicating the size of the marker to use to display 
                the object.
            color: A string indicating the color in which to plot markers 
                representing stars.

        Returns:
            None
        """
        astrometric = self.observer_position.at(self.time).observe(obj)
        ra, dec, _ = astrometric.radec()
        self.ax.scatter(ra.hours, dec.degrees, color)


    def plot_satellite_pass(self, satellite_pass, num=32, color='grey'):
        """ Plot a set of stars on the chart.

        Args:
            satellite_pass: A satpred.SatellitePass object. 
            num: A integer indicating how many coordinates to use in plotting
                a line on the chart representing the satellite's path across
                the sky during the pass.
            color: The color to use to plot the satellite's path.

        Returns:
            None
        """
        timescale = satellite_pass.start_time.ts
        jd_0, jd_1 = satellite_pass.start_time.tt, satellite_pass.end_time.tt
        date = np.linspace(jd_0, jd_1, num)
        time = timescale.tt(jd=date)
        difference = satellite_pass.satellite - satellite_pass.topos
        topocentric = difference.at(time)
        sat_ra, sat_dec, _ = topocentric.radec()
        self.ax.plot(sat_ra.hours, sat_dec.degrees, color)


    def display(self):
        """ Show the chart,

        Returns:
            None
        """
        plt.show()
