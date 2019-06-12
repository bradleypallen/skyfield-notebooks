import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import Star

# Convert visual magnitude to a size for a matplotlib plot marker.
# I fit a few hand-curated examples of marker sizes for magnitudes to
# a third-degree polynomial with the following code:
#
#    >>> x = np.array([-1.44, -0.5, 0., 1., 2., 3., 4., 5.])
#    >>> y = np.array([120., 90., 60., 30., 15., 11., 6., 1.])
#    >>> coeffs = np.polyfit(x, y, 3)
#
# This function is used over the range -2.0 < v <= 6.0; v < -2.0 returns
# size = 160. and v > 6.0 returns size = 1.

def magnitude_to_marker_size(v):
    if v < -2.0:
        return 160.0
    elif v > 6.0:
        return 1.0
    else:
        coeffs = [ -0.39439046,   6.21313285, -33.09853387,  62.07732768 ]
        return coeffs[0] * v**3. + coeffs[1] * v**2. + coeffs[2] * v + coeffs[3]

# Get the pass times for positions to plot

def times_for_satellite_pass(sat_pass):
    ts = sat_pass['start_time'].ts
    jd0, jd1 = sat_pass['start_time'].tt, sat_pass['end_time'].tt
    jd = np.linspace(jd0, jd1, 32)
    return ts.tt(jd=jd)

# Get the satellite's pass positions and alt/az coordinates

def altaz_for_satellite_pass(sat_pass, t):
    difference = sat_pass['satellite'] - sat_pass['topos']
    topocentric = difference.at(t)
    return topocentric.altaz()

# Get alt/az coordinates and magnitudes for bright stars
# NOTE: this is an expensive workaround for the fact that
# Skyfield doesn't handle the one-observer-many-objects case for Apparent.altaz()
# (see https://github.com/skyfielders/python-skyfield/issues/229)

def altaz_and_mag_for_stars(sat_pass, observer, stars):
    theta, r, mag = [], [], []
    for i in range(len(stars)):
        star = Star.from_dataframe(stars.iloc[i])
        app = observer.at(sat_pass['start_time']).observe(star).apparent()
        alt, az, _ = app.altaz()
        theta.append(az.radians)
        r.append(alt.degrees)
        mag.append(stars.iloc[i]['magnitude'])
    return np.array(r), np.array(theta), np.array(mag)

# Get alt/az coordinates for ephemeris objects

def altaz_for_ephemeris_objects(observer, object, t):
    apparent = observer.at(t).observe(object).apparent()
    return apparent.altaz()

# Plot a chart of a satellite pass

def satellite_pass_chart(sat_pass, ephemeris, bright_stars):
    # Create matplotlib chart with polar projection
    ax = plt.subplot(111, projection='polar')
    ax.set_rlim(0, 90)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_yticks(np.arange(0, 105, 15))
    ax.set_yticklabels(ax.get_yticks()[::-1])
    # Gather visualization data
    title_line_1 = f"{sat_pass['satellite'].name} {sat_pass['pass_type']} pass on {sat_pass['date']} {sat_pass['start']} - {sat_pass['end']}"
    title_line_2 = f"View from lat. {sat_pass['topos'].latitude}, long. {sat_pass['topos'].longitude}"
    title_line_3 = f"TZ {sat_pass['timezone']}, peak app. mag. {sat_pass['peak_magnitude']}"
    plt.title(f"{title_line_1}\n\n{title_line_2}\n{title_line_3}")
    t = times_for_satellite_pass(sat_pass)
    sat_alt, sat_az, _ = altaz_for_satellite_pass(sat_pass, t)
    observer = ephemeris['earth'] + sat_pass['topos']
    sun_alt, sun_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['sun'], sat_pass['start_time'])
    moon_alt, moon_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['moon'], sat_pass['start_time'])
    if not sat_pass['pass_type'] == 'daylight':
        mercury_alt, mercury_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['mercury'], sat_pass['start_time'])
        venus_alt, venus_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['venus'], sat_pass['start_time'])
        mars_alt, mars_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['mars'], sat_pass['start_time'])
        jupiter_alt, jupiter_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['JUPITER BARYCENTER'], sat_pass['start_time'])
        saturn_alt, saturn_az, _ = altaz_for_ephemeris_objects(observer, ephemeris['SATURN BARYCENTER'], sat_pass['start_time'])
        stars_alt_degrees, stars_az_radians, stars_v = altaz_and_mag_for_stars(sat_pass, observer, bright_stars)
    # Plot visualization data
    # For visible (i.e., nighttime) passes, plot stars
    if not sat_pass['pass_type'] == 'daylight':
        plt.scatter(stars_az_radians, 90.-stars_alt_degrees, [ magnitude_to_marker_size(v) for v in stars_v ], 'k')
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
