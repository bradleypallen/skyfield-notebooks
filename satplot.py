import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import Star

# Based on the approach outlined in the Millenium Star Atlas, vol. 1, p. xi.

def magnitude_to_marker_size(v):
    if v <= -1.:
        return 132.
    elif v <= 0.0:
        return 112.
    elif v <= 0.5:
        return 92.
    elif v <= 1.:
        return 72.
    elif v <= 1.5:
        return 52.
    else:
        if v <= 2.:
            v = 2.0
        return 1.52 ** (10. ** (0.135*(8.7-v)))
            
def satellite_pass_chart(sat_pass, ephemeris, bright_stars):
    ts = sat_pass['start_time'].ts
    jd0, jd1 = sat_pass['start_time'].tt, sat_pass['end_time'].tt
    jd = np.linspace(jd0, jd1, 32)
    t = ts.tt(jd=jd)
    difference = sat_pass['satellite'] - sat_pass['topos']
    topocentric = difference.at(t)
    sat_alt, sat_az, _ = topocentric.altaz()
    end_topocentric = difference.at(sat_pass['end_time'])
    end_alt, end_az, _ = end_topocentric.altaz()
    observer = ephemeris['earth'] + sat_pass['topos']
    theta, r, mag = [], [], []
    for i in range(len(bright_stars)):
        star = Star.from_dataframe(bright_stars.iloc[i])
        app = observer.at(sat_pass['start_time']).observe(star).apparent()
        alt, az, _ = app.altaz()
        theta.append(az.radians)
        r.append(alt.degrees)
        mag.append(bright_stars.iloc[i]['magnitude'])
    sun_apparent = observer.at(sat_pass['start_time']).observe(ephemeris['sun']).apparent()
    sun_alt, sun_az, _ = sun_apparent.altaz()
    moon_apparent = observer.at(sat_pass['start_time']).observe(ephemeris['moon']).apparent()
    moon_alt, moon_az, _ = moon_apparent.altaz()
    mercury_apparent = observer.at(sat_pass['start_time']).observe(ephemeris['mercury']).apparent()
    mercury_alt, mercury_az, _ = mercury_apparent.altaz()
    venus_apparent = observer.at(sat_pass['start_time']).observe(ephemeris['venus']).apparent()
    venus_alt, venus_az, _ = venus_apparent.altaz()
    mars_apparent = observer.at(sat_pass['start_time']).observe(ephemeris['mars']).apparent()
    mars_alt, mars_az, _ = mars_apparent.altaz()
    jupiter_apparent = observer.at(sat_pass['start_time']).observe(ephemeris['JUPITER BARYCENTER']).apparent()
    jupiter_alt, jupiter_az, _ = jupiter_apparent.altaz()
    saturn_apparent = observer.at(sat_pass['start_time']).observe(ephemeris['SATURN BARYCENTER']).apparent()
    saturn_alt, saturn_az, _ = saturn_apparent.altaz()
    ax = plt.subplot(111, projection='polar')
    ax.set_rlim(0, 90)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    ax.set_yticks(np.arange(0, 105, 15))
    ax.set_yticklabels(ax.get_yticks()[::-1])
    plt.scatter(np.array(theta), 90.-np.array(r), [ magnitude_to_marker_size(v) for v in mag ], 'k')
    plt.scatter(sun_az.radians, 90.-sun_alt.degrees, 400, 'y')
    plt.scatter(moon_az.radians, 90.-moon_alt.degrees, 400, 'silver')
    plt.scatter(mercury_az.radians, 90.-mercury_alt.degrees, 50, 'y')
    plt.scatter(venus_az.radians, 90.-venus_alt.degrees, 150, 'c')
    plt.scatter(mars_az.radians, 90.-mars_alt.degrees, 100, 'r')
    plt.scatter(jupiter_az.radians, 90.-jupiter_alt.degrees, 150, 'y')
    plt.scatter(saturn_az.radians, 90.-saturn_alt.degrees, 125, 'y')
    plt.plot(sat_az.radians, 90.-sat_alt.degrees, 'grey')
    plt.title(f"{sat_pass['satellite'].name} {sat_pass['date']} {sat_pass['start']} - {sat_pass['end']}\n\nView from lat. {sat_pass['topos'].latitude}, long. {sat_pass['topos'].longitude}\nTZ {sat_pass['timezone']}, peak app. mag. {sat_pass['peak_magnitude']}")
    plt.show()
