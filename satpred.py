# -*- coding: utf-8 -*-
"""satpred.py

A Python module for computing satellite passes for
a given satellite and observer.

Todo:
    * Check that the t_0 and t_1 supplied to passes() do not fall inside a pass.

"""


import numpy as np
import pandas as pd
import pytz
from tzwhere import tzwhere
from skyfield import almanac
import extremum
import satpred_utilities


class SatellitePass:
    """A representation of a satellite pass.

    Args:
        sat: A skyfield.sgp4lib.EarthSatellite instance representing the
            the satellite.
        topos: A skyfield.toposlib.Topos instance representing the location of
            the observer.
        timezone: A string describing the time zone that the topos is in, e.g.
            'Europe/Paris'.
        start_time: A skyfield.timelib.Time instance representing the moment
            the satellite pass begins.
        culmination_time: A skyfield.timelib.Time instance representing the moment
            during the pass that the satellite has maximum altitude.
        end_time: A skyfield.timelib.Time instance representing the moment
            the satellite pass ends.
        pass_type: A string representing the type of the pass, i.e. whether it
            is during the daytime, during the night but eclipsed and hence
            not visible, or during the night and visible for at least some
            of the pass.
        peak_magnitude: A float representing the estimated apparent magnitude of
            the satellite during the pass.

    Attributes:
        sat: A skyfield.sgp4lib.EarthSatellite instance representing the
            the satellite.
        topos: A skyfield.toposlib.Topos instance representing the location of
            the observer.
        start_time: A skyfield.timelib.Time instance representing the moment
            the satellite pass begins.
        start_position: A skyfield.positionlib.Geocentric instance representing
            the position of the satellite at the start of the pass.
        culmination_time: A skyfield.timelib.Time instance representing the moment
            during the pass that the satellite has maximum altitude.
        culmination_position: A skyfield.positionlib.Geocentric instance representing
            the position of the satellite at the culmination of the pass.
        end_time: A skyfield.timelib.Time instance representing the moment
            the satellite pass ends.
        end_position: A skyfield.positionlib.Geocentric instance representing
            the position of the satellite at the end of the pass.
        direction_names: A list of strings naming the various compass directions.
        direction_degrees: A list of float representing the various compass
            directions in degrees.
        date: A string that is the date of the start of the pass in ISO 8601
            extended format.
        timezone: A string describing the time zone that the topos is in.
        start: A string that is the local time of the start of the pass in ISO 8601
            extended format.
        start_alt: A float that is the altitude of the satillite at the start of
            the pass in degrees.
        start_az: A string that is the name of the direction closest to the azimuth
            of the satellite at the start of the pass.
        start_d: A float that is the distance in kilometers of the satellite at
            the start of the pass.
        culm: A string that is the local time of the culmination of the pass in
            ISO 8601 extended format.
        culm_alt: A float that is the altitude of the satillite at the culmination
            of the pass in degrees.
        culm_az: A string that is the name of the direction closest to the azimuth
            of the satellite at the culmination of the pass.
        culm_d: A float that is the distance in kilometers of the satellite at
            the culmination of the pass.
        end: A string that is the local time of the end of the pass in ISO 8601
            extended format.
        end_alt: A float that is the altitude of the satillite at the end of
            the pass in degrees.
        end_az: A string that is the name of the direction closest to the azimuth
            of the satellite at the end of the pass.
        end_d: A float that is the distance in kilometers of the satellite at
            the end of the pass.
        pass_type: A string representing the type of the pass, i.e. whether it
            is during the daytime, during the night but eclipsed and hence
            not visible, or during the night and visible for at least some
            of the pass.
        peak_magnitude: A float representing the estimated apparent magnitude of
            the satellite during the pass.
    """

    def __init__(self, sat, topos, timezone, start_time, culmination_time,
                 end_time, pass_type, peak_magnitude):

        difference = sat - topos

        self.satellite = sat
        self.topos = topos
        self.start_time = start_time
        self.start_position = difference.at(start_time)
        self.culmination_time = culmination_time
        self.culmination_position = difference.at(culmination_time)
        self.end_time = end_time
        self.end_position = difference.at(end_time)

        start_local_datetime = self.start_time.astimezone(pytz.timezone(timezone))
        start_alt, start_az, start_d = self.start_position.altaz('standard')
        culm_local_datetime = self.culmination_time.astimezone(pytz.timezone(timezone))
        culm_alt, culm_az, culm_d = self.culmination_position.altaz('standard')
        end_local_datetime = self.end_time.astimezone(pytz.timezone(timezone))
        end_alt, end_az, end_d = self.end_position.altaz('standard')
        self.direction_names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
        self.direction_degrees = [ i*22.5 for i in range(17) ]

        self.date = start_local_datetime.isoformat(' ', timespec='seconds')[:10]
        self.timezone = timezone
        self.start = start_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        self.start_alt = int(np.round(start_alt.degrees))
        self.start_az = self._direction(start_az.degrees)
        self.start_d = int(np.round(start_d.km))
        self.culm = culm_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        self.culm_alt = int(np.round(culm_alt.degrees))
        self.culm_az = self._direction(culm_az.degrees)
        self.culm_d = int(np.round(culm_d.km))
        self.end = end_local_datetime.isoformat(' ', timespec='seconds')[11:19]
        self.end_alt = int(np.round(end_alt.degrees))
        self.end_az = self._direction(end_az.degrees)
        self.end_d = int(np.round(end_d.km))
        self.pass_type = pass_type
        self.peak_magnitude = np.round(peak_magnitude, 1)


    def _direction(self, degrees_az):
        """Return the name of a direction given an azimuth in degrees.

        Used to mimic the way azimuth directions are reported in Heavens Above pass
        prediction tables.

        Args:
            degrees_az: The azimuth in degrees to map to a direction.

        Returns:
            A direction in DIRECTION_NAMES cooresponding to the degrees to which
                the azimuth is nearest.
        """

        degrees = np.asarray(self.direction_degrees)
        idx = (np.abs(degrees - degrees_az)).argmin()
        return self.direction_names[idx]


    def to_json(self):
        """Returns the data in the instance as a JSON dict.

        Useful when converting a satpred.SatelliteEphemeris to a pandas.DataFrame.

        Args:
            None.

        Returns:
            A dictionary containing keys and values cooresponding to the
                instance variables of the SatellitePass and their values.
        """
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

    Attributes:
        passes: A list of satpred.SatellitePass instances.
        pass_types: A list of strings naming the various pass types.

    Returns:
        An instance of a SatelliteEphemeris class.
    """


    def __init__(self, t_0, t_1, sat, topos, earth, sun, visible=True):
        self.sat_minimum_observable_altitude = 10.0
        self.sat_pass_rough_period = 0.0042 # average pass duration of 6 minutes
        time, _ = almanac.find_discrete(t_0, t_1,
                                        self._satellite_pass(sat, topos, earth,
                                                             sun, visible))
        difference = sat - topos
        self.passes = []
        self.pass_types = ['eclipsed', 'visible', 'daylight']
        timezone_str = tzwhere.tzwhere().tzNameAt(topos.latitude.degrees,
                                                  topos.longitude.degrees)
        for i in range(1, len(time), 2):
            start_t, end_t = time[i-1], time[i] # TODO: check that y[i-1] and not y[i]
            culm_t, _ = extremum.find_maximum(start_t, end_t,
                                              lambda t: difference.at(t).altaz('standard')[0].degrees)
            _, mag = extremum.find_minimum(start_t, end_t,
                                           lambda t: satpred_utilities.apparent_magnitude(sat, topos,
                                                                        earth, sun, t))
            if satpred_utilities.civil_twilight(topos, earth, sun, start_t) and satpred_utilities.civil_twilight(topos, earth, sun, end_t):
                if satpred_utilities.umbral_eclipse(sat, earth, sun, start_t) and satpred_utilities.umbral_eclipse(sat, earth, sun, end_t):
                    pass_type = self.pass_types[0]
                else:
                    pass_type = self.pass_types[1]
            else:
                pass_type = self.pass_types[2]
            self.passes.append(SatellitePass(sat, topos, timezone_str, start_t,
                                             culm_t, end_t, pass_type,
                                             np.round(mag, 1)))


    def _satellite_pass(self, sat, topos, earth, sun, visible=True):
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
                    np.logical_and(alt.degrees >= self.sat_minimum_observable_altitude,
                                   satpred_utilities.civil_twilight(topos, earth, sun, time)),
                    np.logical_not(satpred_utilities.umbral_eclipse(sat, earth, sun, time))
                )
            else:
                observable = alt.degrees >= self.sat_minimum_observable_altitude
            return observable
        sat_observable.rough_period = self.sat_pass_rough_period
        return sat_observable


    def to_dataframe(self):
        """Generate a pandas.DataFrame containing humanly-readable information in the satellite ephemeris,
        each row describing a satellite pass.

        Returns:
            A pandas.DataFrame object.

        """
        return pd.DataFrame([p.to_json() for p in self.passes])
