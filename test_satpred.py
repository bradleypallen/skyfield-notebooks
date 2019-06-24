import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz
from skyfield import api, data
import satpred
import skychart

def test_satpred_1():
    tle_line_1 = "1 25544U 98067A   19174.96066161 -.00001641  00000-0 -19667-4 0  9994"
    tle_line_2 = "2 25544  51.6426 327.6827 0008340  79.1947  72.7878 15.51216338176304"
    load = api.Loader('./data')
    ts = load.timescale()
    ephemeris = load('de421.bsp')
    sun = ephemeris['sun']
    earth = ephemeris['earth']
    iss = api.EarthSatellite(tle_line_1, tle_line_2, name='ISS (ZARYA)', ts=ts)
    manhattan_beach_ca_usa = api.Topos(latitude='33.881519 N',
                                       longitude='118.388177 W',
                                       elevation_m=33)
    pacific = pytz.timezone('US/Pacific')
    d0 = datetime.datetime(2019, 6, 1, 0, 0)
    t0 = ts.utc(pacific.localize(d0))
    d1 = datetime.datetime(2019, 6, 11, 0, 0)
    t1 = ts.utc(pacific.localize(d1))
    columns = ['date', 'peak_magnitude', 'start', 'start_alt', 'start_az',
               'start_d', 'culm', 'culm_alt',
               'culm_az', 'culm_d', 'end', 'end_alt', 'end_az', 'end_d']

    df = satpred.SatelliteEphemeris(t0, t1, iss,
                                    manhattan_beach_ca_usa,
                                    earth, sun).to_dataframe()
    assert len(df) == 10
    assert df.iloc[8]['date'] == '2019-06-07'
    assert df.iloc[8]['peak_magnitude'] == -2.8
    assert df.iloc[8]['start'] == '21:08:35'
    assert df.iloc[8]['start_alt'] == 10
    assert df.iloc[8]['start_az'] == 'WNW'
    assert df.iloc[8]['start_d'] == 1493
    assert df.iloc[8]['culm'] == '21:11:46'
    assert df.iloc[8]['culm_alt'] == 42
    assert df.iloc[8]['culm_az'] == 'SW'
    assert df.iloc[8]['culm_d'] == 605
    assert df.iloc[8]['end'] == '21:13:52'
    assert df.iloc[8]['end_alt'] == 18
    assert df.iloc[8]['end_az'] == 'SSE'
    assert df.iloc[8]['end_d'] == 1083
    assert df.iloc[8]['pass_type'] == 'visible'

    satephem = satpred.SatelliteEphemeris(t0, t1, iss, manhattan_beach_ca_usa,
                                          earth, sun, visible=False)
    assert len(satephem.passes) == 39
    df = satephem.to_dataframe()
    assert df.iloc[27]['date'] == '2019-06-07'
    assert df.iloc[27]['start'] == '21:08:35'
    assert df.iloc[27]['start_alt'] == 10
    assert df.iloc[27]['start_az'] == 'WNW'
    assert df.iloc[27]['start_d'] == 1493
    assert df.iloc[27]['culm'] == '21:11:46'
    assert df.iloc[27]['culm_alt'] == 42
    assert df.iloc[27]['culm_az'] == 'SW'
    assert df.iloc[27]['culm_d'] == 605
    assert df.iloc[27]['end'] == '21:14:57'
    assert df.iloc[27]['end_alt'] == 10
    assert df.iloc[27]['end_az'] == 'SSE'
    assert df.iloc[27]['end_d'] == 1490
    assert df.iloc[27]['pass_type'] == 'visible'
