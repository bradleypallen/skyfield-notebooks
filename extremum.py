# -*- coding: utf-8 -*-
"""extremum.py

Routines for calculating the extremum of a given function of time.

Todo:
    * A more robust version of find_extremum() that can deal with local extrema.

"""


import numpy as np


_EXTREMUM_SEARCH_NUM_POINTS = 5
_EXTREMUM_SEARCH_EPSILON = 0.5 / 86400. # A half-second fraction of a Julian day


def find_extremum(t_0, t_1, extremum, function, epsilon=_EXTREMUM_SEARCH_EPSILON,
                  num=_EXTREMUM_SEARCH_NUM_POINTS):
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


def find_minimum(t_0, t_1, function, epsilon=_EXTREMUM_SEARCH_EPSILON,
                 num=_EXTREMUM_SEARCH_NUM_POINTS):
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


def find_maximum(t_0, t_1, function, epsilon=_EXTREMUM_SEARCH_EPSILON,
                 num=_EXTREMUM_SEARCH_NUM_POINTS):
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

