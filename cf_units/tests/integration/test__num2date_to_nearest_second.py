# (C) British Crown Copyright 2016, Met Office
#
# This file is part of cf_units.
#
# cf_units is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cf_units is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with cf_units.  If not, see <http://www.gnu.org/licenses/>.
"""Test function :func:`cf_units._num2date_to_nearest_second`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import unittest
import datetime

import numpy as np
import numpy.testing
import terra.datetime

from cf_units import _num2date_to_nearest_second, Unit


class Test(unittest.TestCase):
    def setup_units(self, calendar):
        self.useconds = Unit('seconds since 1970-01-01',  calendar)
        self.uminutes = Unit('minutes since 1970-01-01', calendar)
        self.uhours = Unit('hours since 1970-01-01', calendar)
        self.udays = Unit('days since 1970-01-01', calendar)

    def check_dates(self, nums, utimes, expected):
        res = np.array([_num2date_to_nearest_second(num, utime)
                        for num, utime in zip(nums, utimes)])
        numpy.testing.assert_array_equal(expected, res)

    def test_scalar(self):
        utime = Unit('seconds since 1970-01-01',  'gregorian')
        num = 5.
        acal = terra.datetime.GregorianNoLeapSecond()
        exp = terra.datetime.datetime(1970, 1, 1, 0, 0, 5,
                                      calendar=acal)
        res = _num2date_to_nearest_second(num, utime)
        self.assertEqual(exp, res)

    def test_sequence(self):
        tunit = Unit('seconds since 1970-01-01',  'gregorian')
        nums = [20., 40., 60., 80, 100.]
        acal = terra.datetime.GregorianNoLeapSecond()
        exp = [terra.datetime.datetime(1970, 1, 1, 0, 0, 20, calendar=acal),
               terra.datetime.datetime(1970, 1, 1, 0, 0, 40, calendar=acal),
               terra.datetime.datetime(1970, 1, 1, 0, 1, calendar=acal),
               terra.datetime.datetime(1970, 1, 1, 0, 1, 20, calendar=acal),
               terra.datetime.datetime(1970, 1, 1, 0, 1, 40, calendar=acal)]
        res = _num2date_to_nearest_second(nums, tunit)
        np.testing.assert_array_equal(exp, res)

    def test_multidim_sequence(self):
        tunit = Unit('seconds since 1970-01-01',  'gregorian')
        nums = [[20., 40., 60.],
                [80, 100., 120.]]
        exp_shape = (2, 3)
        res = _num2date_to_nearest_second(nums, tunit)
        self.assertEqual(exp_shape, res.shape)

    def test_masked_ndarray(self):
        tunit = Unit('seconds since 1970-01-01',  'gregorian')
        nums = np.ma.masked_array([20., 40., 60.], [False, True, False])
        acal = terra.datetime.GregorianNoLeapSecond()
        exp = [terra.datetime.datetime(1970, 1, 1, 0, 0, 20, calendar=acal),
               None,
               terra.datetime.datetime(1970, 1, 1, 0, 1, calendar=acal)]
        res = _num2date_to_nearest_second(nums, tunit)
        np.testing.assert_array_equal(exp, res)

    # Gregorian Calendar tests

    def test_simple_gregorian(self):
        self.setup_units('gregorian')
        nums = [20., 40.,
                75., 150.,
                8., 16.,
                300., 600.]
        utimes = [self.useconds, self.useconds,
                  self.uminutes, self.uminutes,
                  self.uhours, self.uhours,
                  self.udays, self.udays]
        acal = terra.datetime.GregorianNoLeapSecond()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 20, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 40, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 1, 15, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 2, 30, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 8, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 16, calendar=acal),
                    terra.datetime.datetime(1970, 10, 28, calendar=acal),
                    terra.datetime.datetime(1971, 8, 24, calendar=acal)]
        self.check_dates(nums, utimes, expected)

    def test_fractional_gregorian(self):
        self.setup_units('gregorian')
        nums = [5./60., 10./60.,
                15./60., 30./60.,
                8./24., 16./24.]
        utimes = [self.uminutes, self.uminutes,
                  self.uhours, self.uhours,
                  self.udays, self.udays]
        acal = terra.datetime.GregorianNoLeapSecond()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 5, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 10, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 15, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 30, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 8, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 16, calendar=acal)]

        self.check_dates(nums, utimes, expected)

    def test_fractional_second_gregorian(self):
        self.setup_units('gregorian')
        nums = [0.25, 0.5, 0.75,
                1.5, 2.5, 3.5, 4.5]
        utimes = [self.useconds] * 7
        acal = terra.datetime.GregorianNoLeapSecond()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 0, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 1, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 1, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 2, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 3, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 4, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 5, calendar=acal)]

        self.check_dates(nums, utimes, expected)

    # 360 day Calendar tests

    def test_simple_360_day(self):
        self.setup_units('360_day')
        nums = [20., 40.,
                75., 150.,
                8., 16.,
                300., 600.]
        utimes = [self.useconds, self.useconds,
                  self.uminutes, self.uminutes,
                  self.uhours, self.uhours,
                  self.udays, self.udays]
        acal = terra.datetime.G360Day()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 20, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 40, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 1, 15, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 2, 30, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 8, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 16, calendar=acal),
                    terra.datetime.datetime(1970, 11, 1, calendar=acal),
                    terra.datetime.datetime(1971, 9, 1, calendar=acal)]

        self.check_dates(nums, utimes, expected)

    def test_fractional_360_day(self):
        self.setup_units('360_day')
        nums = [5./60., 10./60.,
                15./60., 30./60.,
                8./24., 16./24.]
        utimes = [self.uminutes, self.uminutes,
                  self.uhours, self.uhours,
                  self.udays, self.udays]
        acal = terra.datetime.G360Day()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 5, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 10, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 15, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 30, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 8, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 16, calendar=acal)]

        self.check_dates(nums, utimes, expected)

    def test_fractional_second_360_day(self):
        self.setup_units('360_day')
        nums = [0.25, 0.5, 0.75,
                1.5, 2.5, 3.5, 4.5]
        utimes = [self.useconds] * 7
        acal = terra.datetime.G360Day()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 0, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 1, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 1, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 2, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 3, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 4, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 5, calendar=acal)]

        self.check_dates(nums, utimes, expected)

    # 365 day Calendar tests

    def test_simple_365_day(self):
        self.setup_units('365_day')
        nums = [20., 40.,
                75., 150.,
                8., 16.,
                300., 600.]
        utimes = [self.useconds, self.useconds,
                  self.uminutes, self.uminutes,
                  self.uhours, self.uhours,
                  self.udays, self.udays]
        acal = terra.datetime.G365Day()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 20, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 40, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 1, 15, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 2, 30, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 8, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 16, calendar=acal),
                    terra.datetime.datetime(1970, 10, 28, calendar=acal),
                    terra.datetime.datetime(1971, 8, 24, calendar=acal)]

        self.check_dates(nums, utimes, expected)

    def test_fractional_365_day(self):
        self.setup_units('365_day')
        nums = [5./60., 10./60.,
                15./60., 30./60.,
                8./24., 16./24.]
        utimes = [self.uminutes, self.uminutes,
                  self.uhours, self.uhours,
                  self.udays, self.udays]
        acal = terra.datetime.G365Day()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 5, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 10, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 15, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 30, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 8, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 16, calendar=acal)]

        self.check_dates(nums, utimes, expected)

    def test_fractional_second_365_day(self):
        self.setup_units('365_day')
        nums = [0.25, 0.5, 0.75,
                1.5, 2.5, 3.5, 4.5]
        utimes = [self.useconds] * 7
        acal = terra.datetime.G365Day()
        expected = [terra.datetime.datetime(1970, 1, 1, 0, 0, 0, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 1, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 1, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 2, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 3, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 4, calendar=acal),
                    terra.datetime.datetime(1970, 1, 1, 0, 0, 5, calendar=acal)]

        self.check_dates(nums, utimes, expected)

if __name__ == '__main__':
    unittest.main()
