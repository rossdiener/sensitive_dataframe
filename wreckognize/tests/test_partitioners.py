# pylint: disable=no-self-use

import pytest

import pandas as pd
from pandas import DataFrame
from pandas.util.testing import assert_frame_equal

from wreckognize.partitioners import MondrianPartitioner
from wreckognize.sensitive_dataframe import SensitiveFrame


@pytest.fixture
def lower_df():
    return DataFrame([1, 1, 2, 5, 5], columns=['lower_values'])


@pytest.fixture
def middle_df():
    return DataFrame([1, 1, 3, 5, 5], columns=['middle_values'])


@pytest.fixture
def upper_df():
    return DataFrame([1, 1, 4, 5, 5], columns=['upper_values'])


@pytest.fixture
def wide_df():
    return DataFrame([1, 1, 1000, 1000, 1000], columns=['wide_values'])


@pytest.fixture
def slim_df():
    return DataFrame([1, 1, 1, 2, 2], columns=['slim_values'])


@pytest.fixture
def sample_df(slim_df, wide_df):
    return pd.concat([slim_df['slim_values'], wide_df['wide_values']], axis=1)


class TestMondrianPartitioner():

    def test_oversized_partition(self, middle_df):
        test_mp = MondrianPartitioner(10)
        test_sf = SensitiveFrame(middle_df, quasi_identifiers=['middle_values'])
        assert test_mp.partition_sf(test_sf) == [test_sf]

    def test_simple_partition(self, middle_df):
        test_mp = MondrianPartitioner(2)
        quasi_identifiers = ['middle_values']
        test_sf = SensitiveFrame(middle_df, quasi_identifiers=quasi_identifiers)
        actual_partitions = test_mp.partition_sf(test_sf)

        expected_left_df = DataFrame.from_dict({0: 1, 1: 1}, orient='index')
        expected_left_df.columns = ['middle_values']
        expected_right_df = DataFrame.from_dict({2: 3, 3: 5, 4: 5}, orient='index')
        expected_right_df.columns = ['middle_values']

        assert len(actual_partitions) == 2
        assert actual_partitions[0].quasi_identifiers == quasi_identifiers
        assert actual_partitions[1].quasi_identifiers == quasi_identifiers
        assert_frame_equal(actual_partitions[0], expected_left_df)
        assert_frame_equal(actual_partitions[1], expected_right_df)

    def test_equality_blocked_partition(self, middle_df):
        test_mp = MondrianPartitioner(1)
        quasi_identifiers = ['middle_values']
        test_sf = SensitiveFrame(middle_df, quasi_identifiers=quasi_identifiers)
        actual_partitions = test_mp.partition_sf(test_sf)

        expected_left_df = DataFrame.from_dict({0: 1, 1: 1}, orient='index')
        expected_left_df.columns = ['middle_values']
        expected_middle_df = DataFrame.from_dict({2: 3}, orient='index')
        expected_middle_df.columns = ['middle_values']
        expected_right_df = DataFrame.from_dict({3: 5, 4: 5}, orient='index')
        expected_right_df.columns = ['middle_values']

        assert len(actual_partitions) == 3
        assert actual_partitions[0].quasi_identifiers == quasi_identifiers
        assert actual_partitions[1].quasi_identifiers == quasi_identifiers
        assert actual_partitions[2].quasi_identifiers == quasi_identifiers
        assert_frame_equal(actual_partitions[0], expected_left_df)
        assert_frame_equal(actual_partitions[1], expected_middle_df)
        assert_frame_equal(actual_partitions[2], expected_right_df)

    def test_partition_along_widest_quasi_identifiers(self, sample_df):
        test_mp = MondrianPartitioner(2)
        quasi_identifiers = ['slim_values', 'wide_values']
        test_sf = SensitiveFrame(sample_df, quasi_identifiers=quasi_identifiers)
        actual_partitions = test_mp.partition_sf(test_sf)

        expected_left_df = DataFrame.from_dict({0: [1, 1], 1: [1, 1]}, orient='index')
        expected_left_df.columns = quasi_identifiers
        expected_right_df = DataFrame.from_dict({2: [1, 1000], 3: [2, 1000], 4: [2, 1000]}, orient='index')
        expected_right_df.columns = quasi_identifiers

        assert len(actual_partitions) == 2
        assert sorted(actual_partitions[0].quasi_identifiers) == quasi_identifiers
        assert sorted(actual_partitions[1].quasi_identifiers) == quasi_identifiers
        assert_frame_equal(actual_partitions[0], expected_left_df)
        assert_frame_equal(actual_partitions[1], expected_right_df)

    def test_apply(self, sample_df):
        test_mp = MondrianPartitioner(2)
        quasi_identifiers = ['slim_values', 'wide_values']
        test_sf = SensitiveFrame(sample_df, quasi_identifiers=quasi_identifiers)
        actual_partitions = test_mp.apply([test_sf])

        expected_left_df = DataFrame.from_dict({0: [1, 1], 1: [1, 1]}, orient='index')
        expected_left_df.columns = quasi_identifiers
        expected_right_df = DataFrame.from_dict({2: [1, 1000], 3: [2, 1000], 4: [2, 1000]}, orient='index')
        expected_right_df.columns = quasi_identifiers

        assert test_mp.discernability == 13
        assert len(actual_partitions) == 2
        assert sorted(actual_partitions[0].quasi_identifiers) == quasi_identifiers
        assert sorted(actual_partitions[1].quasi_identifiers) == quasi_identifiers
        assert_frame_equal(actual_partitions[0], expected_left_df)
        assert_frame_equal(actual_partitions[1], expected_right_df)
