import pytest
import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas import Series
from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal

from wreckognize.sensitive_dataframe import SensitiveFrame
from wreckognize.sensitive_dataframe import SensitiveSeries


@pytest.fixture
def sample_df():
    return DataFrame(
        [
            ['crusher', 'commander', 'woman', 'human'],
            ['worf', 'lieutenant', 'man', 'klingon']
        ],
        columns=['name', 'rank', 'gender', 'species']
    )

@pytest.fixture
def sample_sf():
    return SensitiveFrame(
        [
            ['riker', 'commander', 'man', 'human'],
            ['ro', 'ensign', 'woman', 'bejoran']
        ],
        columns=['name', 'rank', 'gender', 'species'],
        quasi_identifiers=['rank', 'gender'],
        sensitive_data=['species']
    )

@pytest.fixture
def sample_sf_two():
    return SensitiveFrame(
        [['starfleet', 2335], ['starfleet', 2340]],
        columns=['organization', 'born'],
        quasi_identifiers=['born'],
        sensitive_data=['born']
    )

@pytest.fixture
def sample_sf_three():
    return SensitiveFrame(
        [['william', 'red'], ['laren', 'red']],
        columns=['given_name', 'uniform'],
        quasi_identifiers=['uniform'],
        sensitive_data=['given_name']
    )

@pytest.fixture
def sample_right_sf():
    return SensitiveFrame(
        [['riker', 'starfleet', 2335],['ro', 'starfleet', 2340]],
        columns=['name', 'organization', 'born'],
        quasi_identifiers=['name', 'born'],
        sensitive_data=['born']
    )


class TestSensitiveSeries:
    
    def test_init_sets_boolean_metadata(self):
        test_ss = SensitiveSeries(name='a', is_quasi_identifier=True)
        assert test_ss.is_quasi_identifier
        assert not test_ss.is_sensitive_data

    def test_metadata_properties(self):
        test_ss = SensitiveSeries(name='a', is_quasi_identifier=True)
        assert test_ss.quasi_identifiers == ['a']
        assert test_ss.sensitive_data == []


class TestSensitiveFrame:

    def test_init_sets_quasi_identifiers(self):
        test_sf = SensitiveFrame(
            columns=['a', 'b'],
            quasi_identifiers=['a', 'b']
        )
        assert test_sf.quasi_identifiers == ['a', 'b']

    def test_init_sets_sensitive_data(self):
        test_sf = SensitiveFrame(
            columns=['c', 'd'],
            sensitive_data=['c', 'd']
        )
        assert test_sf.sensitive_data == ['c', 'd']

    def test_init_sets_multiple_metadata(self):
        test_sf = SensitiveFrame(
            columns=['a', 'b'],
            sensitive_data=['a'],
            quasi_identifiers=['b']
        )
        assert test_sf.sensitive_data == ['a']
        assert test_sf.quasi_identifiers == ['b']

    def test_init_selects_metadata(self):
        test_sf = SensitiveFrame(
            columns=['a'], 
            quasi_identifiers=['a', 'b']
        )
        assert test_sf.quasi_identifiers == ['a']

    def test_init_with_df_keeps_data(self, sample_df):
        test_sf = SensitiveFrame(sample_df)
        expected_df = sample_df
        assert_frame_equal(test_sf, expected_df)
        
    def test_init_cleans_sf_metadata(self, sample_sf):
        test_sf = SensitiveFrame(sample_sf)
        assert test_sf.quasi_identifiers == []
        assert test_sf.sensitive_data == []

    def test_simple_join(self, sample_sf, sample_sf_two):
        test_sf = sample_sf.join(sample_sf_two)
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human', 'starfleet', 2335],
                ['ro', 'ensign', 'woman', 'bejoran', 'starfleet', 2340]
            ],
            columns=['name', 'rank', 'gender', 'species', 'organization', 'born']
        )
        assert test_sf.quasi_identifiers == ['rank', 'gender', 'born']
        assert test_sf.sensitive_data == ['species', 'born']
        assert_frame_equal(test_sf, expected_df)

    def test_join_list(self, sample_sf, sample_sf_two, sample_sf_three):
        test_sf = sample_sf.join([sample_sf_two, sample_sf_three])
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human', 'starfleet', 2335, 'william', 'red'],
                ['ro', 'ensign', 'woman', 'bejoran', 'starfleet', 2340, 'laren', 'red']
            ],
            columns=['name', 'rank', 'gender', 'species', 'organization', 'born', 'given_name', 'uniform']
        )
        assert test_sf.quasi_identifiers == ['rank', 'gender', 'born', 'uniform']
        assert test_sf.sensitive_data == ['species','born','given_name']
        assert_frame_equal(test_sf, expected_df)

    def test_mixed_join(self, sample_sf_three, sample_df, sample_sf_two):
        test_sf = sample_sf_three.join([sample_df, sample_sf_two])
        expected_df = DataFrame(
            [
                ['william', 'red', 'crusher', 'commander', 'woman', 'human', 'starfleet', 2335],
                ['laren', 'red', 'worf', 'lieutenant', 'man', 'klingon', 'starfleet', 2340]
            ],
            columns=['given_name', 'uniform', 'name', 'rank', 'gender', 'species', 'organization', 'born']
        )
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['uniform', 'born']
        assert test_sf.sensitive_data == ['given_name', 'born']

    def test_join_non_metadata(self):
        non_sf = SensitiveFrame(columns=['empty'])
        test_sf = SensitiveFrame(columns=['void']).join(non_sf)
        assert test_sf.quasi_identifiers == []
        assert test_sf.sensitive_data == []

    def test_rename_columns(self, sample_sf_three):
        test_sf = sample_sf_three.rename(columns={'given_name': 'first_name', 'uniform': 'specialty'})
        assert test_sf.sensitive_data == ['first_name']
        assert test_sf.quasi_identifiers == ['specialty']

    def test_rename_inplace(self, sample_sf_three):
        sample_sf_three.rename(columns={'given_name': 'first_name'}, inplace=True)
        assert sample_sf_three.columns.tolist() == ['first_name', 'uniform']
        assert sample_sf_three.sensitive_data == ['first_name']

    def test_rename_non_metadata(self):
        non_sf = SensitiveFrame(columns=['empty'])
        test_sf = non_sf.rename(columns={'empty': 'void'})
        assert test_sf.columns.tolist() == ['void']
        assert test_sf.quasi_identifiers == []
        assert test_sf.sensitive_data == []

    def test_rename_double_metadata(self):
        non_sf = SensitiveFrame(
            columns=['empty'],
            sensitive_data=['empty'],
            quasi_identifiers=['empty']
        )
        test_sf = non_sf.rename(columns={'empty': 'void'})
        assert test_sf.columns.tolist() == ['void']
        assert test_sf.quasi_identifiers == ['void']
        assert test_sf.sensitive_data == ['void']

    def test_simple_drop(self, sample_sf):
        test_sf = sample_sf.drop('rank', axis=1)
        assert test_sf.columns.tolist() == ['name', 'gender', 'species']
        assert test_sf.quasi_identifiers == ['gender']
        assert test_sf.sensitive_data == ['species']
        
    def test_drop_list(self, sample_sf):
        test_sf = sample_sf.drop(['rank','species'], axis=1)
        assert test_sf.columns.tolist() == ['name', 'gender']
        assert test_sf.quasi_identifiers == ['gender']
        assert test_sf.sensitive_data == []

    def test_drop_inplace(self, sample_sf):
        sample_sf.drop('rank', axis=1, inplace=True)
        assert sample_sf.columns.tolist() == ['name', 'gender', 'species']
        assert sample_sf.quasi_identifiers == ['gender']
        assert sample_sf.sensitive_data == ['species']

    def test_simple_get(self, sample_sf):
        test_sf = sample_sf['rank']
        expected_df = Series(['commander', 'ensign'], name='rank')
        assert_series_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['rank']

    def test_get_list(self, sample_sf):
        test_sf = sample_sf[['name', 'rank', 'species']]
        expected_df = DataFrame([['riker', 'commander', 'human'],['ro', 'ensign', 'bejoran']],
            columns=['name', 'rank', 'species']
        )
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['rank']
        assert test_sf.sensitive_data == ['species']

    def test_get_non_metadata(self, sample_sf):
        test_sf = sample_sf['name']
        expected_df = Series(['riker', 'ro'], name='name')
        assert_series_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == []

    def test_get_double_metadata(self, sample_sf_two):
        test_sf = sample_sf_two['born']
        assert test_sf.quasi_identifiers == ['born']
        assert test_sf.sensitive_data == ['born']

    def test_get_and_set_metadata(self, sample_sf):
        test_sf = SensitiveFrame()
        test_sf['grade'] = sample_sf['rank']
        expected_df = DataFrame(['commander', 'ensign'], columns=['grade'])
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['grade']

    def test_get_and_set_non_metadata(self, sample_sf):
        test_sf = SensitiveFrame()
        test_sf['person'] = sample_sf['name']
        expected_df = DataFrame(['riker', 'ro'], columns=['person'])
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.sensitive_data == []        

    def test_get_and_set_from_df(self, sample_df):
        test_sf = SensitiveFrame()
        test_sf['grade'] = sample_df['rank']
        expected_df = DataFrame(['commander', 'lieutenant'], columns=['grade'])
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == []

    def test_get_and_set_double_metadata(self, sample_sf_two):
        test_sf = SensitiveFrame()
        test_sf['birth_year'] = sample_sf_two['born']
        expected_df = DataFrame([2335, 2340], columns=['birth_year'])
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['birth_year']
        assert test_sf.sensitive_data == ['birth_year']

    def test_vertical_concat(self, sample_sf):
        test_sf = pd.concat([sample_sf, sample_sf]).reset_index(drop=True)
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human'],
                ['ro', 'ensign', 'woman', 'bejoran'],
                ['riker', 'commander', 'man', 'human'],
                ['ro', 'ensign', 'woman', 'bejoran']
            ],
            columns=['name', 'rank', 'gender', 'species']
        )
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['rank', 'gender']
        assert test_sf.sensitive_data == ['species']

    def test_horizontal_concat(self, sample_sf, sample_sf_two):
        test_sf = pd.concat([sample_sf, sample_sf_two], axis=1).reset_index(drop=True)
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human', 'starfleet', 2335],
                ['ro', 'ensign', 'woman', 'bejoran', 'starfleet', 2340],
            ],
            columns=['name', 'rank', 'gender', 'species', 'organization', 'born']
        )
        assert_frame_equal(test_sf, expected_df)        
        assert test_sf.quasi_identifiers == ['rank', 'gender', 'born']
        assert test_sf.sensitive_data == ['species', 'born']

    def test_diagonal_concat(self, sample_sf, sample_sf_two):
        test_sf = pd.concat([sample_sf, sample_sf_two]).reset_index(drop=True)
        expected_df = DataFrame(
            [
                [np.nan, 'man', 'riker', np.nan, 'commander', 'human'],
                [np.nan, 'woman', 'ro', np.nan, 'ensign',  'bejoran'],
                [2335, np.nan, np.nan, 'starfleet', np.nan, np.nan],
                [2340, np.nan, np.nan, 'starfleet', np.nan, np.nan],
            ],
            columns=['born', 'gender', 'name', 'organization', 'rank', 'species']
        )
        assert_frame_equal(test_sf, expected_df)        
        assert test_sf.quasi_identifiers == ['born', 'gender', 'rank']
        assert test_sf.sensitive_data == ['born', 'species']

    def test_index_merge(self, sample_sf, sample_sf_two):
        test_sf = sample_sf.merge(sample_sf_two, left_index=True, right_index=True)
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human', 'starfleet', 2335],
                ['ro', 'ensign', 'woman', 'bejoran', 'starfleet', 2340],
            ],
            columns=['name', 'rank', 'gender', 'species', 'organization', 'born']
        )
        assert_frame_equal(test_sf, expected_df)        
        assert test_sf.quasi_identifiers == ['rank', 'gender', 'born']
        assert test_sf.sensitive_data == ['species', 'born']

    def test_on_merge(self, sample_sf, sample_right_sf):
        test_sf = sample_sf.merge(sample_right_sf, on='name')
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human', 'starfleet', 2335],
                ['ro', 'ensign', 'woman', 'bejoran', 'starfleet', 2340],
            ],
            columns=['name', 'rank', 'gender', 'species', 'organization', 'born']
        )
        assert_frame_equal(test_sf, expected_df)        
        assert test_sf.quasi_identifiers == ['rank', 'gender', 'born']
        assert test_sf.sensitive_data == ['species', 'born']

    def test_iloc_columns(self, sample_sf):
        test_sf = sample_sf.iloc[:,[0,1,3]]
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'human'],
                ['ro', 'ensign', 'bejoran'],
            ],
            columns=['name', 'rank', 'species']
        )
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['rank']
        assert test_sf.sensitive_data == ['species']
        
    def test_iloc_rows(self, sample_sf):
        test_sf = sample_sf.iloc[[0],:]
        expected_df = DataFrame(
            [['riker', 'commander', 'man', 'human']],
            columns=['name', 'rank', 'gender', 'species']
        )
        assert_frame_equal(test_sf, expected_df)        
        assert test_sf.quasi_identifiers == ['rank', 'gender']
        assert test_sf.sensitive_data == ['species']

    def test_loc_columns(self, sample_sf):
        test_sf = sample_sf.loc[:, ['name', 'gender', 'species']]
        expected_df = DataFrame(
            [
                ['riker', 'man', 'human'],
                ['ro', 'woman', 'bejoran']
            ],
            columns=['name', 'gender', 'species']
        )
        assert test_sf.quasi_identifiers == ['gender']
        assert test_sf.sensitive_data == ['species']

    def test_sort_values(self, sample_sf):
        test_sf = sample_sf.sort_values(by='species').reset_index(drop=True)
        expected_df = DataFrame(
            [
                ['ro', 'ensign', 'woman', 'bejoran'],
                ['riker', 'commander', 'man', 'human']
            ],
            columns=['name', 'rank', 'gender', 'species']
        )
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['rank', 'gender']
        assert test_sf.sensitive_data == ['species']

    def test_set_column_names(self, sample_sf):
        test_sf = sample_sf
        test_sf.columns = ['name', 'grade', 'sex', 'race']
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human'],
                ['ro', 'ensign', 'woman', 'bejoran']
            ],
            columns=['name', 'grade', 'sex', 'race']
        )
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['grade', 'sex']
        assert test_sf.sensitive_data == ['race']

    def test_append_df_onto_sf(self, sample_sf, sample_df):
        test_sf = sample_sf.append(sample_df, ignore_index=True)
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human'],
                ['ro', 'ensign', 'woman', 'bejoran'],
                ['crusher', 'commander', 'woman', 'human'],
                ['worf', 'lieutenant', 'man', 'klingon']
            ],
            columns=['name', 'rank', 'gender', 'species']
        )
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['rank', 'gender']
        assert test_sf.sensitive_data == ['species']

    def test_append_sf_onto_sf(self, sample_sf):
        test_sf = sample_sf.append(sample_sf, ignore_index=True)
        expected_df = DataFrame(
            [
                ['riker', 'commander', 'man', 'human'],
                ['ro', 'ensign', 'woman', 'bejoran'],
                ['riker', 'commander', 'man', 'human'],
                ['ro', 'ensign', 'woman', 'bejoran'],
            ],
            columns=['name', 'rank', 'gender', 'species']
        )
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['rank', 'gender']
        assert test_sf.sensitive_data == ['species']

    def test_get_numeric_data(self, sample_right_sf):
        test_sf = sample_right_sf._get_numeric_data()
        expected_df = DataFrame([[2335], [2340]], columns=['born'])
        assert_frame_equal(test_sf, expected_df)
        assert test_sf.quasi_identifiers == ['born']        
