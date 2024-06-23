import unittest
import pandas as pd
from utils import DataFrameCollection

class TestDataFrameCollection(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        self.df2 = pd.DataFrame({
            'X': [7, 8, 9],
            'Y': [10, 11, 12]
        })
        self.collection = DataFrameCollection()

    def test_add_dataframe(self):
        self.collection.add_dataframe('first_df', self.df1)
        self.assertIn('first_df', self.collection.dataframes)
        pd.testing.assert_frame_equal(self.collection.get_dataframe('first_df'), self.df1)

    def test_add_invalid_dataframe(self):
        with self.assertRaises(ValueError):
            self.collection.add_dataframe('invalid_df', [1, 2, 3])

    def test_remove_dataframe(self):
        self.collection.add_dataframe('first_df', self.df1)
        self.collection.remove_dataframe('first_df')
        self.assertNotIn('first_df', self.collection.dataframes)

    def test_remove_nonexistent_dataframe(self):
        with self.assertRaises(KeyError):
            self.collection.remove_dataframe('nonexistent_df')

    def test_get_dataframe(self):
        self.collection.add_dataframe('first_df', self.df1)
        df = self.collection.get_dataframe('first_df')
        pd.testing.assert_frame_equal(df, self.df1)

    def test_get_nonexistent_dataframe(self):
        with self.assertRaises(KeyError):
            self.collection.get_dataframe('nonexistent_df')

    def test_list_dataframes(self):
        self.collection.add_dataframe('first_df', self.df1)
        self.collection.add_dataframe('second_df', self.df2)
        df_list = self.collection.list_dataframes()
        self.assertListEqual(df_list, ['first_df', 'second_df'])

    def test_repr(self):
        self.collection.add_dataframe('first_df', self.df1)
