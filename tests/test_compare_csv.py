import unittest
import os
from typing import List, Dict

import compare_csv

TEST1_CSV_PATH = os.path.join('.', 'tests', 'csv', 'test1.csv')
TEST2_CSV_PATH = os.path.join('.', 'tests', 'csv', 'test2.csv')

class TestCompareCompileTime(unittest.TestCase):
    def test(self):
        csv_data = compare_csv.get_csv_data([TEST1_CSV_PATH, TEST2_CSV_PATH])
        header, table = compare_csv.compare_compile_time(csv_data)
        expected_header = ['framework',
         'name',
         'precision',
         'optional model attribute',
         'compile time #1 (secs)',
         'compile time #2 (secs)',
         'compile time #2 - #1 (secs)',
         'compile time #2/#1']
        self.assertEqual(header, expected_header)
        print(table)
        expected_table = [{'framework': 'framework_name1',
                           'name': 'model_name_4',
                           'precision': 'FP16',
                           'optional model attribute': 'OV_FP16-4BIT_1',
                           'compile time #1 (secs)': 60.0100000001,
                           'compile time #2 (secs)': 80.0100000001,
                           'compile time #2 - #1 (secs)': 20.000000000000007,
                           'compile time #2/#1': 1.3332777870349388},
                          {'framework': 'framework_name1',
                           'name': 'model_name_1',
                           'precision': 'FP16',
                           'optional model attribute': 'OV_FP16-4BIT_DEFAULT',
                           'compile time #1 (secs)': 20.0100000001,
                           'compile time #2 (secs)': 30.0010000001,
                           'compile time #2 - #1 (secs)': 9.991,
                           'compile time #2/#1': 1.499300349822592},
                          {'framework': 'framework_name1',
                           'name': 'model_name_3',
                           'precision': 'FP16',
                           'optional model attribute': 'OV_FP16-4BIT_DEFAULT',
                           'compile time #1 (secs)': 'N/A',
                           'compile time #2 (secs)': 50.0010000001,
                           'compile time #2 - #1 (secs)': 'N/A',
                           'compile time #2/#1': 'N/A'},
                          {'framework': 'framework_name1',
                           'name': 'model_name_2',
                           'precision': 'FP16',
                           'optional model attribute': 'OV_FP16-4BIT_1',
                           'compile time #1 (secs)': 40.0100000001,
                           'compile time #2 (secs)': 'N/A',
                           'compile time #2 - #1 (secs)': 'N/A',
                           'compile time #2/#1': 'N/A'}
                          ]
        self.assertEqual(table, expected_table)

if __name__ == "__main__":
    unittest.main()
