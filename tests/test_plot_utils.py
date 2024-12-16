import unittest
from ov_ts_profiler.plot_utils import divide_into_segments_by_y_values

class TestDivideIntoSegmentsByYValues(unittest.TestCase):
    def test_divide_segments_correctly(self):
        x_values = [1, 2, 3, 4, 5]
        y_values = [10, 20, 30, 40, 50]
        n_segments = 2
        segments = list(divide_into_segments_by_y_values(x_values, y_values, n_segments))
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], ([1, 2, 3, 4], [10, 20, 30, 40]))
        self.assertEqual(segments[1], ([5], [50]))

    def test_handles_empty_input(self):
        x_values = []
        y_values = []
        n_segments = 2
        segments = list(divide_into_segments_by_y_values(x_values, y_values, n_segments))
        self.assertEqual(len(segments), 0)

    def test_handles_single_segment(self):
        x_values = [1, 2, 3]
        y_values = [10, 20, 30]
        n_segments = 1
        segments = list(divide_into_segments_by_y_values(x_values, y_values, n_segments))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], ([1, 2, 3], [10, 20, 30]))

    def test_handles_identical_y_values(self):
        x_values = [1, 2, 3]
        y_values = [10, 10, 10]
        n_segments = 2
        segments = list(divide_into_segments_by_y_values(x_values, y_values, n_segments))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], ([1, 2, 3], [10, 10, 10]))

    def test_handles_large_number_of_segments(self):
        x_values = [1, 2, 3, 4, 5]
        y_values = [10, 20, 30, 40, 50]
        n_segments = 10
        segments = list(divide_into_segments_by_y_values(x_values, y_values, n_segments))
        self.assertEqual(len(segments), 5)
        self.assertEqual(segments[0], ([1], [10]))
        self.assertEqual(segments[1], ([2], [20]))
        self.assertEqual(segments[2], ([3], [30]))
        self.assertEqual(segments[3], ([4], [40]))
        self.assertEqual(segments[4], ([5], [50]))

    def test_divide_into_segments_by_y_values_negative_y_values(self):
        x_values = [1, 2, 3, 4, 5]
        y_values = [-10, -20, -30, -40, -50]
        n_segments = 2
        segments = list(divide_into_segments_by_y_values(x_values, y_values, n_segments))
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(len(segment[0]) > 0 for segment in segments))
        self.assertTrue(all(len(segment[1]) > 0 for segment in segments))

    def test_divide_into_segments_by_y_values_zero_y_values(self):
        x_values = [1, 2, 3, 4, 5]
        y_values = [0, 0, 0, 0, 0]
        n_segments = 2
        segments = list(divide_into_segments_by_y_values(x_values, y_values, n_segments))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], (x_values, y_values))

if __name__ == '__main__':
    unittest.main()
