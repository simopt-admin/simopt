"""Test module for the data farming core."""

import unittest

from simopt.data_farming.data_farming_core import Scaler


class TestScaler(unittest.TestCase):
    """Test class for Scaler."""

    def test_init(self) -> None:
        """Test the Scaler class initialization."""
        min_val = 0
        max_val = 100
        precision = 2
        lh_max = 17
        scaler = Scaler(
            original_min=1,
            original_max=lh_max,
            scaled_min=min_val,
            scaled_max=max_val,
            precision=precision,
        )
        self.assertEqual(scaler.original_min, 1)
        self.assertEqual(scaler.original_max, lh_max)
        self.assertEqual(scaler.scaled_min, min_val)
        self.assertEqual(scaler.scaled_max, max_val)
        self.assertEqual(scaler.precision, precision)
        self.assertEqual(scaler.scale_factor, 10**precision)
        self.assertEqual(scaler.scale, (max_val - min_val) / (lh_max - 1))

    def test_scale_value(self) -> None:
        """Test the scale_value method."""
        lh_max = 17
        min_val = 0
        max_val = 100
        precision = 2
        scaler = Scaler(
            original_min=1,
            original_max=lh_max,
            scaled_min=min_val,
            scaled_max=max_val,
            precision=precision,
        )

        # Check Bounds
        with self.assertRaises(ValueError):
            scaler.scale_value(0)  # Below original_min
        self.assertEqual(scaler.scale_value(1), min_val)
        self.assertEqual(scaler.scale_value(lh_max), max_val)
        with self.assertRaises(ValueError):
            scaler.scale_value(lh_max + 1)  # Above original_max

        # Check intermediate values
        for i in range(1, lh_max + 1):
            increment = (max_val - min_val) / (lh_max - 1)  # 6.25 for lh_max=17
            calced_val = min_val + increment * (i - 1)
            rounded_val = round(calced_val, precision)
            self.assertEqual(scaler.scale_value(i), rounded_val)
