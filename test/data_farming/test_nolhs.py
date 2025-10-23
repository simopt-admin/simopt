"""Test module for NOLHS."""

import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from simopt.data_farming.nolhs import NOLHS

STACK_17 = [
    [6, 2, 3, 4, 13, 17, 11, 10, 9, 12, 16, 15, 14, 5, 1, 7, 8],
    [17, 5, 8, 11, 16, 6, 4, 15, 9, 1, 13, 10, 7, 2, 12, 14, 3],
    [14, 15, 2, 6, 8, 7, 17, 13, 9, 4, 3, 16, 12, 10, 11, 1, 5],
    [7, 10, 5, 17, 3, 14, 6, 16, 9, 11, 8, 13, 1, 15, 4, 12, 2],
    [5, 1, 11, 10, 6, 2, 15, 14, 9, 13, 17, 7, 8, 12, 16, 3, 4],
    [16, 6, 14, 3, 1, 13, 8, 11, 9, 2, 12, 4, 15, 17, 5, 10, 7],
    [10, 11, 17, 13, 14, 15, 16, 12, 9, 8, 7, 1, 5, 4, 3, 2, 6],
]


class TestNOLHS(unittest.TestCase):
    """Test class for NOLHS."""

    def test_init_tuple(self) -> None:
        """Test the standard initialization of NOLHS with tuple designs."""
        designs = [(0.0, 10.0, 2), (5.0, 15.0, 1)]
        num_stacks = 1
        nolhs = NOLHS(designs=designs, num_stacks=num_stacks)
        self.assertEqual(nolhs.num_stacks, num_stacks)
        self.assertEqual(nolhs._design_size, len(designs))
        self.assertEqual(nolhs._nolhs_size, 17)
        self.assertEqual(len(nolhs._scalers), len(designs))

    def test_init_path(self) -> None:
        """Test the initialization of NOLHS with a file path."""
        # Create a temporary design file for testing
        with patch(
            "pathlib.Path.open",
            mock_open(read_data="0.0,10.0,2\n5.0,15.0,1\n20.0,30.0,0\n"),
        ):
            nolhs = NOLHS(designs=Path("temp_design_file.txt"), num_stacks=2)
            self.assertEqual(nolhs.num_stacks, 2)
            self.assertEqual(nolhs._design_size, 3)
            self.assertEqual(nolhs._nolhs_size, 17)
            self.assertEqual(len(nolhs._scalers), 3)

    def test_init_empty(self) -> None:
        """Test the initialization of NOLHS with empty designs."""
        designs = []
        nolhs = NOLHS(designs=designs)
        self.assertEqual(nolhs.num_stacks, 1)
        self.assertEqual(nolhs._design_size, 0)
        self.assertEqual(nolhs._nolhs_size, 1)
        self.assertEqual(len(nolhs._scalers), 0)

    def test_init_invalid_num_stacks(self) -> None:
        """Test initialization of NOLHS with invalid number of stacks."""
        designs = [(0.0, 10.0, 2)]
        with self.assertRaises(ValueError):
            NOLHS(designs=designs, num_stacks=0)
        with self.assertRaises(ValueError):
            NOLHS(designs=designs, num_stacks=18)

    def test_set_num_stacks(self) -> None:
        """Test setting the number of stacks."""
        designs = [(0.0, 10.0, 2)]
        nolhs = NOLHS(designs=designs)
        nolhs.num_stacks = 5
        self.assertEqual(nolhs.num_stacks, 5)
        with self.assertRaises(ValueError):
            nolhs.num_stacks = 0
        with self.assertRaises(ValueError):
            nolhs.num_stacks = 18

    def test_generate_design(self) -> None:
        """Test the generate_design method with multiple stacks."""
        # Constants

        stack_len = len(STACK_17[0])
        midrange = stack_len // 2
        scaling = 100.0 / (stack_len - 1)
        designs = [(0.0, 100.0, 3)]

        # Loop vars
        expected_stack_vals = []

        # Loop from 1 to 7 stacks
        for num_stacks in range(1, 8):
            # The index of the stack being added in this iteration
            stack_idx = num_stacks - 1
            current_stack = STACK_17[stack_idx]

            # Incrementally build the expected values list
            if stack_idx == 0:
                # Add all values from the first stack
                expected_stack_vals.extend(current_stack)
            else:
                # Add values from subsequent stacks, skipping midrange
                expected_stack_vals.extend(
                    val
                    for val_idx, val in enumerate(current_stack)
                    if val_idx != midrange
                )

            # Generate the design
            nolhs = NOLHS(designs=designs, num_stacks=num_stacks)
            generated_designs = nolhs.generate_design()

            # Check dimensions
            self.assertEqual(nolhs._design_size, 1)
            self.assertEqual(len(generated_designs), len(expected_stack_vals))

            # Calculate expected scaled values from the *total* list
            expected_values = [(val - 1) * scaling for val in expected_stack_vals]

            # Extract actual values
            actual_values = [design[0] for design in generated_designs]

            # Compare actual and expected values
            self.assertEqual(actual_values, expected_values)

    def test_generate_design_empty(self) -> None:
        """Test the generate_design method with empty designs."""
        nolhs = NOLHS(designs=[])
        generated_designs = nolhs.generate_design()
        self.assertEqual(generated_designs, [])

    def test_generate_design_rounding(self) -> None:
        """Test the generate_design method with rounding precision."""
        stack = STACK_17[0]
        for num_decs in range(5):  # 0 to 4 decimal places
            designs = [(0.0, 1.0, num_decs)]
            nolhs = NOLHS(designs=designs, num_stacks=1)
            generated_designs = nolhs.generate_design()
            for i, design in enumerate(generated_designs):
                # Make sure value is correctly rounded
                expected_value = round((stack[i] - 1) / 16, num_decs)
                self.assertEqual(design[0], expected_value)
                # Make sure value has correct number of decimal places
                decimal_part = (
                    str(design[0]).split(".")[1] if "." in str(design[0]) else ""
                )
                self.assertLessEqual(len(decimal_part), num_decs)
