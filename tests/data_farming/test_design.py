"""Test module for data-farming design generation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simopt.data_farming.design import DesignSpec, NumericRange, build_design
from simopt.data_farming.nolhs import NOLHS
from simopt.directory import solver_directory
from simopt.experiment_base import create_design

N_NOLHS_ROWS = len(NOLHS(designs=[(0.0, 1.0, 2), (1.0, 9.0, 0)]).generate_design())


class TestBuildDesign(unittest.TestCase):
    """Test class for build_design."""

    def test_solver_design(self) -> None:
        """Test varied, crossed, fixed, and default factors of a solver design."""
        spec = DesignSpec(
            kind="solver",
            name="ASTRODF",
            varied={
                "eta_1": NumericRange(min=0.05, max=0.2, decimals=2),
                "lambda_min": NumericRange(min=2, max=10),
            },
            crossed={"easy_solve": [True, False]},
            fixed={"gamma_1": 3.0},
        )
        design = build_design(spec)
        defaults = {k: v["default"] for k, v in solver_directory["ASTRODF"].specifications.items()}

        self.assertEqual(len(design), N_NOLHS_ROWS * 2)
        self.assertEqual({p["easy_solve"] for p in design}, {True, False})
        for point in design:
            self.assertEqual(set(point), set(defaults))
            self.assertIsInstance(point["eta_1"], float)
            self.assertGreaterEqual(point["eta_1"], 0.05)
            self.assertLessEqual(point["eta_1"], 0.2)
            self.assertIsInstance(point["lambda_min"], int)
            self.assertGreaterEqual(point["lambda_min"], 2)
            self.assertLessEqual(point["lambda_min"], 10)
            self.assertEqual(point["gamma_1"], 3.0)
            for name in ("eta_2", "gamma_2", "reuse_points", "use_gradients"):
                self.assertEqual(point[name], defaults[name])

    def test_problem_design(self) -> None:
        """Test a problem design that varies a model factor."""
        spec = DesignSpec(
            kind="problem",
            name="CNTNEWS-1",
            varied={"purchase_price": NumericRange(min=4, max=6, decimals=1)},
        )
        design = build_design(spec)
        self.assertGreater(len(design), 1)
        self.assertEqual(design[0]["budget"], 1000)
        self.assertEqual(design[0]["initial_solution"], [0])
        self.assertIn("salvage_price", design[0])
        self.assertTrue(all(4 <= p["purchase_price"] <= 6 for p in design))

    def test_model_design(self) -> None:
        """Test a model design."""
        spec = DesignSpec(
            kind="model",
            name="MM1",
            varied={"mu": NumericRange(min=2, max=4, decimals=1)},
            fixed={"people": 10},
        )
        design = build_design(spec)
        self.assertTrue(all(p["people"] == 10 and isinstance(p["people"], int) for p in design))
        self.assertTrue(all(2 <= p["mu"] <= 4 for p in design))
        self.assertEqual(design[0]["lambda"], 1.5)

    def test_no_varied_gives_single_row(self) -> None:
        """Test that an empty spec yields one row of defaults."""
        design = build_design(DesignSpec(kind="model", name="MM1"))
        self.assertEqual(len(design), 1)

    def test_invalid_specs(self) -> None:
        """Test that unknown classes, unknown factors, and repeated factors raise."""
        with self.assertRaises(ValueError):
            build_design(DesignSpec(kind="solver", name="NOPE"))
        with self.assertRaises(ValueError):
            build_design(DesignSpec(kind="solver", name="ASTRODF", fixed={"bogus": 1}))
        with self.assertRaises(ValueError):
            build_design(
                DesignSpec(
                    kind="solver",
                    name="ASTRODF",
                    varied={"eta_1": NumericRange(min=0.1, max=0.2, decimals=2)},
                    fixed={"eta_1": 0.1},
                )
            )
        with self.assertRaises(ValueError):
            build_design(
                DesignSpec(
                    kind="problem",
                    name="CNTNEWS-1",
                    varied={"initial_solution": NumericRange(min=0, max=1)},
                )
            )
        with self.assertRaises(ValueError):
            build_design(DesignSpec(kind="solver", name="ASTRODF", crossed={"eta_1": [True]}))

    def test_parity_with_create_design(self) -> None:
        """Test that build_design matches experiment_base.create_design."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("simopt.experiment_base.EXPERIMENT_DIR", Path(tmp_dir)),
        ):
            expected = create_design(
                name="ASTRODF",
                factor_headers=["eta_1", "lambda_min"],
                factor_settings=[(0.05, 0.2, 2), (2, 10, 0)],
                cross_design_factors={"easy_solve": [True, False]},
            )
        spec = DesignSpec(
            kind="solver",
            name="ASTRODF",
            varied={
                "eta_1": NumericRange(min=0.05, max=0.2, decimals=2),
                "lambda_min": NumericRange(min=2, max=10),
            },
            crossed={"easy_solve": [True, False]},
        )
        self.assertEqual(build_design(spec), expected)
