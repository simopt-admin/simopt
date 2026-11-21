"""Global options for SimOpt package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CrnOptions:
    """Options for Common Random Numbers (CRN)."""

    across_budget: bool = True
    across_macroreps: bool = False
    across_x0_xstar: bool = True


DEFAULT_CRN_OPTIONS = CrnOptions()


@dataclass(frozen=True)
class ConfidenceIntervalOptions:
    """Options for computing confidence intervals via bootstrapping."""

    n_bootstraps: int = 100
    confidence_level: float = 0.95
    bias_correction: bool = True


DEFAULT_CONFIDENCE_INTERVAL_OPTIONS = ConfidenceIntervalOptions()


@dataclass(frozen=True)
class MeritOptions:
    """Options for merit = objective + obj_const * deterministic feasibility violation."""

    obj_const: float = 1e6
    feas_tol_upper: float = 1e-5
    feas_tol_lower: float = 1e-8


DEFAULT_MERIT_OPTIONS = MeritOptions()
