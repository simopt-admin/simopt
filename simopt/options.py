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
