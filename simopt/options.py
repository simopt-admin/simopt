"""Global options for SimOpt package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CrnOptions:
    """Options for Common Random Numbers (CRN)."""

    across_budget: bool = True
    across_macroreps: bool = False
    across_x0_xstar: bool = True


DEFAULT_CRN_OPTIONS = CrnOptions()
