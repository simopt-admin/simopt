"""Plot types."""

from enum import Enum
from typing import Self


class PlotType(Enum):
    """Enum class for different types of plots and metrics."""

    ALL = "all"
    MEAN = "mean"
    QUANTILE = "quantile"
    AREA_MEAN = "area_mean"
    AREA_STD_DEV = "area_std_dev"
    SOLVE_TIME_QUANTILE = "solve_time_quantile"
    SOLVE_TIME_CDF = "solve_time_cdf"
    CDF_SOLVABILITY = "cdf_solvability"
    QUANTILE_SOLVABILITY = "quantile_solvability"
    DIFFERENCE_OF_CDF_SOLVABILITY = "difference_of_cdf_solvability"
    DIFFERENCE_OF_QUANTILE_SOLVABILITY = "difference_of_quantile_solvability"
    AREA = "area"
    BOX = "box"
    VIOLIN = "violin"
    TERMINAL_SCATTER = "terminal_scatter"
    FEASIBILITY_SCATTER = "feasibility_scatter"
    FEASIBILITY_VIOLIN = "feasibility_violin"
    ALL_FEASIBILITY_PROGRESS = "all_feasibility_progress"
    MEAN_FEASIBILITY_PROGRESS = "mean_feasibility_progress"
    QUANTILE_FEASIBILITY_PROGRESS = "quantile_feasibility_progress"

    @staticmethod
    def from_str(label: str) -> Self:
        """Converts a string label to a PlotType enum."""
        # Reverse mapping from string to PlotType enum.
        name = label.lower().replace(" ", "_")
        inv_plot_type = {pt.value: pt for pt in PlotType}
        if name in inv_plot_type:
            return inv_plot_type[name]
        error_msg = (
            f"Unknown plot type: {label} ({name}). "
            f"Must be one of {[pt.value for pt in PlotType]}."
        )
        raise ValueError(error_msg)
