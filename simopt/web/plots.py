"""Plot configuration models for the SimOpt web API."""

from typing import Annotated, Optional

from pydantic import BaseModel, Field

from simopt.experiment_base import PlotType


class PlotProgressCurvesConfig(BaseModel):
    """Options for experiment_base.plot_progress_curves (excluding `experiments`)."""

    plot_type: Annotated[
        PlotType,
        Field(
            default=PlotType.ALL,
            description="Type of plot to produce (ALL, MEAN, or QUANTILE).",
        ),
    ]

    beta: Annotated[
        float,
        Field(
            default=0.50,
            description=(
                "Quantile level to plot (0 < beta < 1). Used for QUANTILE plots."
            ),
            gt=0.0,
            lt=1.0,
        ),
    ]

    normalize: Annotated[
        bool,
        Field(
            default=True,
            description="If True, normalize curves by optimality gaps.",
        ),
    ]

    all_in_one: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot all curves in one figure.",
        ),
    ]

    n_bootstraps: Annotated[
        int,
        Field(
            default=100,
            description="Number of bootstrap samples.",
            ge=1,
        ),
    ]

    conf_level: Annotated[
        float,
        Field(
            default=0.95,
            description="Confidence level for CIs (0 < conf_level < 1).",
            gt=0.0,
            lt=1.0,
        ),
    ]

    plot_conf_ints: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot bootstrapped confidence intervals.",
        ),
    ]

    print_max_hw: Annotated[
        bool,
        Field(
            default=True,
            description="If True, print caption with max half-width.",
        ),
    ]

    plot_title: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Custom plot title (used only if all_in_one=True).",
        ),
    ]

    legend_loc: Annotated[
        Optional[str],
        Field(
            default=None,
            description='Legend location (e.g., "best", "lower right").',
        ),
    ]

    ext: Annotated[
        str,
        Field(
            default=".png",
            description='File extension for saved plots (e.g., ".png").',
        ),
    ]

    save_as_pickle: Annotated[
        bool,
        Field(
            default=False,
            description="If True, also save a pickle of the plot.",
        ),
    ]

    solver_set_name: Annotated[
        str,
        Field(
            default="SOLVER_SET",
            description="Label for solver group in plot titles.",
            min_length=1,
        ),
    ]


class PlotSolvabilityCDFConfig(BaseModel):
    """Options for experiment_base.plot_progress_curves (excluding `experiments`)."""

    solve_tol: Annotated[
        float,
        Field(
            default=0.1,
            description=(
                "Optimality gap that defines when a problem is considered solved"
            ),
            gt=0.0,
            le=1.0,
        ),
    ]

    all_in_one: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot all curves in one figure.",
        ),
    ]

    n_bootstraps: Annotated[
        int,
        Field(
            default=100,
            description="Number of bootstrap samples.",
            ge=1,
        ),
    ]

    conf_level: Annotated[
        float,
        Field(
            default=0.95,
            description="Confidence level for CIs (0 < conf_level < 1).",
            gt=0.0,
            lt=1.0,
        ),
    ]

    plot_conf_ints: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot bootstrapped confidence intervals.",
        ),
    ]

    print_max_hw: Annotated[
        bool,
        Field(
            default=True,
            description="If True, print caption with max half-width.",
        ),
    ]

    plot_title: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Custom plot title (used only if all_in_one=True).",
        ),
    ]

    legend_loc: Annotated[
        Optional[str],
        Field(
            default=None,
            description='Legend location (e.g., "best", "lower right").',
        ),
    ]

    ext: Annotated[
        str,
        Field(
            default=".png",
            description='File extension for saved plots (e.g., ".png").',
        ),
    ]

    save_as_pickle: Annotated[
        bool,
        Field(
            default=False,
            description="If True, also save a pickle of the plot.",
        ),
    ]

    solver_set_name: Annotated[
        str,
        Field(
            default="SOLVER_SET",
            description="Label for solver group in plot titles.",
            min_length=1,
        ),
    ]


class PlotAreaScatterplotsConfig(BaseModel):
    """Options for experiment_base.plot_area_scatterplots (excluding `experiments`)."""

    all_in_one: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot all curves in one figure.",
        ),
    ]

    n_bootstraps: Annotated[
        int,
        Field(
            default=100,
            description="Number of bootstrap samples.",
            ge=1,
        ),
    ]

    conf_level: Annotated[
        float,
        Field(
            default=0.95,
            description="Confidence level for CIs (0 < conf_level < 1).",
            gt=0.0,
            lt=1.0,
        ),
    ]

    plot_conf_ints: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot bootstrapped confidence intervals.",
        ),
    ]

    print_max_hw: Annotated[
        bool,
        Field(
            default=True,
            description="If True, print caption with max half-width.",
        ),
    ]

    plot_title: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Custom plot title (used only if all_in_one=True).",
        ),
    ]

    legend_loc: Annotated[
        str,
        Field(
            default="best",
            description='Legend location (e.g., "best", "lower right").',
        ),
    ]

    ext: Annotated[
        str,
        Field(
            default=".png",
            description='File extension for saved plots (e.g., ".png").',
        ),
    ]

    save_as_pickle: Annotated[
        bool,
        Field(
            default=False,
            description="If True, also save a pickle of the plot.",
        ),
    ]

    solver_set_name: Annotated[
        str,
        Field(
            default="SOLVER_SET",
            description="Label for solver group in plot titles.",
            min_length=1,
        ),
    ]

    problem_set_name: Annotated[
        str,
        Field(
            default="PROBLEM_SET",
            description="Label for problem group in plot titles.",
            min_length=1,
        ),
    ]


class PlotSolvabilityProfilesConfig(BaseModel):
    """Options for experiment_base.plot_solvability_profiles."""

    all_in_one: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot all curves in one figure.",
        ),
    ]

    n_bootstraps: Annotated[
        int,
        Field(
            default=100,
            description="Number of bootstrap samples.",
            ge=1,
        ),
    ]

    conf_level: Annotated[
        float,
        Field(
            default=0.95,
            description="Confidence level for CIs (0 < conf_level < 1).",
            gt=0.0,
            lt=1.0,
        ),
    ]

    plot_conf_ints: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot bootstrapped confidence intervals.",
        ),
    ]

    print_max_hw: Annotated[
        bool,
        Field(
            default=True,
            description="If True, print caption with max half-width.",
        ),
    ]

    solve_tol: Annotated[
        float,
        Field(
            default=0.1,
            description=(
                "Optimality gap that defines when a problem is considered solved"
            ),
            gt=0.0,
            le=1.0,
        ),
    ]

    beta: Annotated[
        float,
        Field(
            default=0.5,
            description="Quantile level for quantile solvability (0 < beta < 1).",
            gt=0.0,
            lt=1.0,
        ),
    ]

    ref_solver: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Reference solver for difference plots.",
        ),
    ]

    plot_title: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Custom plot title (used only if all_in_one=True).",
        ),
    ]

    legend_loc: Annotated[
        Optional[str],
        Field(
            default=None,
            description='Legend location (e.g., "best", "lower right").',
        ),
    ]

    ext: Annotated[
        str,
        Field(
            default=".png",
            description='File extension for saved plots (e.g., ".png").',
        ),
    ]

    save_as_pickle: Annotated[
        bool,
        Field(
            default=False,
            description="If True, also save a pickle of the plot.",
        ),
    ]

    solver_set_name: Annotated[
        str,
        Field(
            default="SOLVER_SET",
            description="Label for solver group in plot titles.",
            min_length=1,
        ),
    ]

    problem_set_name: Annotated[
        str,
        Field(
            default="PROBLEM_SET",
            description="Label for problem group in plot titles.",
            min_length=1,
        ),
    ]


class PlotTerminalProgressCurvesConfig(BaseModel):
    """Options for experiment_base.plot_progress_curves (excluding `experiments`)."""

    plot_type: Annotated[
        PlotType,
        Field(
            default=PlotType.VIOLIN,
            description="Type of plot to produce.",
        ),
    ]

    normalize: Annotated[
        bool,
        Field(
            default=True,
            description="If True, normalize curves by optimality gaps.",
        ),
    ]

    all_in_one: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot all curves in one figure.",
        ),
    ]

    plot_title: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Custom plot title (used only if all_in_one=True).",
        ),
    ]

    ext: Annotated[
        str,
        Field(
            default=".png",
            description='File extension for saved plots (e.g., ".png").',
        ),
    ]

    save_as_pickle: Annotated[
        bool,
        Field(
            default=False,
            description="If True, also save a pickle of the plot.",
        ),
    ]

    solver_set_name: Annotated[
        str,
        Field(
            default="SOLVER_SET",
            description="Label for solver group in plot titles.",
            min_length=1,
        ),
    ]


class PlotTerminalScatterplotsConfig(BaseModel):
    """Options for experiment_base.plot_progress_curves (excluding `experiments`)."""

    plot_type: Annotated[
        PlotType,
        Field(
            default=PlotType.TERMINAL_SCATTER,
            description="Type of plot to produce (TERMINAL_SCATTER).",
        ),
    ]

    all_in_one: Annotated[
        bool,
        Field(
            default=True,
            description="If True, plot all curves in one figure.",
        ),
    ]

    plot_title: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Custom plot title (used only if all_in_one=True).",
        ),
    ]

    legend_loc: Annotated[
        Optional[str],
        Field(
            default=None,
            description='Legend location (e.g., "best", "lower right").',
        ),
    ]

    ext: Annotated[
        str,
        Field(
            default=".png",
            description='File extension for saved plots (e.g., ".png").',
        ),
    ]

    save_as_pickle: Annotated[
        bool,
        Field(
            default=False,
            description="If True, also save a pickle of the plot.",
        ),
    ]

    solver_set_name: Annotated[
        str,
        Field(
            default="SOLVER_SET",
            description="Label for solver group in plot titles.",
            min_length=1,
        ),
    ]

    problem_set_name: Annotated[
        str,
        Field(
            default="PROBLEM_SET",
            description="Label for problem group in plot titles.",
            min_length=1,
        ),
    ]
