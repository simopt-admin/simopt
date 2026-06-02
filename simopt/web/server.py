"""FastAPI server for the SimOpt web interface."""

# ruff: noqa: ANN001, ANN201, ANN202, D101, D103, E501

import threading

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import inspect
from pathlib import Path
from typing import Annotated, Any

import matplotlib.pyplot as plt
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from simopt import experiment_base as eb
from simopt.directory import (
    problem_directory,
    solver_directory,
)
from simopt.experiment_base import ProblemsSolvers
from simopt.web.plots import (
    PlotAreaScatterplotsConfig,
    PlotProgressCurvesConfig,
    PlotSolvabilityCDFConfig,
    PlotSolvabilityProfilesConfig,
    PlotTerminalProgressCurvesConfig,
    PlotTerminalScatterplotsConfig,
)


# ── Pydantic request models ──
# These define the expected shape of incoming JSON payloads for each endpoint.
class ProblemRequest(BaseModel):
    name: str
    rename: str | None = None
    fixed_factors: dict[str, Any]
    model_fixed_factors: dict[str, Any] = {}


class SolverRequest(BaseModel):
    name: str
    rename: str | None = None
    fixed_factors: dict[str, Any]


class PlotRequest(BaseModel):
    plot_type: str
    params: dict[str, Any] = {}


class ExperimentParams(BaseModel):
    num_macroreps: int
    num_postreps: int
    num_postnorms: int


class ExperimentRequest(BaseModel):
    experiment_params: ExperimentParams
    problems: list[ProblemRequest]
    solvers: list[SolverRequest]
    plots: list[PlotRequest]


# ── Path configuration ──
# WEB_DIR points to simopt/web; the built frontend is always served from there.
# BASE_DIR points to the repository root for existing result-file storage.
WEB_DIR = Path(__file__).resolve().parent
BASE_DIR = WEB_DIR.parent.parent
RESULTS_DIR = BASE_DIR / "simopt-web" / "results"
DIST_DIR = WEB_DIR / "dist"
STATIC_DIR = DIST_DIR / "assets"

# ── FastAPI app setup ──
app = FastAPI(title="SimOpt API")

# Allow all origins so the frontend (served on the same server) can make API calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POST_REPLICATE_DEFAULTS = getattr(eb, "POST_REPLICATE_DEFAULTS", {})
POST_NORMALIZE_DEFAULTS = getattr(eb, "POST_NORMALIZE_DEFAULTS", {})


# ── Static file serving ──
@app.get("/")
def serve_frontend():
    """Serve the main frontend HTML page."""
    return FileResponse(str(DIST_DIR / "index.html"))


@app.get("/results/{run_id}/experiment.log")
def serve_log(run_id: str):
    """Serve experiment log file for a given run.

    Reads the file fresh on each request to avoid Content-Length mismatches
    that occur when the static file handler caches file size at request start
    while the background thread is still writing to the file.
    """
    path = RESULTS_DIR / run_id / "experiment.log"
    if not path.exists():
        return Response(content="", media_type="text/plain")
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            content = f.read()
        return Response(content=content, media_type="text/plain")
    except Exception:
        return Response(content="", media_type="text/plain")


@app.get("/results/{run_id}/index.html")
def serve_result(run_id: str):
    """Serve the results page for a given run.

    Same rationale as serve_log — reads fresh each time to avoid
    Content-Length errors while update_status() is still rewriting the file.
    """
    path = RESULTS_DIR / run_id / "index.html"
    if not path.exists():
        return Response(content="", media_type="text/html")
    content = path.read_text(encoding="utf-8", errors="replace")
    return Response(content=content, media_type="text/html")


# Mount static directories. Routes defined above take priority over these mounts
# because FastAPI processes explicit routes before static mounts.
RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Name mappings ──
def create_name_mappings():
    """Create bidirectional mappings between abbreviated and full names."""
    solver_abbr_to_full = {}
    solver_full_to_abbr = {}

    for abbr_name in solver_directory:
        solver_cls = solver_directory[abbr_name]
        full_name = getattr(solver_cls, "class_name", abbr_name)
        display_name = f"{abbr_name} ({full_name})" if full_name != abbr_name else abbr_name
        solver_abbr_to_full[abbr_name] = display_name
        solver_full_to_abbr[display_name] = abbr_name

    problem_abbr_to_full = {}
    problem_full_to_abbr = {}

    for abbr_name in problem_directory:
        problem_cls = problem_directory[abbr_name]
        full_name = getattr(problem_cls, "class_name", abbr_name)
        display_name = f"{abbr_name} ({full_name})" if full_name != abbr_name else abbr_name
        problem_abbr_to_full[abbr_name] = display_name
        problem_full_to_abbr[display_name] = abbr_name

    return (
        solver_abbr_to_full,
        solver_full_to_abbr,
        problem_abbr_to_full,
        problem_full_to_abbr,
    )


SOLVER_ABBR_TO_FULL, SOLVER_FULL_TO_ABBR, PROBLEM_ABBR_TO_FULL, PROBLEM_FULL_TO_ABBR = (
    create_name_mappings()
)


# ── Schema endpoints ──
@app.get("/postreplicate_schema")
def postreplicate_schema() -> dict[str, Any]:
    """Returns a simple schema for the post-replicate form."""
    d = {
        "num_post_reps": 100,
        "crn_diff_times": True,
        "crn_diff_macroreps": True,
    }
    d.update(POST_REPLICATE_DEFAULTS or {})
    return {
        "params": [
            {
                "name": "num_post_reps",
                "label": "Number of post-replications",
                "type": "int",
                "default": d["num_post_reps"],
            },
            {
                "name": "crn_diff_times",
                "label": "Use CRN on post-replications for solutions recommended at different times?",
                "type": "bool",
                "default": d["crn_diff_times"],
            },
            {
                "name": "crn_diff_macroreps",
                "label": "Use CRN on post-replications for solutions recommended on different macro-replications?",
                "type": "bool",
                "default": d["crn_diff_macroreps"],
            },
        ]
    }


@app.get("/postnormalize_schema")
def postnormalize_schema() -> dict[str, Any]:
    """Returns a simple schema for the post-normalize form."""
    d = {
        "num_post_reps_init_opt": 100,
        "crn_init_opt": True,
    }
    d.update(POST_NORMALIZE_DEFAULTS or {})
    return {
        "params": [
            {
                "name": "num_post_reps_init_opt",
                "label": "Number of post-replications at initial and optimal solutions",
                "type": "int",
                "default": d["num_post_reps_init_opt"],
            },
            {
                "name": "crn_init_opt",
                "label": "Use CRN on post-replications for initial and optimal solution?",
                "type": "bool",
                "default": d["crn_init_opt"],
            },
        ]
    }


@app.get("/plots")
def list_plots():
    """Returns a flat list of plot names derived from experiment_base.PlotType."""
    if not hasattr(eb, "PlotType"):
        return {"plots": [], "source": "missing PlotType"}

    plot_type_cls = eb.PlotType

    plots = []
    try:
        plots = [member.name for member in plot_type_cls]
        source = "enum"
    except TypeError:
        plots = [n for n in dir(plot_type_cls) if n.isupper()]
        source = "class-attrs"

    return {"plots": plots, "source": source}


def extract_params_from_config(config_cls):
    """Extract parameter info (name, default, description) from a Pydantic BaseModel config."""
    params = []
    if config_cls and hasattr(config_cls, "model_fields"):
        for name, field in config_cls.model_fields.items():
            default = None
            if field.default_factory is not None:
                try:
                    default = field.default_factory()
                except Exception:
                    default = "<factory>"
            else:
                default = field.default

            params.append(
                {
                    "name": name,
                    "default": default,
                    "description": field.description or "",
                }
            )
    return params


# ── Solver and problem info endpoints ─
@app.get("/solvers")
def get_solvers():
    """Return all available solvers with display names."""
    return {"solvers": list(SOLVER_ABBR_TO_FULL.values())}


@app.get("/problems")
def get_problems():
    """Return all available problems with display names."""
    return {"problems": list(PROBLEM_ABBR_TO_FULL.values())}


@app.get("/solver_params/{solver_name}")
def get_solver_params(solver_name: str):
    """Return parameters for a solver (accepts display name)."""
    # Convert display name to abbreviated name
    abbr_name = SOLVER_FULL_TO_ABBR.get(solver_name, solver_name)
    solver_cls = solver_directory.get(abbr_name)
    if solver_cls is None:
        return {"parameters": []}

    params = []
    config_cls = getattr(solver_cls, "config_class", None)
    params += extract_params_from_config(config_cls)

    return {"parameters": params}


@app.get("/problem_params/{problem_name}")
def get_problem_params(problem_name: str):
    """Return parameters for both the problem and its model config (accepts display name)."""
    abbr_name = PROBLEM_FULL_TO_ABBR.get(problem_name, problem_name)
    problem_cls = problem_directory.get(abbr_name)
    if problem_cls is None:
        return {"parameters": []}

    params = []
    config_cls = getattr(problem_cls, "config_class", None)
    params += extract_params_from_config(config_cls)

    model_cls = getattr(problem_cls, "model_class", None)
    if model_cls is not None:
        model_config_cls = getattr(model_cls, "config_class", None)
        params += extract_params_from_config(model_config_cls)
    else:
        try:
            sig = inspect.signature(problem_cls)
            if "model" in sig.parameters:
                model_default = sig.parameters["model"].default
                model_config_cls = getattr(model_default.__class__, "config_class", None)
                params += extract_params_from_config(model_config_cls)
        except Exception:
            pass

    return {"parameters": params}


@app.get("/plot_params/{plot_name}")
def get_plot_params(plot_name: str):
    """Return parameter specs for plots that need them."""
    name = plot_name.strip().upper()
    if name in ["ALL", "MEAN", "QUANTILE"]:
        return {"parameters": extract_params_from_config(PlotProgressCurvesConfig)}
    if name in ["VIOLIN", "BOX"]:
        return {"parameters": extract_params_from_config(PlotTerminalProgressCurvesConfig)}
    if name in [
        "CDF_SOLVABILITY",
        "QUANTILE_SOLVABILITY",
        "DIFFERENCE_OF_CDF_SOLVABILITY",
        "DIFFERENCE_OF_QUANTILE_SOLVABILITY",
    ]:
        return {"parameters": extract_params_from_config(PlotSolvabilityProfilesConfig)}
    if name in ["AREA", "AREA_MEAN", "AREA_STD_DEV"]:
        return {"parameters": extract_params_from_config(PlotAreaScatterplotsConfig)}
    if name == "SOLVE_TIME_CDF":
        return {"parameters": extract_params_from_config(PlotSolvabilityCDFConfig)}
    if name == "TERMINAL_SCATTER":
        return {"parameters": extract_params_from_config(PlotTerminalScatterplotsConfig)}
    return {"parameters": []}


# ── Compatibility checking ──
@app.post("/check_compatibility")
def check_compatibility(payload: dict):
    """Check compatibility between solvers and problems."""
    solvers = payload.get("solvers", [])
    problems = payload.get("problems", [])

    compatibility = {}

    for display_name in solvers:
        abbr_name = SOLVER_FULL_TO_ABBR.get(display_name, display_name)
        solver_cls = solver_directory.get(abbr_name)
        if not solver_cls:
            continue
        solver = solver_cls()
        compatibility[display_name] = {}

        for prob_display_name in problems:
            prob_abbr_name = PROBLEM_FULL_TO_ABBR.get(prob_display_name, prob_display_name)
            problem_cls = problem_directory.get(prob_abbr_name)
            if not problem_cls:
                continue
            problem = problem_cls()

            try:
                exp = ProblemsSolvers(solvers=[solver], problems=[problem])
                err = exp.check_compatibility()
                if err.strip() == "":
                    compatibility[display_name][prob_display_name] = {
                        "compatible": True,
                        "message": "",
                    }
                else:
                    compatibility[display_name][prob_display_name] = {
                        "compatible": False,
                        "message": err,
                    }
            except Exception as e:
                compatibility[display_name][prob_display_name] = {
                    "compatible": False,
                    "message": str(e),
                }

    return {"compatibility": compatibility}


# ── Rerun detection ──
def _check_rerun_logic(payload: dict) -> bool:
    """Returns True if experiment needs to rerun, False if only plots changed."""
    last_run_id = payload.get("last_run_id")
    if not last_run_id:
        print("check_rerun: no last_run_id, needs rerun")
        return True

    config_path = RESULTS_DIR / last_run_id / "experiment_config.json"
    experiments_path = RESULTS_DIR / last_run_id / "experiments.pkl"

    if not config_path.exists() or not experiments_path.exists():
        print(
            f"check_rerun: missing files - config:{config_path.exists()} pkl:{experiments_path.exists()}"
        )
        return True

    import json

    with config_path.open() as f:
        saved = json.load(f)

    new_problems = [
        {
            "name": PROBLEM_FULL_TO_ABBR.get(p["name"], p["name"]),
            "fixed_factors": p.get("fixed_factors", {}),
            "model_fixed_factors": p.get("model_fixed_factors", {}),
        }
        for p in payload.get("problems", [])
    ]
    new_solvers = [
        {
            "name": SOLVER_FULL_TO_ABBR.get(s["name"], s["name"]),
            "fixed_factors": s.get("fixed_factors", {}),
        }
        for s in payload.get("solvers", [])
    ]
    new_params = payload.get("experiment_params", {})

    problems_match = saved["problems"] == new_problems
    solvers_match = saved["solvers"] == new_solvers
    params_match = saved["experiment_params"] == new_params

    print(f"check_rerun: problems={problems_match}, solvers={solvers_match}, params={params_match}")
    if not problems_match:
        print(f"  saved problems: {saved['problems']}")
        print(f"  new problems:   {new_problems}")
    if not solvers_match:
        print(f"  saved solvers:  {saved['solvers']}")
        print(f"  new solvers:    {new_solvers}")
    if not params_match:
        print(f"  saved params:   {saved['experiment_params']}")
        print(f"  new params:     {new_params}")

    return not (problems_match and solvers_match and params_match)


# ── Plot generation ──
def generate_plots(
    plots_config,
    all_experiments,
    needed_solver_indices,
    needed_problem_indices,
    solver_idx_map,
    problem_idx_map,
    solvers_config,
    problems_config,
    folder,
):
    """Shared plot generation logic used by both run_experiment_async and run_plots_only."""
    from simopt.experiment_base import (
        plot_area_scatterplots,
        plot_progress_curves,
        plot_solvability_cdfs,
        plot_solvability_profiles,
        plot_terminal_progress,
        plot_terminal_scatterplots,
    )
    from simopt.plot_type import PlotType

    plot_files = []

    for plot_cfg in plots_config:
        plot_type_name = plot_cfg.get("plot_type", "MEAN").upper()
        plot_params = plot_cfg.get("params", {})
        plot_solvers = plot_cfg.get("solvers")
        plot_problems = plot_cfg.get("problems")

        # Map selected indices to experiment array positions
        if plot_solvers:
            plot_solver_abbrs = [SOLVER_FULL_TO_ABBR.get(s, s) for s in plot_solvers]
            orig_solver_indices = [
                i for i, s in enumerate(solvers_config) if s["name"] in plot_solver_abbrs
            ]
            solver_exp_indices = [solver_idx_map[i] for i in orig_solver_indices]
        else:
            solver_exp_indices = list(range(len(needed_solver_indices)))

        if plot_problems:
            plot_problem_abbrs = [PROBLEM_FULL_TO_ABBR.get(p, p) for p in plot_problems]
            orig_problem_indices = [
                i for i, p in enumerate(problems_config) if p["name"] in plot_problem_abbrs
            ]
            problem_exp_indices = [problem_idx_map[i] for i in orig_problem_indices]
        else:
            problem_exp_indices = list(range(len(needed_problem_indices)))

        if not solver_exp_indices or not problem_exp_indices:
            continue

        if plot_type_name in ["ALL", "MEAN", "QUANTILE"]:
            # Generate progress curves for each problem
            for exp_prob_idx in problem_exp_indices:
                try:
                    plt.figure(figsize=(10, 6))

                    all_in_one = plot_params.get("all_in_one", True)
                    normalize = plot_params.get("normalize", False)

                    plot_type_map = {
                        "ALL": PlotType.ALL,
                        "MEAN": PlotType.MEAN,
                        "QUANTILE": PlotType.QUANTILE,
                    }
                    plot_type_enum = plot_type_map.get(plot_type_name, PlotType.MEAN)

                    plot_progress_curves(
                        [
                            all_experiments[exp_prob_idx][exp_solver_idx]
                            for exp_solver_idx in solver_exp_indices
                        ],
                        plot_type=plot_type_enum,
                        all_in_one=all_in_one,
                        normalize=normalize,
                    )
                    actual_prob_idx = needed_problem_indices[exp_prob_idx]
                    filename = f"{plot_type_name.lower()}_progress_curves_problem_{actual_prob_idx + 1}.png"
                    plt.savefig(folder / filename, dpi=150, bbox_inches="tight")
                    plt.close()
                    plot_files.append(filename)
                    print(f"  Saved {filename}")
                except Exception as e:
                    print(
                        f"Error generating {plot_type_name} plot for problem {exp_prob_idx + 1}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    continue

        elif plot_type_name in ["VIOLIN", "BOX"]:
            # Generate terminal progress plots (BOX or VIOLIN) for each problem
            for exp_prob_idx in problem_exp_indices:
                try:
                    plt.figure(figsize=(10, 6))

                    # Extract parameters with defaults
                    normalize = plot_params.get("normalize", True)
                    all_in_one = plot_params.get("all_in_one", True)

                    # Determine which PlotType to use
                    plot_type_enum = PlotType.VIOLIN if plot_type_name == "VIOLIN" else PlotType.BOX

                    plot_terminal_progress(
                        [
                            all_experiments[exp_prob_idx][exp_solver_idx]
                            for exp_solver_idx in solver_exp_indices
                        ],
                        plot_type=plot_type_enum,
                        normalize=normalize,
                        all_in_one=all_in_one,
                    )
                    actual_prob_idx = needed_problem_indices[exp_prob_idx]
                    filename = f"{plot_type_name.lower()}_progress_curves_problem_{actual_prob_idx + 1}.png"
                    plt.savefig(folder / filename, dpi=150, bbox_inches="tight")
                    plt.close()
                    plot_files.append(filename)
                    print(f"  Saved {filename}")
                except Exception as e:
                    print(
                        f"Error generating {plot_type_name} plot for problem {exp_prob_idx + 1}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    continue

        elif plot_type_name in ["AREA", "AREA_MEAN", "AREA_STD_DEV"]:
            # Generate area scatterplots for each problem
            if len(problem_exp_indices) < 2:
                print(f"Warning: {plot_type_name} requires multiple problems. Skipping.")
                continue
            try:
                print(f"Generating {plot_type_name} plot...")
                plt.figure(figsize=(10, 6))
                # Extract parameters with defaults
                all_in_one = plot_params.get("all_in_one", True)
                n_bootstraps = plot_params.get("n_bootstraps", 100)
                conf_level = plot_params.get("conf_level", 0.95)
                plot_conf_ints = plot_params.get("plot_conf_ints", True)
                print_max_hw = plot_params.get("print_max_hw", True)
                solver_set_name = plot_params.get("solver_set_name", "SOLVER_SET")
                problem_set_name = plot_params.get("problem_set_name", "PROBLEM_SET")

                plot_type_map = {
                    "AREA": PlotType.AREA,
                    "AREA_MEAN": PlotType.AREA_MEAN,
                    "AREA_STD_DEV": PlotType.AREA_STD_DEV,
                }
                plot_type_enum = plot_type_map[plot_type_name]

                filtered_experiments = [
                    [
                        all_experiments[exp_prob_idx][exp_solver_idx]
                        for exp_solver_idx in solver_exp_indices
                    ]
                    for exp_prob_idx in problem_exp_indices
                ]

                plot_area_scatterplots(
                    filtered_experiments,
                    all_in_one=all_in_one,
                    n_bootstraps=n_bootstraps,
                    conf_level=conf_level,
                    plot_conf_ints=plot_conf_ints,
                    print_max_hw=print_max_hw,
                    solver_set_name=solver_set_name,
                    problem_set_name=problem_set_name,
                )
                filename = f"{plot_type_name.lower()}_area_scatterplot.png"
                plt.savefig(folder / filename, dpi=150, bbox_inches="tight")
                plt.close()
                plot_files.append(filename)
                print(f"  Saved {filename}")
            except Exception as e:
                print(f"Error generating {plot_type_name} plot: {e}")
                import traceback

                traceback.print_exc()

        elif plot_type_name in [
            "CDF_SOLVABILITY",
            "QUANTILE_SOLVABILITY",
            "DIFFERENCE_OF_CDF_SOLVABILITY",
            "DIFFERENCE_OF_QUANTILE_SOLVABILITY",
        ]:
            # Solvability profiles require multiple problems
            if len(problem_exp_indices) < 2:
                print(f"Warning: {plot_type_name} requires multiple problems. Skipping.")
                continue
            try:
                print(f"Generating {plot_type_name} plot...")
                plt.figure(figsize=(10, 6))
                # Extract parameters with defaults
                all_in_one = plot_params.get("all_in_one", True)
                n_bootstraps = plot_params.get("n_bootstraps", 100)
                conf_level = plot_params.get("conf_level", 0.95)
                plot_conf_ints = plot_params.get("plot_conf_ints", False)  # Disabled by default
                print_max_hw = plot_params.get("print_max_hw", False)
                solve_tol = plot_params.get("solve_tol", 0.1)
                beta = plot_params.get("beta", 0.5)
                ref_solver = plot_params.get("ref_solver", None)
                solver_set_name = plot_params.get("solver_set_name", "SOLVER_SET")
                problem_set_name = plot_params.get("problem_set_name", "PROBLEM_SET")
                # Map plot type name to PlotType enum
                plot_type_map = {
                    "CDF_SOLVABILITY": PlotType.CDF_SOLVABILITY,
                    "QUANTILE_SOLVABILITY": PlotType.QUANTILE_SOLVABILITY,
                    "DIFFERENCE_OF_CDF_SOLVABILITY": PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
                    "DIFFERENCE_OF_QUANTILE_SOLVABILITY": PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
                }
                plot_type_enum = plot_type_map[plot_type_name]

                filtered_experiments = [
                    [
                        all_experiments[exp_prob_idx][exp_solver_idx]
                        for exp_solver_idx in solver_exp_indices
                    ]
                    for exp_prob_idx in problem_exp_indices
                ]

                plot_solvability_profiles(
                    filtered_experiments,
                    plot_type=plot_type_enum,
                    all_in_one=all_in_one,
                    n_bootstraps=n_bootstraps,
                    conf_level=conf_level,
                    plot_conf_ints=plot_conf_ints,
                    print_max_hw=print_max_hw,
                    solve_tol=solve_tol,
                    beta=beta,
                    ref_solver=ref_solver,
                    solver_set_name=solver_set_name,
                    problem_set_name=problem_set_name,
                )
                filename = f"{plot_type_name.lower()}_solvability_profile.png"
                plt.savefig(folder / filename, dpi=150, bbox_inches="tight")
                plt.close()
                plot_files.append(filename)
                print(f"  Saved {filename}")
            except Exception as e:
                print(f"Error generating {plot_type_name} plot: {e}")
                import traceback

                traceback.print_exc()

        elif plot_type_name == "SOLVE_TIME_CDF":
            # Generate solvability CDF plots for each problem
            for exp_prob_idx in problem_exp_indices:
                try:
                    plt.figure(figsize=(10, 6))

                    # Extract parameters with defaults
                    solve_tol = plot_params.get("solve_tol", 0.1)
                    all_in_one = plot_params.get("all_in_one", True)
                    n_bootstraps = plot_params.get("n_bootstraps", 100)
                    conf_level = plot_params.get("conf_level", 0.95)
                    plot_conf_ints = plot_params.get(
                        "plot_conf_ints", False
                    )  # Disabled by default to avoid bootstrap errors
                    print_max_hw = plot_params.get("print_max_hw", False)

                    plot_solvability_cdfs(
                        [
                            all_experiments[exp_prob_idx][exp_solver_idx]
                            for exp_solver_idx in solver_exp_indices
                        ],
                        solve_tol=solve_tol,
                        all_in_one=all_in_one,
                        n_bootstraps=n_bootstraps,
                        conf_level=conf_level,
                        plot_conf_ints=plot_conf_ints,
                        print_max_hw=print_max_hw,
                    )
                    filename = f"solvability_cdf_problem_{exp_prob_idx + 1}.png"
                    plt.savefig(folder / filename, dpi=150, bbox_inches="tight")
                    plt.close()
                    plot_files.append(filename)
                    print(f"  Saved {filename}")
                except Exception as e:
                    print(
                        f"Error generating SOLVE_TIME_CDF plot for problem {exp_prob_idx + 1}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    continue

        elif plot_type_name == "TERMINAL_SCATTER":
            # Generate terminal scatterplot (requires multiple problems)
            if len(problem_exp_indices) < 2:
                print("Warning: TERMINAL_SCATTER requires multiple problems. Skipping.")
                continue

            try:
                print("Generating TERMINAL_SCATTER plot...")
                plt.figure(figsize=(10, 6))

                # Extract parameters with defaults
                all_in_one = plot_params.get("all_in_one", True)
                solver_set_name = plot_params.get("solver_set_name", "SOLVER_SET")
                problem_set_name = plot_params.get("problem_set_name", "PROBLEM_SET")

                filtered_experiments = [
                    [
                        all_experiments[exp_prob_idx][exp_solver_idx]
                        for exp_solver_idx in solver_exp_indices
                    ]
                    for exp_prob_idx in problem_exp_indices
                ]

                plot_terminal_scatterplots(
                    filtered_experiments,
                    all_in_one=all_in_one,
                    solver_set_name=solver_set_name,
                    problem_set_name=problem_set_name,
                )
                filename = "terminal_scatterplot.png"
                plt.savefig(folder / filename, dpi=150, bbox_inches="tight")
                plt.close()
                plot_files.append(filename)
                print(f"  Saved {filename}")
            except Exception as e:
                print(f"Error generating TERMINAL_SCATTER plot: {e}")
                import traceback

                traceback.print_exc()

    return plot_files


# ── Output capture ──
def setup_print_capture(log_file):
    """Sets up stdout capture to log file. Returns (original_stdout, capture_instance)."""
    import json as _json
    import sys
    import threading

    write_lock = threading.Lock()

    class PrintCapture:
        def __init__(self, original) -> None:
            self.original = original
            self.buf = ""

        def write(self, text):
            self.original.write(text)
            self.buf += text
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                line = line.strip()
                if line:
                    entry = {
                        "time": __import__("datetime").datetime.now().strftime("%H:%M:%S"),
                        "level": "INFO",
                        "msg": line,
                    }
                    with write_lock, log_file.open("a") as f:
                        f.write(_json.dumps(entry) + "\n")

        def flush(self):
            self.original.flush()

        def isatty(self):
            return False

    original_stdout = sys.stdout
    sys.stdout = PrintCapture(original_stdout)
    return original_stdout


# ── Experiment runner ──
def run_experiment_async(run_id: str, payload: dict):
    """Run the experiment in a background thread."""
    folder = RESULTS_DIR / run_id
    print(f"Results folder: {folder}")
    print(f"Folder exists: {folder.exists()}")

    import json as _json
    import sys

    log_file = folder / "experiment.log"

    original_stdout = setup_print_capture(log_file)
    import logging as _logging

    class PrintForwardHandler(_logging.Handler):
        def emit(self, record):
            if record.name.startswith(("matplotlib", "PIL", "urllib", "findfont")):
                return
            msg = self.format(record)
            if msg.startswith("findfont:"):
                return
            print(msg)

    log_handler = PrintForwardHandler()
    log_handler.setFormatter(_logging.Formatter("%(message)s"))
    log_handler.setLevel(_logging.DEBUG)

    root_logger = _logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(_logging.DEBUG)
    root_logger.addHandler(log_handler)

    try:
        update_status(folder, "Running experiments...")

        exp_params = payload.get("experiment_params", {})
        num_macroreps = exp_params.get("num_macroreps", 10)
        num_postreps = exp_params.get("num_postreps", 100)
        num_postnorms = exp_params.get("num_postnorms", 200)

        problems_config = payload.get("problems", [])
        solvers_config = payload.get("solvers", [])
        plots_config = payload.get("plots", [])

        # Convert display names to abbreviated names
        for solver_cfg in solvers_config:
            display_name = solver_cfg["name"]
            abbr_name = SOLVER_FULL_TO_ABBR.get(display_name, display_name)
            solver_cfg["name"] = abbr_name
            print(f"Converted solver: '{display_name}' -> '{abbr_name}'")

        for prob_cfg in problems_config:
            display_name = prob_cfg["name"]
            abbr_name = PROBLEM_FULL_TO_ABBR.get(display_name, display_name)
            prob_cfg["name"] = abbr_name
            print(f"Converted problem: '{display_name}' -> '{abbr_name}'")

        from simopt.experiment_base import ProblemSolver, post_normalize

        # Validate solver and problem names
        for solver_cfg in solvers_config:
            if solver_cfg["name"] not in solver_directory:
                raise ValueError(
                    f"Solver '{solver_cfg['name']}' not found in solver directory. Available solvers: {list(solver_directory.keys())[:10]}"
                )

        for prob_cfg in problems_config:
            if prob_cfg["name"] not in problem_directory:
                raise ValueError(
                    f"Problem '{prob_cfg['name']}' not found in problem directory. Available problems: {list(problem_directory.keys())[:10]}"
                )

        needed_solver_indices = set()
        needed_problem_indices = set()

        for plot_cfg in plots_config:
            plot_solvers = plot_cfg.get("solvers")
            plot_problems = plot_cfg.get("problems")

            if plot_solvers:
                plot_solver_abbrs = [SOLVER_FULL_TO_ABBR.get(s, s) for s in plot_solvers]
                for i, s in enumerate(solvers_config):
                    if s["name"] in plot_solver_abbrs:
                        needed_solver_indices.add(i)
            else:
                needed_solver_indices.update(range(len(solvers_config)))

            if plot_problems:
                plot_problem_abbrs = [PROBLEM_FULL_TO_ABBR.get(p, p) for p in plot_problems]
                for i, p in enumerate(problems_config):
                    if p["name"] in plot_problem_abbrs:
                        needed_problem_indices.add(i)
            else:
                needed_problem_indices.update(range(len(problems_config)))

        # Convert to sorted lists
        needed_solver_indices = sorted(needed_solver_indices)
        needed_problem_indices = sorted(needed_problem_indices)

        solver_idx_map = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(needed_solver_indices)
        }
        problem_idx_map = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(needed_problem_indices)
        }

        print(
            f"Running experiments for {len(needed_solver_indices)} solvers and {len(needed_problem_indices)} problems"
        )

        # Run experiments for each problem
        all_experiments = []
        for prob_idx in needed_problem_indices:
            prob_cfg = problems_config[prob_idx]
            print(f"Running problem {prob_idx + 1}: {prob_cfg['name']}...")

            experiments_same_problem = []

            for solver_idx in needed_solver_indices:
                solver_cfg = solvers_config[solver_idx]
                print(
                    f"Creating ProblemSolver with solver={solver_cfg['name']}, problem={prob_cfg['name']}"
                )
                print(f"  Solver factors: {solver_cfg.get('fixed_factors', {})}")
                print(f"  Problem factors: {prob_cfg.get('fixed_factors', {})}")

                experiment = ProblemSolver(
                    solver_name=solver_cfg["name"],
                    solver_rename=solver_cfg.get("rename", solver_cfg["name"]),
                    solver_fixed_factors=solver_cfg.get("fixed_factors", {}),
                    problem_name=prob_cfg["name"],
                    problem_rename=prob_cfg.get("rename", prob_cfg["name"]),
                    problem_fixed_factors=prob_cfg.get("fixed_factors", {}),
                    model_fixed_factors=prob_cfg.get("model_fixed_factors", {}),
                )

                print(f"Running experiment with {num_macroreps} macroreps...")
                experiment.run(n_macroreps=num_macroreps)
                print(f"Post-replicating with {num_postreps} postreps...")
                experiment.post_replicate(n_postreps=num_postreps)
                experiments_same_problem.append(experiment)

            # Post-normalize
            print(f"Post-normalizing with {num_postnorms} postnorms...")
            post_normalize(
                experiments=experiments_same_problem,
                n_postreps_init_opt=num_postnorms,
            )

            all_experiments.append(experiments_same_problem)
            import pickle

            with (folder / "experiments.pkl").open("wb") as f:
                pickle.dump(all_experiments, f)
            with (folder / "index_maps.pkl").open("wb") as f:
                pickle.dump(
                    {
                        "needed_solver_indices": needed_solver_indices,
                        "needed_problem_indices": needed_problem_indices,
                        "solver_idx_map": solver_idx_map,
                        "problem_idx_map": problem_idx_map,
                    },
                    f,
                )

        print("Generating plots...")
        plot_files = generate_plots(
            plots_config=plots_config,
            all_experiments=all_experiments,
            needed_solver_indices=needed_solver_indices,
            needed_problem_indices=needed_problem_indices,
            solver_idx_map=solver_idx_map,
            problem_idx_map=problem_idx_map,
            solvers_config=solvers_config,
            problems_config=problems_config,
            folder=folder,
        )

        config_to_save = {
            "problems": [
                {
                    "name": p["name"],
                    "fixed_factors": p.get("fixed_factors", {}),
                    "model_fixed_factors": p.get("model_fixed_factors", {}),
                }
                for p in problems_config
            ],
            "solvers": [
                {"name": s["name"], "fixed_factors": s.get("fixed_factors", {})}
                for s in solvers_config
            ],
            "experiment_params": exp_params,
        }
        with (folder / "experiment_config.json").open("w") as f:
            _json.dump(config_to_save, f)

        # Create final results page with plots
        update_status(folder, "Complete!", plot_files)
        print(f"Experiment {run_id} completed successfully!")

    except Exception as e:
        error_msg = f"Error: {e!s}"
        update_status(folder, error_msg)
        print(f"Experiment {run_id} failed: {error_msg}")
        import traceback

        traceback.print_exc()

    finally:
        sys.stdout = original_stdout
        root_logger.removeHandler(log_handler)
        root_logger.setLevel(original_level)


# ── Results page generation ──
def update_status(
    folder: Path,
    status: str,
    plot_files: list[str] | None = None,
):
    """Update the results page with current status and plots."""
    run_id = folder.name
    plots_html = ""
    if plot_files:
        plot_cards = ""
        minimized_icons = ""
        preview_data = ""
        for i, plot_file in enumerate(plot_files):
            plot_id = f"plot_{i}"
            label = plot_file.replace("_", " ").replace(".png", "")
            plot_cards += f"""
            <div class="plot-card" id="{plot_id}">
                <div class="plot-card-header">
                    <span class="plot-label">{label}</span>
                    <button class="collapse-btn" onclick="collapsePlot('{plot_id}')" title="Minimize plot">-</button>
                </div>
                <img src="/results/{run_id}/{plot_file}" alt="{plot_file}">
            </div>
            """
            minimized_icons += f"""
            <div class="mini-icon" id="mini_{plot_id}" style="display:none;"
                 onmouseenter="expandPreview('{plot_id}', this)"
                 onmouseleave="collapsePreview()"
                 onclick="restorePlot('{plot_id}')">
                <div class="mini-thumbnail-wrapper">
                    <img src="{plot_file}" alt="{label}" class="mini-thumbnail">
                    <div class="mini-overlay">
                        <span class="mini-restore-hint">click to restore</span>
                    </div>
                </div>
                <span class="mini-label">{label}</span>
            </div>
            """
            preview_data += (
                f'"{plot_id}": {{"src": "/results/{run_id}/{plot_file}", "label": "{label}"}},'
            )

        plots_html = f"""
        <div id="plots-grid">{plot_cards}</div>
        <div id="minimized-tray">{minimized_icons}</div>
        <div id="global-preview" style="display:none;">
            <img id="global-preview-img" src="" alt="">
            <p id="global-preview-label"></p>
        </div>
        <script>var previewData = {{ {preview_data} }};</script>
        """

    status_class = (
        "success"
        if status == "Complete!"
        else ("error" if status.startswith("Error:") else "running")
    )
    is_running_js = "true" if status_class == "running" else "false"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Results - {run_id}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', sans-serif; background: #f9fafb; min-height: 100vh; }}
        .header {{ background: #e5e7eb; padding: 1.5rem 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
        .header h1 {{ color: #0f172a; font-size: 2rem; font-weight: 700; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 0 2rem 6rem; }}
        .card {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem; }}
        .card h2 {{ color: #2563eb; font-size: 1.2rem; margin-bottom: 1rem; }}
        .status {{ padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }}
        .status.running {{ background: #eff6ff; border: 1px solid #93c5fd; }}
        .status.success {{ background: #dcfce7; border: 1px solid #86efac; }}
        .status.error   {{ background: #fee2e2; border: 1px solid #fca5a5; }}
        .status p {{ margin: 0.5rem 0; }}
        .status.running p {{ color: #1e40af; }}
        .status.success p {{ color: #166534; }}
        .status.error   p {{ color: #991b1b; }}
        .status strong {{ font-weight: 600; }}

        /* ── Log panel ── */
        .log-panel {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #e2e8f0;
            overflow: hidden;
        }}
        .log-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.5rem;
            background: white;
            border-bottom: 1px solid #e2e8f0;
        }}
        .log-header-left {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .log-title {{
            color: #2563eb;
            font-size: 1.2rem;
            font-weight: 600;
        }}
        .log-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            flex-shrink: 0;
            box-shadow: 0 0 5px #22c55e;
            animation: pulse 1.5s infinite;
        }}
        .log-dot.idle {{
            background: #9ca3af;
            box-shadow: none;
            animation: none;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.35; }}
        }}
        .log-controls {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}
        .log-btn {{
            background: white;
            border: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 500;
            padding: 0.25rem 0.65rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.15s, color 0.15s, border-color 0.15s;
        }}
        .log-btn:hover {{ background: #f1f5f9; color: #374151; border-color: #cbd5e1; }}
        #log-body {{
            padding: 0.85rem 1.5rem;
            height: 260px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.75rem;
            line-height: 1.75;
            background: #f1f5f9;
        }}
        #log-body::-webkit-scrollbar {{ width: 5px; }}
        #log-body::-webkit-scrollbar-track {{ background: #e2e8f0; }}
        #log-body::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 3px; }}
        .log-entry {{
            display: flex;
            gap: 0.75rem;
            padding: 0.1rem 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .log-entry:last-child {{ border-bottom: none; }}
        .log-time {{ color: #9ca3af; flex-shrink: 0; min-width: 58px; }}
        .log-msg {{ color: #374151; word-break: break-word; }}
        .log-msg .kw-success {{ color: #16a34a; font-weight: 600; }}
        .log-msg .kw-error   {{ color: #dc2626; font-weight: 600; }}
        .log-msg .kw-running {{ color: #2563eb; }}
        .log-msg .kw-saved   {{ color: #7c3aed; }}
        .log-empty {{ color: #9ca3af; font-style: italic; padding: 0.5rem 0; }}

        .log-toggle {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            background: white;
            transition: background 0.15s;
        }}
        .log-toggle:hover {{ background: #f8fafc; }}
        .log-toggle-left {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .log-toggle-right {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .log-chevron {{
            color: #9ca3af;
            font-size: 0.8rem;
            transition: transform 0.2s ease;
        }}
        .log-chevron.open {{ transform: rotate(180deg); }}
        .log-body-wrapper {{ display: none; border-top: 1px solid #e2e8f0; }}
        .log-body-wrapper.open {{ display: block; }}

        /* ── Plot grid ── */
        #plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}
        .plot-card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: visible;
            transition: box-shadow 0.2s ease;
        }}
        .plot-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
        .plot-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 0.9rem;
            border-bottom: 1px solid #f1f5f9;
        }}
        .plot-label {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            text-transform: capitalize;
            letter-spacing: 0.02em;
        }}
        .collapse-btn {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: 1.5px solid #cbd5e1;
            background: #f8fafc;
            color: #64748b;
            font-size: 1.1rem;
            line-height: 1;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.15s, border-color 0.15s, color 0.15s;
            flex-shrink: 0;
        }}
        .collapse-btn:hover {{ background: #fee2e2; border-color: #fca5a5; color: #dc2626; }}
        .plot-card img {{ width: 100%; height: auto; border-radius: 0 0 10px 10px; display: block; }}

        /* ── Minimized tray ── */
        #minimized-tray {{
            position: fixed;
            bottom: 0; left: 0; right: 0;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 10px 20px;
            background: rgba(241,245,249,0.92);
            backdrop-filter: blur(8px);
            border-top: 1px solid #e2e8f0;
            z-index: 100;
        }}
        #minimized-tray:empty {{ display: none; }}
        .mini-icon {{
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 3px;
            cursor: pointer;
            transition: transform 0.15s ease;
        }}
        .mini-icon:hover {{ transform: translateY(-3px); }}
        .mini-thumbnail-wrapper {{
            position: relative;
            width: 64px; height: 48px;
            border-radius: 6px;
            overflow: hidden;
            border: 1.5px solid #cbd5e1;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
            opacity: 0.55;
            transition: opacity 0.15s, border-color 0.15s, box-shadow 0.15s;
        }}
        .mini-icon:hover .mini-thumbnail-wrapper {{
            opacity: 1;
            border-color: #2563eb;
            box-shadow: 0 2px 8px rgba(37,99,235,0.2);
        }}
        .mini-thumbnail {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
        .mini-overlay {{
            position: absolute; inset: 0;
            background: rgba(15,23,42,0);
            display: flex; align-items: center; justify-content: center;
            transition: background 0.15s;
        }}
        .mini-icon:hover .mini-overlay {{ background: rgba(15,23,42,0.35); }}
        .mini-restore-hint {{
            color: white; font-size: 0.6rem; font-weight: 600;
            opacity: 0; transition: opacity 0.15s;
            text-transform: uppercase; letter-spacing: 0.05em;
        }}
        .mini-icon:hover .mini-restore-hint {{ opacity: 1; }}
        .mini-label {{
            font-size: 0.6rem; color: #94a3b8;
            text-align: center; max-width: 70px;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            text-transform: capitalize;
        }}
        .mini-icon:hover .mini-label {{ color: #2563eb; }}

        /* ── Global preview ── */
        #global-preview {{
            display: none;
            position: fixed;
            width: 380px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.18), 0 0 0 1px rgba(37,99,235,0.12);
            padding: 0.75rem;
            pointer-events: none;
            z-index: 9999;
            animation: previewIn 0.15s ease;
        }}
        #global-preview::after {{
            content: '';
            position: absolute;
            top: 100%;
            left: var(--arrow-left, 50%);
            transform: translateX(-50%);
            border: 7px solid transparent;
            border-top-color: white;
        }}
        #global-preview img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
        #global-preview p {{ font-size: 0.78rem; color: #64748b; text-align: center; margin-top: 0.5rem; font-weight: 500; text-transform: capitalize; }}
        @keyframes previewIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    <script>
        var allLogs = [];
        var autoScroll = true;
        var lastLogCount = 0;
        var hasReloadedAfterCompletion = false;
        var lastCompletionDetected = false;

        function fetchLogs() {{
            fetch('experiment.log?t=' + Date.now())
                .then(function(r) {{ return r.text(); }})
                .catch(function() {{ return ''; }})
                .then(function(text) {{
                    if (!text.trim()) return;
                    var lines = text.trim().split('\\n');
                    if (lines.length === lastLogCount) return;
                    lastLogCount = lines.length;
                    allLogs = [];
                    lines.forEach(function(line) {{
                        try {{ allLogs.push(JSON.parse(line)); }} catch(e) {{}}
                    }});
                    renderLogs();
                    var completed = allLogs.some(function(e) {{
                        return e.msg && (
                            e.msg.includes("completed successfully") ||
                            e.msg.includes("Complete!")
                        );
                    }});
                    if (completed && !lastCompletionDetected) {{
                        lastCompletionDetected = true;
                        var reloadKey = 'reloaded_' + window.location.pathname;
                        if (!sessionStorage.getItem(reloadKey)) {{
                            sessionStorage.setItem(reloadKey, '1');
                            setTimeout(function() {{
                                window.location.reload();
                            }}, 1200);
                        }}
                    }}
                }});
        }}

        function colorize(msg) {{
            msg = msg.replace(/(completed successfully|Complete!)/gi, '<span class="kw-success">$1</span>');
            msg = msg.replace(/(error|failed|exception)/gi, '<span class="kw-error">$1</span>');
            msg = msg.replace(/(Running|Starting|Creating|Post-replicating|Post-normalizing)/gi, '<span class="kw-running">$1</span>');
            msg = msg.replace(/(Saved [^ ]+[.]png)/gi, '<span class="kw-saved">$1</span>');
            return msg;
        }}

        function escHtml(s) {{
            return String(s)
                .replace(/&/g,'&amp;').replace(/</g,'&lt;')
                .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
        }}

        function renderLogs() {{
            var body = document.getElementById('log-body');
            if (allLogs.length === 0) {{
                body.innerHTML = '<div class="log-empty">Waiting for output...</div>';
                return;
            }}
            var atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 60;
            body.innerHTML = allLogs.map(function(e) {{
                return '<div class="log-entry">' +
                    '<span class="log-time">' + escHtml(e.time) + '</span>' +
                    '<span class="log-msg">' + colorize(escHtml(e.msg)) + '</span>' +
                    '</div>';
            }}).join('');
            if (autoScroll && atBottom) {{
                body.scrollTop = body.scrollHeight;
            }}
        }}

        var logOpen = false;
        function toggleLog() {{
            logOpen = !logOpen;
            var wrapper = document.getElementById('log-body-wrapper');
            var chevron = document.getElementById('log-chevron');
            if (logOpen) {{
                wrapper.classList.add('open');
                chevron.classList.add('open');
            }} else {{
                wrapper.classList.remove('open');
                chevron.classList.remove('open');
            }}
        }}

        function toggleAutoScroll() {{
            autoScroll = !autoScroll;
            var btn = document.getElementById('autoscroll-btn');
            btn.textContent = autoScroll ? 'Auto-scroll: ON' : 'Auto-scroll: OFF';
            btn.style.color = autoScroll ? '#22c55e' : '#64748b';
        }}

        function clearDisplay() {{
            allLogs = [];
            renderLogs();
        }}

        // ── Plot functions ──
        function collapsePlot(plotId) {{
            document.getElementById(plotId).style.display = 'none';
            document.getElementById('mini_' + plotId).style.display = 'flex';
            document.getElementById('minimized-tray').style.display = 'flex';
        }}

        function restorePlot(plotId) {{
            collapsePreview();
            document.getElementById('mini_' + plotId).style.display = 'none';
            var card = document.getElementById(plotId);
            card.style.display = '';
            card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}

        function expandPreview(plotId, iconEl) {{
            var preview = document.getElementById('global-preview');
            var data = previewData[plotId];
            if (!data) return;
            document.getElementById('global-preview-img').src = data.src;
            document.getElementById('global-preview-img').alt = data.label;
            document.getElementById('global-preview-label').textContent = data.label;
            preview.style.display = 'block';
            preview.style.visibility = 'hidden';
            preview.style.left = '0px';
            preview.style.top = '0px';
            var previewW = preview.offsetWidth;
            var previewH = preview.offsetHeight;
            var margin = 12;
            var iconRect = iconEl.getBoundingClientRect();
            var left = iconRect.left + iconRect.width / 2 - previewW / 2;
            var top = iconRect.top - previewH - 14;
            var clampedLeft = Math.max(margin, Math.min(left, window.innerWidth - previewW - margin));
            preview.style.left = clampedLeft + 'px';
            preview.style.top = top + 'px';
            var arrowLeft = (iconRect.left + iconRect.width / 2) - clampedLeft;
            preview.style.setProperty('--arrow-left', arrowLeft + 'px');
            preview.style.visibility = 'visible';
        }}

        function collapsePreview() {{
            document.getElementById('global-preview').style.display = 'none';
        }}

        // ── Init ──
        function saveLogState() {{
            sessionStorage.setItem('logOpen', logOpen ? '1' : '0');
        }}

        window.addEventListener('beforeunload', saveLogState);

        window.addEventListener('DOMContentLoaded', function() {{
            var savedOpen = sessionStorage.getItem('logOpen');
            if (savedOpen === '1') {{
                logOpen = true;
                document.getElementById('log-body-wrapper').classList.add('open');
                document.getElementById('log-chevron').classList.add('open');
            }}
            fetchLogs();
            if ({is_running_js}) {{
                setInterval(fetchLogs, 1500);
            }}
        }});
    </script>
</head>
<body>
    <div class="header"><h1>Results</h1></div>
    <div class="container">

        <div class="card">
            <h2>Experiment Details</h2>
            <div class="status {status_class}">
                <p><strong>Experiment ID:</strong> {run_id}</p>
                <p><strong>Status:</strong> {status}</p>
            </div>
        </div>

        <div class="log-panel">
            <div style="display:flex; align-items:center; justify-content:space-between; padding:0.75rem 1.5rem; background:white; border-bottom:1px solid #e2e8f0; cursor:pointer;" onclick="toggleLog()">
                <div style="display:flex; align-items:center; gap:0.6rem;">
                    <div class="log-dot" id="log-dot"></div>
                    <span class="log-title">Output Log</span>
                    <span class="log-chevron" id="log-chevron">&#9660;</span>
                </div>
                <div style="display:flex; gap:0.5rem;">
                    <button class="log-btn" id="autoscroll-btn"
                        onclick="event.stopPropagation(); toggleAutoScroll();"
                        style="color:#16a34a; border-color:#bbf7d0; background:#f0fdf4;">Auto-scroll: ON</button>
                    <button class="log-btn"
                        onclick="event.stopPropagation(); clearDisplay();">Clear</button>
                </div>
            </div>
            <div class="log-body-wrapper" id="log-body-wrapper">
                <div id="log-body">
                    <div class="log-empty">Waiting for output...</div>
                </div>
            </div>
        </div>

        {plots_html}

    </div>
    <div id="global-preview" style="display:none;">
        <img id="global-preview-img" src="" alt="">
        <p id="global-preview-label"></p>
    </div>
</body>
</html>"""

    with (folder / "index.html").open("w") as f:
        f.write(html_content)


# ── Plot-only rerun ──
def run_plots_only(run_id: str, payload: dict):
    """Regenerate plots only using saved experiment objects."""
    import pickle

    folder = RESULTS_DIR / run_id

    import sys

    log_file = folder / "experiment.log"

    original_stdout = setup_print_capture(log_file)

    try:
        print("Reusing existing experiment results, generating new plots only...")
        update_status(folder, "Generating plots...")

        with (folder / "experiments.pkl").open("rb") as f:
            all_experiments = pickle.load(f)
        with (folder / "index_maps.pkl").open("rb") as f:
            maps = pickle.load(f)

        needed_solver_indices = maps["needed_solver_indices"]
        needed_problem_indices = maps["needed_problem_indices"]
        solver_idx_map = maps["solver_idx_map"]
        problem_idx_map = maps["problem_idx_map"]

        plot_files = []

        plots_config = payload.get("plots", [])
        solvers_config = payload.get("solvers", [])
        problems_config = payload.get("problems", [])

        # Convert display names
        for solver_cfg in solvers_config:
            solver_cfg["name"] = SOLVER_FULL_TO_ABBR.get(solver_cfg["name"], solver_cfg["name"])
        for prob_cfg in problems_config:
            prob_cfg["name"] = PROBLEM_FULL_TO_ABBR.get(prob_cfg["name"], prob_cfg["name"])

        plot_files = generate_plots(
            plots_config=plots_config,
            all_experiments=all_experiments,
            needed_solver_indices=needed_solver_indices,
            needed_problem_indices=needed_problem_indices,
            solver_idx_map=solver_idx_map,
            problem_idx_map=problem_idx_map,
            solvers_config=solvers_config,
            problems_config=problems_config,
            folder=folder,
        )

        update_status(folder, "Complete!", plot_files)
        print("Plots generated successfully!")

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback

        traceback.print_exc()
        update_status(folder, f"Error: {e!s}")
    finally:
        sys.stdout = original_stdout


# ── API endpoints ──
@app.post("/api/run")
def run_experiment(payload: Annotated[dict[str, Any], Body()]):
    from datetime import datetime

    needs_rerun = _check_rerun_logic(payload)

    if needs_rerun:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = RESULTS_DIR / run_id
        folder.mkdir(parents=True, exist_ok=True)
        update_status(folder, "Initializing...")
        thread = threading.Thread(target=run_experiment_async, args=(run_id, payload))
        thread.daemon = True
        thread.start()
    else:
        run_id = payload.get("last_run_id")
        if not isinstance(run_id, str):
            return {"error": "Missing last_run_id for plot-only run"}
        folder = RESULTS_DIR / run_id
        print(f"Skipping rerun, generating plots only for {run_id}")
        thread = threading.Thread(target=run_plots_only, args=(run_id, payload))
        thread.daemon = True
        thread.start()

    return {"id": run_id}


@app.get("/api/results/{experiment_id}")
def get_results(experiment_id: str):
    """Get results for an experiment."""
    path = Path(f"svelte-app/results/{experiment_id}")
    images = [f"svelte-app/results/{experiment_id}/{p.name}" for p in path.glob("*.png")]
    return {"images": images}
