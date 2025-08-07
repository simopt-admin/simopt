"""Modern web-based GUI for SimOpt Library using Flask and HTMX."""

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Import SimOpt modules
import simopt.directory as directory

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global variables to store current state
current_problems = []
current_solvers = []
current_experiment = None
experiments = []


@app.route("/")
def index():
    """Main page with menu options."""
    return render_template("index.html")


@app.route("/experiments")
def experiments_page():
    """Simulation Optimization Experiments page."""
    # Get available problems and solvers
    problems = list(directory.problem_directory.keys())
    solvers = list(directory.solver_directory.keys())
    return render_template(
        "experiments.html", problems=problems, solvers=solvers, experiments=experiments
    )


@app.route("/data-farming")
def data_farming_page():
    """Data Farming page."""
    return render_template("data_farming.html")


@app.route("/api/problems")
def get_problems():
    """Get available problems."""
    problems = []
    for key, problem_class in directory.problem_directory.items():
        try:
            problem = problem_class()
            problems.append(
                {
                    "key": key,
                    "name": problem.name,
                    "objective": "Single" if problem.n_objectives == 1 else "Multiple",
                    "constraint": "Unconstrained"
                    if not hasattr(problem, "constraint")
                    else "Constrained",
                    "variable_type": "Continuous"
                    if problem.variable_type == "continuous"
                    else "Discrete",
                    "gradient_available": getattr(problem, "gradient_available", False),
                }
            )
        except Exception:
            # Skip problems that can't be instantiated
            continue
    return jsonify(problems)


@app.route("/api/solvers")
def get_solvers():
    """Get available solvers."""
    solvers = []
    for key, solver_class in directory.solver_directory.items():
        try:
            solver = solver_class()
            solvers.append(
                {
                    "key": key,
                    "name": solver.name,
                    "gradient_needed": getattr(solver, "gradient_needed", False),
                    "discrete": getattr(solver, "discrete", False),
                    "constraint_type": getattr(solver, "constraint_type", "box"),
                }
            )
        except Exception:
            # Skip solvers that can't be instantiated
            continue
    return jsonify(solvers)


@app.route("/api/add-problem", methods=["POST"])
def add_problem():
    """Add a problem to the current workspace."""
    global current_problems
    problem_key = request.json.get("problem_key")

    if problem_key in directory.problem_directory:
        if problem_key not in [p["key"] for p in current_problems]:
            problem_class = directory.problem_directory[problem_key]
            problem = problem_class()
            current_problems.append(
                {"key": problem_key, "name": problem.name, "instance": problem}
            )
        return jsonify({"success": True, "problems": current_problems})

    return jsonify({"success": False, "error": "Problem not found"}), 400


@app.route("/api/add-solver", methods=["POST"])
def add_solver():
    """Add a solver to the current workspace."""
    global current_solvers
    solver_key = request.json.get("solver_key")

    if solver_key in directory.solver_directory:
        if solver_key not in [s["key"] for s in current_solvers]:
            solver_class = directory.solver_directory[solver_key]
            solver = solver_class()
            current_solvers.append(
                {"key": solver_key, "name": solver.name, "instance": solver}
            )
        return jsonify({"success": True, "solvers": current_solvers})

    return jsonify({"success": False, "error": "Solver not found"}), 400


@app.route("/api/remove-problem", methods=["POST"])
def remove_problem():
    """Remove a problem from the current workspace."""
    global current_problems
    problem_key = request.json.get("problem_key")
    current_problems = [p for p in current_problems if p["key"] != problem_key]
    return jsonify({"success": True, "problems": current_problems})


@app.route("/api/remove-solver", methods=["POST"])
def remove_solver():
    """Remove a solver from the current workspace."""
    global current_solvers
    solver_key = request.json.get("solver_key")
    current_solvers = [s for s in current_solvers if s["key"] != solver_key]
    return jsonify({"success": True, "solvers": current_solvers})


@app.route("/api/create-experiment", methods=["POST"])
def create_experiment():
    """Create a new experiment."""
    global experiments
    data = request.json

    experiment_name = data.get("experiment_name", "Experiment")
    design_type = data.get("design_type", "nolhs")
    num_stacks = int(data.get("num_stacks", 1))

    if not current_problems or not current_solvers:
        return jsonify(
            {"success": False, "error": "Need at least one problem and one solver"}
        ), 400

    # Create experiment
    experiment = {
        "name": experiment_name,
        "problems": current_problems.copy(),
        "solvers": current_solvers.copy(),
        "design_type": design_type,
        "num_stacks": num_stacks,
        "created": True,
    }

    experiments.append(experiment)

    return jsonify({"success": True, "experiment": experiment})


@app.route("/api/clear-workspace", methods=["POST"])
def clear_workspace():
    """Clear the current workspace."""
    global current_problems, current_solvers
    current_problems = []
    current_solvers = []
    return jsonify({"success": True})


@app.route("/components/workspace")
def workspace_component():
    """Return updated workspace component."""
    return render_template(
        "components/workspace.html", problems=current_problems, solvers=current_solvers
    )


@app.route("/components/experiments-list")
def experiments_list_component():
    """Return updated experiments list component."""
    return render_template("components/experiments_list.html", experiments=experiments)


def main():
    """Run the web GUI."""
    # Parse command line
    log_level = logging.INFO
    debug = False

    for arg in sys.argv:
        if arg == "--debug":
            log_level = logging.DEBUG
            debug = True
            break
        if arg == "--silent":
            log_level = logging.CRITICAL
            break

    debug_format = "%(levelname)s: %(message)s"
    logging.basicConfig(level=log_level, format=debug_format)
    logging.info("Web GUI started")

    # Create templates and static directories if they don't exist
    Path("templates").mkdir(exist_ok=True)
    Path("templates/components").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("static/css").mkdir(exist_ok=True)
    Path("static/js").mkdir(exist_ok=True)

    app.run(debug=debug, host="127.0.0.1", port=6000)


if __name__ == "__main__":
    main()
