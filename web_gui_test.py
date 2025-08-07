"""Test version of the web GUI that works without full SimOpt dependencies."""

from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

# Mock data for testing
mock_problems = [
    {
        "key": "ASTRO",
        "name": "Astro Problem",
        "objective": "Single",
        "constraint": "Box",
        "variable_type": "Continuous",
        "gradient_available": True,
    },
    {
        "key": "MM1-1",
        "name": "M/M/1 Queue Problem",
        "objective": "Single",
        "constraint": "Unconstrained",
        "variable_type": "Continuous",
        "gradient_available": False,
    },
    {
        "key": "CNTNV",
        "name": "Continuous News Vendor",
        "objective": "Single",
        "constraint": "Box",
        "variable_type": "Continuous",
        "gradient_available": True,
    },
]

mock_solvers = [
    {
        "key": "NELDMD",
        "name": "Nelder-Mead",
        "gradient_needed": False,
        "discrete": False,
        "constraint_type": "box",
    },
    {
        "key": "SPSA",
        "name": "Simultaneous Perturbation SA",
        "gradient_needed": False,
        "discrete": False,
        "constraint_type": "box",
    },
    {
        "key": "ASTRO-DF",
        "name": "ASTRO-DF",
        "gradient_needed": False,
        "discrete": False,
        "constraint_type": "box",
    },
]

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
    problems = [p["key"] for p in mock_problems]
    solvers = [s["key"] for s in mock_solvers]
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
    return jsonify(mock_problems)


@app.route("/api/solvers")
def get_solvers():
    """Get available solvers."""
    return jsonify(mock_solvers)


@app.route("/api/add-problem", methods=["POST"])
def add_problem():
    """Add a problem to the current workspace."""
    global current_problems
    problem_key = request.json.get("problem_key")

    # Find the problem in mock data
    problem_data = next((p for p in mock_problems if p["key"] == problem_key), None)
    if problem_data and problem_key not in [p["key"] for p in current_problems]:
        current_problems.append({"key": problem_key, "name": problem_data["name"]})
        return jsonify({"success": True, "problems": current_problems})

    return jsonify(
        {"success": False, "error": "Problem not found or already added"}
    ), 400


@app.route("/api/add-solver", methods=["POST"])
def add_solver():
    """Add a solver to the current workspace."""
    global current_solvers
    solver_key = request.json.get("solver_key")

    # Find the solver in mock data
    solver_data = next((s for s in mock_solvers if s["key"] == solver_key), None)
    if solver_data and solver_key not in [s["key"] for s in current_solvers]:
        current_solvers.append({"key": solver_key, "name": solver_data["name"]})
        return jsonify({"success": True, "solvers": current_solvers})

    return jsonify(
        {"success": False, "error": "Solver not found or already added"}
    ), 400


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


if __name__ == "__main__":
    print("Starting SimOpt Web GUI Test Server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")

    # Create directories if they don't exist
    Path("templates").mkdir(exist_ok=True)
    Path("templates/components").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("static/css").mkdir(exist_ok=True)
    Path("static/js").mkdir(exist_ok=True)

    app.run(debug=True, host="127.0.0.1", port=6000)
