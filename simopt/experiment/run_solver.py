def run(self, n_macroreps: int, n_jobs: int = -1) -> None:
    """Runs the solver on the problem for a given number of macroreplications.

    Note:
        RNGs for random problem instances are reserved but currently unused.
        This method is under development.

    Args:
        n_macroreps (int): Number of macroreplications to run.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
            -1: use all available cores
            1: run sequentially

    Raises:
        ValueError: If `n_macroreps` is not positive.
    """
    # Local Imports
    from functools import partial

    # Value checking
    if n_macroreps <= 0:
        error_msg = "Number of macroreplications must be positive."
        raise ValueError(error_msg)

    msg = f"Running Solver {self.solver.name} on Problem {self.problem.name}."
    logging.info(msg)

    # Initialize variables
    self.n_macroreps = n_macroreps
    self.all_recommended_xs = [[] for _ in range(n_macroreps)]
    self.all_intermediate_budgets = [[] for _ in range(n_macroreps)]
    self.timings = [0.0 for _ in range(n_macroreps)]

    # Create, initialize, and attach random number generators
    #     Stream 0: reserved for taking post-replications
    #     Stream 1: reserved for bootstrapping
    #     Stream 2: reserved for overhead ...
    #         Substream 0: rng for random problem instance
    #         Substream 1: rng for random initial solution x0 and
    #                      restart solutions
    #         Substream 2: rng for selecting random feasible solutions
    #         Substream 3: rng for solver's internal randomness
    #     Streams 3, 4, ..., n_macroreps + 2: reserved for
    #                                         macroreplications
    # rng0 = MRG32k3a(s_ss_sss_index=[2, 0, 0])  # Currently unused.
    rng_list = [MRG32k3a(s_ss_sss_index=[2, i + 1, 0]) for i in range(3)]
    self.solver.attach_rngs(rng_list)

    # Start a timer
    function_start = time.time()

    logging.debug("Starting macroreplications")

    # Start the macroreplications in parallel (async)
    run_multithread_partial = partial(
        self.run_multithread, solver=self.solver, problem=self.problem
    )

    if n_jobs == 1:
        results: list[tuple] = [run_multithread_partial(i) for i in range(n_macroreps)]
    else:
        results: list[tuple] = Parallel(n_jobs=n_jobs)(
            delayed(run_multithread_partial)(i) for i in range(n_macroreps)
        )  # type: ignore

    for mrep, recommended_xs, intermediate_budgets, timing in results:
        self.all_recommended_xs[mrep] = recommended_xs
        self.all_intermediate_budgets[mrep] = intermediate_budgets
        self.timings[mrep] = timing

    runtime = round(time.time() - function_start, 3)
    logging.info(f"Finished running {n_macroreps} mreps in {runtime} seconds.")

    self.has_run = True
    self.has_postreplicated = False
    self.has_postnormalized = False

    # Save ProblemSolver object to .pickle file if specified.
    if self.create_pickle:
        file_name = self.file_name_path.name
        self.record_experiment_results(file_name=file_name)


def run_multithread(self, mrep: int, solver: Solver, problem: Problem) -> tuple:
    """Runs one macroreplication of the solver on the problem.

    Args:
        mrep (int): Index of the macroreplication.
        solver (Solver): The simulation-optimization solver to run.
        problem (Problem): The problem to solve.

    Returns:
        tuple: A tuple containing:
            - int: Macroreplication index.
            - list: Recommended solutions.
            - list: Intermediate budgets.
            - float: Runtime for the macroreplication.

    Raises:
        ValueError: If `mrep` is negative.
    """
    # Value checking
    if mrep < 0:
        error_msg = "Macroreplication index must be non-negative."
        raise ValueError(error_msg)

    logging.debug(
        f"Macroreplication {mrep + 1}: "
        f"Starting Solver {solver.name} on Problem {problem.name}."
    )
    # Create, initialize, and attach RNGs used for simulating solutions.
    progenitor_rngs = [
        MRG32k3a(s_ss_sss_index=[mrep + 3, ss, 0]) for ss in range(problem.model.n_rngs)
    ]
    # Create a new set of RNGs for the solver based on the current macroreplication.
    # Tried re-using the progentior RNGs, but we need to match the number needed by
    # the solver, not the problem
    solver_rngs = [
        MRG32k3a(
            s_ss_sss_index=[
                mrep + 3,
                problem.model.n_rngs + rng_index,
                0,
            ]
        )
        for rng_index in range(len(solver.rng_list))
    ]

    # Set progenitor_rngs and rng_list for solver.
    solver.solution_progenitor_rngs = progenitor_rngs
    solver.rng_list = solver_rngs

    # logging.debug([rng.s_ss_sss_index for rng in progenitor_rngs])
    # Run the solver on the problem.
    tic = time.perf_counter()
    recommended_solns, intermediate_budgets = solver.run(problem=problem)
    toc = time.perf_counter()
    runtime = toc - tic
    logging.debug(
        f"Macroreplication {mrep + 1}: "
        f"Finished Solver {solver.name} on Problem {problem.name} "
        f"in {runtime:0.4f} seconds."
    )

    # Trim the recommended solutions and intermediate budgets
    recommended_solns, intermediate_budgets = trim_solver_results(
        problem=problem,
        recommended_solutions=recommended_solns,
        intermediate_budgets=intermediate_budgets,
    )
    # Sometimes we end up with numpy scalar values in the solutions,
    # so we convert them to Python scalars. This is especially problematic
    # when trying to dump the solutions to human-readable files as numpy
    # scalars just spit out binary data.
    # TODO: figure out where numpy scalars are coming from and fix it
    solutions = [tuple([float(x) for x in soln.x]) for soln in recommended_solns]
    # Return tuple (rec_solns, int_budgets, runtime)
    return (
        mrep,
        solutions,
        intermediate_budgets,
        runtime,
    )
