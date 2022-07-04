#!/usr/bin/env python
"""
Summary
-------
Provide base classes for solvers, problems, and models.

Listing
-------
Solver : class
Problem : class
Model : class
Solution : class
"""

import numpy as np
from copy import deepcopy


from rng.mrg32k3a import MRG32k3a


class Solver(object):
    """
    Base class to implement simulation-optimization solvers.

    Attributes
    ----------
    name : string
        name of solver
    objective_type : string
        description of objective types:
            "single" or "multi"
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_needed : bool
        indicates if gradient of objective function is needed
    factors : dict
        changeable factors (i.e., parameters) of the solver
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    rng_list : list of rng.MRG32k3a objects
        list of RNGs used for the solver's internal purposes
    solution_progenitor_rngs : list of rng.MRG32k3a objects
        list of RNGs used as a baseline for simulating solutions

    Arguments
    ---------
    fixed_factors : dict
        dictionary of user-specified solver factors
    """
    def __init__(self, fixed_factors):
        # Set factors of the solver.
        # Fill in missing factors with default values.
        self.factors = fixed_factors
        for key in self.specifications:
            if key not in fixed_factors:
                self.factors[key] = self.specifications[key]["default"]

    def __eq__(self, other):
        """
        Check if two solvers are equivalent.

        Arguments
        ---------
        other : base.Solver object
            other Solver object to compare to self

        Returns
        -------
        bool
            Are the two solvers equivalent?
        """
        if type(self) == type(other):
            if self.factors == other.factors:
                return True
            else:
                # print("Solver factors do not match.")
                return False
        else:
            # print("Solver types do not match.")
            return False

    def attach_rngs(self, rng_list):
        """
        Attach a list of random-number generators to the solver.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            list of random-number generators used for the solver's internal purposes
        """
        self.rng_list = rng_list

    def solve(self, problem):
        """
        Run a single macroreplication of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve

        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """
        raise NotImplementedError

    def check_crn_across_solns(self):
        return True

    def check_solver_factor(self, factor_name):
        """
        Determine if the setting of a solver factor is permissible.

        Arguments
        ---------
        factor_name : string
            name of factor for dictionary lookup (i.e., key)

        Returns
        -------
        is_permissible : bool
            indicates if solver factor is permissible
        """
        is_permissible = True
        is_permissible *= self.check_factor_datatype(factor_name)
        is_permissible *= self.check_factor_list[factor_name]()
        return is_permissible
        # raise NotImplementedError

    def check_solver_factors(self):
        """
        Determine if the joint settings of solver factors are permissible.

        Returns
        -------
        is_simulatable : bool
            indicates if solver factors are permissible
        """
        return True
        # raise NotImplementedError

    def check_factor_datatype(self, factor_name):
        """
        Determine if a factor's data type matches its specification.

        Arguments
        ---------
        factor_name : string
            string corresponding to name of factor to check

        Returns
        -------
        is_right_type : bool
            indicates if factor is of specified data type
        """
        is_right_type = isinstance(self.factors[factor_name], self.specifications[factor_name]["datatype"])
        return is_right_type

    def create_new_solution(self, x, problem):
        """
        Create a new solution object with attached rngs primed
        to simulate replications.

        Arguments
        ---------
        x : tuple
            vector of decision variables
        problem : base.Problem object
            problem being solved by the solvers

        Returns
        -------
        new_solution : base.Solution object
            new solution
        """
        # Create new solution with attached rngs.
        new_solution = Solution(x, problem)
        new_solution.attach_rngs(rng_list=self.solution_progenitor_rngs, copy=True)
        # Manipulate progenitor rngs to prepare for next new solution.
        if not self.factors["crn_across_solns"]:  # If CRN are not used ...
            # ...advance each rng to start of the substream = current substream + # of model RNGs.
            for rng in self.solution_progenitor_rngs:
                for _ in range(problem.model.n_rngs):
                    rng.advance_substream()
        return new_solution

    def rebase(self, n_reps):
        """
        Rebase the progenitor rngs to start at a later subsubstream index.

        Arguments
        ---------
        n_reps : int >= 0
            substream index to skip to
        """
        new_rngs = []
        for rng in self.solution_progenitor_rngs:
            stream_index = rng.s_ss_sss_index[0]
            substream_index = rng.s_ss_sss_index[1]
            new_rngs.append(MRG32k3a(s_ss_sss_index=[stream_index, substream_index, n_reps]))
        self.solution_progenitor_rngs = new_rngs


class Problem(object):
    """
    Base class to implement simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of problem
    dim : int
        number of decision variables
    n_objectives : int
        number of objectives
    n_stochastic_constraints : int
        number of stochastic constraints
    minmax : tuple of int (+/- 1)
        indicator of maximization (+1) or minimization (-1) for each objective
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    lower_bounds : tuple
        lower bound for each decision variable
    upper_bounds : tuple
        upper bound for each decision variable
    gradient_available : bool
        indicates if gradient of objective function is available
    optimal_value : float
        optimal objective function value
    optimal_solution : tuple
        optimal solution
    model : Model object
        associated simulation model that generates replications
    model_default_factors : dict
        default values for overriding model-level default factors
    model_fixed_factors : dict
        combination of overriden model-level factors and defaults
    model_decision_factors : set of str
        set of keys for factors that are decision variables
    rng_list : list of rng.MRG32k3a objects
        list of RNGs used to generate a random initial solution
        or a random problem instance
    factors : dict
        changeable factors of the problem
            initial_solution : tuple
                default initial solution from which solvers start
            budget : int > 0
                max number of replications (fn evals) for a solver to take
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed_factors : dict
        subset of user-specified non-decision factors to pass through to the model
    """
    def __init__(self, fixed_factors, model_fixed_factors):
        # Set factors of the problem.
        # Fill in missing factors with default values.
        self.factors = fixed_factors
        for key in self.specifications:
            if key not in fixed_factors:
                self.factors[key] = self.specifications[key]["default"]
        # Set subset of factors of the simulation model.
        # Fill in missing model factors with problem-level default values.
        for key in self.model_default_factors:
            if key not in model_fixed_factors:
                model_fixed_factors[key] = self.model_default_factors[key]
        self.model_fixed_factors = model_fixed_factors
        # super().__init__()

    def __eq__(self, other):
        """
        Check if two problems are equivalent.

        Arguments
        ---------
        other : base.Problem object
            other Problem object to compare to self

        Returns
        -------
        bool
            Are the two problems equivalent?
        """
        if type(self) == type(other):
            if self.factors == other.factors:
                # Check if non-decision-variable factors of models are the same.
                non_decision_factors = set(self.model.factors.keys()) - self.model_decision_factors
                for factor in non_decision_factors:
                    if self.model.factors[factor] != other.model.factors[factor]:
                        # print("Model factors do not match")
                        return False
                return True
            else:
                # print("Problem factors do not match.")
                return False
        else:
            # print("Problem types do not match.")
            return False

    def check_initial_solution(self):
        if len(self.factors["initial_solution"]) != self.dim:
            return False
        elif not self.check_deterministic_constraints(x=self.factors["initial_solution"]):
            return False
        else:
            return True

    def check_budget(self):
        return self.factors["budget"] > 0

    def check_problem_factor(self, factor_name):
        """
        Determine if the setting of a problem factor is permissible.

        Arguments
        ---------
        factor_name : string
            name of factor for dictionary lookup (i.e., key)

        Returns
        -------
        is_permissible : bool
            indicates if problem factor is permissible
        """
        is_permissible = True
        is_permissible *= self.check_factor_datatype(factor_name)
        is_permissible *= self.check_factor_list[factor_name]()
        return is_permissible
        # raise NotImplementedError

    def check_problem_factors(self):
        """
        Determine if the joint settings of problem factors are permissible.

        Returns
        -------
        is_simulatable : bool
            indicates if problem factors are permissible
        """
        return True
        # raise NotImplementedError

    def check_factor_datatype(self, factor_name):
        """
        Determine if a factor's data type matches its specification.

        Arguments
        ---------
        factor_name : string
            string corresponding to name of factor to check

        Returns
        -------
        is_right_type : bool
            indicates if factor is of specified data type
        """
        is_right_type = isinstance(self.factors[factor_name], self.specifications[factor_name]["datatype"])
        return is_right_type

    def attach_rngs(self, rng_list):
        """
        Attach a list of random-number generators to the problem.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            list of random-number generators used to generate a random initial solution
            or a random problem instance
        """
        self.rng_list = rng_list

    def vector_to_factor_dict(self, vector):
        """
        Convert a vector of variables to a dictionary with factor keys

        Arguments
        ---------
        vector : tuple
            vector of values associated with decision variables

        Returns
        -------
        factor_dict : dictionary
            dictionary with factor keys and associated values
        """
        raise NotImplementedError

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert a dictionary with factor keys to a vector
        of variables.

        Arguments
        ---------
        factor_dict : dictionary
            dictionary with factor keys and associated values

        Returns
        -------
        vector : tuple
            vector of values associated with decision variables
        """
        raise NotImplementedError

    def response_dict_to_objectives(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of objectives.

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        objectives : tuple
            vector of objectives
        """
        raise NotImplementedError

    def response_dict_to_stoch_constraints(self, response_dict):
        """
        Convert a dictionary with response keys to a vector
        of left-hand sides of stochastic constraints: E[Y] >= 0

        Arguments
        ---------
        response_dict : dictionary
            dictionary with response keys and associated values

        Returns
        -------
        stoch_constraints : tuple
            vector of LHSs of stochastic constraint
        """
        stoch_constraints = ()
        return stoch_constraints

    def deterministic_objectives_and_gradients(self, x):
        """
        Compute deterministic components of objectives for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_objectives : tuple
            vector of deterministic components of objectives
        det_objectives_gradients : tuple
            vector of gradients of deterministic components of objectives
        """
        det_objectives = (0,) * self.n_objectives
        det_objectives_gradients = tuple([(0,) * self.dim for _ in range(self.n_objectives)])
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints
        for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic
            constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = (0,) * self.n_stochastic_constraints
        det_stoch_constraints_gradients = tuple([(0,) * self.dim for _ in range(self.n_stochastic_constraints)])
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic
        constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        # Check box constraints.
        return bool(np.prod([self.lower_bounds[idx] <= x[idx] <= self.upper_bounds[idx] for idx in range(len(x))]))

    def get_random_solution(self, rand_sol_rng):
        """
        Generate a random solution for starting or restarting solvers.

        Arguments
        ---------
        rand_sol_rng : rng.MRG32k3a object
            random-number generator used to sample a new random solution

        Returns
        -------
        x : tuple
            vector of decision variables
        """
        pass

    def simulate(self, solution, m=1):
        """
        Simulate `m` i.i.d. replications at solution `x`.

        Arguments
        ---------
        solution : Solution object
            solution to evalaute
        m : int
            number of replications to simulate at `x`
        """
        if m < 1:
            print('--* Error: Number of replications must be at least 1. ')
            print('--* Aborting. ')
        else:
            # pad numpy arrays if necessary
            if solution.n_reps + m > solution.storage_size:
                solution.pad_storage(m)
            # set the decision factors of the model
            self.model.factors.update(solution.decision_factors)
            for _ in range(m):
                # generate one replication at x
                responses, gradients = self.model.replicate(solution.rng_list)
                # convert gradient subdictionaries to vectors mapping to decision variables
                # TEMPORARILY COMMENT OUT GRADIENTS
                # vector_gradients = {keys: self.factor_dict_to_vector(gradient_dict) for (keys, gradient_dict) in gradients.items()}
                # convert responses and gradients to objectives and gradients and add
                # to those of deterministic components of objectives
                solution.objectives[solution.n_reps] = [sum(pairs) for pairs in zip(self.response_dict_to_objectives(responses), solution.det_objectives)]
                # solution.objectives_gradients[solution.n_reps] = [[sum(pairs) for pairs in zip(stoch_obj, det_obj)] for stoch_obj, det_obj in zip(self.response_dict_to_objectives(vector_gradients), solution.det_objectives_gradients)]
                if self.n_stochastic_constraints > 0:
                    # convert responses and gradients to stochastic constraints and gradients and add
                    # to those of deterministic components of stochastic constraints
                    solution.stoch_constraints[solution.n_reps] = [sum(pairs) for pairs in zip(self.response_dict_to_stoch_constraints(responses), solution.det_stoch_constraints)]
                    # solution.stoch_constraints_gradients[solution.n_reps] = [[sum(pairs) for pairs in zip(stoch_stoch_cons, det_stoch_cons)] for stoch_stoch_cons, det_stoch_cons in zip(self.response_dict_to_stoch_constraints(vector_gradients), solution.det_stoch_constraints_gradients)]
                # increment counter
                solution.n_reps += 1
                # advance rngs to start of next subsubstream
                for rng in solution.rng_list:
                    rng.advance_subsubstream()
            # update summary statistics
            solution.recompute_summary_statistics()

    def simulate_up_to(self, solutions, n_reps):
        """
        Simulate a set of solutions up to a given number of replications.

        Arguments
        ---------
        solutions : set
            a set of base.Solution objects
        n_reps : int > 0
            common number of replications to simulate each solution up to
        """
        for solution in solutions:
            # If more replications needed, take them.
            if solution.n_reps < n_reps:
                n_reps_to_take = n_reps - solution.n_reps
                self.simulate(solution=solution, m=n_reps_to_take)


class Model(object):
    """
    Base class to implement simulation models (models) featured in
    simulation-optimization problems.

    Attributes
    ----------
    name : string
        name of model
    n_rngs : int
        number of random-number generators used to run a simulation replication
    n_responses : int
        number of responses (performance measures)
    factors : dict
        changeable factors of the simulation model
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    check_factor_list : dict
        switch case for checking factor simulatability

    Arguments
    ---------
    fixed_factors : dict
        dictionary of user-specified model factors
    """
    def __init__(self, fixed_factors):
        # set factors of the simulation model
        # fill in missing factors with default values
        self.factors = fixed_factors
        for key in self.specifications:
            if key not in fixed_factors:
                self.factors[key] = self.specifications[key]["default"]

    def __eq__(self, other):
        """
        Check if two models are equivalent.

        Arguments
        ---------
        other : base.Model object
            other Model object to compare to self

        Returns
        -------
        bool
            Are the two models equivalent?
        """
        if type(self) == type(other):
            if self.factors == other.factors:
                return True
            else:
                # print("Model factors do not match.")
                return False
        else:
            # print("Model types do not match.")
            return False

    def check_simulatable_factor(self, factor_name):
        """
        Determine if a simulation replication can be run with the given factor.

        Arguments
        ---------
        factor_name : string
            name of factor for dictionary lookup (i.e., key)

        Returns
        -------
        is_simulatable : bool
            indicates if model specified by factors is simulatable
        """
        is_simulatable = True
        is_simulatable *= self.check_factor_datatype(factor_name)
        is_simulatable *= self.check_factor_list[factor_name]()
        return is_simulatable
        # raise NotImplementedError

    def check_simulatable_factors(self):
        """
        Determine if a simulation replication can be run with the given factors.

        Returns
        -------
        is_simulatable : bool
            indicates if model specified by factors is simulatable
        """
        return True
        # raise NotImplementedError

    def check_factor_datatype(self, factor_name):
        """
        Determine if a factor's data type matches its specification.

        Returns
        -------
        is_right_type : bool
            indicates if factor is of specified data type
        """
        is_right_type = isinstance(self.factors[factor_name], self.specifications[factor_name]["datatype"])
        return is_right_type

    def replicate(self, rng_list):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            rngs for model to use when simulating a replication

        Returns
        -------
        responses : dict
            performance measures of interest
        gradients : dict of dicts
            gradient estimate for each response
        """
        raise NotImplementedError


class Solution(object):
    """
    Base class for solutions represented as vectors of decision variables
    and dictionaries of decision factors.

    Attributes
    ----------
    x : tuple
        vector of decision variables
    dim : int
        number of decision variables describing `x`
    decision_factors : dict
        decision factor names and values
    rng_list : list of rng.MRG32k3a objects
        rngs for model to use when running replications at the solution
    n_reps : int
        number of replications run at the solution
    det_objectives : tuple
        deterministic components added to objectives
    det_objectives_gradients : tuple of tuples (# objectives x dimension)
        gradients of deterministic components added to objectives
    det_stoch_constraints : tuple
        deterministic components added to LHS of stochastic constraints
    det_stoch_constraints_gradients : tuple (# stochastic constraints x dimension)
        gradients of deterministics components added to LHS stochastic constraints
    storage_size : int
        max number of replications that can be recorded in current storage
    objectives : numpy array (# replications x # objectives)
        objective(s) estimates from each replication
    objectives_gradients : numpy array (# replications x # objectives x dimension)
        gradient estimates of objective(s) from each replication
    stochastic_constraints : numpy array (# replications x # stochastic constraints)
        stochastic constraint estimates from each replication
    stochastic_constraints_gradients : numpy array (# replications x # stochastic constraints x dimension)
        gradient estimates of stochastic constraints from each replication

    Arguments
    ---------
    x : tuple
        vector of decision variables
    problem : Problem object
        problem to which x is a solution
    """
    def __init__(self, x, problem):
        super().__init__()
        self.x = x
        self.dim = len(x)
        self.decision_factors = problem.vector_to_factor_dict(x)
        self.n_reps = 0
        self.det_objectives, self.det_objectives_gradients = problem.deterministic_objectives_and_gradients(self.x)
        self.det_stoch_constraints, self.det_stoch_constraints_gradients = problem.deterministic_stochastic_constraints_and_gradients(self.x)
        init_size = 100  # initialize numpy arrays to store up to 100 replications
        self.storage_size = init_size
        # Raw data
        self.objectives = np.zeros((init_size, problem.n_objectives))
        self.objectives_gradients = np.zeros((init_size, problem.n_objectives, problem.dim))
        if problem.n_stochastic_constraints > 0:
            self.stoch_constraints = np.zeros((init_size, problem.n_stochastic_constraints))
            self.stoch_constraints_gradients = np.zeros((init_size, problem.n_stochastic_constraints, problem.dim))
        else:
            self.stoch_constraints = None
            self.stoch_constraints_gradients = None
        # Summary statistics
        # self.objectives_mean = np.full((problem.n_objectives), np.nan)
        # self.objectives_var = np.full((problem.n_objectives), np.nan)
        # self.objectives_stderr = np.full((problem.n_objectives), np.nan)
        # self.objectives_cov = np.full((problem.n_objectives, problem.n_objectives), np.nan)
        # self.objectives_gradients_mean = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_var = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_stderr = np.full((problem.n_objectives, problem.dim), np.nan)
        # self.objectives_gradients_cov = np.full((problem.n_objectives, problem.dim, problem.dim), np.nan)
        # self.stoch_constraints_mean = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_var = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_stderr = np.full((problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_cov = np.full((problem.n_stochastic_constraints, problem.n_stochastic_constraints), np.nan)
        # self.stoch_constraints_gradients_mean = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_var = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_stderr = np.full((problem.n_stochastic_constraints, problem.dim), np.nan)
        # self.stoch_constraints_gradients_cov = np.full((problem.n_stochastic_constraints, problem.dim, problem.dim), np.nan)

    def attach_rngs(self, rng_list, copy=True):
        """
        Attach a list of random-number generators to the solution.

        Arguments
        ---------
        rng_list : list of rng.MRG32k3a objects
            list of random-number generators used to run simulation replications
        """
        if copy:
            self.rng_list = [deepcopy(rng) for rng in rng_list]
        else:
            self.rng_list = rng_list

    def pad_storage(self, m):
        """
        Append zeros to numpy arrays for summary statistics.

        Arguments
        ---------
        m : int
            number of replications to simulate
        """
        # Size of data storage
        n_objectives = len(self.det_objectives)
        base_pad_size = 100  # default is to append space for 100 more replications
        # if more space needed, append in multiples of 100
        pad_size = int(np.ceil(m / base_pad_size)) * base_pad_size
        self.storage_size += pad_size
        self.objectives = np.concatenate((self.objectives, np.zeros((pad_size, n_objectives))))
        self.objectives_gradients = np.concatenate((self.objectives_gradients, np.zeros((pad_size, n_objectives, self.dim))))
        if self.stoch_constraints is not None:
            n_stochastic_constraints = len(self.det_stoch_constraints)
            self.stoch_constraints = np.concatenate((self.stoch_constraints, np.zeros((pad_size, n_stochastic_constraints))))
            self.stoch_constraints_gradients = np.concatenate((self.stoch_constraints_gradients, np.zeros((pad_size, n_stochastic_constraints, self.dim))))

    def recompute_summary_statistics(self):
        """
        Recompute summary statistics of the solution.
        """
        self.objectives_mean = np.mean(self.objectives[:self.n_reps], axis=0)
        if self.n_reps > 1:
            self.objectives_var = np.var(self.objectives[:self.n_reps], axis=0, ddof=1)
            self.objectives_stderr = np.std(self.objectives[:self.n_reps], axis=0, ddof=1) / np.sqrt(self.n_reps)
            self.objectives_cov = np.cov(self.objectives[:self.n_reps], rowvar=False, ddof=1)
        # TEMPORARILY COMMENT OUT GRADIENTS
        # self.objectives_gradients_mean = np.mean(self.objectives_gradients[:self.n_reps], axis=0)
        # if self.n_reps > 1:
            # self.objectives_gradients_var = np.var(self.objectives_gradients[:self.n_reps], axis=0, ddof=1)
            # self.objectives_gradients_stderr = np.std(self.objectives_gradients[:self.n_reps], axis=0, ddof=1) / np.sqrt(self.n_reps)
            # self.objectives_gradients_cov = np.array([np.cov(self.objectives_gradients[:self.n_reps, obj], rowvar=False, ddof=1) for obj in range(len(self.det_objectives))])
        if self.stoch_constraints is not None:
            self.stoch_constraints_mean = np.mean(self.stoch_constraints[:self.n_reps], axis=0)
            self.stoch_constraints_var = np.var(self.stoch_constraints[:self.n_reps], axis=0, ddof=1)
            self.stoch_constraints_stderr = np.std(self.stoch_constraints[:self.n_reps], axis=0, ddof=1) / np.sqrt(self.n_reps)
            self.stoch_constraints_cov = np.cov(self.stoch_constraints[:self.n_reps], rowvar=False, ddof=1)
            # self.stoch_constraints_gradients_mean = np.mean(self.stoch_constraints_gradients[:self.n_reps], axis=0)
            # self.stoch_constraints_gradients_var = np.var(self.stoch_constraints_gradients[:self.n_reps], axis=0, ddof=1)
            # self.stoch_constraints_gradients_stderr = np.std(self.stoch_constraints_gradients[:self.n_reps], axis=0, ddof=1) / np.sqrt(self.n_reps)
            # self.stoch_constraints_gradients_cov = np.array([np.cov(self.stoch_constraints_gradients[:self.n_reps, stcon], rowvar=False, ddof=1) for stcon in range(len(self.det_stoch_constraints))])
