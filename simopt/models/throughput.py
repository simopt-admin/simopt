"""
Summary
-------
Simulate throughput maximizaion.
A detailed description of the problem can be found `here `

"""
import numpy as np

from base import Model, Problem


class Throughput(Model):
    """
    A model that simulates a working station with an n-stage flow line and a finite buffer storage, 
    an infinite number of jobs in front of Station. A single server at each station with 
    an Exponential(x) service time. Returns

        - the time width between last 50 jobs 
        - the throughput of the process

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
    fixed_factors : nested dict
        fixed factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors={}):
        self.name = "Thput"
        self.n_rngs = 3
        self.n_responses = 3
        self.specifications = {
            "lambda": {
                "description": "Parameter of the buffer allocation \
                                distribution.",
                "datatype": float,
                "default": 1.5
            },
            "mu": {
                "description": "Rate parameter of service time \
                                distribution.",
                "datatype": float,
                "default": 3.0
            },
            "warmup": {
                "description": "Number of people as warmup before \
                                collecting statistics",
                "datatype": int,
                "default": 2000
            },
            "n": {
                "description": "The number of the station \
                                distribution.",
                "datatype": int,
                "default": 3
            },
            "jobs": {
                "description": "Number required for the  \
                                next 50 jobs",
                "datatype": int,
                "default": 50
            }
        }
        self.check_factor_list = {
            "lambda": self.check_lambda,
            "mu": self.check_mu,
            "warmup": self.check_warmup,
            "n": self.check_n,
            "jobs": self.check_jobs
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_lambda(self):
        return self.factors["lambda"] > 0

    def check_mu(self):
        return self.factors["mu"] > 0

    def check_warmup(self):
        return self.factors["warmup"] >= 0

    def check_jobs(self):
        return self.factors["jobs"] >= 1
    
    def check_n(self):
        return self.factors["n"] >= 1

    def check_simulatable_factors(self):
        # demo for condition that queue must be stable
        # return self.factors["mu"] > self.factors["lambda"]
        return True

    
    def replicate(rng, runlength, processing_rate):
        """
        Simulate a single replication for the current model factors.

        Arguments
        ---------
        rng : mrg32k3a.MRG32k3a
            rng for model to use when simulating a replication
        runlength : float
            how long (in hours) to run a single replication of the model
        processing_rate : float
            rate parameter lambda for the exponential distribution used
            to generate random processing times for three stations.

        Returns
        -------
        responses : dict
            performance measures of interest
            "avg_waiting_time" = average waiting time
        WIP_values : numpy array of ints
            List of work-in-process (WIP) values as it changes.
        WIP_times : numpy array of floats
            List of times at which the WIP changes.
        throughput : float
            total number of parts produced
        """
        
        service_rng = rng
        station1_ptime = rng
        buffer_size = rng
        
        terminate = False

    # For each part, track 4 quantities:
    #   - Time at which the part begins processing at Station 1 (i.e., enters the system).
    #   - Time at which the part ends processing at Station 1.
    #   - Time at which the part begins processing at Station 2
    #   - Time at which the part ends processing at Station 2 (i.e., exits the system).
    part_times = []

    # Corresponds to first part    
    part_number = 0

    # Store these in a list of lists: outer level = each part, inner level = 4 values.

    # When we encounter a part that enters the system after the end of the horizon,
    # we terminate the simulation.
    while not terminate:

        if part_number == 0:
            # Record the first part's experience.
            begin_proc_station1 = 0
            end_proc_station1 = station1_ptime
            begin_proc_station2 = station1_ptime
            end_proc_station2 = station1_ptime + rng.expovariate(processing_rate)
            parts_experience = [begin_proc_station1, end_proc_station1, begin_proc_station2, end_proc_station2]
            part_times.append(parts_experience)

        else:
            # Record the part's experience.

            if part_number <= buffer_size + 1:
                # If one of the first few parts, blocking is not possible.
                # Part begins processing at first time when Station 1 ends processing of previous part
                begin_proc_station1 = part_times[part_number - 1][1]

            else:
                # Part begins processing at first time when Station 1 ends processing of previous part
                # AND becomes unblocked.
                # Previous part completes service
                #   = part_times[part_number - 1][1].
                # Station 1 becomes unblocked when a part sufficiently in front of the current part
                # starts processing at Station 2.            
                #   = part_times[part_number - buffer_size - 1][2]
                begin_proc_station1 = max(part_times[part_number - 1][1], part_times[part_number - buffer_size - 1][2])

            # Part ends processing at Station 1 at first time when processing time is up.
            end_proc_station1 = begin_proc_station1 + station1_ptime

            # Part begins processing at Station 2 when it finishes processing at Station 1 AND 
            # previous part has completed processing at Station 2.
            begin_proc_station2 = max(end_proc_station1, part_times[part_number - 1][3])

            # Part ends processing at Station 2 when processing time is up
            end_proc_station2 = begin_proc_station2 + rng.expovariate(processing_rate)

            # Concatenate results
            parts_experience = [begin_proc_station1, end_proc_station1, begin_proc_station2, end_proc_station2]
            part_times.append(parts_experience)

        # If we have passed the time horizon, terminate.
        if parts_experience[0] > runlength:
            terminate = True

        # IF YOU WANT TO TRACK THE PER-PART STATISTICS, UNCOMMENT THIS.
        # print(f"Part {part_number}: {parts_experience}")

        # Move on the next part.
        part_number += 1

    # Record summary statistics.
    enter_times = np.array([parts_experience[0] for parts_experience in part_times if parts_experience[0] < runlength])
    exit_times = np.array([parts_experience[3] for parts_experience in part_times if parts_experience[3] < runlength])

    # Calculate throughput.
    throughput = np.sum(exit_times < runlength)

    # Construct the WIP trajectory.
    WIP_change_times = np.concatenate((enter_times, exit_times))
    WIP_increments = np.concatenate((np.ones(len(enter_times)), -np.ones(len(exit_times))))
    # (Sort the WIP change times and make corresponding swaps to the increments array.)
    ordering = np.argsort(WIP_change_times)
    WIP_times = np.sort(WIP_change_times)
    sorted_WIP_increments = WIP_increments[ordering]
    WIP_values = np.cumsum(sorted_WIP_increments)
    
    return WIP_values, WIP_times, throughput



"""
Summary
-------
Minimize the mean sojourn time of an M/M/1 queue.
"""


class MM1MinMeanSojournTime(Problem):
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
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)

    Arguments
    ---------
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed_factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="MM1-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 1
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.lower_bounds = (0,)
        self.upper_bounds = (np.inf,)
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None  # (2.75,)
        self.model_default_factors = {
            "warmup": 50,
            "people": 200
        }
        self.model_decision_variables = {"mu"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (5,)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 1000
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = Throughput(self.model_fixed_factors)

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
        factor_dict = {
            "mu": vector[0]
        }
        return factor_dict

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
        vector = (factor_dict["mu"],)
        return vector

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
        objectives = (response_dict["avg_sojourn_time"],)
        return objectives

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
        stoch_constraints = (response_dict["frac_cust_wait"],)
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
        det_objectives = (0.1 * (x[0]**2),)
        det_objectives_gradients = ((0.2 * x[0],),)
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
        det_stoch_constraints = (0.5,)
        det_stoch_constraints_gradients = ((0,),)
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
        return x[0] > 0

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
        # Generate an Exponential(rate = 1/3) r.v.
        x = (rand_sol_rng.expovariate(1 / 3),)
        return x
