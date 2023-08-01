"""
Summary
-------
Sequentially searches each coordinate axis for a local minimum.
Can handle stochastic constraints.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/randomsearch.html>`_.
"""
from ..base import Solver


class CoordinateSearch(Solver):
    """
    A solver that sequentially searches each coordinate axis for a local optimum.
    Takes Nk replications at each solution.

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
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used for the solver's internal purposes

    Arguments
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver

    See also
    --------
    base.Solver
    """
    def __init__(self, name="COORDSRCH", fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "discrete"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "ini_sample_size": {
                "description": "initial sample size",
                "datatype": int,
                "default": 10
            },
            "sample_size_slope": {
                "description": "sample size increment per iteration",
                "datatype": int,
                "default": 1
            },
            "m0": {
                "description": "2^m0 is the maximum step size the line search may take",
                "datatype": int,
                "default": 4
            },
            "zmax": {
                "description": "zmax controls the maximum distance from x^*_k-1 and z ≤ zmax+2^m0",
                "datatype": int,
                "default": 50
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "ini_sample_size": self.check_ini_sample_size,
            "sample_size_slope": self.check_sample_size_slope,
            "m0": self.check_m0,
            "zmax": self.check_zmax
        }
        super().__init__(fixed_factors)

    def check_ini_sample_size(self):
        return self.factors["ini_sample_size"] > 0
    
    def check_sample_size_slope(self):
        return self.factors["sample_size_slope"] >= 0
    
    def check_m0(self):
        return self.factors["m0"] >= 0
    
    def check_zmax(self):
        return self.factors["zmax"] > 0

    def update_tuple(self, x, i, delta_x):
        x = list(x)
        x[i] += delta_x
        return tuple(x)
    
    def find_in_visited(self, x, visited_solns):
        for i in range(len(visited_solns)):
            if visited_solns[i].x == x:
                return i
        return -1

    def solve(self, problem):
        """
        Run a single macroreplication of a solver on a problem.

        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        crn_across_solns : bool
            indicates if CRN are used when simulating different solutions

        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(new_x, problem)
        initial_solution = new_solution
        problem.simulate(new_solution, self.factors["ini_sample_size"])
        expended_budget = self.factors["ini_sample_size"]
        intermediate_budgets = [expended_budget]
        recommended_solns = [new_solution]
        visited_solns = [new_solution]
        k = 0
        while expended_budget < problem.factors["budget"]:
            i = k % problem.dim
            Nk = self.factors["ini_sample_size"] + k*self.factors["sample_size_slope"]
            k += 1
            # STEP 0, LINE SEARCH (Hong, 2005).
            minus1_x = self.update_tuple(new_x, i, -1)
            plus1_x = self.update_tuple(new_x, i, 1)
            if (not problem.check_deterministic_constraints(minus1_x)) and (not problem.check_deterministic_constraints(plus1_x)):
                continue
            elif problem.check_deterministic_constraints(plus1_x):
                y = 1
                y_x = plus1_x
            else:
                y = -1
                y_x = minus1_x
            # Update y_solution with the current Nk. Create a new solution if not visited before.
            i_visited = self.find_in_visited(y_x,visited_solns)
            if i_visited >= 0:
                y_solution = visited_solns[i_visited]
            else:
                y_solution = self.create_new_solution(y_x, problem)
                visited_solns.append(y_solution)
            expended_budget += (Nk - y_solution.n_reps)
            problem.simulate_up_to({y_solution},Nk)
            # Update new_solution with the current Nk.
            expended_budget += (Nk - new_solution.n_reps)
            problem.simulate_up_to({new_solution},Nk)
            if problem.minmax * y_solution.objectives_mean > problem.minmax * new_solution.objectives_mean:
                d = y
                z0 = 1
            else:
                d = -y
                z0 = 0
            y0 = d * z0
            # STEP 1, LINE SEARCH (Hong, 2005).
            y0_x = self.update_tuple(new_x, i, y0)
            # Update y0_solution with the current Nk. Create a new solution if necessary.
            i_visited = self.find_in_visited(y0_x,visited_solns)
            if i_visited >= 0:
                y0_solution = visited_solns[i_visited]
            else:
                y0_solution = self.create_new_solution(y0_x, problem)
                visited_solns.append(y0_solution)
            expended_budget += (Nk - y0_solution.n_reps)
            problem.simulate_up_to({y0_solution},Nk)
            y0d_x = self.update_tuple(new_x, i, y0+d)
            # Update y0d_solution with the current Nk. Create a new solution if necessary.
            i_visited = self.find_in_visited(y0d_x,visited_solns)
            if i_visited >= 0:
                y0d_solution = visited_solns[i_visited]
            else:
                y0d_solution = self.create_new_solution(y0d_x, problem)
                visited_solns.append(y0d_solution)
            expended_budget += (Nk - y0d_solution.n_reps)
            problem.simulate_up_to({y0d_solution},Nk)
            if i_visited >= 0 and problem.minmax * y0d_solution.objectives_mean <= problem.minmax * y0_solution.objectives_mean:
                new_x = y0_x
                new_solution = y0_solution
            else:    
                m = self.factors["m0"]
                print("k=",k," i=",i, " new_x=",new_x,sep='',end=' ')
                print(new_solution.objectives_mean,y_solution.objectives_mean,y0d_solution.objectives_mean,y0_solution.objectives_mean)
                print("y=",y," d=",d," z0=",z0," y0=",y0," y_x=",y_x," y0d_x=",y0d_x," y0_x=",y0_x,sep='')
                # STEP 2, LINE SEARCH (Hong, 2005).
                while True:
                    z = z0 + 2**m
                    y = d * z
                    y_x = self.update_tuple(new_x, i, y)
                    print("m=",m," z=",z," y=",y," y_x=",y_x,sep='')
                    if (not problem.check_deterministic_constraints(y_x)) and m == 0:
                        new_x = y0_x
                        new_solution = y0_solution
                        break
                    elif (not problem.check_deterministic_constraints(y_x)) and m > 0:
                        m -= 1
                    else:
                        # Update y_solution with the current Nk. Create a new solution if necessary.
                        i_visited = self.find_in_visited(y_x,visited_solns)
                        if i_visited >= 0:
                            y_solution = visited_solns[i_visited]
                        else:
                            y_solution = self.create_new_solution(y_x, problem)
                            visited_solns.append(y_solution)
                        expended_budget += (Nk - y_solution.n_reps)
                        problem.simulate_up_to({y_solution},Nk)
                        # STEP 3, LINE SEARCH (Hong, 2005).
                        if problem.minmax * y_solution.objectives_mean > problem.minmax * y0_solution.objectives_mean and z < self.factors["zmax"]:
                            z0 = z
                        elif problem.minmax * y_solution.objectives_mean > problem.minmax * y0_solution.objectives_mean and z >= self.factors["zmax"]:
                            new_x = y_x
                            new_solution = y_solution
                            break
                        elif problem.minmax * y_solution.objectives_mean <= problem.minmax * y0_solution.objectives_mean and m == 0:
                            new_x = y0_x
                            new_solution = y0_solution
                            break
                        else:
                            m -= 1
            print("new_x=",new_x,sep='')
            # If problem is not fully constrained, make sure new_solution is not worse than initial_solution.
            if problem.minmax * new_solution.objectives_mean >= problem.minmax * initial_solution.objectives_mean:
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
            else:
                print("The new solution is worse than x0!")
                new_x = self.factors["initial_solution"]
                new_solution = initial_solution
            print()
        print("total expended budget:",expended_budget)
        return recommended_solns, intermediate_budgets