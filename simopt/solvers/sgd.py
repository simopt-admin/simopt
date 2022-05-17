"""
Summary
-------
Randomly sample solutions from the feasible region.
Can handle stochastic constraints.
"""
from base import Solver
import numpy as np

# how to choose step size, what constraints to handle, need to project back into feasible region


class SGD(Solver):
    """
    A solver that randomly samples solutions from the feasible region.
    Take a fixed number of replications at each solution.

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
    def __init__(self, name="SGD", fixed_factors={}):
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.specifications = {
            "crn_across_solns": {
                "description": "Use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "sample_size": {
                "description": "Sample size per solution",
                "datatype": int,
                "default": 1
            },
            "step_size": {
                "description": "Step size in normalized negative gradient direction",
                "datatype": float,
                "default": 0.1
            },
            'use_fin_diff': {
                "description": "Step size in normalized negative gradient direction",
                "datatype": bool,
                "default": False
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "sample_size": self.check_sample_size
        }
        super().__init__(fixed_factors)
        self.gradient_needed = not self.factors['use_fin_diff']

    def check_sample_size(self):
        return self.factors["sample_size"] > 0

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
        # get problem bounds
        lower_bound = np.array(problem.lower_bounds)
        upper_bound = np.array(problem.upper_bounds)
        
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        # Sequentially generate random solutions and simulate them.
        while expended_budget < problem.factors["budget"]:
            if expended_budget == 0:
                # Start at initial solution and record as best.
                new_x = problem.factors["initial_solution"]
                new_solution = self.create_new_solution(new_x, problem)
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
            else:
                
                if self.factors['use_fin_diff']:
                    forward = [int(new_x[i] == lower_bound[i]) for i in range(problem.dim)]
                    backward = [int(new_x[i] == upper_bound[i]) for i in range(problem.dim)]
                    # BdsCheck: 1 stands for forward, -1 stands for backward, 0 means central diff
                    BdsCheck = np.subtract(forward, backward)
                    avg_gradient, bud = self.finite_diff(new_x, new_solution, .01, BdsCheck, 
                                                         problem, 1, 1, expended_budget, lower_bound, upper_bound)
                else:
                    avg_gradient = np.sum(recommended_solns[-1].objectives_gradients_mean,axis=0)
                
                
                # Identify new solution to simulate.
                
                
                old_x = recommended_solns[-1].x
                
                #print(avg_gradient)
                
                new_x = old_x + problem.minmax[0]*self.factors['step_size']*avg_gradient/np.linalg.norm(avg_gradient)
                new_x = self.box_project(new_x, lower_bound, upper_bound)
                new_solution = self.create_new_solution(new_x, problem)
            # Simulate new solution and update budget.
            problem.simulate(new_solution, self.factors["sample_size"])
            expended_budget += self.factors["sample_size"]
            # Check for improvement relative to incumbent best solution.
            
            '''
            # Also check for feasibility w.r.t. stochastic constraints.
            if (problem.minmax * new_solution.objectives_mean
                    > problem.minmax * best_solution.objectives_mean and
                    all(new_solution.stoch_constraints_mean[idx] >= 0 for idx in range(problem.n_stochastic_constraints))):
                # If better, record incumbent solution as best.
                best_solution = new_solution
                recommended_solns.append(new_solution)
                intermediate_budgets.append(expended_budget)
           '''
            best_solution = new_solution
            recommended_solns.append(new_solution)
            intermediate_budgets.append(expended_budget)
        return recommended_solns, intermediate_budgets
    
    # Finite difference for calculating gradients
    def finite_diff(self, new_x, new_solution, delta_T, BdsCheck, problem, r, NumOfEval, expended_budget, lower_bound, upper_bound):
        fn = -1 * problem.minmax[0] * new_solution.objectives_mean
        # Store values for each dimension
        FnPlusMinus = np.zeros((problem.dim, 3)) 
        grad = np.zeros(problem.dim)

        while (np.all((grad == 0))):
            if expended_budget >= problem.factors["budget"]:
                break
            for i in range(problem.dim):
                # initialization
                x1 = list(new_x)
                x2 = list(new_x)
                steph1 = delta_T # forward stepsize
                steph2 = delta_T # backward stepsize
                
                # check variable bounds
                if x1[i] + steph1 > upper_bound[i]:
                    steph1 = np.abs(upper_bound[i] - x1[i])
                if x2[i] - steph2 < lower_bound[i]:
                    steph2 = np.abs(x2[i] - lower_bound[i])

                # decide stepsize
                if BdsCheck[i] == 0:   #central diff
                    FnPlusMinus[i, 2] = min(steph1, steph2)
                    x1[i] = x1[i] + FnPlusMinus[i, 2]
                    x2[i] = x2[i] - FnPlusMinus[i, 2]
                elif BdsCheck[i] == 1:    # forward diff
                    FnPlusMinus[i, 2] = steph1
                    x1[i] = x1[i] + FnPlusMinus[i, 2]
                else:    # backward diff
                    FnPlusMinus[i, 2] = steph2
                    x2[i] = x2[i] - FnPlusMinus[i,2]

                x1_solution = self.create_new_solution(tuple(x1), problem)
                if BdsCheck[i] != -1:
                    problem.simulate_up_to([x1_solution], r)
                    fn1 = -1 * problem.minmax[0] * x1_solution.objectives_mean
                    FnPlusMinus[i, 0] = fn1 # first column is f(x+h,y)
                x2_solution = self.create_new_solution(tuple(x2), problem)
                if BdsCheck[i] != 1:
                    problem.simulate_up_to([x2_solution], r)
                    fn2 = -1 * problem.minmax[0] * x2_solution.objectives_mean
                    FnPlusMinus[i, 1] = fn2 # second column is f(x-h,y)
                
                # Calculate gradient
                if BdsCheck[i] == 0:
                    grad[i] = (fn1 - fn2) / (2 * FnPlusMinus[i,2])
                elif BdsCheck[i] == 1:
                    grad[i] = (fn1 - fn) / FnPlusMinus[i, 2]
                elif BdsCheck[i] == -1:
                    grad[i] = (fn - fn2) / FnPlusMinus[i, 2]
                            
            expended_budget += NumOfEval * r
            
            # print('expended_budget', expended_budget)
        return grad, expended_budget
    
    def box_project(self, x,lower_bound,upper_bound):
        return np.maximum(np.minimum(x,upper_bound),lower_bound)
