"""
Summary
-------
Simulate demand at facilities.
"""
from mmap import mmap
from re import L
import numpy as np

from base import Model, Problem
import math as math

class EmergencyRoom(Model):
    """
    A model that simulates an emergency room within a hospital.
    Returns the total cost and whether or not the average wait
    time of critical patients exceeds a given amount of time.

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
        details of each factor (for GUI and data validation)
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
        self.name = "Emergency Room"
        self.n_rngs = 9 # how many rngs?
        self.n_responses = 1
        self.specifications = {
            "employee_allocations": {
                "description": "Allocations of each type of employee: receptionists, doctors, lab technicians,"
                "treatment nurses, and ER nurses, respectively.",
                "datatype": list,
                "default": [1, 4, 3, 2, 5]
            },
            "employee_max": {
                "description": "Maximums of each type of employee: receptionists, doctors, lab technicians,"
                "treatment nurses, and ER nurses, respectively.",
                "datatype": list,
                "default": [3, 6, 5, 6, 12]
            },
            "employee_min": {
                "description": "Minimums of each type of employee: receptionists, doctors, lab technicians,"
                "treatment nurses, and ER nurses, respectively.",
                "datatype": list,
                "default": [1, 4, 2, 2, 8]
            },
            "employee_salaries": {
                "description": "Yearly salaries for each type of employee: receptionists, doctors, "
                "lab technicians, treatment nurses, and ER nurses, respectively.",
                "datatype": list,
                "default": [40000, 120000, 50000, 35000, 35000]
            },
            "st_reception": {
                "description": "Exponential distribution parameter for reception service time.",
                "datatype": float,
                "default": 7.5
            },
            "st_labtests_min": {
                "description": "Service time minimum for triangular distribution of lab tests.",
                "datatype": int,
                "default": 10
            },
            "st_labtests_mode": {
                "description": "Service time mode for triangular distribution of lab tests.",
                "datatype": int,
                "default": 20
            },
            "st_labtests_max": {
                "description": "Service time maximum for triangular distribution of lab tests.",
                "datatype": int,
                "default": 30
            },
            "st_er": {
                "description": "Exponential distribution parameter for emergency room service time.",
                "datatype": float,
                "default": 90
            },
            "st_exam": {
                "description": "Exponential distribution parameter for examination room service time.",
                "datatype": float,
                "default": 15
            },
            "st_reexam": {
                "description": "Exponential distribution parameter for re-examination process service time.",
                "datatype": float,
                "default": 9
            },
            "st_tr_min": {
                "description": "Service time minimum for triangular distribution of treatments.",
                "datatype": int,
                "default": 20
            },
            "st_tr_mode": {
                "description": "Service time mode for triangular distribution of treatments.",
                "datatype": int,
                "default": 28
            },
            "st_tr_max": {
                "description": "Service time maximum for triangular distribution of treatments.",
                "datatype": int,
                "default": 30
            },
            "walkin_rates": {
                "description": "List of arrival rates for walk ins. Each element represents the arrival"
                "rate for the next two hours in a 24 hour day starting at hour 0.",
                "datatype": list,
                "default": [5.25, 3.8, 3, 4.8, 7, 8.25, 9, 7.75, 7.75, 8, 6.5, 3.25]
            },
            "prob_extra": {
                "description": "Probability that extra tests are needed.",
                "datatype": float,
                "default": 0.5
            },
            "prob_tn": {
                "description": "Probability that treatment is needed.",
                "datatype": float,
                "default": 0.8
            },
            "prob_majorinjury": {
                "description": "Probability that a patient has a major injury.",
                "datatype": float,
                "default": 0.5
            },
            "warm_period": {
                "description": "Warm up period for simulation, in days.",
                "datatype": int,
                "default": 4
            },
            "run_time": {
                "description": "Run time of simulation, in days",
                "datatype": int,
                "default": 100
            },
            "amb_arr": {
                "description": "The rate per hour of ambulance arrivals",
                "datatype": int,
                "default": 2
            },

        }
        self.check_factor_list = {
            "employee_allocations": self.check_employee_allocations,
            "employee_max": self.check_employee_max,
            "employee_min": self.check_employee_min,
            "employee_salaries": self.check_employee_salaries,
            "st_reception": self.check_st_reception,
            "st_labtests_min": self.check_st_labtests_min,
            "st_labtests_mode": self.check_st_labtests_mode,
            "st_labtests_max": self.check_st_labtests_max,
            "st_er": self.check_st_er,
            "st_exam": self.check_st_exam,
            "st_reexam": self.check_st_reexam,
            "st_tr_min": self.check_st_tr_min,
            "st_tr_mode": self.check_st_tr_mode,
            "st_tr_max": self.check_st_tr_max,
            "walkin_rates": self.check_walkin_rates,
            "prob_extra": self.check_prob_extra,
            "prob_tn": self.check_prob_tn,
            "prob_majorinjury": self.check_prob_majorinjury,
            "warm_period": self.check_warm_period,
            "run_time": self.check_run_time,
            "amb_arr": self.check_amb_arr
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)
    def check_simulatable_factors(self):
        if len(self.factors["employee_allocations"]) != len(self.factors["employee_max"]):
            return False
        elif len(self.factors["employee_allocations"]) != len(self.factors["employee_min"]):
            return False
        elif len(self.factors["employee_allocations"]) != len(self.factors["employee_salaries"]):
            return False
        else:
            return True

    def check_employee_allocations(self):
        for i in range(len(self.factors["employee_allocations"])):
            if self.factors["employee_allocations"][i] > self.factors["employee_max"][i]:
                return False
            elif self.factors["employee_allocations"][i] < self.factors["employee_min"][i]:
                return False
            elif i+1 == len(self.factors["employee_allocations"]):
                return np.all(self.factors["employee_allocations"]) > 0
        return False

    def check_employee_min(self):
        return np.all(self.factors["employee_min"]) > 0

    def check_employee_max(self):
        return np.all(self.factors["employee_max"]) > 0
    
    def check_employee_salaries(self):
        return np.all(self.factors["employee_salaries"]) > 0

    def check_st_reception(self):
        return self.factors["st_reception"] > 0
    
    def check_st_labtests_min(self):
        return self.factors["st_labtests_min"] > 0

    def check_st_labtests_mode(self):
        return self.factors["st_labtests_mode"] > 0

    def check_st_labtests_max(self):
        return self.factors["st_labtests_max"] > 0

    def check_st_er(self):
        return self.factors["st_er"] > 0

    def check_st_exam(self):
        return self.factors["st_exam"] > 0

    def check_n_fac(self):
        return self.factors["n_fac"] > 0
    
    def check_walkin_rates(self):
        return np.all(self.factors["walkin_rates"]) > 0

    def check_st_reexam(self):
        return self.factors["st_reexam"] > 0

    def check_st_tr_min(self):
        return self.factors["st_tr_min"] > 0
    
    def check_st_tr_mode(self):
        return self.factors["st_tr_mode"] > 0

    def check_st_tr_max(self):
        return self.factors["st_tr_max"] > 0

    def check_prob_extra(self):
        return self.factors["prob_extra"] > 0

    def check_prob_tn(self):
        if self.factors["prob_tn"] < 0 or self.factors["prob_tn"] > 1:
            return False
        return True

    def check_prob_majorinjury(self):
        return self.factors["prob_majorinjury"] > 0

    def check_warm_period(self):
        return self.factors["warm_period"] > 0
    
    def check_run_time(self):
        return self.factors["run_time"] > 0

    def check_amb_arr(self):
        return self.factors["amb_arr"] > 0

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
            "stockout_flag" = a binary variable
                 0 : all facilities satisfy the demand
                 1 : at least one of the facilities did not satisfy the demand
            "n_fac_stockout" = the number of facilities which cannot satisfy the demand
            "n_cut" = the number of toal demand which cannot be satisfied
        gradients : dict of dicts
            gradient estimates for each response
        """
        # Designate RNG for demands.
        arrivals_rng = rng_list[0]
        reception_rng = rng_list[1]
        lab_rng = rng_list[2]
        exam_rng = rng_list[3]
        mm_rng = rng_list[4]
        treatment_rng = rng_list[5]
        er_rng = rng_list[6]
        tr_rng = rng_list[7]
        amb_rng = rng_list[8]


        arrival_times = []
        arrival_amb = []
        # Generate random arrival times at facilities from truncated multivariate normal distribution.
        t = arrivals_rng.expovariate(self.factors["walkin_rates"][0])

        while t <= (self.factors["warm_period"]) * 60:  #non stationary poisson process, so this is wrong but could make a function 
            if t <= 120:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][0])
            elif t <= 240:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][1])
            elif t <= 360:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][2])
            elif t <= 480:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][3])
            elif t <= 600:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][4])
            elif t <= 720:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][5])
            elif t <= 840:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][6])
            elif t <= 960:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][7])
            elif t <= 1080:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][8])
            elif t <= 1200:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][9])
            elif t <= 1320:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][10])
            elif t <= 1440:
                arrival_times.append(t)
                t += arrivals_rng.expovariate(self.factors["walkin_rates"][11])

        t = arrivals_rng.expovariate(self.factors["amb_arr"])
        print("hello")
        while t <= (self.factors["warm_period"]) * 60:
            arrival_amb.append(t)
            t += amb_rng.expovariate(self.factors["amb_arr"])

        reception_time = []
        t = reception_rng.expovariate(self.factors["st_reception"])

        for i in range(len(arrival_times)):  # time that people are in reception
            reception_time.append(t)
            t = reception_rng.expovariate(self.factors["st_reception"])

        wait_times_rec = []
        receptionists = []
        arr_ind = 0

        for i in range(self.factors["employee_allocations"][0]):  # create a list that has indeces for every receptionist
            receptionists.append(math.inf)

        clock_rec = 0
        rec_ind = 0
        rec_queue = []
        next_queue = 0
        print("first")
        while arr_ind < len(arrival_times):  # wait times for receptionist
            if min(receptionists) <= arrival_times[arr_ind]:
                clock_rec = min(receptionists)
                if len(rec_queue) > 0:
                    rec_ind = receptionists.index(min(receptionists))
                    next_queue = rec_queue.pop(0)
                    receptionists[rec_ind] = clock_rec + reception_time[arr_ind]
                    wait_times_rec.append(clock_rec - next_queue)
                    arr_ind += 1
                elif len(rec_queue) == 0:
                    rec_ind = receptionists.index(min(receptionists))
                    receptionists[rec_ind] = math.inf
                else:
                    print("Error in receptionist loop")

            elif min(receptionists) > arrival_times[arr_ind]:
                clock_rec = arrival_times[arr_ind]
                if len(rec_queue) == 0:
                    for i in range(len(receptionists)):
                        if receptionists[i] == math.inf:
                            rec_ind = i
                        elif receptionists[i] != math.inf:
                            rec_ind = -1
                        else:
                            print("Error in receptionist loop")
                    if rec_ind >= 0:
                        receptionists[rec_ind] = clock_rec + reception_time[arr_ind]
                        wait_times_rec.append(0)
                        arr_ind += 1
                    elif rec_ind == -1:
                        rec_queue.append(clock_rec)
                        arr_ind += 1
                    else:
                        print("Error in receptionist loop")
                elif len(rec_queue) > 0:
                    rec_queue.append(clock_rec)
                    arr_ind += 1
                else:
                    print("Error in receptionist loop.")
      
        while len(rec_queue) > 0: # this empties the queue at the end of the day once we are done with all arrivals
            clock_rec = min(receptionists)
            rec_ind = receptionists.index(min(receptionists))
            next_queue = rec_queue.pop(0)
            receptionists[rec_ind] = next_queue + receptionists[rec_ind]
            rec_ind += 1
            wait_times_rec.append(clock - next_queue)  # need to add to wait time for receptionist

        post_rec = []
        for i in range(len(wait_times_rec)):  # when patients are done with receptionists and what time they go into system with amb_arr
            post_rec.append(arrival_times[i] + wait_times_rec[i])

        system = post_rec + arrival_amb
        system.sort()

        # while len(system) < (len(post_rec) + len(arrival_amb)):  # making list for entire system before patients go into exam room
        #     if post_rec[counter1] <= arrival_amb[counter2] and counter1 != (len(post_rec)-1) and counter2 != (len(arrival_amb)-1):
        #         system.append(post_rec[counter1])
        #         counter1 += 1
        #     elif post_rec[counter1] > arrival_amb[counter2] and counter1 != (len(post_rec)-1) and counter2 != (len(arrival_amb)-1):
        #         system.append(arrival_amb[counter2])
        #         counter2 += 1
        #     elif counter2 == (len(arrival_amb)-1):
        #         system.append(post_rec[counter1])
        #         counter1 += 1
        #     elif counter1 == (len(post_rec)-1):
        #         system.append(arrival_amb[counter2])
        #         counter2 += 1
        #     else:
        #         print("Error in system list configuration.")  # need to mark er and walk in list to track time for er
        t = 0
        treatment = []
        p2 = 0
        major = []
        minor = []
        t_major = 0
        t_minor = 0
        # should I make two separate lists and add a 0 if not major and if major add a time
        # this way in the while loop I look for 0 or time and according to which I send to the right nurse
        for i in range(len(system)):  # calculates whether someone needs treatment
                p = self.factors["prob_tn"]  # Prob of needing treatment, 80%
                if treatment_rng.choices([0, 1], [1-p, p]) == [1]:  # Determining if someone needs treatment
                    # determine if er or tr
                    p2 = self.factors["prob_majorinjury"]
                    if mm_rng.choices([0, 1], [1-p2, p2]) == [1]: # major injury
                        t_major = er_rng.expovariate(1/self.factors["st_er"])
                        t_minor = 0
                        t = 1
                    else: 
                        t_minor = tr_rng.triangular(self.factors["st_tr_min"],self.factors["st_tr_max"],self.factors["st_tr_mode"])
                        t_major = 0
                        t =1
                else:
                    t = 0  # No treatment needed
                    t_major = 0
                    t_minor = 0
                treatment.append(t)  # treatment list says whether or not they need treatment
                major.append(t_major)  # if they need treatment, while loop whill check major and minor list to see who to go to
                minor.append(t_minor)
        exam_wt = []
        for i in range(len(system)):  # i is each individual 
                t = exam_rng.expovariate(self.factors["st_exam"])
                exam_wt.append(t)
        # go to exam room with x doctors (first queue), if extra tests go to extra tests with x technicians (second queue)
        # reexam process if they got extra tests (cant do it again)
        # emergency room or treatment room and nurse depending on which one... third queue
        # if someone goes out of exam queue they either go to extra tests or next queue if queue empty they go if not go into 

        system_waittime = []
        doctors = []
        lab_techs = []
        er_nurse = []
        treat_nurse = []

        for i in range(self.factors["employee_allocations"][1]):
            doctors.append(math.inf)
        for i in range(self.factors["employee_allocations"][2]):
            lab_techs.append(math.inf)
        for i in range(self.factors["employee_allocations"][3]):
            er_nurse.append(math.inf)
        for i in range(self.factors["employee_allocations"][4]):
            treat_nurse.append(math.inf)
        # !!!! need to change because some don't need treatment
        sys_ind = 0
        doc_ind = 0
        doc_queue = []
        minor_queue = []
        major_queue = []
        for i in range(len(system)):
            minor.append(0)
            major.append(0)
        types = []
        next_event = 0
        for i in range(4):
            types.append(0)
        clock = 0
        popped = 0
        treat_ind = 0
        next_patient = 0
        exam_ind = 0
        doc_wt = []
        treatment_wt = []
        mm_ind = 0
        while len(system_waittime) < len(system): # problems with loop include: sys_ind becomes too 
            # large and can't empty out the system, there are some instances when the major and minor 
            # lists are equal to 0 and treatment is equal to one which should not be possible
            # there also needs to be a way to track people who come in through ambulance, I did not do this
            types[0] = system[sys_ind]
            types[1] = min(doctors)
            types[2] = min(er_nurse)
            types[3] = min(treat_nurse)
            next_event = types.index(min(types))
            if next_event == 0:  # next event is from the queue entering system, go to doctors in exam room
                clock = system[sys_ind]
                # need a way to mark someone as coming from the reexam so they don't go again, maybe append as -1 and the clock after?
                if len(doc_queue) > 0:
                    doc_queue.append(clock)
                elif len(doc_queue) == 0:
                    for i in range(len(doctors)):
                        if doctors[i] == math.inf:
                            doctors[i] = clock + exam_wt[sys_ind]
                            doc_wt.append(0)
                            popped = 1
                            break
                        elif doctors[i] != math.inf:
                            doc_queue.append(clock) # append time for 
                sys_ind += 1
            elif next_event == 1:
                clock = min(doctors)
                if len(doc_queue) > 0:
                    next_patient = doc_queue.pop(0)
                    doc_wt.append(clock - next_patient)
                    doc_ind = doctors.index(min(doctors))
                    doctors[doc_ind] = next_patient + exam_wt[exam_ind]
                    exam_ind += 1
                elif len(doc_queue) == 0:
                    doc_ind = doctors.index(min(doctors))
                    doctors[doc_ind] = math.inf
                # print(treat_ind)
                if treatment[treat_ind] == 0:  # they don't need treatment
                    # they go out of system, wait time recorded at the end
                    treat_ind += 1
                else:  # need to figure out if major or minor
                    if major[mm_ind] > 0:
                            # add to er nurse queue or to er nurse if open
                        if len(major_queue) > 0:
                            major_queue.append(clock)  # should be clock
                        elif len(major_queue) == 0: # problem! nobody is being sent to the er
                            for i in range(len(er_nurse)):
                                if er_nurse[i] == math.inf:
                                    er_nurse[i] = clock + major[mm_ind]
                                    treatment_wt.append(0)
                                    mm_ind += 1
                                    popped = 1
                                    break
                                elif er_nurse[i] != math.inf:
                                    popped = 0
                            if popped == 0:
                                major_queue.append(clock)
                        else:
                            print("Error in major queue")
                    elif minor[mm_ind] > 0:
                        # add to tr nurse queue or to tr nurse if open
                        if len(minor_queue) > 0:
                            minor_queue.append(clock)
                        elif len(minor_queue) == 0:
                            for i in range(len(treat_nurse)):
                                if treat_nurse[i] == math.inf:
                                    treat_nurse[i] = clock + minor[mm_ind]
                                    mm_ind += 1
                                    popped = 1
                                    break
                                elif treat_nurse[i] != math.inf:
                                    popped = 0
                            if popped == 0:
                                minor_queue.append(clock)
                    else:
                        print("Error in major minor next event == 1")
                        print(major)
                        print(minor)
                    treat_ind += 1
                    # need to make doctor available again with math.inf
            elif next_event == 2:
                # er nurse opens up
                # take wait time and nurse is open
                print("2")
                if len(major_queue) == 0:
                    er_nurse[er_nurse.index(min(er_nurse))] = math.inf
                elif len(major_queue) > 0:
                    next_queue = major_queue.pop(0)
                    er_nurse[er_nurse.index(min(er_nurse))] = clock + major[mm_ind]
                    treatment_wt.append(clock - next_queue)
                    mm_ind += 1


            elif next_event == 3:
                print("3")
                clock = min(treat_nurse)
                # tr nurse opens up
                # take wait time and nurse is open
                if len(minor_queue) == 0:
                    treat_nurse[treat_nurse.index(min(treat_nurse))] = math.inf
                elif len(minor_queue) > 0:
                    next_queue = minor_queue.pop(0)
                    treat_nurse[treat_nurse.index(min(treat_nurse))] = clock + minor[mm_ind]
                    treatment_wt.append(clock - next_queue)
                    mm_ind += 1



        total_cost = 0
        avg_er_time = True
        responses = {'total_cost': total_cost,  # Compose responses and gradients.
                     'avg_er_time': avg_er_time
                     }
        gradients = {response_key: {factor_key: np.nan for factor_key in self.specifications} for response_key in responses}
        return responses, gradients
#need to generate amount of employees in a given sim
# stochastic constraint average has to be less than 2 hours
"""
Summary
-------
Determine the total cost and average amount of time critical patients spend waiting in the hospital.
Need to meet the constraints of the problem which include a certain amount of each different kind of employee 
specified in factors and a specific amount of time specified by the hospital in factors.
"""


class MinCost(Problem):
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
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="FACSIZE-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 1
        self.minmax = (-1,)
        self.constraint_type = "stochastic"
        self.variable_type = "continuous"
        self.lower_bounds = (0, 0, 0)
        self.upper_bounds = (np.inf, np.inf, np.inf)
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None  # (185, 185, 185)
        self.model_default_factors = {}
        self.model_decision_factors = {"capacity"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (300, 300, 300)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            },
            "installation_costs": {
                "description": "Cost to install a unit of capacity at each facility.",
                "datatype": tuple,
                "default": (1, 1, 1)
            },
            "epsilon": {
                "description": "Maximum allowed probability of stocking out.",
                "datatype": float,
                "default": 0.05
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "installation_costs": self.check_installation_costs,
            "epsilon": self.check_epsilon
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = EmergencyRoom(self.model_fixed_factors)

    def check_installation_costs(self):
        if len(self.factors["installation_costs"]) != self.model.factors["n_fac"]:
            return False
        elif any([elem < 0 for elem in self.factors["installation_costs"]]):
            return False
        else:
            return True

    def check_epsilon(self):
        return 0 <= self.factors["epsilon"] <= 1

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
            "capacity": vector[:]
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
        vector = tuple(factor_dict["capacity"])
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
        objectives = (0,)
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
        stoch_constraints = (-response_dict["stockout_flag"],)
        return stoch_constraints

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = (self.factors["epsilon"],)
        det_stoch_constraints_gradients = ((0,),)
        return det_stoch_constraints, det_stoch_constraints_gradients

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
        det_objectives = (np.dot(self.factors["installation_costs"], x),)
        det_objectives_gradients = ((self.factors["installation_costs"],),)
        return det_objectives, det_objectives_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return np.all(x > 0)

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
        x = tuple([300*rand_sol_rng.random() for _ in range(self.dim)])
        return x


"""
Summary
-------
Maximize the probability of not stocking out subject to a budget
constraint on the total cost of installing capacity.
"""


class FacilitySizingMaxService(Problem):
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
    name : str
        user-specified name for problem
    fixed_factors : dict
        dictionary of user-specified problem factors
    model_fixed factors : dict
        subset of user-specified non-decision factors to pass through to the model

    See also
    --------
    base.Problem
    """
    def __init__(self, name="FACSIZE-2", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.dim = 3
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (1,)
        self.constraint_type = "deterministic"
        self.variable_type = "continuous"
        self.lower_bounds = (0, 0, 0)
        self.upper_bounds = (np.inf, np.inf, np.inf)
        self.gradient_available = False
        self.optimal_value = None
        self.optimal_solution = None  # (175, 179, 143)
        self.model_default_factors = {}
        self.model_decision_factors = {"capacity"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (100, 100, 100)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 10000
            },
            "installation_costs": {
                "description": "Cost to install a unit of capacity at each facility.",
                "datatype": tuple,
                "default": (1, 1, 1)
            },
            "installation_budget": {
                "description": "Total budget for installation costs.",
                "datatype": float,
                "default": 500.0
            }
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "installation_costs": self.check_installation_costs,
            "installation_budget": self.check_installation_budget
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and over-riden defaults.
        self.model = FacilitySize(self.model_fixed_factors)

    def check_installation_costs(self):
        if len(self.factors["installation_costs"]) != self.model.factors["n_fac"]:
            return False
        elif any([elem < 0 for elem in self.factors["installation_costs"]]):
            return False
        else:
            return True

    def check_installation_budget(self):
        return self.factors["installation_budget"] > 0

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
            "capacity": vector[:]
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
        vector = tuple(factor_dict["capacity"])
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
        objectives = (1 - response_dict["stockout_flag"],)
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
        stoch_constraints = None
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
        det_objectives = (0,)
        det_objectives_gradients = ((0, 0, 0),)
        return det_objectives, det_objectives_gradients

    def deterministic_stochastic_constraints_and_gradients(self, x):
        """
        Compute deterministic components of stochastic constraints for a solution `x`.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        det_stoch_constraints : tuple
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
        return det_stoch_constraints, det_stoch_constraints_gradients

    def check_deterministic_constraints(self, x):
        """
        Check if a solution `x` satisfies the problem's deterministic constraints.

        Arguments
        ---------
        x : tuple
            vector of decision variables

        Returns
        -------
        satisfies : bool
            indicates if solution `x` satisfies the deterministic constraints.
        """
        return (np.dot(self.factors["installation_costs"], x) <= self.factors["installation_budget"])

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
        # Generate random solution using acceptable/rejection.
        # TO DO: More efficiently sample uniformly from the simplex.
        while True:
            x = tuple([self.factors["installation_budget"]*rand_sol_rng.random() for _ in range(self.dim)])
            if self.check_deterministic_constraints(x):
                break
        return x
