#imports
import math
import numpy as np
import matplotlib.pyplot as plt

from simopt.base import (
    ConstraintType,
    Model,
    Objective,
    Problem,
    RepResult,
    VariableType,
)

#model simulation
def ccbaby(x, runlength, seed, other=None):
    """
    Python translation of the MATLAB function CCBaby.

    Parameters
    ----------
    x : array-like
        Vector of the number of agents starting shifts at times j/2.
    runlength : int
        Number of days to simulate.
    seed : int
        Positive integer random seed.
    other : unused
        Included only to match original signature.

    Returns
    -------
    fn : float
        Mean total cost across simulated days.
    FnVar : float
        Variance of total cost divided by number of days.
    FnGrad : float
        NaN (not used, matches MATLAB code).
    FnGradCov : float
        NaN (not used, matches MATLAB code).
    constraint : np.ndarray
        Mean hourly performance measure across days.
    ConstraintCov : np.ndarray
        Covariance matrix of hourly performance measures.
    ConstraintGrad : float
        NaN (not used, matches MATLAB code).
    ConstraintGradCov : float
        NaN (not used, matches MATLAB code).
    """

    FnGrad = np.nan
    FnGradCov = np.nan
    constraint = np.nan
    ConstraintCov = np.nan
    ConstraintGrad = np.nan
    ConstraintGradCov = np.nan

    #C sets maximum number of agents allowed to start at any time slot
    #x is your staffing decision vector, here it is converted into a NumPy array of integers
    C = 100
    x = np.asarray(x, dtype=int)
    hk = np.array([3.0, 3.5, 4.0, 4.5])

    #if anything about the inputs is invalid, stop the simulation and return NaN
    #checks if staffing value is negative, if any entry exceeds the max number of agents allowed to start,
    #if the simulation runs for at least one day, random seed must be positive and integer
    if (
        x.ndim != 2
        or x.shape[1] != 4
        or np.any(x < 0)
        or np.any(x > C)
        or runlength <= 0
        or seed <= 0
        or int(seed) != seed
    ):
        print("x must be a 2D nonneg integer array with 4 columns: x[j,k]")
        fn = np.nan
        FnVar = np.nan
        return (
            fn,
            FnVar,
            FnGrad,
            FnGradCov,
            constraint,
            ConstraintCov,
            ConstraintGrad,
            ConstraintGradCov,
        )

    # Parameters
    lambdaMax = 1000              # Arrival upper bound
    serviceShape = 18             # Gamma shape for service times
    serviceScale = 1 / 3 / 60     # Gamma scale for service times (hours)
    patienceShape = 4             # Gamma shape for patience times
    patienceScale = 1 / 2 / 60    # Gamma scale for patience times (hours)
    TrunkMax = 150                # Number of trunk lines
    Salary = 18                   # Agent hourly wage
    nDays = int(runlength)
    tMax = 16.5                   # Length of a day
    m = 16                        # Number of constraints (not otherwise used)
    
    #converts staffing plan into a working variable, then sets variable for total agents in system
    xjk = x
    nAgents = int(np.sum(xjk))

    print("x matrix:")
    print(xjk)
    print("x shape:", xjk.shape)
    print("total agents =", nAgents)
    print("agents by start time =", np.sum(xjk, axis=1))
    print("agents by lunch choice =", np.sum(xjk, axis=0))

    #storage arrays to store (# of hours in a day, # of days), and a 1D array of length nDays
    PerformanceMeasure = np.zeros((math.ceil(tMax), nDays))
    TotalCost = np.zeros(nDays)

    #create 5 separate random number generators to mimic separate MATLAB streams
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(4)

    dummy_rng = np.random.default_rng(child_seeds[0])
    service_rng = np.random.default_rng(child_seeds[1])
    patience_rng = np.random.default_rng(child_seeds[2])
    ar_rng = np.random.default_rng(child_seeds[3])

    #generates exponentially distributed time gapes between customer call arrivals
    Dummy = dummy_rng.exponential(scale=1 / lambdaMax, size=(nDays, 100000))

    #pre-generates random values for service times, customer patience, and arrival acceptance
    sTime = service_rng.gamma(shape=serviceShape, scale=serviceScale, size=(nDays, 100000))
    pTime = patience_rng.gamma(shape=patienceShape, scale=patienceScale, size=(nDays, 100000))

    AR = ar_rng.random(size=(nDays, 100000))

    #counters for each day
    D = np.ones(nDays, dtype=int)
    S = np.ones(nDays, dtype=int)
    P = np.ones(nDays, dtype=int)
    A = np.ones(nDays, dtype=int)

    #convert MATLAB 1-based counters to Python 0-based by subtracting 1 when indexing
    #simulate one full day of call center operations, then repeat for the next day
    for k in range(nDays):
        served_immediate = 0
        served_wait = 0
        abandoned = 0
        blocked = 0
        #stores current state of every agent for that day, including shift start time,
        #lunch start time, next time agent is available, departure/off-duty time,
        #whether lunch has happened already
        Agents = np.zeros((nAgents, 5))
        #stores call information, including call arrival time, time service starts,
        #time call leaves the system
        Calls = np.zeros((3, 1))

        #create start times and break times for agents
        #converts decision variable into actual agents
        #for each start time, and for each lunch option, create the specificed number
        #of agents
        num = 0
        max_j = int((tMax - 8.5) * 2)

        for j in range(max_j + 1):
            for k_idx in range(4):
                num_agents_starting = xjk[j, k_idx] if j < xjk.shape[0] else 0
                for _ in range(num_agents_starting):
                    #each agent gets assigned these 3 things
                    start_time = j/2
                    lunch_time = start_time + hk[k_idx]

                    Agents[num, 0] = start_time
                    Agents[num, 1] = lunch_time
                    Agents[num, 2] = start_time
                    num += 1
              
        if k == 0:
          print("\nFirst 10 agents (start time, lunch time):")
          print(Agents[:10, 0:2])

        #first arrival via acceptance-rejection
        #finds first customer of the day
        nextCall = 0.0
        dummy = 0.0           #running timeof candidate arrivals
        #generates candidate arrivals and accepts only some
        while nextCall == 0:
            dummy += Dummy[k, D[k] - 1]
            D[k] += 1
            #time-varying arrival rate at that moment so that call intensity depends on time of day
            funValue = 500 + 500 * math.sin((3 * math.pi * dummy - 16 * math.pi) / 32)
            a = AR[k, A[k] - 1]
            A[k] += 1
            #random acceptance-rejection number
            if a <= funValue / 1000:
                nextCall = dummy
                nCalls = 1
                Calls[0, 0] = nextCall

        trunks = np.zeros(TrunkMax)

        #as long as the next call arrives before the end of the day, process it
        while nextCall < tMax:
            #if there isn't enough room in Calls var, add another column
            if Calls.shape[1] < nCalls:
                Calls = np.hstack([Calls, np.zeros((3, 1))])

            #stores call's arrival time
            Calls[0, nCalls - 1] = nextCall
            minTime = 25.0
            minAgent = -1
            callTaken = False

            #check trunk availability
            minTrunkIndex = np.argmin(trunks)
            minTrunk = trunks[minTrunkIndex]

            #a blocked call immediately leaves the system at arrival time, this is consistent with "busy signal"
            if minTrunk >= nextCall:
                Calls[2, nCalls - 1] = nextCall
                callTaken = True
                blocked += 1

            #assigns random charactertistics to current caller, including
            #how long service will take if answered, and caller's patience time
            serviceTime = sTime[k, S[k] - 1]
            patienceTime = pTime[k, P[k] - 1]
            S[k] += 1
            P[k] += 1
            
            #checks agent statuses to decide who can take the call
            for i in range(nAgents):
                #update agent's status if available by nextCall and not already left
                if Agents[i, 2] <= nextCall and Agents[i, 3] == 0 and not callTaken:
                    #check if Agent left for lunch break
                    if Agents[i, 1] <= nextCall and Agents[i, 4] == 0:
                        Agents[i, 2] = nextCall + 0.5
                        Agents[i, 4] = 1

                    #check if Agent left job already/already off shift
                    if Agents[i, 0] + 8.5 <= nextCall and Agents[i, 0] + 8.5 < 16.5:
                        Agents[i, 3] = Agents[i, 0] + 8.5

                #if agent available after those two checks, they take the call
                if Agents[i, 2] <= nextCall and Agents[i, 3] == 0 and not callTaken:
                    Calls[1, nCalls - 1] = nextCall
                    #update agent's next available time, mark and record taken call
                    Agents[i, 2] = nextCall + serviceTime
                    callTaken = True
                    served_immediate += 1
                    Calls[2, nCalls - 1] = nextCall + serviceTime

                    #check whether lunch happens during the call
                    #if so, then agent starts lunch after call is over
                    if Agents[i, 2] >= Agents[i, 1] and Agents[i, 4] == 0:
                        Agents[i, 2] += 0.5
                        Agents[i, 4] = 1

                    #check whether shift end happens during call
                    if Agents[i, 2] >= Agents[i, 0] + 8.5 and Agents[i, 0] + 8.5 < 16.5:
                        Agents[i, 3] = Agents[i, 2]

                    break

                #track next available agent
                if Agents[i, 2] <= minTime and Agents[i, 3] == 0:
                    minTime = Agents[i, 2]
                    minAgent = i

            #no immediately available agents
            if minTime < nextCall + patienceTime and not callTaken and minAgent != -1:
                Calls[1, nCalls - 1] = Agents[minAgent, 2]
                Agents[minAgent, 2] = Agents[minAgent, 2] + serviceTime
                Calls[2, nCalls - 1] = Agents[minAgent, 2]
                served_wait += 1

                if Agents[minAgent, 2] >= Agents[minAgent, 1] and Agents[minAgent, 4] == 0:
                    Agents[minAgent, 2] += 0.5
                    Agents[minAgent, 4] = 1

                if (
                    Agents[minAgent, 2] >= Agents[minAgent, 0] + 8.5
                    and Agents[minAgent, 0] + 8.5 < 16.5
                ):
                    Agents[minAgent, 3] = Agents[minAgent, 2]
            elif not callTaken:
                Calls[2, nCalls - 1] = nextCall + patienceTime
                abandoned += 1

            #generate next arrival/call
            #advances simulation clock to the next real customer arrival
            #so the loop can process the next calll
            success = False
            while not success:
                dummy += Dummy[k, D[k] - 1]
                D[k] += 1
                funValue = 500 + 500 * math.sin((3 * math.pi * dummy - 16 * math.pi) / 32)
                a = AR[k, A[k] - 1]
                A[k] += 1
                if a <= funValue / 1000:
                    nextCall = dummy
                    success = True

            trunks[minTrunkIndex] = Calls[2, nCalls - 1]
            nCalls += 1

        #update departure times for last agents
        #some agents may not yet have a recorded departure time- this fills that in
        for i in range(nAgents):
            if Agents[i, 3] == 0:
                if Agents[i, 2] > 16.5:
                    Agents[i, 3] = Agents[i, 2]
                else:
                    Agents[i, 3] = Agents[i, 0] + 8.5

        #compute days outputs
        callsPerHour = np.zeros(math.ceil(tMax))
        under20sPerHour = np.zeros(math.ceil(tMax))

        for i in range(nCalls - 1):
            hour = int(math.floor(Calls[0, i]))
            callsPerHour[hour] += 1
            wait_time = Calls[1, i] - Calls[0, i]
            if 0 <= wait_time <= 0.0056:
                under20sPerHour[hour] += 1
        print("callsPerHour:", callsPerHour.astype(int))
        print("under20sPerHour:", under20sPerHour.astype(int))
        print("under20 <= calls each hour:", np.all(under20sPerHour <= callsPerHour))

        print(f"\n--- Day {k+1} ---")
        print("total calls:", nCalls - 1)
        print("served immediately:", served_immediate)
        print("served after waiting:", served_wait)
        print("abandoned:", abandoned)
        print("blocked:", blocked)
        print("sum check:", served_immediate + served_wait + abandoned + blocked)
        #cost calculation
        TotalCost[k] = Salary * np.sum(Agents[:, 3] - Agents[:, 0])
        PerformanceMeasure[:, k] = 0.8 * callsPerHour - under20sPerHour
        print("daily cost:", TotalCost[k])

    fn = np.mean(TotalCost)
    FnVar = np.var(TotalCost, ddof=1) / nDays if nDays > 1 else 0.0

    Performance = np.zeros((math.ceil(tMax), 2))
    Performance[:, 0] = np.mean(PerformanceMeasure, axis=1)
    if nDays > 1:
        Performance[:, 1] = 1.96 * np.std(PerformanceMeasure, axis=1, ddof=1) / np.sqrt(nDays)
    else:
        Performance[:, 1] = 0.0

    constraint = Performance[:, 0]
    ConstraintCov = np.cov(PerformanceMeasure)

    return (
        fn,
        FnVar,
        FnGrad,
        FnGradCov,
        constraint,
        ConstraintCov,
        ConstraintGrad,
        ConstraintGradCov,
    )

#model class
class CallCenter(Model):
    """SimOpt wrapper for the ccbaby call center simulation."""

    class_name_abbr = "CALLCENTER"
    class_name = "Call Center Staffing"
    n_rngs = 1
    n_responses = 2

    def __init__(self, fixed_factors=None):
        super().__init__(fixed_factors)

    def replicate(self):
        x = self.factors["staffing_matrix"]
        runlength = self.factors.get("runlength", 1)
        seed = self.factors.get("seed", 123)

        (
            fn,
            FnVar,
            FnGrad,
            FnGradCov,
            constraint,
            ConstraintCov,
            ConstraintGrad,
            ConstraintGradCov,
        ) = ccbaby(x=x, runlength=runlength, seed=seed)

        responses = {
            "total_cost": fn,
            "constraint_lhs_by_hour": constraint,
        }

        return responses, {}

#optimization problem
class CallCenterMinCost(Problem):
    """Simulation-optimization problem for call center staffing."""

    class_name_abbr = "CALLCENTER-1"
    class_name = "Minimize Staffing Cost Subject to Service Levels"

    model_class = CallCenter
    n_objectives = 1
    n_stochastic_constraints = 16

    minmax = (-1,)
    constraint_type = ConstraintType.STOCHASTIC
    variable_type = VariableType.DISCRETE
    gradient_available = False

    optimal_value = None
    optimal_solution = None

    model_default_factors = {}
    model_decision_factors = {"staffing_matrix"}

    @property
    def dim(self):
        #17 start times: 0, 0.5, ..., 8
        #4 lunch options: 3, 3.5, 4, 4.5
        return 17 * 4

    @property
    def lower_bounds(self):
        return (0,) * self.dim

    @property
    def upper_bounds(self):
        return (100,) * self.dim

    def vector_to_factor_dict(self, vector):
        """
        Convert flat optimization vector into 17 x 4 staffing matrix.
        """
        x_matrix = np.array(vector, dtype=int).reshape((17, 4))
        return {
            "staffing_matrix": x_matrix,
        }

    def factor_dict_to_vector(self, factor_dict):
        """
        Convert 17 x 4 staffing matrix back into flat vector.
        """
        return tuple(np.array(factor_dict["staffing_matrix"]).flatten())

    def replicate(self, _x):
        """
        Run one simulation replication at solution x.
        """
        responses, _ = self.model.replicate()

        objective = responses["total_cost"]

        stochastic_constraints = responses["constraint_lhs_by_hour"]

        return RepResult(
            objectives=[Objective(stochastic=objective)],
            stochastic_constraints=[
                Objective(stochastic=value) for value in stochastic_constraints
            ],
        )
    
    #checks simple bounds and integrality
    def check_deterministic_constraints(self, x):

        if not super().check_deterministic_constraints(x):
            return False

        x_matrix = np.array(x).reshape((17, 4))

        # Optional: total staffing cap
        if np.sum(x_matrix) > 1000:
            return False

        return True

    #generate random feasible staffing plan
    def get_random_solution(self, rand_sol_rng):

        return tuple(
            rand_sol_rng.integer_random_vector_from_simplex(
                n_elements=self.dim,
                summation=200,
                with_zero=True,
            )
        )

#function making number of agents dependent on demand
#this makes an alternative ("better") staffing plan than the random feasible one above
def make_demand_based_x():
    hk = np.array([3.0, 3.5, 4.0, 4.5])
    start_times = np.arange(0, 8.5, 0.5)

    x = np.zeros((17, 4), dtype=int)

    #estimated number of agents needed by half-hour
    times = np.arange(0, 16.5, 0.5)
    demand = 500 + 500 * np.sin((3 * np.pi * times - 16 * np.pi) / 32)

    #each agent handles about 10 calls/hour if avg service time is 6 minutes
    required_agents = np.ceil(demand / 10).astype(int)

    #add buffer
    required_agents = np.ceil(required_agents * 1.25).astype(int)

    #put more shift starts before the peak
    starts_by_time = np.array([
        2, 4, 6, 8, 10, 12, 14, 16, 18,
        18, 16, 14, 12, 10, 8, 6, 4
    ])

    #split each start time across the 4 lunch options
    for j in range(17):
        base = starts_by_time[j] // 4
        remainder = starts_by_time[j] % 4
        x[j, :] = base
        x[j, :remainder] += 1

    return x

#plot staffing vs. demand to test
def plot_staffing_vs_demand(x):
    x = np.asarray(x, dtype=int)

    hk = np.array([3.0, 3.5, 4.0, 4.5])
    times = np.arange(0, 16.5, 0.5)

    working_agents = []

    for t in times:
        count = 0

        for j in range(17):
            start_time = j / 2

            for k in range(4):
                lunch_start = start_time + hk[k]
                lunch_end = lunch_start + 0.5
                shift_end = start_time + 8.5

                #agent is working if in shift, but not at lunch
                is_on_shift = start_time <= t < shift_end
                is_at_lunch = lunch_start <= t < lunch_end

                if is_on_shift and not is_at_lunch:
                    count += x[j, k]

        working_agents.append(count)

    working_agents = np.array(working_agents)

    #arrival-rate demand function from simulation
    demand = 500 + 500 * np.sin((3 * np.pi * times - 16 * np.pi) / 32)

    #plot agents working vs. demand
    plt.figure(figsize=(10, 5))
    plt.plot(times, working_agents, marker="o", label="Agents working")
    plt.plot(times, demand, marker="x", label="Demand / arrival rate")

    plt.xlabel("Time of day")
    plt.ylabel("Count")
    plt.title("Agents Working vs. Demand by Half-Hour")
    plt.legend()
    plt.grid(True)
    plt.show()

#test
if __name__ == "__main__":

    x = make_demand_based_x()

    print("x matrix:")
    print(x)
    print("total agents:", np.sum(x))

    plot_staffing_vs_demand(x)

    fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov = ccbaby(
        x=x,
        runlength=2,
        seed=123
    )

    print("Objective value:", fn)
    print("Constraints:", constraint)