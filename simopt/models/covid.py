"""
Summary
-------
Under some testing and vaccination policy, simulate the spread of COVID-19
over a period of time by multinomial distribution.
A detailed description of the model/problem can be found
`here <https://simopt.readthedocs.io/en/latest/covid.html>`_.
"""
import numpy as np

from ..base import Model, Problem


class COVID_vac(Model):
    """
    A model that simulates the spread of COVID disease under isolation policy and vaccination policy
    with a multinomial distribution. 
    Returns the total number of infected people and the total number of patients who have observable symptoms 
    during the whole period. 

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
        fixed_factors of the simulation model

    See also
    --------
    base.Model
    """
    def __init__(self, fixed_factors={}):
        self.name = "COVID_vac"
        self.n_rngs = 7
        self.n_responses = 6
        self.factors = fixed_factors
        self.specifications = {
            "num_groups": {
                "description": "Number of groups.",
                "datatype": int,
                "default": 3
                # "default": 7
            },
            "p_trans": {
                "description": "Probability of transmission per interaction.",
                "datatype": float,
                "default": 0.018
            },
            "p_trans_vac": {
                "description": "Probability of transmission per interaction if being vaccinated",
                "datatype": float,
                "default": 0.01
            },
            "inter_rate": {
                "description": "Interaction rates between two groups per day",
                "datatype": tuple,
                "default": (10.58, 5, 2, 4, 6.37, 3, 6.9, 4, 2)
                # "default": (10.57, 4, 1, 1, 2, 1, 1, 6, 12.8, 2, 2, 3.5, 1, 1, 2, 2.5, 8.4, 1.5, 1.5, 1, 2, 1.5, 1.5, 1,
                #             11.61, 3.31, 3.5, 1, 1, 4.29, 4.91, 2.8, 2.9, 2.22, 1, 2.3, 1, 1, 1, 1, 2.2, 8.1, 2.75, 1, 1, 2.7, 1, 2.99, 1)  # For 7 groups
            },
            "group_size": {
                "description": "Size of each group.",
                "datatype": tuple,
                "default": (8123, 4921, 3598)
                # "default": (8123, 3645, 4921, 3598, 3598, 1907, 4478)  #total 30270, for 7 gruops
            },
            "lamb_exp_inf": {
                "description": "Mean number of days from exposed to infectious.",
                "datatype": float,
                "default": 2.0
            },
            "lamb_inf_sym": {
                "description": "Mean number of days from infectious to symptomatic.",
                "datatype": float,
                "default": 3.0
            },
            "lamb_sym": {
                "description": "Mean number of days from symptomatic/asymptomatic to recovered.",
                "datatype": float,
                "default": 12.0
            },
            "n": {
                "description": "Number of days to simulate.",
                "datatype": int,
                "default": 200
            },
            "init_infect_percent": {
                "description": "Initial prevalance level.",
                "datatype": tuple,
                "default": (0.00200, 0.00121, 0.0008)
                # "default": (0.00200, 0.00121, 0.0008, 0.002, 0.00121, 0.008, 0.002)  # For 7 gruops
            },
            "asymp_rate": {
                "description": "Probability of being asymptomatic.",
                "datatype": float,
                "default": 0.35
            },
            "asymp_rate_vac": {
                "description": "Probability of being asymptomatic for vaccinated individuals.",
                "datatype": float,
                "default": 0.5
            },
            "vac": {
                "description": "The fraction of people being vaccinated in each group.",
                "datatype": tuple,
                "default": (0.8, 0.3, 0)
                # "default": (1/7,1/7,1/7,1/7,1/7,1/7,1/7)  # For 7 groups
            },
            "total_vac": {
                "description": "The total number of vaccines.",
                "datatype": int,
                "default": 8000
            },
            "freq": {
                "description": "Testing frequency of each group.",
                "datatype": tuple,
                "default": (0/7, 0/7, 0/7)
                # "default": (0/7,0/7,0/7,0/7,0/7,0/7,0/7)  # For 7 groups
            },
            "freq_vac": {
                "description": "Testing frequency of each group.",
                "datatype": tuple,
                "default": (0/7, 0/7, 0/7, 0.8, 0.3, 0)
                # "default": (0/7,0/7,0/7,0/7,0/7,0/7,0/7,0,0,0,0,0,0,0)  # For 7 groups
            },
            "false_neg": {
                "description": "False negative rate.",
                "datatype": float,
                "default": 0.12
            },
            "w": {
                "description": "Protection weight for each group.",
                "datatype": tuple,
                "default": (1, 1, 1)
            }
        }

        self.check_factor_list = {
            "num_groups": self.check_num_groups,
            "p_trans": self.check_p_trans,
            "p_trans_vac": self.check_p_trans_vac,
            "inter_rate": self.check_inter_rate,
            "group_size": self.check_group_size,
            "lamb_exp_inf": self.check_lamb_exp_inf,
            "lamb_inf_sym": self.check_lamb_inf_sym,
            "lamb_sym": self.check_lamb_sym,
            "n": self.check_n,
            "init_infect_percent": self.check_init_infect_percent,
            "asymp_rate": self.check_asymp_rate,
            "asymp_rate_vac": self.check_asymp_rate_vac,
            "vac": self.check_vac,
            "total_vac": self.check_total_vac,
            "freq": self.check_freq,
            "freq_vac": self.check_freq_vac,
            "false_neg": self.check_false_neg,
            "w": self.check_w
        }
        # Set factors of the simulation model.
        super().__init__(fixed_factors)

    def check_num_groups(self):
        return self.factors["num_groups"] > 0

    def check_p_trans(self):
        return self.factors["p_trans"] > 0

    def check_p_trans_vac(self):
        return self.factors["p_trans_vac"] > 0

    def check_inter_rate(self):
        return all(np.array(self.factors["inter_rate"]) >= 0) & (len(self.factors["inter_rate"]) == self.factors["num_groups"]**2)

    def check_group_size(self):
        return all(np.array(self.factors["group_size"]) >= 0) & (len(self.factors["group_size"]) == self.factors["num_groups"])

    def check_lamb_exp_inf(self):
        return self.factors["lamb_exp_inf"] > 0

    def check_lamb_inf_sym(self):
        return self.factors["lamb_inf_sym"] > 0

    def check_lamb_sym(self):
        return self.factors["lamb_sym"] > 0

    def check_n(self):
        return self.factors["n"] > 0

    def check_init_infect_percent(self):
        return all(np.array(self.factors["init_infect_percent"]) >= 0) & (len(self.factors["init_infect_percent"]) == self.factors["num_groups"])

    def check_asymp_rate(self):
        return (self.factors["asymp_rate"] > 0) & (self.factors["asymp_rate"] < 1)

    def check_asymp_rate_vac(self):
        return (self.factors["asymp_rate_vac"] > 0) & (self.factors["asymp_rate_vac"] < 1)

    def check_vac(self):
        return all(np.array(self.factors["vac"]) >= 0) & (len(self.factors["init_infect_percent"]) == self.factors["num_groups"])

    def check_total_vac(self):
        return self.factors["total_vac"] > 0

    def check_freq(self):
        return all(np.array(self.factors["freq"]) >= 0) & (len(self.factors["freq"]) == self.factors["num_groups"])

    def check_freq_vac(self):
        return all(np.array(self.factors["freq_vac"]) >= 0) & (len(self.factors["freq_vac"]) == self.factors["num_groups"] * 2)

    def check_false_neg(self):
        return (self.factors["false_neg"] > 0) & (self.factors["false_neg"] < 1)
    
    def check_w(self):
        return (self.factors["w"] >= 0)

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
            "num_infected" = Number of infected individuals per day
            "num_susceptible" = Number of susceptible individuals per day
            "num_exposed" = Number of exposed individuals per day
            "num_recovered" = Number of recovered individuals per day
            "num_symptomatic" = Number of symptomatic individuals per day
            "total_symp" = Total number of symptomatic individuals
        """
        # Designate random number generator for generating Poisson random variables.
        poisson_numexp_rng = rng_list[0]
        binomial_asymp_rng = rng_list[1]
        multinomial_exp_rng = rng_list[2]
        multinomial_inf_rng = rng_list[3]
        multinomial_symp_rng = rng_list[4]
        multinomial_asymp_rng = rng_list[5]
        multinomial_iso_rng = rng_list[6]

        # Reshape the transmission rate
        inter_rate = np.reshape(np.array(self.factors["inter_rate"]), (self.factors["num_groups"], self.factors["num_groups"]))
        # Calculate the transmission rate
        t_rate = inter_rate * self.factors["p_trans"]
        t_rate = np.sum(t_rate, axis=1)
        t_rate_vac = inter_rate * self.factors["p_trans_vac"]
        t_rate_vac = np.sum(t_rate_vac, axis=1)

        # Initialize states, each row is one day, each column is one group
        susceptible = np.zeros((self.factors["n"], self.factors["num_groups"]))
        sus_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        sus_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))

        exposed_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        exposed_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))

        infectious_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        infectious_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))

        asymptomatic_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        asymptomatic_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))

        symptomatic_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        symptomatic_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))

        isolation_exp_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_inf_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_symp_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_asymp_non_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))

        isolation_exp_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_inf_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_symp_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        isolation_asymp_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))

        recovered = np.zeros((self.factors["n"], self.factors["num_groups"]))

        # Initialize the performance measures of interest
        num_infected = np.zeros((self.factors["n"], self.factors["num_groups"]))
        num_exposed = np.zeros((self.factors["n"], self.factors["num_groups"]))
        num_recovered = np.zeros((self.factors["n"], self.factors["num_groups"]))
        num_susceptible = np.zeros((self.factors["n"], self.factors["num_groups"]))
        last_susceptible = np.zeros(self.factors["num_groups"])
        num_symptomatic = np.zeros((self.factors["n"], self.factors["num_groups"]))
        tot_num_syptomatic = 0
        tot_num_isolated = 0
        num_isolation = np.zeros((self.factors["n"], self.factors["num_groups"]))

        # Number of newly coming people for each day, each state
        inf_arr = np.zeros((self.factors["n"], self.factors["num_groups"]))
        sym_arr = np.zeros((self.factors["n"], self.factors["num_groups"]))
        asym_arr = np.zeros((self.factors["n"], self.factors["num_groups"]))
        inf_arr_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        sym_arr_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))
        asym_arr_vac = np.zeros((self.factors["n"], self.factors["num_groups"]))

        # Number of newly isolated people for each day, each state
        exp_arr_iso = np.zeros((self.factors["n"], self.factors["num_groups"]))
        inf_arr_iso = np.zeros((self.factors["n"], self.factors["num_groups"]))
        asym_arr_iso = np.zeros((self.factors["n"], self.factors["num_groups"]))
        exp_arr_vac_iso = np.zeros((self.factors["n"], self.factors["num_groups"]))
        inf_arr_vac_iso = np.zeros((self.factors["n"], self.factors["num_groups"]))
        asym_arr_vac_iso = np.zeros((self.factors["n"], self.factors["num_groups"]))

        # Get the current vaccine policy
        vac = self.factors["freq_vac"][self.factors["num_groups"]:]
        freq = self.factors["freq_vac"][:self.factors["num_groups"]]

        # Add day 0 num infections
        infectious_non_vac[0, :] = np.ceil(np.multiply(list(self.factors["group_size"]), list(self.factors["init_infect_percent"])))
        infectious_vac[0, :] = np.multiply(np.multiply(list(self.factors["group_size"]), list(self.factors["init_infect_percent"])), 0)
        susceptible[0, :] = np.subtract(list(self.factors["group_size"]), infectious_non_vac[0, :])
        sus_vac[0, :] = np.ceil(np.multiply(list(self.factors["group_size"]), list(vac)))
        sus_non_vac[0, :] = np.subtract(susceptible[0, :], sus_vac[0, :])

        for g in range(self.factors["num_groups"]):
            infect = infectious_vac[0, g] + infectious_non_vac[0, g]
            num_infected[0][g] = np.sum(infect)
            num_susceptible[0][g] = sus_vac[0, g] + sus_non_vac[0, g]

        # Initialization -- calculate the probability distribution for alias
        # For exposed:
        p_exp = []  # Min=1, max=7 (by original setting)
        exp_max = 7  # Min=1, max=7 (by original setting)
        lamb_exp = self.factors["lamb_exp_inf"]
        p0_e = lamb_exp ** 0 * np.exp(-lamb_exp) / 1
        for i in range(1, 1 + exp_max):
            p0_e = p0_e * lamb_exp / i
            p_exp.append(p0_e)
        p_exp = p_exp / np.sum(p_exp)  # Normalize to sum equal to 1

        dict_exp = {}
        for i in range(1, 1 + exp_max):
            dict_exp[i] = p_exp[i - 1]

        exp_table, exp_alias, exp_value = multinomial_exp_rng.alias_init(dict_exp)  # New alias: exp_value

        # For infected:
        p_inf = []
        inf_max = 8
        lamb_inf = self.factors["lamb_inf_sym"]
        p0_i = lamb_inf ** 0 * np.exp(-lamb_inf) / 1
        for i in range(1, 1 + inf_max):
            p0_i = p0_i * lamb_inf / i
            p_inf.append(p0_i)
        p_inf = p_inf / np.sum(p_inf)

        dict_inf = {}
        for i in range(1, 1 + inf_max):
            dict_inf[i] = p_inf[i - 1]

        inf_table, inf_alias, inf_value = multinomial_inf_rng.alias_init(dict_inf)

        # For symptomatic:
        p_sym = []
        asym_sym_max = 20
        lamb_asy_sym = self.factors["lamb_sym"]
        p0_as_s = lamb_asy_sym ** 0 * np.exp(-lamb_asy_sym) / 1
        for i in range(1, 1 + asym_sym_max):
            p0_as_s = p0_as_s * lamb_asy_sym / i
            p_sym.append(p0_as_s)
        p_sym = p_sym / np.sum(p_sym)

        dict_sym = {}
        for i in range(1, 1 + asym_sym_max):
            dict_sym[i] = p_sym[i - 1]

        sym_table, sym_alias, sym_value = multinomial_symp_rng.alias_init(dict_sym)
        asym_table, asym_alias, asym_value = multinomial_asymp_rng.alias_init(dict_sym)

        # Isolation schedule generator
        def calculate_iso_mul(day_max, pb, n):  # Note: day_max is the number of days before change, n is the number of people
            p = []
            p0 = pb  # pb = freq*(1-self.factors["false_neg"])
            p.append(p0)
            for i in range(1, day_max):
                p0 = p0 * p0
                p.append(p0)
            p.append(1 - np.sum(p))  # p is a list with length day_max+1

            dict_p = {}
            for i in range(len(p)):
                dict_p[i] = p[i]

            table, alias, value = multinomial_iso_rng.alias_init(dict_p)

            # For base case that we do not do testing
            if pb == 0:
                day_iso = np.zeros(len(p)).astype(int)
                gen_iso_num = np.zeros(len(p)).astype(int)
            else:
                gen_iso = []
                for i in range(n):
                    gen_iso.append(multinomial_iso_rng.alias(table, alias, value))
                day_iso, gen_iso_num = np.unique(gen_iso, return_counts=True)  # Note, day_iso start from today. last entry represents no isolation.

            return day_iso, gen_iso_num

        # Disease schedule generator
        def calculate_mul(n, inf_table, inf_alias, inf_value):
            gen_inf = []
            for i in range(n):
                gen_inf.append(multinomial_inf_rng.alias(inf_table, inf_alias, inf_value))
            day_inf, gen_inf_num = np.unique(gen_inf, return_counts=True)

            return day_inf, gen_inf_num

        # Loop through day 1 - day n-1
        for day in range(0, self.factors['n']):

            if day == 0:
                num_exp_non_vac = np.array(infectious_non_vac[0, :], int)
                num_exp_vac = np.array(infectious_vac[0, :], int)

            else:
                # Update the states from the day before
                sus_vac[day, :] += sus_vac[day - 1, :]
                sus_non_vac[day, :] += sus_non_vac[day - 1, :]
                exposed_vac[day, :] += exposed_vac[day - 1, :]
                exposed_non_vac[day, :] += exposed_non_vac[day - 1, :]
                infectious_vac[day, :] += infectious_vac[day - 1, :]
                infectious_non_vac[day, :] += infectious_non_vac[day - 1, :]
                asymptomatic_vac[day, :] += asymptomatic_vac[day - 1, :]
                asymptomatic_non_vac[day, :] += asymptomatic_non_vac[day - 1, :]
                symptomatic_vac[day, :] += symptomatic_vac[day - 1, :]
                symptomatic_non_vac[day, :] += symptomatic_non_vac[day - 1, :]
                isolation_exp_non_vac[day, :] += isolation_exp_non_vac[day - 1, :]
                isolation_exp_vac[day, :] += isolation_exp_vac[day - 1, :]
                isolation_inf_non_vac[day, :] += isolation_inf_non_vac[day - 1, :]
                isolation_inf_vac[day, :] += isolation_inf_vac[day - 1, :]
                isolation_symp_non_vac[day, :] += isolation_symp_non_vac[day - 1, :]
                isolation_symp_vac[day, :] += isolation_symp_vac[day - 1, :]
                isolation_asymp_non_vac[day, :] += isolation_asymp_non_vac[day - 1, :]
                isolation_asymp_vac[day, :] += isolation_asymp_vac[day - 1, :]
                recovered[day, :] += recovered[day - 1, :]

                # Potential new exposed people (include those exposed but healthy and exposed will become infectious later)
                new_exp_vac = np.multiply(np.multiply(t_rate_vac, (infectious_vac[day, :] + infectious_non_vac[day, :] + asymptomatic_vac[day, :] + asymptomatic_non_vac[day, :])), (sus_vac[day, :] / (sus_vac[day, :] + sus_non_vac[day, :] + exposed_vac[day, :] + exposed_non_vac[day, :] + infectious_vac[day, :] + infectious_non_vac[day, :] + asymptomatic_vac[day, :] + asymptomatic_non_vac[day, :] + recovered[day, :])))
                new_exp_non_vac = np.multiply(np.multiply(t_rate, (infectious_vac[day, :] + infectious_non_vac[day, :] + asymptomatic_vac[day, :] + asymptomatic_non_vac[day, :])), (sus_non_vac[day, :] / (sus_vac[day, :] + sus_non_vac[day, :] + exposed_vac[day, :] + exposed_non_vac[day, :] + infectious_vac[day, :] + infectious_non_vac[day, :] + asymptomatic_vac[day, :] + asymptomatic_non_vac[day, :] + recovered[day, :])))

                num_exp_vac = [poisson_numexp_rng.poissonvariate(new_exp_vac[i]) for i in range(self.factors["num_groups"])]  # Number of newly exposed people (will have disease) this day
                num_exp_non_vac = [poisson_numexp_rng.poissonvariate(new_exp_non_vac[i]) for i in range(self.factors["num_groups"])]

                exposed_vac[day, :] = np.add(exposed_vac[day, :], num_exp_vac)
                exposed_non_vac[day, :] = np.add(exposed_non_vac[day, :], num_exp_non_vac)

                sus_vac[day, :] = np.subtract(sus_vac[day, :], num_exp_vac)  # Update the suspectible people (healthy)
                sus_non_vac[day, :] = np.subtract(sus_non_vac[day, :], num_exp_non_vac)

            # Get vector of testing frequency from decision variable
            freq = self.factors["freq_vac"][:self.factors["num_groups"]]

            # For exposed_non_vac:
            for g in range(self.factors["num_groups"]):

                if day > 0:
                    # Exp_num: unscheduled people in state exposed
                    exp_num = num_exp_non_vac[g]

                    day_exp, gen_exp_num = calculate_mul(exp_num, exp_table, exp_alias, exp_value)

                    for i in range(len(gen_exp_num)):
                        n = gen_exp_num[i]  # Subgroup of new comings according to their schedule
                        day_max = day_exp[i]
                        day_move_iso_exp, num_iso_exp = calculate_iso_mul(day_max, freq[g] * (1 - self.factors["false_neg"]), n)  # Calculate isolation schedule for new comings
                        # The last entry in num_iso_exp represents the number of people not isolated.
                        for j in range(len(num_iso_exp) - 1):
                            if day + day_move_iso_exp[j] < self.factors["n"]:
                                exposed_non_vac[day + day_move_iso_exp[j], g] -= num_iso_exp[j]  # Remove from exposed state
                                isolation_exp_non_vac[day + day_move_iso_exp[j], g] += num_iso_exp[j]  # Move to isolatioin exposed
                                exp_arr_iso[day + day_move_iso_exp[j], g] += num_iso_exp[j]   # Track the new comings for isolation exposed(to do disease schedule)
                                gen_exp_num[i] -= num_iso_exp[j]   # Remove them from the free newcoming
                                tot_num_isolated += num_iso_exp[j]

                    # Update for free new exposed people:
                    if day > 0:
                        for i in range(len(gen_exp_num)):
                            if day + day_exp[i] < self.factors["n"]:
                                infectious_non_vac[day + day_exp[i], g] += gen_exp_num[i]  # Move to infectious state
                                exposed_non_vac[day + day_exp[i], g] -= gen_exp_num[i]  # Remove from exposed state
                                inf_arr[day + day_exp[i], g] += gen_exp_num[i]  # Track the new comings for infectious state

                    # Schedule for isolated_exposed:
                    exp_iso_num = int(exp_arr_iso[day, g])
                    day_exp_iso, gen_exp_num_iso = calculate_mul(exp_iso_num, exp_table, exp_alias, exp_value)  # Disease schedule for isolation_exposed

                    if day > 0:
                        for i in range(len(gen_exp_num_iso)):
                            if day + day_exp_iso[i] < self.factors["n"]:
                                isolation_inf_non_vac[day + day_exp_iso[i], g] += gen_exp_num_iso[i]  # Move to isolation_infectious
                                isolation_exp_non_vac[day + day_exp_iso[i], g] -= gen_exp_num_iso[i]  # Remove from isolation_exposed
                                inf_arr_iso[day + day_exp_iso[i], g] += gen_exp_num_iso[i]  # Track the new comings for isolation_infectious

                # Get the number of new comings in infectious/symp/asymp states:
                if day == 0:
                    inf_num = int(infectious_non_vac[day, g])
                    asym_num = 0
                    sym_num = 0
                    # We do not do test at day0

                else:
                    inf_num = int(inf_arr[day, g])  # The number of new comings for infectious state
                    asym_num = int(asym_arr[day, g])
                    sym_num = int(sym_arr[day, g])

                # Do schedule for the free unscheduled people in infectious state
                day_inf, gen_inf_num = calculate_mul(inf_num, inf_table, inf_alias, inf_value)

                # Do test and isolation for new coming people:
                for i in range(len(gen_inf_num)):
                    # The number of new coming individuals that will become infectious in day_max days
                    n = gen_inf_num[i]
                    day_max = day_inf[i]
                    # Assume the day of leaving state do not do test at this state
                    day_move_iso_inf, num_iso_inf = calculate_iso_mul(day_max, freq[g] * (1 - self.factors["false_neg"]), n)  # Store the isolation schedule for this group of new comings
                    for j in range(len(num_iso_inf) - 1):  # Move them to isolation at corrosponding days
                        if day + day_move_iso_inf[j] < self.factors["n"]:
                            infectious_non_vac[day + day_move_iso_inf[j], g] -= num_iso_inf[j]
                            isolation_inf_non_vac[day + day_move_iso_inf[j], g] += num_iso_inf[j]
                            inf_arr_iso[day + day_move_iso_inf[j], g] += num_iso_inf[j]  # Track the number of new isolated
                            gen_inf_num[i] -= num_iso_inf[j]  # Remove them from the free newcoming of day i subgroup
                            tot_num_isolated += num_iso_inf[j]

                # Move the number of people to the symp/asymp at correponding days
                for i in range(len(gen_inf_num)):
                    if day + day_inf[i] < self.factors["n"]:
                        if gen_inf_num[i] != 0:
                            # Generate the number of asymptomatic people
                            asym_num_inf = binomial_asymp_rng.binomialvariate(int(gen_inf_num[i]), self.factors["asymp_rate"])
                            sym_num_inf = gen_inf_num[i] - asym_num_inf
                            symptomatic_non_vac[day + day_inf[i], g] += sym_num_inf  # Move to symptomatic state
                            asymptomatic_non_vac[day + day_inf[i], g] += asym_num_inf  # Move to asymptomatic state
                            sym_arr[day + day_inf[i], g] += sym_num_inf
                            asym_arr[day + day_inf[i], g] += asym_num_inf
                            tot_num_syptomatic += sym_num_inf  # Count the total symptomatic number
                            infectious_non_vac[day + day_inf[i], g] -= gen_inf_num[i]  # Remove from infectious state

                # Schedule for infectious isolated
                if day > 0:
                    inf_iso_num = int(inf_arr_iso[day, g])
                    day_inf_iso, gen_inf_num_iso = calculate_mul(inf_iso_num, inf_table, inf_alias, inf_value)

                    # If we only update the next day condition and then check the next
                    for i in range(len(gen_inf_num_iso)):
                        if day + day_inf_iso[i] < self.factors["n"]:
                            if gen_inf_num_iso[i] != 0:
                                asym_num_inf = binomial_asymp_rng.binomialvariate(int(gen_inf_num_iso[i]), self.factors["asymp_rate"])  # Number of people will go to isolation_asymp
                                sym_num_inf = gen_inf_num_iso[i] - asym_num_inf
                                symptomatic_non_vac[day + day_inf_iso[i], g] += sym_num_inf  # All symptomatic people are isolated
                                isolation_asymp_non_vac[day + day_inf_iso[i], g] += asym_num_inf
                                sym_arr[day + day_inf_iso[i], g] += sym_num_inf  # All symptomatic people are isolated
                                asym_arr_iso[day + day_inf_iso[i], g] += asym_num_inf
                                tot_num_syptomatic += sym_num_inf
                                isolation_inf_non_vac[day + day_inf_iso[i], g] -= gen_inf_num_iso[i]

                # For symptomatic:
                day_sym, gen_sym_num = calculate_mul(sym_num, sym_table, sym_alias, sym_value)

                for t in range(len(gen_sym_num)):
                    if day + day_sym[t] < self.factors["n"]:
                        symptomatic_non_vac[day + day_sym[t], g] -= gen_sym_num[t]
                        recovered[day + day_sym[t], g] += gen_sym_num[t]

                # For asymptomatic:
                day_asym, gen_asym_num = calculate_mul(asym_num, asym_table, asym_alias, asym_value)

                for i in range(len(gen_asym_num)):
                    n = gen_asym_num[i]
                    day_max = day_asym[i]
                    day_move_iso_asym, num_iso_asym = calculate_iso_mul(day_max, freq[g] * (1 - self.factors["false_neg"]), n)
                    for j in range(len(num_iso_asym) - 1):
                        if day + day_move_iso_asym[j] < self.factors["n"]:
                            asymptomatic_non_vac[day + day_move_iso_asym[j], g] -= num_iso_asym[j]
                            isolation_asymp_non_vac[day + day_move_iso_asym[j], g] += num_iso_asym[j]
                            asym_arr_iso[day + day_move_iso_asym[j], g] += num_iso_asym[j]
                            gen_asym_num[i] -= num_iso_asym[j]
                            tot_num_isolated += num_iso_asym[j]

                for t in range(len(gen_asym_num)):
                    if day + day_asym[t] < self.factors["n"]:
                        asymptomatic_non_vac[day + day_asym[t], g] -= gen_asym_num[t]
                        recovered[day + day_asym[t], g] += gen_asym_num[t]

                # For asymptomatic isolation:
                if day > 0:
                    asym_num_iso = int(asym_arr_iso[day, g])
                    day_asym_iso, gen_asym_num_iso = calculate_mul(asym_num_iso, asym_table, asym_alias, asym_value)

                    for t in range(len(gen_asym_num_iso)):
                        if day + day_asym_iso[t] < self.factors["n"]:
                            isolation_asymp_non_vac[day + day_asym_iso[t], g] -= gen_asym_num_iso[t]
                            recovered[day + day_asym_iso[t], g] += gen_asym_num_iso[t]

            # For vaccinated group:
            # For exposed_vac:
            for g in range(self.factors["num_groups"]):

                if day > 0:
                    # Exp_num: unscheduled people in state exposed
                    exp_num = num_exp_vac[g]
                    day_exp, gen_exp_num = calculate_mul(exp_num, exp_table, exp_alias, exp_value)

                    for i in range(len(gen_exp_num)):
                        n = gen_exp_num[i]  # Subgroup of new comings according to their schedule
                        day_max = day_exp[i]
                        day_move_iso_exp, num_iso_exp = calculate_iso_mul(day_max, freq[g] * (1 - self.factors["false_neg"]), n)  # Calculate isolation schedule for new comings
                        # The last entry in num_iso_exp represents the number of people not isolated.
                        for j in range(len(num_iso_exp) - 1):
                            if day + day_move_iso_exp[j] < self.factors["n"]:
                                exposed_vac[day + day_move_iso_exp[j], g] -= num_iso_exp[j]  #Remove from exposed state
                                isolation_exp_vac[day + day_move_iso_exp[j], g] += num_iso_exp[j]  # Move to isolatioin exposed
                                exp_arr_vac_iso[day + day_move_iso_exp[j], g] += num_iso_exp[j]
                                gen_exp_num[i] -= num_iso_exp[j]  # Remove them from the free newcoming
                                tot_num_isolated += num_iso_exp[j]

                    # Update for free new exposed people:
                    if day > 0:
                        for i in range(len(gen_exp_num)):
                            if day + day_exp[i] < self.factors["n"]:
                                infectious_vac[day + day_exp[i], g] += gen_exp_num[i]
                                exposed_vac[day + day_exp[i], g] -= gen_exp_num[i]
                                inf_arr_vac[day + day_exp[i], g] += gen_exp_num[i]

                    # Schedule for isolated_exposed:
                    exp_iso_num = int(exp_arr_vac_iso[day, g])
                    day_exp_iso, gen_exp_num_iso = calculate_mul(exp_iso_num, exp_table, exp_alias, exp_value)

                    if day > 0:
                        for i in range(len(gen_exp_num_iso)):
                            if day + day_exp_iso[i] < self.factors["n"]:
                                isolation_inf_vac[day + day_exp_iso[i], g] += gen_exp_num_iso[i]
                                isolation_exp_vac[day + day_exp_iso[i], g] -= gen_exp_num_iso[i]
                                inf_arr_vac_iso[day + day_exp_iso[i], g] += gen_exp_num_iso[i]

                # For infectious:
                if day == 0:
                    inf_num = int(infectious_vac[day, g])
                    asym_num = 0
                    sym_num = 0
                    # We do not do test at day 0

                else:
                    inf_num = int(inf_arr_vac[day, g])
                    asym_num = int(asym_arr_vac[day, g])
                    sym_num = int(sym_arr_vac[day, g])

                # Do schedule for the free unscheduled people in infectious state
                day_inf, gen_inf_num = calculate_mul(inf_num, inf_table, inf_alias, inf_value)

                # Do test and isolation for new coming people:
                for i in range(len(gen_inf_num)):
                    # The number of new coming individuals that will become infectious in day_max days
                    n = gen_inf_num[i]
                    day_max = day_inf[i]
                    # Assume the day of leaving state do not do test at this state
                    day_move_iso_inf, num_iso_inf = calculate_iso_mul(day_max, freq[g] * (1 - self.factors["false_neg"]), n)  # Store the isolation schedule for this group of new comings
                    for j in range(len(num_iso_inf) - 1):  # Move them to isolation at corrosponding days
                        if day + day_move_iso_inf[j] < self.factors["n"]:
                            infectious_vac[day + day_move_iso_inf[j], g] -= num_iso_inf[j]
                            isolation_inf_vac[day + day_move_iso_inf[j], g] += num_iso_inf[j]
                            inf_arr_vac_iso[day + day_move_iso_inf[j], g] += num_iso_inf[j]  # Track the number of new isolated
                            gen_inf_num[i] -= num_iso_inf[j]  # Remove them from the free newcoming of day i subgroup
                            tot_num_isolated += num_iso_inf[j]

                # Move the number of people to the symp/asymp at correponding days
                for i in range(len(gen_inf_num)):
                    if day + day_inf[i] < self.factors["n"]:
                        if gen_inf_num[i] != 0:
                            # Generate the number of asymptomatic people
                            asym_num_inf = binomial_asymp_rng.binomialvariate(int(gen_inf_num[i]), self.factors["asymp_rate_vac"])
                            sym_num_inf = gen_inf_num[i] - asym_num_inf
                            symptomatic_vac[day + day_inf[i], g] += sym_num_inf
                            asymptomatic_vac[day + day_inf[i], g] += asym_num_inf
                            sym_arr_vac[day + day_inf[i], g] += sym_num_inf
                            asym_arr_vac[day + day_inf[i], g] += asym_num_inf
                            tot_num_syptomatic += sym_num_inf
                            infectious_vac[day + day_inf[i], g] -= gen_inf_num[i]

                # Schedule for infectious isolated
                if day > 0:
                    inf_iso_num = int(inf_arr_vac_iso[day, g])
                    day_inf_iso, gen_inf_num_iso = calculate_mul(inf_iso_num, inf_table, inf_alias, inf_value)

                    # If we only update the next day condition and then check the next
                    for i in range(len(gen_inf_num_iso)):
                        if day + day_inf_iso[i] < self.factors["n"]:
                            if gen_inf_num_iso[i] != 0:
                                asym_num_inf = binomial_asymp_rng.binomialvariate(int(gen_inf_num_iso[i]), self.factors["asymp_rate_vac"])
                                sym_num_inf = gen_inf_num_iso[i] - asym_num_inf
                                symptomatic_vac[day + day_inf_iso[i], g] += sym_num_inf  # All symptomatic people are isolated
                                isolation_asymp_vac[day + day_inf_iso[i], g] += asym_num_inf
                                sym_arr_vac[day + day_inf_iso[i], g] += sym_num_inf  # All symptomatic people are isolated
                                asym_arr_vac_iso[day + day_inf_iso[i], g] += asym_num_inf
                                tot_num_syptomatic += sym_num_inf
                                isolation_inf_vac[day + day_inf_iso[i], g] -= gen_inf_num_iso[i]

                # For symptomatic:
                day_sym, gen_sym_num = calculate_mul(sym_num, sym_table, sym_alias, sym_value)

                for t in range(len(gen_sym_num)):
                    if day + day_sym[t] < self.factors["n"]:
                        symptomatic_vac[day + day_sym[t], g] -= gen_sym_num[t]
                        recovered[day + day_sym[t], g] += gen_sym_num[t]

                # For asymptomatic:
                day_asym, gen_asym_num = calculate_mul(asym_num, asym_table, asym_alias, asym_value)

                for i in range(len(gen_asym_num)):
                    n = gen_asym_num[i]
                    day_max = day_asym[i]
                    day_move_iso_asym, num_iso_asym = calculate_iso_mul(day_max, freq[g] * (1 - self.factors["false_neg"]), n)
                    for j in range(len(num_iso_asym) - 1):
                        if day + day_move_iso_asym[j] < self.factors["n"]:
                            asymptomatic_vac[day + day_move_iso_asym[j], g] -= num_iso_asym[j]
                            isolation_asymp_vac[day + day_move_iso_asym[j], g] += num_iso_asym[j]
                            asym_arr_vac_iso[day + day_move_iso_asym[j], g] += num_iso_asym[j]
                            gen_asym_num[i] -= num_iso_asym[j]
                            tot_num_isolated += num_iso_asym[j]

                for t in range(len(gen_asym_num)):
                    if day + day_asym[t] < self.factors["n"]:
                        asymptomatic_vac[day + day_asym[t], g] -= gen_asym_num[t]
                        recovered[day + day_asym[t], g] += gen_asym_num[t]

                # For asymptomatic isolation:
                if day > 0:
                    asym_num_iso = int(asym_arr_vac_iso[day, g])
                    day_asym_iso, gen_asym_num_iso = calculate_mul(asym_num_iso, asym_table, asym_alias, asym_value)

                    for t in range(len(gen_asym_num_iso)):
                        if day + day_asym_iso[t] < self.factors["n"]:
                            isolation_asymp_vac[day + day_asym_iso[t], g] -= gen_asym_num_iso[t]
                            recovered[day + day_asym_iso[t], g] += gen_asym_num_iso[t]

            # Update performance measures
            for g in range(self.factors["num_groups"]):
                num_exposed[day][g] = np.sum(exposed_non_vac[day, g] + exposed_vac[day, g] + isolation_exp_non_vac[day, g] + isolation_exp_vac[day, g])
                num_susceptible[day][g] = np.sum(sus_vac[day, g] + sus_non_vac[day, g])
                num_symptomatic[day][g] = np.sum(symptomatic_vac[day, g] + symptomatic_non_vac[day, g])  # All symp people are isolated. isolated_symp = symp
                num_recovered[day][g] = np.sum(recovered[day, g])
                num_infected[day][g] = np.sum(infectious_non_vac[day, g] + infectious_vac[day, g] + symptomatic_non_vac[day, g] + symptomatic_vac[day, g] + asymptomatic_non_vac[day, g] + asymptomatic_vac[day, g] +
                                              isolation_inf_non_vac[day, g] + isolation_inf_vac[day, g] + isolation_asymp_non_vac[day, g] + isolation_asymp_vac[day, g])
                num_isolation[day][g] = np.sum(isolation_exp_non_vac[day, g] + isolation_inf_non_vac[day, g] + symptomatic_non_vac[day, g] + isolation_asymp_non_vac[day, g] +
                                               isolation_exp_vac[day, g] + isolation_inf_vac[day, g] + symptomatic_vac[day, g] + isolation_asymp_vac[day, g])

        # Compose responses and gradients.
        last_susceptible = num_susceptible[self.factors["n"] - 1]  # Number of suspectible(unaffected) people at the end of the period
        inf = self.factors["group_size"] - last_susceptible  # Number of total infected people during the period
        w = np.array(self.factors["w"])  # Protection-rate for each group, here we heavy scale for the group3 as example w=[1,1,5]; if no scaling, set w=np.ones
        tot_infected = np.dot(w, inf)  # After adjusting for the protection rate, the weighted objective value (adjusted infected)

        responses = {"num_infected": num_infected,
                     "num_exposed": num_exposed,
                     "num_susceptible": num_susceptible,
                     "num_recovered": num_recovered,
                     "num_symptomatic": num_symptomatic,
                     "free_population": last_susceptible,
                     "num_isolation": num_isolation,
                     "total_symp": tot_num_syptomatic,
                     "total_isolated": tot_num_isolated + tot_num_syptomatic,
                     "total_infected": tot_infected}
        gradients = {response_key:
                     {factor_key: np.nan for factor_key in self.specifications}
                     for response_key in responses
                     }
        return responses, gradients


"""
Summary
-------
Minimize the average number of daily infected people.
"""


class CovidMinInfectVac(Problem):
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
    def __init__(self, name="COVID-vac-test-1", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_fixed_factors = {}
        self.model_decision_factors = {"freq_vac"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                # "default": (0/7, 0/7, 0/7, 0/7, 0/7, 0/7, 0/7, 0, 0, 0, 0, 0, 0, 0)  #for 7 groups
                "default": (0/7, 0/7, 0/7, 0, 0, 0)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 600
            },
            "vaccine_cap": {
                "description": "Maximum number of vaccines.",
                "datatype": int,
                "default": 8000
            },
            "testing_cap": {
                "description": "Maximum testing capacity per day.",
                "datatype": int,
                "default": 5000
            },
            "pen_coef": {
                "description": "Penalty coefficient.",
                "datatype": int,
                "default": 100000
            },
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "testing_cap": self.check_testing_cap,
            "vaccine_cap": self.check_vaccine_cap,
            "pen_coef": self.check_pen_coef
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = COVID_vac(self.model_fixed_factors)
        self.dim = len(self.model.factors["group_size"])
        self.lower_bounds = (0,) * self.dim
        self.upper_bounds = (1,) * self.dim

    def check_vaccine_cap(self):
        return self.factors["vaccine_cap"] > 0

    def check_testing_cap(self):
        return self.factors["testing_cap"] > 0

    def check_pen_coef(self):
        return self.factors["pen_coef"] > 0

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
            "freq_vac": vector
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
        vector = (factor_dict["freq_vac"], )
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
        objectives = (response_dict["total_symp"], )
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
        # Checking:
        # print(self.factors['pen_coef'])
        # for g in range(self.dim):
        #     print(np.dot(self.model.factors["group_size"][g],x[g]))
        # print(np.sum(np.dot(self.model.factors["group_size"][g],x[g]) for g in range(self.dim)))
        # print(max(np.sum(np.dot(self.model.factors["group_size"][g],x[g]) for g in range(self.dim)) - self.factors["vaccine_cap"],0))

        x_v = x[self.dim: self.dim * 2]
        x_t = x[: self.dim]
        det_objectives_test = (self.factors["pen_coef"] * (max(np.sum(np.ceil(np.dot(self.model.factors["group_size"][g], x_t[g])) for g in range(self.dim)) - self.factors["testing_cap"], 0)),)
        det_objectives_vac = (self.factors["pen_coef"] * (max(np.sum(np.ceil(np.dot(self.model.factors["group_size"][g], x_v[g])) for g in range(self.dim)) - self.factors["vaccine_cap"], 0)),)
        det_objectives = max(det_objectives_test, det_objectives_vac)
        det_objectives_gradients = ((0,),)
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
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
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
        return np.all(x >= 0)

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
        # If uniformly find freq & vac:
        # x = tuple([rand_sol_rng.uniform(0,1) for _ in range(self.dim)])
        # f = tuple([rand_sol_rng.uniform(0,1) for _ in range(self.dim)])
        # dx = f + x

        # By redistribution method: (reallocate the resource(vac/freq) if some group have more resource than their population size)
        # Linear constraint
        # For vaccine:
        x = tuple(rand_sol_rng.integer_random_vector_from_simplex(self.model.factors["num_groups"], self.factors["vaccine_cap"]))
        v = np.divide(x, self.model.factors["group_size"])

        x = np.array(x)  # For the temporal calculation
        v = np.array(v)

        n = len(self.model.factors["group_size"])
        while True:
            xn = x - np.array(self.model.factors["group_size"])
            idx = np.argmax(xn)
            if v[idx] > 1:
                remain = x[idx] - self.model.factors["group_size"][idx]
                x[idx] -= remain
                x += int(np.floor(remain / n))
                short_idx = np.argmax(np.array(self.model.factors["group_size"]) - x)
                rr = remain - int(np.floor(remain / 3) * n)
                if rr > 0:
                    x[short_idx] += rr
                v = np.divide(x, self.model.factors["group_size"])
            else:
                break

        x = tuple(x)
        v = tuple(v)

        # If modify testing freq by redistribution, for testing:
        xf = tuple(rand_sol_rng.integer_random_vector_from_simplex(self.model.factors["num_groups"], self.factors["testing_cap"]))
        f = np.divide(xf, self.model.factors["group_size"])

        xf = np.array(xf)  # For the temporal calculation
        f = np.array(f)

        n = len(self.model.factors["group_size"])
        while True:
            xn = xf - np.array(self.model.factors["group_size"])
            idx = np.argmax(xn)
            if f[idx] > 1:
                remain = xf[idx] - self.model.factors["group_size"][idx]
                xf[idx] -= remain
                xf += int(np.floor(remain / n))
                short_idx = np.argmax(np.array(self.model.factors["group_size"]) - xf)
                rr = remain - int(np.floor(remain / 3) * n)
                if rr > 0:
                    xf[short_idx] += rr
                f = np.divide(xf, self.model.factors["group_size"])
            else:
                break

        xf = tuple(xf)
        f = tuple(f)

        print('initial f, v: ', f, v)

        # Final decision variable freq_vac:
        dx = f + v

        return dx


# Other problem classes:
# Sub-problem 2: problem with more groups.
class CovidMinInfectVac2(Problem):
    """
    Base class to implement simulation-optimization problems.
    In this problem classes, consider there are 7 groups in total. 
    Thus, to study this problem class, one need to modify the model part default values to those corresponding to 7 groups.

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
    def __init__(self, name="COVID-vac-test-2", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_fixed_factors = {}
        self.model_decision_factors = {"freq_vac"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (0/7, 0/7, 0/7, 0/7, 0/7, 0/7, 0/7, 0, 0, 0, 0, 0, 0, 0)  #for 7 groups
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 600
            },
            "vaccine_cap": {
                "description": "Maximum number of vaccines.",
                "datatype": int,
                "default": 8000
            },
            "testing_cap": {
                "description": "Maximum testing capacity per day.",
                "datatype": int,
                "default": 5000
            },
            "pen_coef": {
                "description": "Penalty coefficient.",
                "datatype": int,
                "default": 100000
            },
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "testing_cap": self.check_testing_cap,
            "vaccine_cap": self.check_vaccine_cap,
            "pen_coef": self.check_pen_coef
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = COVID_vac(self.model_fixed_factors)
        self.dim = len(self.model.factors["group_size"])
        self.lower_bounds = (0,) * self.dim
        self.upper_bounds = (1,) * self.dim

    def check_vaccine_cap(self):
        return self.factors["vaccine_cap"] > 0

    def check_testing_cap(self):
        return self.factors["testing_cap"] > 0

    def check_pen_coef(self):
        return self.factors["pen_coef"] > 0

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
            "freq_vac": vector
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
        vector = (factor_dict["freq_vac"], )
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
        objectives = (response_dict["total_infected"], )
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
        # Checking:
        # print(self.factors['pen_coef'])
        # for g in range(self.dim):
        #     print(np.dot(self.model.factors["group_size"][g],x[g]))
        # print(np.sum(np.dot(self.model.factors["group_size"][g],x[g]) for g in range(self.dim)))
        # print(max(np.sum(np.dot(self.model.factors["group_size"][g],x[g]) for g in range(self.dim)) - self.factors["vaccine_cap"],0))

        x_v = x[self.dim: self.dim * 2]
        x_t = x[: self.dim]
        det_objectives_test = (self.factors["pen_coef"] * (max(np.sum(np.ceil(np.dot(self.model.factors["group_size"][g], x_t[g])) for g in range(self.dim)) - self.factors["testing_cap"], 0)),)
        det_objectives_vac = (self.factors["pen_coef"] * (max(np.sum(np.ceil(np.dot(self.model.factors["group_size"][g], x_v[g])) for g in range(self.dim)) - self.factors["vaccine_cap"], 0)),)
        det_objectives = max(det_objectives_test, det_objectives_vac)
        det_objectives_gradients = ((0,),)
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
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
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
        return np.all(x >= 0)

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
        # If uniformly find freq & vac:
        # x = tuple([rand_sol_rng.uniform(0,1) for _ in range(self.dim)])
        # f = tuple([rand_sol_rng.uniform(0,1) for _ in range(self.dim)])
        # dx = f + x

        # By redistribution method: (reallocate the resource(vac/freq) if some group have more resource than their population size)
        # Linear constraint
        # For vaccine:
        x = tuple(rand_sol_rng.integer_random_vector_from_simplex(self.model.factors["num_groups"], self.factors["vaccine_cap"]))
        v = np.divide(x, self.model.factors["group_size"])

        x = np.array(x)  # For the temporal calculation
        v = np.array(v)

        n = len(self.model.factors["group_size"])
        while True:
            xn = x - np.array(self.model.factors["group_size"])
            idx = np.argmax(xn)
            if v[idx] > 1:
                remain = x[idx] - self.model.factors["group_size"][idx]
                x[idx] -= remain
                x += int(np.floor(remain / n))
                short_idx = np.argmax(np.array(self.model.factors["group_size"]) - x)
                rr = remain - int(np.floor(remain / 3) * n)
                if rr > 0:
                    x[short_idx] += rr
                v = np.divide(x, self.model.factors["group_size"])
            else:
                break

        x = tuple(x)
        v = tuple(v)

        # If modify testing freq by redistribution, for testing:
        xf = tuple(rand_sol_rng.integer_random_vector_from_simplex(self.model.factors["num_groups"], self.factors["testing_cap"]))
        f = np.divide(xf, self.model.factors["group_size"])

        xf = np.array(xf)  # For the temporal calculation
        f = np.array(f)

        n = len(self.model.factors["group_size"])
        while True:
            xn = xf - np.array(self.model.factors["group_size"])
            idx = np.argmax(xn)
            if f[idx] > 1:
                remain = xf[idx] - self.model.factors["group_size"][idx]
                xf[idx] -= remain
                xf += int(np.floor(remain / n))
                short_idx = np.argmax(np.array(self.model.factors["group_size"]) - xf)
                rr = remain - int(np.floor(remain / 3) * n)
                if rr > 0:
                    xf[short_idx] += rr
                f = np.divide(xf, self.model.factors["group_size"])
            else:
                break

        xf = tuple(xf)
        f = tuple(f)

        print('initial f, v: ', f, v)

        # Final decision variable freq_vac:
        dx = f + v

        return dx


# Sub-problem 3: problem that consider different protection weight for different groups.
class CovidMinInfectVac3(Problem):
    """
    Base class to implement simulation-optimization problems.
    This problem studies the case when different groups are influenced by the disease to various degree.
    To study this problem, one need to modify the factor "w" in model classes to assign a protection weight for each group.

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
    def __init__(self, name="COVID-vac-test-3", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_fixed_factors = {}
        self.model_decision_factors = {"freq_vac"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (0/7, 0/7, 0/7, 0, 0, 0)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 600
            },
            "vaccine_cap": {
                "description": "Maximum number of vaccines.",
                "datatype": int,
                "default": 8000
            },
            "testing_cap": {
                "description": "Maximum testing capacity per day.",
                "datatype": int,
                "default": 5000
            },
            "pen_coef": {
                "description": "Penalty coefficient.",
                "datatype": int,
                "default": 100000
            },
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "testing_cap": self.check_testing_cap,
            "vaccine_cap": self.check_vaccine_cap,
            "pen_coef": self.check_pen_coef
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = COVID_vac(self.model_fixed_factors)
        self.dim = len(self.model.factors["group_size"])
        self.lower_bounds = (0,) * self.dim
        self.upper_bounds = (1,) * self.dim

    def check_vaccine_cap(self):
        return self.factors["vaccine_cap"] > 0

    def check_testing_cap(self):
        return self.factors["testing_cap"] > 0

    def check_pen_coef(self):
        return self.factors["pen_coef"] > 0

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
            "freq_vac": vector
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
        vector = (factor_dict["freq_vac"], )
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
        objectives = (response_dict["total_infected"], )  # here, we use the weighted number of infected people as objective;
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
        # Checking:
        # print(self.factors['pen_coef'])
        # for g in range(self.dim):
        #     print(np.dot(self.model.factors["group_size"][g],x[g]))
        # print(np.sum(np.dot(self.model.factors["group_size"][g],x[g]) for g in range(self.dim)))
        # print(max(np.sum(np.dot(self.model.factors["group_size"][g],x[g]) for g in range(self.dim)) - self.factors["vaccine_cap"],0))

        x_v = x[self.dim: self.dim * 2]
        x_t = x[: self.dim]
        det_objectives_test = (self.factors["pen_coef"] * (max(np.sum(np.ceil(np.dot(self.model.factors["group_size"][g], x_t[g])) for g in range(self.dim)) - self.factors["testing_cap"], 0)),)
        det_objectives_vac = (self.factors["pen_coef"] * (max(np.sum(np.ceil(np.dot(self.model.factors["group_size"][g], x_v[g])) for g in range(self.dim)) - self.factors["vaccine_cap"], 0)),)
        det_objectives = max(det_objectives_test, det_objectives_vac)
        det_objectives_gradients = ((0,),)
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
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
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
        return np.all(x >= 0)

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
        # If uniformly find freq & vac:
        # x = tuple([rand_sol_rng.uniform(0,1) for _ in range(self.dim)])
        # f = tuple([rand_sol_rng.uniform(0,1) for _ in range(self.dim)])
        # dx = f + x

        # By redistribution method: (reallocate the resource(vac/freq) if some group have more resource than their population size)
        # Linear constraint
        # For vaccine:
        x = tuple(rand_sol_rng.integer_random_vector_from_simplex(self.model.factors["num_groups"], self.factors["vaccine_cap"]))
        v = np.divide(x, self.model.factors["group_size"])

        x = np.array(x)  # For the temporal calculation
        v = np.array(v)

        n = len(self.model.factors["group_size"])
        while True:
            xn = x - np.array(self.model.factors["group_size"])
            idx = np.argmax(xn)
            if v[idx] > 1:
                remain = x[idx] - self.model.factors["group_size"][idx]
                x[idx] -= remain
                x += int(np.floor(remain / n))
                short_idx = np.argmax(np.array(self.model.factors["group_size"]) - x)
                rr = remain - int(np.floor(remain / 3) * n)
                if rr > 0:
                    x[short_idx] += rr
                v = np.divide(x, self.model.factors["group_size"])
            else:
                break

        x = tuple(x)
        v = tuple(v)

        # If modify testing freq by redistribution, for testing:
        xf = tuple(rand_sol_rng.integer_random_vector_from_simplex(self.model.factors["num_groups"], self.factors["testing_cap"]))
        f = np.divide(xf, self.model.factors["group_size"])

        xf = np.array(xf)  # For the temporal calculation
        f = np.array(f)

        n = len(self.model.factors["group_size"])
        while True:
            xn = xf - np.array(self.model.factors["group_size"])
            idx = np.argmax(xn)
            if f[idx] > 1:
                remain = xf[idx] - self.model.factors["group_size"][idx]
                xf[idx] -= remain
                xf += int(np.floor(remain / n))
                short_idx = np.argmax(np.array(self.model.factors["group_size"]) - xf)
                rr = remain - int(np.floor(remain / 3) * n)
                if rr > 0:
                    xf[short_idx] += rr
                f = np.divide(xf, self.model.factors["group_size"])
            else:
                break

        xf = tuple(xf)
        f = tuple(f)

        print('initial f, v: ', f, v)

        # Final decision variable freq_vac:
        dx = f + v

        return dx


# Sub-problem 4: with given testing frequency for each group, find the optimal vaccination policy.
class CovidMinInfectVac4(Problem):
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
    def __init__(self, name="COVID-vac-test-4", fixed_factors={}, model_fixed_factors={}):
        self.name = name
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_available = True
        self.optimal_value = None
        self.optimal_solution = None
        self.model_default_factors = {}
        self.model_fixed_factors = {}
        self.model_decision_factors = {"freq_vac"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {
                "description": "Initial solution from which solvers start.",
                "datatype": tuple,
                "default": (1/7, 1/7, 1/7, 0, 0, 0)
            },
            "budget": {
                "description": "Max # of replications for a solver to take.",
                "datatype": int,
                "default": 600
            },
            "vaccine_cap": {
                "description": "Maximum number of vaccines.",
                "datatype": int,
                "default": 8000
            },
            "testing_cap": {
                "description": "Maximum testing capacity per day.",
                "datatype": int,
                "default": 5000
            },
            "testing_freq": {
                "description": "Testing frequency for each group each day.",
                "datatype": tuple,
                "default": (1/7, 1/7, 1/7)
            },
            "pen_coef": {
                "description": "Penalty coefficient.",
                "datatype": int,
                "default": 100000
            },
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
            "testing_cap": self.check_testing_cap,
            "vaccine_cap": self.check_vaccine_cap,
            "testing_freq": self.check_testing_freq,
            "pen_coef": self.check_pen_coef
        }
        super().__init__(fixed_factors, model_fixed_factors)
        # Instantiate model with fixed factors and overwritten defaults.
        self.model = COVID_vac(self.model_fixed_factors)
        self.dim = len(self.model.factors["group_size"])
        self.lower_bounds = (0,) * self.dim
        self.upper_bounds = (1,) * self.dim

    def check_vaccine_cap(self):
        return self.factors["vaccine_cap"] > 0

    def check_testing_cap(self):
        return self.factors["testing_cap"] > 0
    
    def check_testing_freq(self):
        return all(np.array(self.factors["testing_freq"]) >= 0)

    def check_pen_coef(self):
        return self.factors["pen_coef"] > 0

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
            "freq_vac": vector
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
        vector = (factor_dict["freq_vac"], )
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
        objectives = (response_dict["total_infected"], )
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
        # Checking:
        # print(self.factors['pen_coef'])
        # for g in range(self.dim):
        #     print(np.dot(self.model.factors["group_size"][g],x[g]))
        # print(np.sum(np.dot(self.model.factors["group_size"][g],x[g]) for g in range(self.dim)))
        # print(max(np.sum(np.dot(self.model.factors["group_size"][g],x[g]) for g in range(self.dim)) - self.factors["vaccine_cap"],0))

        x_v = x[self.dim: self.dim * 2]
        x_t = x[: self.dim]
        det_objectives_test = (self.factors["pen_coef"] * (max(np.sum(np.ceil(np.dot(self.model.factors["group_size"][g], x_t[g])) for g in range(self.dim)) - self.factors["testing_cap"], 0)),)
        det_objectives_vac = (self.factors["pen_coef"] * (max(np.sum(np.ceil(np.dot(self.model.factors["group_size"][g], x_v[g])) for g in range(self.dim)) - self.factors["vaccine_cap"], 0)),)
        det_objectives = max(det_objectives_test, det_objectives_vac)
        det_objectives_gradients = ((0,),)
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
            vector of deterministic components of stochastic constraints
        det_stoch_constraints_gradients : tuple
            vector of gradients of deterministic components of
            stochastic constraints
        """
        det_stoch_constraints = None
        det_stoch_constraints_gradients = None
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
        return np.all(x >= 0)

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
        # If uniformly find freq & vac:
        # x = tuple([rand_sol_rng.uniform(0,1) for _ in range(self.dim)])
        # f = tuple([rand_sol_rng.uniform(0,1) for _ in range(self.dim)])
        # dx = f + x

        # By redistribution method: (reallocate the resource(vac/freq) if some group have more resource than their population size)
        # Linear constraint
        # For vaccine:
        x = tuple(rand_sol_rng.integer_random_vector_from_simplex(self.model.factors["num_groups"], self.factors["vaccine_cap"]))
        v = np.divide(x, self.model.factors["group_size"])

        x = np.array(x)  # For the temporal calculation
        v = np.array(v)

        n = len(self.model.factors["group_size"])
        while True:
            xn = x - np.array(self.model.factors["group_size"])
            idx = np.argmax(xn)
            if v[idx] > 1:
                remain = x[idx] - self.model.factors["group_size"][idx]
                x[idx] -= remain
                x += int(np.floor(remain / n))
                short_idx = np.argmax(np.array(self.model.factors["group_size"]) - x)
                rr = remain - int(np.floor(remain / 3) * n)
                if rr > 0:
                    x[short_idx] += rr
                v = np.divide(x, self.model.factors["group_size"])
            else:
                break

        x = tuple(x)
        v = tuple(v)

        # with given testing policy for the period:
        f = self.factors["testing_freq"]

        xf = tuple(xf)
        f = tuple(f)

        print('initial f, v: ', f, v)

        # Final decision variable freq_vac:
        dx = f + v

        return dx