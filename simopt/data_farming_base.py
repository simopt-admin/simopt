from directory import oracle_directory
from rng.mrg32k3a import MRG32k3a
from copy import deepcopy


class DesignPoint(object):
    """
    Base class for design points represented as dictionaries of factors.

    Attributes
    ----------
    oracle : Oracle object
        oracle to simulate
    oracle_factors : dict
        oracle factor names and values
    n_reps : int
        number of replications run at a design point
    responses : dict
        responses observed from replications
    gradients : dict of dict
        gradients of responses (w.r.t. oracle factors) observed from replications

    Arguments
    ---------
    oracle : Oracle object
        oracle with factors oracle_factors
    """
    def __init__(self, oracle):
        super().__init__()
        # create separate copy of Oracle object for use at this design point
        self.oracle = deepcopy(oracle)
        self.oracle_factors = self.oracle.factors
        self.n_reps = 0
        self.responses = {}
        self.gradients = {}

    def simulate(self, m=1):
        """
        Simulate m replications for the current oracle factors.
        Append results to the responses and gradients dictionaries.

        Arguments
        ---------
        m : int > 0
            number of macroreplications to run at the design point
        """
        for _ in range(m):
            # generate a single replication of oracle, as described by design point
            responses, gradients = self.oracle.replicate()
            # if first replication, set up recording responses and gradients
            if self.n_reps == 0:
                self.responses = {response_key: [] for response_key in responses}
                self.gradients = {response_key: {factor_key: [] for factor_key in gradients[response_key]} for response_key in responses}
            # append responses and gradients
            for key in self.responses:
                self.responses[key].append(responses[key])
            for outerkey in self.gradients:
                for innerkey in self.gradients[outerkey]:
                    self.gradients[outerkey][innerkey].append(gradients[outerkey][innerkey])
            # increment counter
            self.n_reps += 1
            # advance rngs to start of next subsubstream
            for rng in self.oracle.rng_list:
                print(rng.s_ss_sss_index)
                rng.advance_subsubstream()
            print("on to next run")


class DataFarmingExperiment(object):
    """
    Base class for data-farming experiments consisting of an oracle
    and design of associated factors.

    Attributes
    ----------
    oracle : Oracle object
        oracle on which the experiment is run
    design : list of DesignPoint objects
        list of design points forming the design
    n_design_pts : int
        number of design points in the design

    Arguments
    ---------
    oracle : oracle name
        name of oracle on which the experiment is run
    oracle_fixed_factors : dictionary
        non-default values of oracle factors that will not be varied
    design_filename : string
        name of file containing design matrix
    """
    def __init__(self, oracle_name, oracle_fixed_factors={}, design_filename=None):
        # initialize oracle object with fixed factors
        self.oracle = oracle_directory[oracle_name](fixed_factors=oracle_fixed_factors)
        # HARD-CODED FOR GIVEN DESIGN MATRIX.
        # WILL LATER READ IN DESIGN MATRIX FROM FILE
        design_table = [[1, 3], [2, 4], [3, 5]]
        self.n_design_pts = len(design_table)
        # create all design points
        self.design = []
        for i in range(self.n_design_pts):
            # HARD-CODED TO PARSE DESIGN TABLE IN A CERTAIN WAY
            # WILL LATER GENERALIZE TO READ FACTOR NAMES FROM COLUMN HEADERS
            # parse oracle factors for next design point
            design_pt_factors = {"lambda": design_table[i][0], "mu": design_table[i][1]}
            # update oracle factors according to next design point
            self.oracle.factors.update(design_pt_factors)
            # create new design point and add to design
            self.design.append(DesignPoint(self.oracle))

    def run(self, n_reps=10, crn_across_design_pts=True):
        """
        Run a fixed number of macroreplications at each design point.

        Arguments
        ---------
        n_reps : int
            number of replications run at each design point
        crn_across_design_pts : Boolean
            use CRN across design points?
        """
        # setup random number generators for oracle
        # use stream 0 for all runs; start with substreams 0, 1, ..., oracle.n_rngs-1
        rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(self.oracle.n_rngs)]
        # all design points will share the same random number generator objects
        # simulate n_reps replications from each design point
        for design_pt in self.design:
            # attach random number generators
            design_pt.oracle.attach_rngs(rng_list)
            # simulate n_reps replications from each design point
            design_pt.simulate(n_reps)
            # manage random number streams
            if crn_across_design_pts is True:
                # reset rngs to start of current substream
                for rng in rng_list:
                    rng.reset_substream()
            else:  # if not using CRN
                # advance rngs to starts of next set of substreams
                for rng in rng_list:
                    for _ in range(len(rng_list)):
                        rng.advance_substream()
