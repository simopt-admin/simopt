## Welcome to SimOpt!

This branch contains a previous stable version of the testbed written in MATLAB.

> ⚠️ MATLAB versions of this testbed are no longer supported

This [WSC paper](https://eckman.engr.tamu.edu/wp-content/uploads/sites/233/2020/01/SimOptRevisions-paper.pdf) from 2019 gives a detailed overview of this version and how to run experiments; see Section 3 in particular.

The purpose of the SimOpt testbed is to encourage development and constructive comparison of simulation-optimization (SO) solvers (algorithms). We are particularly interested in the finite-time performance of solvers, rather than the asymptotic results that one often finds in related literature.

For the purposes of this site, we define simulation as a very general technique for estimating statistical measures of complex systems. A system is modeled as if the probability distributions of the underlying random variables were known. Realizations of these random variables are then drawn randomly from these distributions. Each replication gives one observation of the system response, i.e., an evaluation of the objective function. By simulating a system in this fashion for multiple replications and aggregating the responses, one can compute statistics and use them for evaluation and design.

The paper [Pasupathy and Henderson (2006)](https://www.informs-sim.org/wsc06papers/028.pdf) explains the original motivation for the testbed, and the follow-up paper [Pasupathy and Henderson (2011)](https://www.informs-sim.org/wsc11papers/363.pdf) describes an earlier interface for MATLAB implementations of problems and solvers. The paper [Dong et al. (2017)](https://www.informs-sim.org/wsc17papers/includes/files/179.pdf) conducts an experimental comparison of several solvers in SimOpt and analyzes their relative performance. The recent Winter Simulation Conference paper [Eckman et al. (2019)](https://www.informs-sim.org/wsc19papers/374.pdf) describes in detail the recent changes to the architecture of SimOpt and the control of random number streams.

## Instructions

### Steps to run one or more solvers on one or more problems:

**1.** Clone or fork the [simopt](https://github.com/simopt-admin/simopt) repository.

**2.** Open MATLAB and set the path to the [*Experiments*](https://github.com/simopt-admin/simopt/tree/master/Experiments) folder.

**3.** From the MATLAB terminal, run the command

`RunWrapper({'ProblemName1', 'ProblemName2', ...}, {'SolverName1', 'SolverName2', ...}, NumberOfMacroreplications)`

This will run multiple macroreplications of each specified solver on each specified problem. The outputs from each solver-problem pair will be saved in a MATLAB workspace (.mat) file in the [RawData](https://github.com/simopt-admin/simopt/tree/master/Experiments/RawData) folder.

For example, the command `RunWrapper({'SAN'},{'NELDMD'},10)` will run 10 macroreplications of the [Nelder-Mead](https://github.com/simopt-admin/simopt/tree/master/Solvers/NELDMD) solver on the [Stochastic Activity Network](https://github.com/simopt-admin/simopt/tree/master/Problems/SAN) problem.
The outputs will be stored in a file named `RawData_NELDMD_on_SAN.mat`.

**4.** From the MATLAB terminal, run the command

`PostWrapper({'ProblemName1', 'ProblemName2', ...}, {'SolverName1', 'SolverName2', ...}, NumberOfPostReplications)`

This will read in the data produced by the previous run of `RunWrapper.m` and take fresh replications in a post-processing step. The outputs from each solver-problem pair will be saved in a MATLAB workspace (.mat) file in the [PostData](https://github.com/simopt-admin/simopt/tree/master/Experiments/PostData) folder.

For example, the command `PostWrapper({'SAN'},{'NELDMD'},50)` will take 50 post-replications at each reported solution from the macroreplications of the Nelder-Mead solver on the Stochastic Activity Network. The outputs will be stored in a file named `PostData_NELDMD_on_SAN.mat`.

**5.** From the MATLAB terminal, run the command

`PlotWrapper({'ProblemName1', 'ProblemName2', ...}, {'SolverName1', 'SolverName2', ...})`

This will read in the data produce by the previous run of `PostWrapper.m' and produce convergence plots with mean performance (95% confidence intervals) and median performance (0.25-0.75 quantiles). The plots for each problem will be saved in MATLAB figure (.fig) files in the [Plots](https://github.com/simopt-admin/simopt/tree/master/Experiments/Plots) folder.

For example, the command `PlotWrapper({'SAN'},{'NELDMD'})` will produce convergence plots of the Nelder-Mead solver on the Stochastic Activity Network based on data stored in `PostData_NELDMD_on_SAN.mat`. The two plots produced will be saved as `SAN_MeanCI.fig` and `SAN_Quantile.fig`.

## External Links

Here are a few links to relevant external websites on simulation optimization.

**Related Testbeds**
* [Stochastic Programming Testbed](http://users.iems.northwestern.edu/~jrbirge/html/dholmes/post.html#post_2)
* [Test Problems for Non-Linear, Stochastic, Mixed-Integer Optimization](http://www.ise.ufl.edu/uryasev/research/testproblems/)

**New to Simulation?**
* [Definition of Simulation](http://www.me.utexas.edu/~jensen/ORMM/models/unit/simulate/index.html)
* [Winter Simulation Conference](http://meetings2.informs.org/wordpress/wsc2018/)
* [INFORMS Simulation Society](https://informs-sim.org/)

## Acknowledgments

An earlier website for [SimOpt](http://www.simopt.org) was developed through work supported by the National Science Foundation under grant nos. DMI-0400287, CMMI-0800688 and CMMI-1200315.

Recent work on the development of SimOpt has been supported by the National Science Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, IIS-1247696, and TRIPODS+X DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038, and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.

Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).
