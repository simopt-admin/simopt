# Welcome to SimOpt!
SimOpt is a testbed of simulation-optimization problems and solvers.
Its purpose is to encourage the development and constructive comparison of simulation-optimization (SO) solvers (algorithms). We are particularly interested in the finite-time performance of solvers, rather than the asymptotic results that one often finds in related literature.

For the purposes of this project, we define simulation as a very general technique for estimating statistical measures of complex systems. A system is modeled as if the probability distributions of the underlying random variables were known. Realizations of these random variables are then drawn randomly from these distributions. Each replication gives one observation of the system response, i.e., an evaluation of the objective function or stochastic constraints. By simulating a system in this fashion for multiple replications and aggregating the responses, one can compute statistics and use them for evaluation and design.

Several papers have discussed the development of SimOpt and experiments run on the testbed:
* [Pasupathy and Henderson (2006)](https://www.informs-sim.org/wsc06papers/028.pdf) explains the original motivation for the testbed.
* [Pasupathy and Henderson (2011)](https://www.informs-sim.org/wsc11papers/363.pdf) describes an earlier interface for MATLAB implementations of problems and solvers.
* [Dong et al. (2017)](https://www.informs-sim.org/wsc17papers/includes/files/179.pdf) conducts an experimental comparison of several solvers in SimOpt and analyzes their relative performance.
* [Eckman et al. (2019)](https://www.informs-sim.org/wsc19papers/374.pdf) describes in detail changes to the architecture of the MATLAB version of SimOpt and the control of random number streams.
* [Eckman et al. (2021)](https://eckman.engr.tamu.edu/wp-content/uploads/sites/233/2021/09/SimOpt-metrics-paper.pdf) introduces the design of experiments for comparing solvers; this design has been implemented in the latest Python version of SimOpt.

## Documentation
Full documentation for the source code can be found **[here](https://simopt.readthedocs.io/en/latest/index.html)**. Descriptions of the problems and solvers are under development. 

## Getting Started
The most straightforward way to interact with the library is to [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this repository. *(If you anticipate making improvements or contributions to SimOpt, you should first [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository so that you can later request your changes be integrated via GitHub's pull request feature.)*

Download a copy of the cloned repository to your local machine and navigate to the `simopt/simopt` folder in your preferred integrated development environment (IDE). You will need to make sure that you have the following dependencies installed: Python 3, `numpy`, `scipy`, `matplotlib`, `pandas`, `math`, `random`, and `copy`.

The `demo` folder contains a handful of useful scripts that can be easily modified, as directed in the comments.

* `demo_model.py`: Run multiple replications of a simulation model and report its responses.

* `demo_problem.py`: Run multiple replications of a given solution for an SO problem and report its objective function values and left-hand sides of stochastic constraints.

* `demo_solver_problem.py`: Run multiple macroreplications of a solver on a problem, save the outputs to a .pickle file in the `experiments/outputs` folder, and save plots of the results to .png files in the `experiments/plots` folder.

* `demo_data_farming_model.py`: Create a design over model factors, run multiple replications at each design point, and save the results to a comma separated value (.csv) file in the `data_farming_experiments` folder.

* `demo_sscont_experiment.py`: Run multiple solvers on multiple versions of an (s, S) inventory problem and produce plots.

A graphical user interface (GUI) is also under development.

## Contributing
Users can contribute problems and solver to SimOpt by using [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) in GitHub or corresponding with the developers. The core development team currently consists of David Eckman (Texas A&M University), Shane Henderson (Cornell University), and Sara Shashaani (North Carolina State University).

## Acknowledgments
An earlier website for SimOpt, [http://www.simopt.org](http://www.simopt.org), was developed through work supported by the National Science Foundation under grant nos. DMI-0400287, CMMI-0800688 and CMMI-1200315.
Recent work on the development of SimOpt has been supported by the National Science Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, IIS-1247696, and TRIPODS+X DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038, and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).
