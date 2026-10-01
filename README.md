![SimOpt Logo](https://raw.githubusercontent.com/simopt-admin/simopt/master/.github/resources/logo_full_magnifying_glass.png)

## About the Project
SimOpt is a testbed of simulation-optimization problems and solvers. Its purpose is to encourage the development and constructive comparison of simulation-optimization (SO) solvers (algorithms). We are particularly interested in the finite-time performance of solvers, rather than the asymptotic results that one often finds in related literature.

For the purposes of this project, we define simulation as a very general technique for estimating statistical measures of complex systems. A system is modeled as if the probability distributions of the underlying random variables were known. Realizations of these random variables are then drawn randomly from these distributions. Each replication gives one observation of the system response, i.e., an evaluation of the objective function or stochastic constraints. By simulating a system in this fashion for multiple replications and aggregating the responses, one can compute statistics and use them for evaluation and design.

Several papers have discussed the development of SimOpt and experiments run on the testbed:
* [Eckman et al. (2024)](https://ieeexplore.ieee.org/document/10408734) studies feasibility metrics for stochastically constrained simulation-optimization problems in preparation for introducing related metrics in SimOpt.
* [Shashaani et al. (2024)](https://dl.acm.org/doi/10.1145/3680282) conducts a large data-farming experiment over solver factors to learn relationships between their settings and a solver's finite-time performance.
* [Eckman et al. (2023)](https://pubsonline.informs.org/doi/10.1287/ijoc.2023.1273) is the most up-to-date publication about SimOpt and describes the code architecture and how users can interact with the library.
* [Eckman et al. (2023)](https://pubsonline.informs.org/doi/10.1287/ijoc.2022.1261) introduces the design of experiments for comparing solvers; this design has been implemented in the latest Python version of SimOpt. For detailed description of the terminology used in the library, e.g., factors, macroreplications, post-processing, solvability plots, etc., see this paper.
* [Eckman et al. (2019)](https://www.informs-sim.org/wsc19papers/374.pdf) describes in detail changes to the architecture of the MATLAB version of SimOpt and the control of random number streams.
* [Dong et al. (2017)](https://www.informs-sim.org/wsc17papers/includes/files/179.pdf) conducts an experimental comparison of several solvers in SimOpt and analyzes their relative performance.
* [Pasupathy and Henderson (2011)](https://www.informs-sim.org/wsc11papers/363.pdf) describes an earlier interface for MATLAB implementations of problems and solvers.
* [Pasupathy and Henderson (2006)](https://www.informs-sim.org/wsc06papers/028.pdf) explains the original motivation for the testbed.

## Code
### Python
- The [`master branch`](https://github.com/simopt-admin/simopt/tree/master) contains the source code for the latest stable release of the testbed
- The [`development branch`](https://github.com/simopt-admin/simopt/tree/development) contains the latest code for the testbed, but may contain more bugs than the master branch

### Matlab
> ⚠️ MATLAB versions of this testbed are no longer supported
- The [`matlab branch`](https://github.com/simopt-admin/simopt/tree/matlab) contains a previous stable version of the testbed written in MATLAB

## Documentation
Full documentation for the source code can be found on our **[readthedocs page](https://simopt.readthedocs.io/en/latest/index.html)**.

[![Documentation Status](https://readthedocs.org/projects/simopt/badge/?version=latest)](https://simopt.readthedocs.io/en/latest/?badge=latest)

## Getting Started
### Requirements
- [Miniconda or Anaconda](https://www.anaconda.com/download)
    - If you already have a compatible IDE (such as VS Code), we've found that Miniconda will work fine at 1/10 of the size of Anaconda. Otherwise, you may need the Spyder IDE that comes with the full Anaconda distribution.
    - It is ***highly recommended*** to check the box during installation to add Python/Miniconda/Anaconda to your system PATH.
    - If you know you have Python installed but are getting a `Command not found` error when trying to use Python commands, then you may need to [add Python to your PATH](https://realpython.com/add-python-to-path/).
- [VS Code](https://code.visualstudio.com/download) (optional)
    - This is a lightweight IDE that is compatible with Miniconda.
- [Git](https://git-scm.com/downloads) (optional)
    - If you don't have Git installed, you can download the code as a zip file instead

### Downloading Source Code
There are two ways to download a copy of the source code onto your machine:
1. Download the code in a zip file by clicking the green `<> Code` button above repo contents and clicking the `Download ZIP` option, then unzip the code to a folder on your computer. This does not require `git` to be installed but makes downloading updates to the repository more challenging.
![image](https://github.com/user-attachments/assets/3c45804c-f8b0-48ed-b32c-a443550c6ef5)

1. [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the branch you'd like to download to a folder on your computer. This requires `git` to be installed but makes downloading updates to the repository much easier.

If you do not need the source code for SimOpt, you may install the library as a Python package instead. See the [Package](#package) and [Basic Example](#basic-example) sections for more details about this option.

The `notebooks` folder includes several useful Jupyter notebooks and scripts that are easy to customize. You can either run the scripts as standalone programs or open the notebooks in JupyterLab or VS Code. A description of the contents is provided below:

| File                                     | Description                                                                                                                                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `demo_model.py`                          | Run multiple replications of a simulation model and report its responses                                                                                                                                           |
| `demo_problem.py`                        | Run multiple replications of a given solution for an SO problem and report its objective function values and left-hand sides of stochastic constraints                                                             |
| `demo_problem_solver.py`                 | Run multiple macroreplications of a solver on a problem, save the outputs to a `.pickle` file in the `experiments/outputs` folder, and save plots of the results to `.png` files in the `experiments/plots` folder |
| `demo_problems_solvers.py`               | Run multiple macroreplications of groups of problem-solver pairs and save the outputs and plots                                                                                                                    |
| `demo_data_farming_model.py`             | Create a design over model factors, run multiple replications at each design point, and save the results to a comma separated value (`.csv`) file in the `data_farming_experiments` folder                         |
| `demo_san-sscont-ironorecont_experiment` | Run multiple solvers on multiple versions of (s, S) inventory, iron ore, and stochastic activiy network problems and produce plots                                                                                 |

### Environment Setup

After downloading the source code, you will need to configure the conda environment to run the code. This can be done by running the following command in the terminal:

#### Windows (Command Prompt)
```cmd
setup_simopt.bat
```

#### Windows (PowerShell)
```powershell
cmd /c setup_simopt.bat
```

#### MacOS/Linux
```bash
chmod +x setup_simopt.sh && ./setup_simopt.sh
```

This script will create a new conda environment called `simopt` and install all necessary packages. To activate the environment, run the following command in the terminal:

```bash
conda activate simopt
```

If you wish to update the environment with the latest compatible packages, you can simply rerun the setup script.

## Web Interface

To open the web interface, run `simopt web` (or `python -m simopt web`) and open http://localhost:8000.
From the web interface you can build and run problem-solver experiments, generate plots, and create data-farming designs for solvers, problems, and models.

The previous Tkinter GUI has been removed; it is preserved on the `archive/tkinter` branch.

## Package
The `simoptlib` package is available to download through the Python Packaging Index (PyPI) and can be installed from the terminal with the following command:
```
python -m pip install simoptlib
```

## Basic Example
After installing `simoptlib`, the package's main modules can be imported from the Python console (or in code):
```
import simopt
from simopt import models, solvers, experiment_base
```

The following snippet of code will run 10 macroreplications of the Random Search solver ("RNDSRCH") on the Continuous Newsvendor problem ("CNTNEWS-1"):
```python
myexperiment = simopt.experiment_base.ProblemSolver("RNDSRCH", "CNTNEWS-1")
myexperiment.run(n_macroreps=10)
```

The results will be saved to a .pickle file in a folder called `experiments/outputs`. To post-process the results, by taking, for example 200 postreplications at each recommended solution, run the following:
```python
myexperiment.post_replicate(n_postreps=200)
simopt.experiment_base.post_normalize([myexperiment], n_postreps_init_opt=200)
```

A .txt file summarizing the progress of the solver on each macroreplication can be produced:
```python
myexperiment.log_experiment_results()
```

A .txt file called `RNDSRCH_on_CNTNEWS-1_experiment_results.txt` will be saved in a folder called `experiments/logs`.

One can then plot the mean progress curve of the solver (with confidence intervals) with the objective function values shown on the y-axis:
```python
simopt.experiment_base.plot_progress_curves(
    experiments=[myexperiment],
    plot_type=simopt.experiment_base.PlotType.MEAN,
    normalize=False,
)
```

The Python scripts in the `notebooks` folder provide more guidance on how to run common experiments using the library.

One can also use the SimOpt graphical user interface by running the following from the terminal:
```bash
python -m simopt
```

## Contributing
You can contribute problems and solvers to SimOpt (or fix other coding bugs) by [forking](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository and initiating [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) in GitHub to request that your changes be integrated.


## Authors
The core development team currently consists of 
- [**David Eckman**](https://eckman.engr.tamu.edu) (Texas A&M University)
- [**Sara Shashaani**](https://shashaani.wordpress.ncsu.edu) (North Carolina State University)
- [**Shane Henderson**](https://people.orie.cornell.edu/shane/) (Cornell University)
- [**Cen Wang**](https://cenwangumass.github.io/) (Texas A&M University)
- [**Emily Liu**](https://github.com/eyl48) (Cornell University)

Previous maintainer:

- [**William Grochocinski**](https://github.com/Grochocinski) (North Carolina State University)

## Citation
To cite this work, please use the `CITATION.cff` file or use the built-in citation generator:
![GitHub's built-in citation generator](https://github.com/user-attachments/assets/b8b49544-eb74-469e-aa37-68c2c0c3708b)


## Acknowledgments
An earlier website for SimOpt ([http://www.simopt.org](http://www.simopt.org)) was developed through work supported by the following grants:
- National Science Foundation
    - [DMI-0400287](https://www.nsf.gov/awardsearch/showAward?AWD_ID=0400287)
    - [CMMI-0800688](https://www.nsf.gov/awardsearch/showAward?AWD_ID=0800688)
    - [CMMI-1200315](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1200315)

Recent work on the development of SimOpt has been supported by the following grants
- National Science Foundation
    - [IIS-1247696](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1247696)
    - [CMMI-1254298](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1254298)
    - [CMMI-1536895](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1536895)
    - [CMMI-1537394](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1537394)
    - [DGE-1650441](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1650441)
    - [DMS-1839346](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1839346) (TRIPODS+X)
    - [CMMI-2206972](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2206972)
    - [OAC-2410948](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2410948)
    - [OAC-2410949](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2410949)
    - [OAC-2410950](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2410950)
- Air Force Office of Scientific Research
    - FA9550-12-1-0200
    - FA9550-15-1-0038
    - FA9550-16-1-0046
- Army Research Office
    - W911NF-17-1-0094

*Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).*
