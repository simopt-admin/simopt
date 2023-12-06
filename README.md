# Welcome to SimOpt!

SimOpt is a testbed of simulation-optimization problems and solvers. Its purpose is to encourage the development and constructive comparison of simulation-optimization (SO) solvers (algorithms). We are particularly interested in the finite-time performance of solvers, rather than the asymptotic results that one often finds in related literature.

For the purposes of this project, we define simulation as a very general technique for estimating statistical measures of complex systems. A system is modeled as if the probability distributions of the underlying random variables were known. Realizations of these random variables are then drawn randomly from these distributions. Each replication gives one observation of the system response, i.e., an evaluation of the objective function or stochastic constraints. By simulating a system in this fashion for multiple replications and aggregating the responses, one can compute statistics and use them for evaluation and design.

Several papers have discussed the development of SimOpt and experiments run on the testbed:
* [Eckman et al. (2023)](https://pubsonline.informs.org/doi/10.1287/ijoc.2023.1273) is the most up-to-date publication about SimOpt and describes the code architecture and how users can interact with the library.
* [Eckman et al. (2023)](https://pubsonline.informs.org/doi/10.1287/ijoc.2022.1261) introduces the design of experiments for comparing solvers; this design has been implemented in the latest Python version of SimOpt. For detailed description of the terminology used in the library, e.g., factors, macroreplications, post-processing, solvability plots, etc., see this paper.
* [Eckman et al. (2019)](https://www.informs-sim.org/wsc19papers/374.pdf) describes in detail changes to the architecture of the MATLAB version of SimOpt and the control of random number streams.
* [Dong et al. (2017)](https://www.informs-sim.org/wsc17papers/includes/files/179.pdf) conducts an experimental comparison of several solvers in SimOpt and analyzes their relative performance.
* [Pasupathy and Henderson (2011)](https://www.informs-sim.org/wsc11papers/363.pdf) describes an earlier interface for MATLAB implementations of problems and solvers.
* [Pasupathy and Henderson (2006)](https://www.informs-sim.org/wsc06papers/028.pdf) explains the original motivation for the testbed.


## Code
* The `master` branch contains the source code for the latest version of the testbed, which is written in Python.
* The `matlab` branch contains a previous stable version of the testbed written in MATLAB.

## Documentation
Full documentation for the source code can be found **[here](https://simopt.readthedocs.io/en/latest/index.html)**.

## Getting Started
The most straightforward way to interact with the library is to [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this repository. Alternatively, you can choose to install the library as a Python package; see the sections titled **Package** and **Basic Example** below for more details about this option.

Download a copy of the cloned repository to your local machine and navigate to the `simopt` folder in your preferred integrated development environment (IDE). You will need to make sure that you have the following dependencies installed: Python 3, `numpy`, `scipy`, `matplotlib`, `pandas`, `seaborn`, and `mrg32k3a`. Run the command ``` python -m pip install numpy scipy matplotlib pandas seaborn mrg32k3a``` to install them from the terminal.

The `demo` folder contains a handful of useful scripts that can be easily modified, as directed in the comments.

* `demo_model.py`: Run multiple replications of a simulation model and report its responses.

* `demo_problem.py`: Run multiple replications of a given solution for an SO problem and report its objective function values and left-hand sides of stochastic constraints.

* `demo_problem_solver.py`: Run multiple macroreplications of a solver on a problem, save the outputs to a .pickle file in the `experiments/outputs` folder, and save plots of the results to .png files in the `experiments/plots` folder.

* `demo_problems_solvers.py`: Run multiple macroreplications of groups of problem-solver pairs and save the outputs and plots.

* `demo_data_farming_model.py`: Create a design over model factors, run multiple replications at each design point, and save the results to a comma separated value (.csv) file in the `data_farming_experiments` folder.

* `demo_san-sscont-ironorecont_experiment`: Run multiple solvers on multiple versions of (s, S) inventory, iron ore, and stochastic activiy network problems and produce plots.

## Graphical User Interface (GUI) - User Guide

### Installation
To start up the GUI, navigate to the outer `simopt` directory and run the command ``` python3 -m simopt.GUI ``` from the terminal. The GUI depends on Python 3, `numpy`, `scipy`, `matplotlib`, `Pillow`, and `tkinter`. Run the command ``` python3 -m pip install numpy scipy matplotlib Pillow tkinter ``` to install them from the terminal.

### Overview
From the GUI, you can create a specified **problem-solver pair** or a **problem-solver group**, run macroreplications, and generate plots. The main page provides ways to create or continue working with experiments:

1. Create an individual **problem-solver pair** with customized problem and solver factors.
2. Load a .pickle file of a previously created **problem-solver pair**.
3. Create a **problem-solver group**. 

At the bottom of the main page, there is a workspace containing all **problem-solver pair**s and **problem-solver group**s. The first tab lists the **problem-solver pair**s ready to be run or post-replicated, the second tab lists the **problem-solver group**s made from the cross-design or by generating a **problem-solver group** from partial set of **problem-solver pair** in the first tab, and the third tab lists those **problem-solver pair**s that are ready to be post-normalized and prepared for plotting.


#### 1. Creating a **problem-solver pair**
This is the main way to add **problem-solver pair**s to the queue in the workspace.
1. First, select a solver from the "Solver" dropdown list. Each of the solvers has an abbreviation for the type of problems the solver can handle. Once a solver is selected, the "Problem" list will be sorted and show only the problems that work with the selected solver.
2. Change factors associated with the solver as necessary. The first factor is a customizable name for the solver that use can specify. 
3. All solvers with unique combinations of factors must have unique names, i.e., no two solvers can have the same name, but different factors. If you want to use the same solver twice for a problem but with different solver factors, make sure you change the name of the solver accordingly. For example, if you want to create two **problem-solver pair**s with the same problem and solver but with or without CRN for the solver, you can change the name of the solver of choice for each pair to reflect that. This name will appear in the queue within the workspace below.
4. Select a problem from the "Problem" dropdown list.
Each problem has an abbreviation indicating which types of solver is compatible to solve it. The letters in the abbreviation stand for:
    <table>
        <tr>
          <th> Objective </th>
          <th> Constraint </th>
          <th> Variable </th>
          <th> Direct Gradient Observations </th>
        </tr>
        <tr>
          <td> Single (S) </td>
          <td> Unconstrained (U) </td>
          <td> Discrete (D) </td>
          <td> Available (G) </td>
        </tr>
      <tr>
          <td> Multiple (M) </td>
          <td> Box (B) </td>
          <td> Continuous (C) </td>
          <td> Not Available (N) </td>
        </tr>
      <tr>
          <td>  </td>
          <td> Deterministic (D) </td>
          <td> Mixed (M)  </td>
          <td>  </td>
        </tr>
      <tr>
          <td>  </td>
          <td> Stochastic (S) </td>
          <td> </td>
          <td>  </td>
        </tr>
    </table>
5. Change factors associated with the problem and model as necessary.
6. All problems with unique factors must have unique names, i.e., no two problems can have the same name, but different factors. If you want to use the same problem twice for a solver but with different problem or model factors, make sure you change the name of the problem accordingly. This name will appear in the queue within the workspace below.
7.  The number of macroreplications can be modified in the top-left corner. The default is 10.
8.  Select the "Add **problem-solver pair**" button, which only appears when a solver and problem is selected. The **problem-solver pair** will be added in the "Queue of **problem-solver pair**s."


#### 2. Loading a **problem-solver pair** from a file
Instead of creating a **problem-solver pair** from scratch, you can load one from a \*.pickle file:
1. In the top left corner, click "Load a **problem-solver pair**". Your file system will pop up, and you can navigate to and select an appropriate \*.pickle file. The GUI will throw an error if the selected file is not a \*.pickle file.
2. Once a **problem-solver pair** object is loaded, it will be added to the "Queue of **problem-solver pair**s".
3. The Run and Post-Process buttons will be updated to accurately reflect whether the **problem-solver pair** has already been run and/or post-processed.

#### 3. Creating a **problem-solver group**
Currently, **problem-solver group**s can only be created within the GUI or command line; they cannot be loaded from a file. 

You can create a **problem-solver group** and add a new item to the "Queue of **problem-solver group**s" in two ways. The first is a quick grouping of problems and solvers that are compatible with their default factors: 
of problems and solvers with their default factors.
1. Click the "Create a **problem-solver group**" button.
2. Check the compatibility of the Problems and Solvers being selected. Note that solvers with deterministic constraint type can not handle problems with stochastic constraints (e.g., ASTRO-DF cannot be run on FACSIZE-2).
3. Specify the number of macroreplications - the default is 10.
4. Click "Confirm Cross-Design **problem-solver group**."
5. The pop-up window will disappear, and the **problem-solver pair**s frame will automatically switch to the "Queue of **problem-solver group**s".
6. To exit out of the **problem-solver group** pop-up without creating a **problem-solver group**, click the red "x" in the top-left corner of the window.

The second is converting a list of **problem-solver pair**s into a **problem-solver group** by a cross-design: 
1. Select the **problem-solver pair**s of interest from the "Queue of **problem-solver pair**s". 
2. Clicking the "Convert to a **problem-solver group**" button. This will complete the cross-design for the partial list and create a new row in the "Queue of **problem-solver group**s".


### Running a **problem-solver pair** or a **problem-solver group** 
To run a **problem-solver pair** or a **problem-solver group**, click the "Run" button in the "Queue of **problem-solver pair**s" or "Queue of **problem-solver group**s". Once the **problem-solver pair** or **problem-solver group** has been run, the "Run" button becomes disabled.
**Note:** Running a **problem-solver pair** can take anywhere from a couple of seconds to a couple of minutes depending on the **problem-solver pair** and the number of macroreplications.

### Post-Processing and Post-Normalization
Post-processing happens before post-normalizing and after the run is complete. You can specify the number of post-replications and the (proxy) optimal solution or function value.  After post-normalization is complete, the Plotting window appears.
To exit out of the Post-Process/Normalize pop-up without post-processing or post-normalizing, click the red "x" in the top-left corner of the window.

#### - **problem-solver pair**
**problem-solver pair**s can be post-processed from the "Queue of **problem-solver pair**s" tab by clicking "Post-Process." Adjust Post-Processing factors as necessary. Only **problem-solver pair**s that have already been run can be post-processed. After post-processing, click the "Post-Normalize by Problem" tab to select which **problem-solver pair**s to post-normalize together.
* Only **problem-solver pair**s with the same problem can be post-normalized together.
* Once all **problem-solver pair**s of interest are selected, click the "Post-Normalize Selected" button at the bottom of the GUI (this button only appears when in the Post-Normalize tab).
* In the new pop-up form, update any values necessary and click "Post-Normalize" when the **problem-solver pair**s are ready to be post-normalized.

#### - **problem-solver group**
**problem-solver group**s are post-processed and post-normalized at the same time. In the "Queue of **problem-solver group**s" tab, click the "Post-Process" button for the specific **problem-solver group**, then change any values necessary, then click "Post-Process".

### Plotting
The Plotting page is identical for both **problem-solver pair**s and **problem-solver group**s. Currently, multiple **problem-solver pair**s with the same problem can be plotted together, and any problem-solver pair from a single **problem-solver group** can be plotted together:
1. On the left side, select one or more problems from the problem list.
2. Select solvers from the solver list.
3. On the right side, select a plot type and adjust plot parameters and settings.
There are 5 settings common to most plot types: Confidence Intervals, Number of Bootstrap Samples, Confidence Level, Plot Together, and Print Max HW.
The type of plots that are currently available in the GUI are: Mean Progress Curve, Quantile Progress Curve, Solve Time CDF, Scatter Plot, CDF Solvability, Quantile Solvability, CDF Difference Plot, Quantile Difference Plot, Terminal Box/Violin, and Terminal Scatter.
4. Click "Add."
5. All plots will show in the plotting queue, along with information about their parameters and where the file is saved.
6. To view one plot, click "View Plot." All plots can be viewed together by clicking "See All Plots" at the bottom of the page.
7. To return to the main page, click the red "x" in the top-left corner of the window.

## Contributing
You can contribute problems and solvers to SimOpt (or fix other coding bugs) by [forking](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository and initiating [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) in GitHub to request that your changes be integrated.

We also maintain a short list of next steps for improving the libary:

### Short-Term To-Do List
1. **Accelerating and Hardening the Source Code:** profiling, exception tracking, unit testing.
1. **Random Variate Generation:** adding to the random-variate generating capabilities of the `mrg32k3a` package that supports SimOpt.
1. **Robustness to User Specification:** additional exception handling for changing model/problem/solver factors.
1. **Improving the Diversity of Solvers and Problems:** we especially welcome more solvers for problems with discrete feasible regions and/or deterministic linear constraints and/or stochastic constraints and/or multiple objectives.
1. **Parallelization for Large-Scale Experiments:** using `OpenMP` to parallelize macroreplications.
1. **Performance Metrics and Plots:** new metrics and plots for problems with stochastic constraints and/or multiple objectives.

## Package
The `simoptlib` package is available to download through the Python Packaging Index (PyPI) and can be installed from the terminal with the following command:

    python -m pip install simoptlib

## Basic Example

After installing `simoptlib`, the package's main modules can be imported from the Python console (or in code):

    import simopt
    from simopt import models, solvers, experiment_base

The following snippet of code will run 10 macroreplications of the Random Search solver ("RNDSRCH") on the Continuous Newsvendor problem ("CNTNEWS-1"):

    myexperiment = simopt.experiment_base.ProblemSolver("RNDSRCH", "CNTNEWS-1")
    myexperiment.run(n_macroreps=10)

The results will be saved to a .pickle file in a folder called `experiments/outputs`. To post-process the results, by taking, for example 200 postreplications at each recommended solution, run the following:

    myexperiment.post_replicate(n_postreps=200)
    simopt.experiment_base.post_normalize([myexperiment], n_postreps_init_opt=200)

A .txt file summarizing the progress of the solver on each macroreplication can be produced:
    
    myexperiment.log_experiment_results()

A .txt file called `RNDSRCH_on_CNTNEWS-1_experiment_results.txt` will be saved in a folder called `experiments/logs`.

One can then plot the mean progress curve of the solver (with confidence intervals) with the objective function values shown on the y-axis:
    
    simopt.experiment_base.plot_progress_curves(experiments=[myexperiment], plot_type="mean", normalize=False)

The Python scripts in the `demo` folder provide more guidance on how to run common experiments using the library.

One can also use the SimOpt graphical user interface by running the following from the terminal:

    python -m simopt.GUI

## Authors
The core development team currently consists of 

- [**David Eckman**](https://eckman.engr.tamu.edu) (Texas A&M University)
- [**Sara Shashaani**](https://shashaani.wordpress.ncsu.edu) (North Carolina State University)
- [**Shane Henderson**](https://people.orie.cornell.edu/shane/) (Cornell University)


## Citation
To cite this work, please use
```
@misc{simoptgithub,
  author = {D. J. Eckman and S. G. Henderson and S. Shashaani and R. Pasupathy},
  title = {{SimOpt}},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/simopt-admin/simopt}},
  commit = {4c5de2e7576a596ea20979636cb034e75fada3f4}
}
```

## Acknowledgments
An earlier website for SimOpt, [http://www.simopt.org](http://www.simopt.org), was developed through work supported by the National Science Foundation under grant nos. DMI-0400287, CMMI-0800688 and CMMI-1200315.
Recent work on the development of SimOpt has been supported by the National Science Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, CMMI-2206972, IIS-1247696, and TRIPODS+X DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038, and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).
