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

## Graphical User Interface (GUI) - User Guide

### Installation
To start up the GUI, navigate to the `simopt/simopt` directory and run the command ``` python3 GUI.py ``` from the terminal.
The GUI depends on Python 3, `numpy`, `scipy`, `matplotlib`, `Pillow`, and `tkinter`.

### Overview
From the GUI's main page, a user can create a specified problem-solver pair (referred to in the GUI as an **Experiment**), run macroreplications, and generate plots.

The top of the main page provides three ways to create or continue working with an existing Experiment:

1. Create an individual Experiment for a single problem-solver pair.
2. Load a .pickle file of a previously created Experiment.
3. Generate a cross-design Experiment, which is a collection of problem-solver pairs (referred to in the GUI as a **Meta-Experiment**).

At the bottom of the main page, there is a frame containing all Experiments. The Experiments are listed in different tabs: the first tab lists the individual problem-solver pairs ready to be run or post-replicated, the second tab lists the Meta-Experiments made from the cross-design, and the third tab lists those Experiments that are ready to be post-normalized and prepared for plotting.

### Adding Experiments
This section will explain how to add Experiments to the Experiments or MetaExperiments queue.

#### Loading an Experiment from a File
1. In the top left corner, you can click "Load File". Your file system will pop up, and you can navigate to an appropriate \*.pickle file to select. (The GUI will throw an error is the selected file is not a \*.pickle file.
2. Once an Experiment is loaded, it will be added to the Queue of Experiments.
3. The Run and Post-Process buttons will be updated to accurately reflect whether the Experiment has already been run and/or post-processed.

#### Creating an Experiment
Instead of loading an existing Experiment, you can create one from the main page of the GUI:
1. First, select a solver from the "Solver" dropdown list. Each of the solvers contain an abbreviation for the type of problems the solver can handle. Once a solver is selected it will sort through the "Problems" and show only the problems that work with this solver. 
2. Change factors associated with the solver as necessary.
3. All solvers with unique factors must have unique names (no two solvers can have the same name, but different factors). If you want to use the same solver twice for a problem but with different solver factors, make sure you change the name of the solver - the last solver factor - accordingly.
4. Select a problem from the "Problem" dropdown list. Each problem contain an abbreviation that determines which type of solver it can work with. The letters in the abbreviation stand for: 
<table>
    <tr>
      <th> Objective </th>
      <th> Constraint </th>
      <th> Variable </th>
      <th> Gradient </th>
    </tr>
    <tr>
      <td> Single (S) </td>
      <td> Unconstrained (U) </td>
      <td> Discrete (D) </td>
      <td> Gradients Available (G) </td>
    </tr>
  <tr>
      <td> Multi (M) </td>
      <td> Box (B) </td>
      <td> Continuous (C) </td>
      <td> Gradients Not Available (NG) </td>
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
6. All problems with unique factors must have unique names (no two problems can have the same name, but different factors). If you want to use the same problem twice for a solver but with different problem or model factors, make sure you change the name of the problem - the last problem factor - accordingly.
7. The number of macroreplications can be modified in the top-left corner. The default is 10 macroreplicatons.
8. Select the "Add Experiment" button, which only appears when a solver and problem is selected. Then, a new Experiment will be added in the "Queue of Experiments".

#### Creating a Cross-Design Experiment
By cross-designing an Experiment, you can create a MetaExperiment, which will be added to the "Queue of Meta-Experiments". (Currently, Meta-Experiments can only be created and not loaded from a file.)
1. Click the "Cross-Design Experiments" button.
2. Check the compatibility of the Problems and Solvers being selected. Note that deterministic solvers can not handle problems with stochastic constraints (e.g., ASTRO-DF cannot be run on FACSIZE-2).
3. Specify the number of macroreplications - the default is 10.
4. Click "Confirm Cross-Design Experiment."
5. The pop-up window will disappear, and the Experiments frame will automatically switch to the "Queue of Meta-Experiments".
6. To exit out of the Meta-Experiment pop-up without creating an Experiment, click the red "x" in the top-left corner of the window.

### Run an Experiment
To run an Experiment or Meta-Experiment that has not already been run, click the "Run Exp." button in the "Queue of Experiments" or "Queue of MetaExperiments". Once the Experiment has been run, it cannot be re-run.
**Note:** Running an Experiment can take anywhere a couple of seconds to a couple of minutes depending on the Experiment and the number of macroreplications.

### Post-Processing and Post-Normalization
Post-processing happens before post-normalizing. After post-normalization is complete, the Plotting window appears.
To exit out of the Post-Process/ Normalize pop-up without post-processing or post-normalizing, click the red "x" in the top-left corner of the window.

#### Experiments
Experiments can be post-processed from the "Queue of Experiments" tab by clicking "Post-Process." Adjust Post-Processing factors as necessary. Only Experiments that have already been run and have not yet been post-processed can be post-processed. <br>
After post-processing, click the "Post-Normalize by Problem" tab to select which Experiments to post-normalize together.
* Only Experiments with the same Problem can be post-normalized together.
* Once all Experiments are checked, click the "Post-Normalize Selected" button at the bottom of the GUI (this button only appears when in the Post-Normalize Tab).
* Update any values necessary and click "Post-Normalize" when the Experiment(s) are ready to be post-normalized.

#### Meta-Experiments
Meta-Experiments are post-processed and post-normalized at the same time.
* Click the "Post-Process" button for the specific Meta-Experiment, then change any values necessary, then click "Post-Process".

### Plotting Experiments
The Plotting page is the same for both Experiments and Meta Experiments. Currently, multiple Experiments with the same Problem can be plotted together, and any problem- solver pair from one Meta-Experiment can be plotted. (The ability to produce plots based on multiple problems *and* multiple solvers using the GUI is currently under development.) To return to the main page, click the red "x" in the top-left corner of the window.
1. On the left side, select one or more problems from the problem list.
2. Select solvers from the solver list.
3. On the right side, select a plot type and adjust plot parameters and settings. 
There are 3 plot settings for all plots: Confident Intervals, Plot Together, and Print Max HW. 
The type of plots that are currently available in the GUI are: Mean Progress Curve, Quantile Progress Curve, Solve Time CDF, Scatter Plot, CDF Solvability, Quantile Solvability, CDF Difference Plot, Quantile Difference Plot, Box, Violin, and Terminal Scatter. 
4. Click "Add."
5. All plots will show in the plotting queue, along with information about their parameters and where the file is saved at.
6. To view one plot, click "View Plot," or all plots can be viewed together by clicking "See All Plots" at the bottom of the page.

## Contributing
Users can contribute problems and solver to SimOpt by using [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) in GitHub or corresponding with the developers. The core development team currently consists of David Eckman (Texas A&M University), Shane Henderson (Cornell University), and Sara Shashaani (North Carolina State University).

## Acknowledgments
An earlier website for SimOpt, [http://www.simopt.org](http://www.simopt.org), was developed through work supported by the National Science Foundation under grant nos. DMI-0400287, CMMI-0800688 and CMMI-1200315.
Recent work on the development of SimOpt has been supported by the National Science Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, IIS-1247696, and TRIPODS+X DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038, and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).
