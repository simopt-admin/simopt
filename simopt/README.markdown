# Welcome to SimOpt!
SimOpt is a testbed of simulation-optimization problems and solvers.
Its purpose is to encourage the development and constructive comparison of simulation-optimization (SO) solvers (algorithms). We are particularly interested in the finite-time performance of solvers, rather than the asymptotic results that one often finds in related literature.

For the purposes of this site, we define simulation as a very general technique for estimating statistical measures of complex systems. A system is modeled as if the probability distributions of the underlying random variables were known. Realizations of these random variables are then drawn randomly from these distributions. Each replication gives one observation of the system response, i.e., an evaluation of the objective function. By simulating a system in this fashion for multiple replications and aggregating the responses, one can compute statistics and use them for evaluation and design.

Several papers have discussed the development of SimOpt and experiments run on the testbed:
* [Pasupathy and Henderson (2006)](https://www.informs-sim.org/wsc06papers/028.pdf) explains the original motivation for the testbed.
* [Pasupathy and Henderson (2011)](https://www.informs-sim.org/wsc11papers/363.pdf) describes an earlier interface for MATLAB implementations of problems and solvers.
* [Dong et al. (2017)](https://www.informs-sim.org/wsc17papers/includes/files/179.pdf) conducts an experimental comparison of several solvers in SimOpt and analyzes their relative performance.
* [Eckman et al. (2019)](https://www.informs-sim.org/wsc19papers/374.pdf) describes in detail changes to the architecture of the MATLAB version of SimOpt and the control of random number streams.
* [Eckman et al. (2021)](https://eckman.engr.tamu.edu/wp-content/uploads/sites/233/2021/09/SimOpt-metrics-paper.pdf) introduces the design of experiments for comparing solvers; this design has been implemented in the latest Python version of SimOpt.

## Documentation
Full documentation for the source code and descriptions of the problems and solvers can be found **[here](https://simopt.readthedocs.io/en/latest/index.html)**.

## Getting Started
Please make sure you have the following dependencies installed: Python 3, numpy, scipy, and matplotlib.
Then clone the repo.
To see an example of how to run an Experiment of a solver on a problem, please view or run demo/demo\_solver\_problem.py.
A graphical user interface (GUI) is now available.

# User Guide for the Graphical User Interface of SimOpt
The main page for the GUI has all the options necessary to add an Experiment to the list or load an existing Experiment and generate desirable plots for it. At the top of the main page are the three ways to create or continue working with an existing Experiment:

* Load a .pickle file of a previously created Experiment.
* Generate a cross-design Experiment, which is a collection of problem-solver pairs or a MetaExperiment.
* Create an individual Experiment on a single problem-solver pair.

At the bottom, there is a frame containing all Experiments. The Experiments are listed in different tabs: the first tab lists the individual problem-solver pairs ready to be run or post-replicated, the second tab lists the (currently called) MetaExperiments made from the cross-design, and the third tab lists the or if they are ready to be post-normalized and prepared for plotting.

## Installation
This GUI depends on Python 3, numpy, scipy, matplotlib, Pillow, and tkinter. <br>
To run the GUI, navigate to the Simopt/simopt directory and run ``` python3 GUI.py ```

## Adding Experiments
This section will explain how to add Experiments to the Experiments or MetaExperiments queue.

### Loading a File
* In the top left corner, you can click "Load File". Your file system will pop up, and you can navigate to an appropriate \*.pickle file to select.
* The file must be \*.pickle, and if it is not, the GUI will throw an error.
* Once an Experiment is loaded, it will be added to the Queue of Experiments.
* The Run and Post-Process buttons will be updated to accurately represent if the Experiment has already been run and post-processed.

### Creating an Experiment through the GUI
Instead of loading in an existing Experiment, you can create one from the main page of the GUI:
* First, select a solver from the "Solver" dropdown list.
* Change factors associated with the solver as necessary.
* All solvers with unique factors must have unique names (no two solvers can have the same name, but different factors). If you want to use the same solver twice for a problem but with different solver factors, make sure you change the name of the solver - the last solver factor - accordingly.
* Select a problem from the "Problem" dropdown list.
* Change factors associated with the Problem and Model as necessary.
* All problems with unique factors must have unique names (no two problems can have the same name, but different factors). If you want to use the same problem twice for a solver but with different problem or model factors, make sure you change the name of the problem - the last problem factor - accordingly.
* The default number of macroreplications is 10. You can modify this in the top left corner.
* Select the "Add Experiment" button, which only appears when a solver and Problem is selected. Then, a new Experiment will be added in the "Queue of Experiments".

### Creating a Cross-Design Experiment
By cross-designing an Experiment, you can create a MetaExperiment, which will be added to the Queue of Meta Experiments. Meta Experiments can only be created and not loaded from a file.
* Click the "Cross-Design Experiments" button.
* Check the compatibility of the Problems and Solvers being selected. Note that deterministic solvers can not handle stochastic problems (ex: FACSIZE-2 can not be solved with ASTRODF).
* The default number of macroreplicaitions is 10, but you can modify that.
* Click "Confirm Cross-Design Experiment."
* The pop-up window will disappear, and the Experiments frame will automatically switch to the "Queue of MetaExperiments".
* To exit out of the MetaExperiment pop-up without creating an Experiment, click the red x in the top left corner of the window.

## Run an Experiment
To run an Experiment or MetaExperiment that has not already been run, click the "Run Exp." button in the "Queue of Experiments" or "Queue of MetaExperiments". Once the Experiment is run, it can not be rerun.
**Note:** running an Experiment could take a couple of seconds to a couple of minutes depending on the Experiment and the number of macroreplications.

## Post-Processing and Post-Normalization
Post Processing happens before post normalizing. After post-normalization is complete, the Plotting window appears.
To exit out of the Post Process/ Normalize pop-up without post-processing or normalizing, click the red x in the top left corner of the window.

### Experiments
Experiments can be post-processed from the "Queue of Experiments" tab by clicking "Post-Process." Adjust Post Processing factors as necessary. Only Experiments that have already been run and have not yet been post-processed can be post-processed. <br>
After post-processing, click the "Post-Normalize by Problem" tab to select which Experiments to post-normalize together.
* Only Experiments with the same Problem can be post-normalized Together.
* Once all Experiments are checked, click the "Post Normalize Selected" button at the bottom of the GUI (this button only appears when in the Post Normalize Tab)
* Update any values necessary and click "Post-Normalize" when the Experiment or Experiments are ready to be post-normalized

### MetaExperiments
MetaExperiments are post-processed and post-normalized at the same time.
* Click the Post Process button for the specific Meta Experiment, then change any values necessary, then click "Post-Process"

## Plotting Experiments
The Plotting page is the same for both Experiments and Meta Experiments. Currently, multiple Experiments with the same Problem can be plotted together, and any problem/ solver from one Meta Experiment can be plotted together. To return to the main page, click the red x in the top left corner of the window.
* On the left side, select one or more problems from the problem list
* Select solvers from the solver list
* On the right side, select a plot type and adjust plot parameters and setting
* Click "Add."
* All plots will show in the plotting queue, along with information about their parameters and where the file is saved at.
* To view one plot, click "View Plot," or all plots can be viewed together by clicking "See All Plots" at the bottom of the page.


## Note to Users
You can contribute problems and solver to SimOpt by using pull requests in GitHub or corresponding with the developers.
* To do: as of now, the plotting functionality of multiple problems and multiple solvers is under construction.

## Acknowledgments
An earlier website for SimOpt, [http://www.simopt.org](http://www.simopt.org), was developed through work supported by the National Science Foundation under grant nos. DMI-0400287, CMMI-0800688 and CMMI-1200315.
Recent work on the development of SimOpt has been supported by the National Science Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, IIS-1247696, and TRIPODS+X DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038, and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).
