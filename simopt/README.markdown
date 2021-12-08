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
To see an example of how to run an experiment of a solver on a problem, please view or run demo/demo\_solver\_problem.py.
A graphical user interface (GUI) is now available.

# User Guide for the Graphical User Interface of SimOpt
The main page for the GUI has all the options necessary to add an experiment to the list and generate desirable plots for it. At the top are the three ways to create or continue working with an experiment.
There are three main ways to start an experiment:
Load a pickle file of a previously created experiment.
Cross-design experiment of a collection of problem-solver pairs.
Create an individual experiment on a single problem-solver pair. 
At the bottom, there is a frame containing all experiments. The experiments are in different tabs, depending on if they are single problem-solver pairs, a (currently called) MetaExperiment made from a cross-design), or if they are ready to be postnormalized and prepared for plotting. 

## Installation
This GUI depends on Python 3, numpy, scipy, matplotlib, Pillow, and tkinter. <br>
To run the GUI, navigate to the Simopt/simopt directory and run ``` python3 GUI.py ```

## Adding Experiments
This section will explain how to add experiments to the Experiments or MetaExperiments queue.
### Load a File
* In the top left corner, a user can click "Load File". Their file system will pop up, and the user can navigate to an appropriate \*.pickle file to select.
* The file must be \*.pickle, and if it is not, the GUI will throw an error.
* Once an experiment is loaded, it will be added to the Queue of Experiments. 
* The Run and Postprocess buttons will be updated to accurately represent if the experiment has already been run and postprocessed.

### Creating an Experiment through the GUI
Instead of loading in an experiment, the user can create one from the main page of the GUI:
* First, select a solver from the "Solver" dropdown.
* Change factors associated with the solver as necessary.
* All solvers with unique factors must have unique names (no two solvers can have the same name, but different factors)
* Select a problem from the "Problem" dropdown.
* Change factors associated with the Problem and Model as necessary. 
* All problems with unique factors must have unique names (no two problems can have the same name, but different factors)
* The default number of macroreplications is 10. The user can modify this in the top left corner
* Select the "Add Experiment" button, which only appears when a solver and Problem is selected.

### Cross-Design Experiment
By cross-designing an experiment, the user creates a MetaExperiment, which will be added to the Queue of Meta Experiments. Meta Experiments can only be created and not loaded from a file. 
* Click Cross-Design Experiment
* Check which Problems and Solvers the experiment should contain
* Note that deterministic solvers can not handle stochastic problems (ex: FACSIZE-2 can not be solved with ASTRODF)
* The default number of Macroreplicaitions is 10, but that can be adjusted
* Click "Confirm Cross-Design Experiment." 
* The pop-up window will disappear, and the Experiments frame will automatically switch to Queue of Meta Experiments
To exit out of the Meta Experiment pop-up without creating an experiment, click the red x in the top left corner of the window.

## Run an Experiment
To run an experiment or meta experiment that has not already been run, click the "Run Exp." button in the queue of experiments or queue of meta experiments. Once the experiment is run, it can not be rerun. 
**Note:** running an experiment could take a couple of seconds to a couple of minutes depending on the experiment and the number of macroreplications.

## Postprocess and Postnormalize
Post Processing happens before post normalizing. After postnormalization is complete, the Plotting window appears.
To exit out of the Post Process/ Normalize pop-up without postprocessing or normalizing, click the red x in the top left corner of the window.
### Experiments
Experiments can be post-processed from the queue of Experiments tab by clicking "Postprocess." Adjust Post Processing factors as necessary. Only Experiments that have already been run and have not yet been postprocessed can be postprocessed. <br> 
After Post Processing, click the "Postnormalize by Problem" tab to select which experiments to postnormalize together.
* Only Experiments with the same Problem can be postnormalized Together.
* Once all Experiments are checked, click the "Post Normalize Selected" button at the bottom of the GUI (this button only appears when in the Post Normalize Tab)
* Update any values necessary and click "Post Normalize" when the experiment or experiments are ready to be postnormalized

### Meta Experiments
Meta Experiments are postprocessed and postnormalized at the same time.
* Click the Post Process button for the specific Meta Experiment, then change any values necessary, then click "Postprocess"

## Plotting Experiments
The Plotting page is the same for both Experiments and Meta Experiments. Currently, multiple Experiments with the same Problem can be plotted together, and any problem/ solver from one Meta Experiment can be plotted together. To return to the main page, click the red x in the top left corner of the window.
* On the left side, select one or more problems from the problem list 
* Select solvers from the solver list
* On the right side, select a plot type and adjust plot parameters and setting 
* Click "Add."
* All plots will show in the plotting queue, along with information about their parameters and where the file is saved at. 
* To view one plot, click "View Plot," or all plots can be viewed together by clicking "See All Plots" at the bottom of the page.


## Note to Users

Users can contribute problems and solver to SimOpt by using pull requests in GitHub or corresponding with the developers.

## Acknowledgments
An earlier website for SimOpt, [http://www.simopt.org](http://www.simopt.org), was developed through work supported by the National Science Foundation under grant nos. DMI-0400287, CMMI-0800688 and CMMI-1200315.
Recent work on the development of SimOpt has been supported by the National Science Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, IIS-1247696, and TRIPODS+X DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038, and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).
