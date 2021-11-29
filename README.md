# Welcome to SimOpt! 

SimOpt is a testbed of simulation-optimization problems and solvers.
* The `simopt` folder contains the source code for the latest version of the testbed, which is written in Python.
* The `MATLAB` folder contains a previous stable version of the testbed written in MATLAB.

# User Guide for SimOpt
The main page for the GUI has all the options necessary to start working. At the top are the three ways to start working with an experiment.
There are three main ways to start an experient, load from a file, cross-design an experient, or create an individual experiement. 
At the bottom, there is a frame containing all experiments. The expereiemnts are in different tabs, depending on if they are inidividual, a meta experiment (made from a cross design expereiment), or if they have ready to be Post- Normalized and plotted. 

## Adding Experiments to the experiment queue
This section will explain how to add experiments to the queue of Experimenets or Meta Experiments.
### Load a File
* In the top left corner a user can click "Load File". Their file system will pop up and the user can naviage to an appriopriate .pickle file to select.
* The file must be .pickle, and if it is not, the GUI will throw an error.
* Once an experiment is loaded, it will be added to the Queue of Experiments. 
* The buttons for Run and Post Process will be updated to accuretly represent if the experiment has already been run and post processed.

### Creating an Experiment through the GUI
Instead of loading in an experiment, the user can create one from the main page of the GUI
* First selct a sovler from the "Solver" drop down.
  * Change factors accossiated with the solver as necessary.
  * All solvers with unique factors, must have unique names (no two solvers can have the same name, but different factors)
* Select a problem from the "Problem" drop down.
  * Change factors accossiated wit hthe Problem and Model as necessary. 
  * All problems with unique factors, must have unique names (no two problems can have the same name, but different factors)
* The default number of macroreplications is 10, the user can adjust this in the top left corner
* Select the "Add Experiment" Button (This button will only show up once a solver and problem is selected)

### Cross-Design Experiment
By cross-designing and experiment, the user is creating a Meta Experiment, and it will be add to the Queue of Meta Experiments. Meta Experiments can only be created this way, and not loaded from a file. 
* Click Cross-Design Experiment
* Check which Problems and Solvers the experiment should contain
* Note that deterministic sovlers can not handle stochastic problems (ex: FACSIZE-2 can not be solved with ASTRODF)
* The default number of Macroreplicaitions is 10, but that can be adjusted
* Click "Confirm Cross-Design Experiment" 
* The pop up window will disappear and the Experiments frame will automatically switch to Queue of Meta Experiments
To exit out of the Meta Experiment pop-up without creating an experiment, click the red x in the top left corner of the window.

## Run an Experiment
To run an experiment or meta experiment that has not already been run click the "Run Exp." button in the queue of experiments or queue of meta experiments. Once the experiment is run, it can not be run again. **Note:** running an experiment could take a couple of seconds to a couple of minutes depending on the experiment and the number of macroreplications.

## Post Process and Post Normalize
Post Processing happens before post normalizing. After Post Normalizing the Plotting window is automatically shown.
To exit out of the Post Process/ Normalize pop-up without post processing or normalizing, click the red x in the top left corner of the window.
### Experiments
Experiments can be post-processed from the queue of Experiments tab, by clicking "Post Process". Adjust Post Processing factors as necessary. Only Experiments that have already been run and have not yet been post processed can be post processed. <br> 
After Post Processing, click the "Post Normalize by Problem" tab to select which experiments to post normalize together.
* Only Experiments with the same Problem can be Post-Normalized Together.
* Once all Experiments are checked, click the "Post Normalize Selected" button at the bottom of the GUI (this button only appears when in the Post Normalize Tab)
* Update any values necessary and click "Post Normalize" when the expermient or experiments are ready to be post normalized
### Meta Experiments
Meta Experiments are post processed and post normalized at the same time.
* Click the Post Process button for the specific Meta Experiment, then change any values necessary, then click "Post Process"

## Plotting Experiments
The Plotting page is the same for both Experiments and Meta Experiments. Currently multiple Experimetns with the same Problem can be plotted together, and any problem/ solver from one Meta Experiment can be plotted together. To return to the main page, simply click the red x in the top left corner of the window.
* On the left side, select a problem from the problem list 
* Select solvers from the solver list
* One the right side select a plot type and adjust plot parameters and setting 
* Click "Add"
* All plots will show in the plotting queue, along with infomration about thier parameters and where the file is saved at. 
* To veiw one plot click "View Plot", or all plots can be viewed together by clicking "See All Plots" at the bottom of the page

