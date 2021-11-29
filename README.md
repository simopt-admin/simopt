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


