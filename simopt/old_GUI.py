# Necessary libraries or files needed to be imported:

import tkinter as tk

from directory import problem_directory
from directory import solver_directory
from wrapper_base import Experiment

# Beginning of code:

# initializes window
window = tk.Tk()

def get_selected():
    # Problem selected, Solution selected, and Macro Replications valid
    if problemVar.get() != "----" and solverVar.get() != "----" and macroEntry.get().isnumeric() != False:
        # creates blank list to store selections
        selected = []
        # grabs problemVar (whatever is selected our of OptionMenu)
        selected.append(problemVar.get())
        # grabs solverVar (" ")
        selected.append(solverVar.get())
        # grabs number of macro replications
        selected.append(macroEntry.get())

        # complete experiment with given arguments
        solver_name = solverVar.get()
        problem_name = problemVar.get()
        # solver_fixed_factors = dictionary

        # resets problemVar to default value
        problemVar.set("----")
        # resets solverVar
        solverVar.set("----")

        # Macro Replications is a positive integer
        if int(macroEntry.get()) != 0:
            # resets current entry from index 0 to length of entry
            macroEntry.delete(0, len(macroEntry.get()))
            # resets macroEntry textbox
            macroEntry.insert(index=tk.END, string="10")
        
        # Macro Replications is zero
        if int(macroEntry.get()) == 0:
            message = "Please enter a positive (non zero) integer for the number of Macro Replications, example: 10"
            createError(message)

        # prints selected (list) in console/terminal
        print(selected)

        myexperiment = Experiment(solver_name, problem_name, solver_fixed_factors={"sample_size": 50})
        myexperiment.run(n_macroreps= int(macroEntry.get()))
        # myexperiment.post_replicate(n_postreps=200, n_postreps_init_opt=200, crn_across_budget=True, crn_across_macroreps=False)
        # myexperiment.plot_progress_curves(plot_type="all", normalize=False)
        # myexperiment.plot_progress_curves(plot_type="mean", normalize=True)

        postProcessingWindow()



        # returns for future use
        return selected 

    # Problem NOT selected, but Solution selected
    elif problemVar.get() == "----" and solverVar.get() != "----":
        message = "You have not selected a Problem!"
        createError(message)

    # Problem selected, but Solution NOT selected
    elif problemVar.get() != "----" and solverVar.get() == "----":
        message = "You have not selected a Solver!"
        createError(message)
    
    # Macro Replications not numeric or negative
    elif macroEntry.get().isnumeric() == False:
        message = "Please enter a positive (non zero) integer for the number of Macro Replications, example: 10"
        createError(message)

    # Neither Problem nor Solution selected
    else:
        # resets problemVar to default value
        problemVar.set("----")
        # resets solverVar
        solverVar.set("----")
        # prints to console/terminal
        message = "You have not selected all required fields, check for '*' near input boxes."
        createError(message)

def createError(str):
    # initialize error window
    errorWindow = tk.Tk()

    errorLabel = tk.Label(master = errorWindow,
                        # aesthetics of window
                        text = str,
                        foreground= "red",
                        font = "Calibri 11 bold")
    # not used below, but since there is not grid, must use here
    errorLabel.pack()

    # title of window
    errorWindow.title("Error Window")
    # starting size of window
    errorWindow.geometry("700x50")
    # required
    errorWindow.mainloop()

def postProcessingWindow():
    postProcessing = tk.Tk()

    # this will change spacing vertically
    postProcessing.rowconfigure([0, 1, 2, 3, 4, 5],
                    minsize = 100,
                    weight = 1)

    # this will change spacing horizontally
    postProcessing.columnconfigure([0, 1, 2, 3, 4, 5],
                            minsize = 400,
                            weight = 1)

    # n_postreps : int
    n_postrepsLabel = tk.Label(master = postProcessing,
                            text = "Number of postreplications to take at each recommended solution",
                            font = "Calibri 11 bold",
                            wraplength = "250")
    n_postrepsLabel.grid(row=1, column=0)

    n_postrepsVar = tk.StringVar(postProcessing)
    n_postrepsEntry = tk.Entry(master=postProcessing, textvariable = n_postrepsVar, justify = tk.LEFT, width = 12)
    n_postrepsEntry.insert(index=tk.END, string="8")
    n_postrepsEntry.grid(row=1, column=0, sticky='e')

    # n_postreps_init_opt : int
    n_postreps_init_optLabel = tk.Label(master = postProcessing,
                                text = "Number of postreplications to take at initial x\u2070 and optimal x\u002A",
                                font = "Calibri 11 bold",
                                wraplength = "200")
    n_postreps_init_optLabel.grid(row=2, column=0)

    n_postreps_init_optVar = tk.StringVar(postProcessing)
    n_postreps_init_optEntry = tk.Entry(master=postProcessing, textvariable = n_postreps_init_optVar, justify = tk.LEFT, width = 12)
    n_postreps_init_optEntry.insert(index=tk.END, string="6")
    n_postreps_init_optEntry.grid(row=2, column=0, sticky='e')

    # crn_across_budget : boolean
    crn_across_budgetLabel = tk.Label(master = postProcessing,
                                    text = "Use CRN for post-replications at solutions recommended at different times?",
                                    font = "Calibri 11 bold",
                                    wraplength = "300")
    crn_across_budgetLabel.grid(row=1, column=1)

    crn_across_budgetList = ["True", "False"]
    # stays the same, has to change into a special type of variable via tkinter function
    crn_across_budgetVar = tk.StringVar(postProcessing)
    # sets the default OptionMenu selection
    crn_across_budgetVar.set("----")
    # creates drop down menu, for tkinter, it is called "OptionMenu"
    crn_across_budgetMenu = tk.OptionMenu(postProcessing, crn_across_budgetVar, *crn_across_budgetList)
    crn_across_budgetMenu.grid(row=1, column=1, sticky='e')

    # crn_across_macroreps : boolean
    crn_across_macrorepsLabel = tk.Label(master = postProcessing,
                                        text = "Use CRN for post-replications at solutions recommended on different macroreplications?",
                                        font = "Calibri 11 bold",
                                        wraplength = "250")
    crn_across_macrorepsLabel.grid(row=2, column=1)

    crn_across_macrorepsList = ["True", "False"]
    # stays the same, has to change into a special type of variable via tkinter function
    crn_across_macrorepsVar = tk.StringVar(postProcessing)
    # sets the default OptionMenu selection
    crn_across_macrorepsVar.set("----")
    # creates drop down menu, for tkinter, it is called "OptionMenu"
    crn_across_macrorepsMenu = tk.OptionMenu(postProcessing, crn_across_macrorepsVar, *crn_across_macrorepsList)
    crn_across_macrorepsMenu.grid(row=2, column=1, sticky='e')

    postProcessingRunLabel = tk.Label(master=postProcessing, # window label is used for
                        text = "When ready, press the 'Run' button below:",
                        font = "Calibri 11 bold")
    postProcessingRunLabel.grid(row=3, column=1)

    postProcessingRunButton = tk.Button(master=postProcessing, # window button is used in
                        # aesthetic of button and specific formatting options
                        text = "Run", 
                        width = 15, # width of button
                        bd = 5, # boarder size
                        command = test) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click
    postProcessingRunButton.grid(row=3, column=1, sticky='s')



    # postProcessing's title
    postProcessing.title("Post Processing Information")
    # starting size of the postProcessing
    postProcessing.geometry("1450x500")
    # must be included, allows postProcessing to show
    postProcessing.mainloop()


def test():
    print("postProcessing is connected")

    postProcessingList = []
    print(postProcessingList)

# this will change spacing vertically
window.rowconfigure([0, 1, 2, 3, 4, 5],
                    minsize = 100,
                    weight = 1)
# this will change spacing horizontally
window.columnconfigure([0, 1, 2, 3, 4],
                        minsize = 100,
                        weight = 1)

instructionLabel = tk.Label(master=window, # window label is used in
                            text = "Welcome to SimOpt \n Please complete the fields below to run your experiment: \n Please note: '*' are required fields",
                            font = "Calibri 15 bold")
instructionLabel.grid(row=0,column=1) # position

problemLabel = tk.Label(master=window, # window label is used in
                        text = "Please select the type of Problem:*",
                        font = "Calibri 11 bold")
problemLabel.grid(row=1, column=0) # position

# from experiments.inputs.all_factors.py:
problemList = problem_directory
# stays the same, has to change into a special type of variable via tkinter function
problemVar = tk.StringVar(window)
# sets the default OptionMenu value
problemVar.set("----")
# creates drop down menu, for tkinter, it is called "OptionMenu"
problemMenu = tk.OptionMenu(window, problemVar, *problemList)
# stick = 's' means it sticks to the south end of the alloted grid space
problemMenu.grid(row=1, column=0, sticky='s')

solverLabel = tk.Label(master=window, # window label is used in
                        text = "Please select the type of Solver:*",
                        font = "Calibri 11 bold")
solverLabel.grid(row=2, column=0)

# from experiments.inputs.all_factors.py:
solverList = solver_directory
# stays the same, has to change into a special type of variable via tkinter function
solverVar = tk.StringVar(window)
# sets the default OptionMenu selection
solverVar.set("----")
# creates drop down menu, for tkinter, it is called "OptionMenu"
solverMenu = tk.OptionMenu(window, solverVar, *solverList)
solverMenu.grid(row=2, column=0, sticky='s')

runButton = tk.Button(master=window, # window button is used in
                    # aesthetic of button and specific formatting options
                     text = "Run", 
                     width = 15, # width of button
                     bd = 5, # boarder size
                     command = get_selected) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click
runButton.grid(row=3, column=0, sticky='s')

runLabel = tk.Label(master=window, # window label is used for
                    text = "When ready, press the 'Run' button below:",
                    font = "Calibri 11 bold")
runLabel.grid(row=3, column=0)

macroLabel = tk.Label(master=window,
                    text = "Number of Macro Replications:*",
                    font = "Calibri 11 bold")
macroLabel.grid(row=3, column=1)

macroVar = tk.StringVar(window)
macroEntry = tk.Entry(master=window, textvariable = macroVar, justify = tk.LEFT)
macroEntry.insert(index=tk.END, string="10")
macroEntry.grid(row=3, column=1, sticky='e')

# window's title
window.title("SimOpt Application")
# starting size of the window
window.geometry("1450x500")
# must be included, allows window to show
window.mainloop()
