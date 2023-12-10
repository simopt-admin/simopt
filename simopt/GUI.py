from email.policy import default
from os import path
from random import expovariate
import tkinter as tk
from tkinter import NONE, Place, ttk, Scrollbar, filedialog, simpledialog, Listbox
from timeit import timeit
from functools import partial
from tkinter.constants import FALSE, MULTIPLE, S
import time
from xml.dom.minidom import parseString
from PIL import ImageTk, Image
import traceback
import pickle
import ast
import os
import sys
import csv
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

import pandas as pd

from simopt.data_farming_base import DataFarmingExperiment, DesignPoint, DataFarmingMetaExperiment

from .directory import problem_directory, problem_unabbreviated_directory, solver_directory, solver_unabbreviated_directory, model_directory, model_unabbreviated_directory, model_problem_unabbreviated_directory, model_problem_class_directory
from .experiment_base import ProblemSolver, ProblemsSolvers, post_normalize, find_missing_experiments, make_full_metaexperiment, plot_progress_curves, plot_solvability_cdfs, plot_area_scatterplots, plot_solvability_profiles, plot_terminal_progress, plot_terminal_scatterplots



class Main_Menu_Window(tk.Tk):
    def __init__(self, master):
        
        self.master = master
         
     
        self.title_label = tk.Label(master = self.master, text = "Welcome to SimOpt Library Graphic User Interface", font = "Calibri 15 bold", 
                                    justify="center", width = 50)
        self.title_label.place(relx = .1, rely = .05)
       
        
        # Button to open original main window to run experiments across solvers & problems
        self.experiment_button = tk.Button( master = self.master, text = 'Problem-Solver Experiment', 
                                           font = 'Calibri 13', width = 50, command = self.open_experiment_window)
        self.experiment_button.place( relx = .15, rely = .2)
        self.experiment_button.configure( background = 'light gray')
        
        
        # Button to open model data farming window
        self.datafarm_model_button = tk.Button(master = self.master, text = 'Model Data Farming (beta)', 
                                           font = 'Calibri 13', width = 50, command = self.open_model_datafarming)
        self.datafarm_model_button.place( relx = .15, rely = .3) 
        self.datafarm_model_button.configure( background = 'light gray')
        
        
        # Commented out for demo
        # # Button to open solver & problem data farming window
        # self.datafarm_prob_sol_button = tk.Button(master = self.master, text = 'Solver Data Farming', 
        #                                    font = 'Calibri 13', width = 50, command = self.open_prob_sol_datafarming)
        # self.datafarm_prob_sol_button.place( relx = .15, rely = .4) 
        # self.datafarm_prob_sol_button.configure( background = 'light gray')
        
    def open_experiment_window(self):
        
        self.create_experiment_window = tk.Toplevel(self.master)
        self.create_experiment_window.geometry("1200x1000")
        self.create_experiment_window.title("SimOpt Library Graphical User Interface - Problem-Solver Experiment")
        self.experiment_app = Experiment_Window(self.create_experiment_window)
    
    def open_model_datafarming(self):
        self.datafarming_window = tk.Toplevel(self.master)
        self.datafarming_window.geometry("1500x850")
        self.datafarming_window.title("SimOpt Library Graphical User Interface - Model Data Farming")
        self.datafarming_app = Data_Farming_Window(self.datafarming_window, self)
        
    def open_prob_sol_datafarming(self):
        self.solver_datafarm_window = tk.Toplevel(self.master)
        self.solver_datafarm_window.geometry("1000x850")
        self.solver_datafarm_window.title("SimOpt Library Graphical User Interface - Solver Data Farming")
        self.solver_datafarm_app = Solver_Datafarming_Window(self.solver_datafarm_window)
    
        
        
        
        
        
    


class Experiment_Window(tk.Toplevel):
    """
    Main window of the GUI

    Attributes
    ----------
    self.frame : Tkinter frame that contains the GUI widgets
    self.experiment_master_list : 2D array list that contains queue of experiment object arguments
    self.widget_list : Current method to clear, view/edit, and run individual experiments
        * this functionality is currently not enabled, possible contraint of the GUI framework
    self.experiment_object_list : List that contains matching experiment objects to every sublist from self.experiment_master_list
    self.problem_var : Variable that contains selected problem (use .get() method to obtain value for)
    self.solver_var : Variable that contains selected solver (use .get() method to obtain value for)
    self.maco_var : Variable that contains inputted number of macroreplications (use .get() method to obtain value for)

    Functions
    ---------
    show_problem_factors(self, *args) : displays additional information on problem and oracle factors
            connected to : self.problem_menu <- ttk.OptionMenu
    show_solver_factors(self, *args) : displays additional information on solver factors
            connected to : self.solver_menu <- ttk.OptionMenu
    run_single_function(self, *args) : completes single-object experiment and invokes Post_Processing_Window class
            connected to : self.run_button <- ttk.Button
    crossdesign_function(self) : invokes Cross_Design_Window class
            connected to : self.crossdesign_button <- ttk.Button
    clearRow_function(self) : ~not functional~ meant to clear a single row of the experiment queue
            connected to : self.clear_button_added <- ttk.Button, within self.add_experiment
    clear_queue(self) : clears entire experiment queue and resets all lists containing experiment data
            connected to : self.clear_queue_button <- ttk.Button
    add_experiment(self) : adds experiment to experiment queue
            connected to : self.add_button <- ttk.Button
    confirm_problem_factors(self) : used within run_single_function, stores all problem factors in a dictionary
            return : problem_factors_return | type = list | contains = [problem factor dictionary, None or problem rename]
    confirm_oracle_factors(self) : used within run_single_function, stores all oracle factors in a dictionary
            return : oracle_factors_return | type = list | contains = [oracle factor dictionary]
    confirm_solver_factors(self) : used within run_single_function, stores all solver factors in a dictionary
            return : solver_factors_return | type = list | contains = [solver factor dictionary, None or solver rename]
    onFrameConfigure_queue(self, event) : creates scrollbar for the queue notebook
    onFrameConfigure_factor_problem(self, event) : creates scrollbar for the problem factors notebook
    onFrameConfigure_factor_solver(self, event) : creates scrollbar for the solver factors notebook
    onFrameConfigure_factor_oracle(self, event) : creates scrollbar for the oracle factor notebook
    test_function(self, *args) : placeholder function to make sure buttons, OptionMenus, etc are connected properly
    """

    def __init__(self, master):

        #problem.model_decision_factors

        self.master = master

        self.frame = tk.Frame(self.master)
        self.count_meta_experiment_queue = 0
        self.experiment_master_list = []
        self.meta_experiment_master_list = []
        self.widget_list = []
        self.experiment_object_list = []
        self.count_experiment_queue = 1
        self.normalize_list = []
        self.normalize_list2 = []
        self.widget_meta_list = []
        self.widget_norm_list = []
        self.post_norm_exp_list = []
        self.prev = 60
        self.meta_experiment_macro_reps = []
        self.check_box_list = []
        self.check_box_list_var = []
        self.list_checked_experiments = []

        self.instruction_label = tk.Label(master=self.master, # window label is used in
                            text = "Welcome to SimOpt Library Graphic User Interface\n Please Load or Add Your Problem-Solver Pair(s): ",
                            font = "Calibri 15 bold",
                            justify="center")

        self.problem_label = tk.Label(master=self.master, # window label is used in
                        text = "Select Problem:",
                        font = "Calibri 13")

        self.or_label = tk.Label(master=self.master, # window label is used in
            text = " OR ",
            font = "Calibri 13")
        self.or_label2 = tk.Label(master=self.master, # window label is used in
            text = " OR Select Problem and Solver from Below:",
            font = "Calibri 13")
        self.or_label22 = tk.Label(master=self.master, # window label is used in
            text = "Select from Below:",
            font = "Calibri 12")

        # from experiments.inputs.all_factors.py:
        self.problem_list = problem_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.problem_var = tk.StringVar(master=self.master)
        # sets the default OptionMenu value

        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.problem_menu = ttk.OptionMenu(self.master, self.problem_var, "Problem", *self.problem_list, command=self.show_problem_factors)

        self.solver_label = tk.Label(master=self.master, # window label is used in
                        text = "Select Solver:",
                        font = "Calibri 13")

        # from experiments.inputs.all_factors.py:
        self.solver_list = solver_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.solver_var = tk.StringVar(master=self.master)
        # sets the default OptionMenu value

        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.solver_menu = ttk.OptionMenu(self.master, self.solver_var, "Solver", *self.solver_list, command=self.show_solver_factors)
        
        #self.macro_label = tk.Label(master=self.master,
        #                text = "Number of Macroreplications:",
          #              font = "Calibri 13")

        self.macro_definition = tk.Label(master=self.master,
                        text = "",
                        font = "Calibri 13")

        self.macro_definition_label = tk.Label(master=self.master,
                                                  text = "Number of Macroreplications:",
                                                  font = "Calibri 13",
                                                  width = 25)
        
        self.macro_var = tk.StringVar(self.master)
        self.macro_entry = ttk.Entry(master=self.master, textvariable = self.macro_var, justify = tk.LEFT, width=10)
        self.macro_entry.insert(index=tk.END, string="10")

        self.add_button = ttk.Button(master=self.master,
                                    text = "Add Problem-Solver Pair",
                                    width = 15,
                                    command=self.add_experiment)

        self.clear_queue_button = ttk.Button(master=self.master,
                                    text = "Clear All Problem-Solver Pairs",
                                    width = 15,
                                    command = self.clear_queue)#(self.experiment_added, self.problem_added, self.solver_added, self.macros_added, self.run_button_added))

        self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Create Problem-Solver Group",
                                            width = 50,
                                            command = self.crossdesign_function)
        

        self.pickle_file_load_button = ttk.Button(master=self.master,
                                                text = "Load Problem-Solver Pair",
                                                width = 50,
                                                command = self.load_pickle_file_function)

        self.attribute_description_label = tk.Label(master=self.master,
                                                    text = "Attribute Description Label for Problems:\n Objective (Single [S] or Multiple [M])\n Constraint (Unconstrained [U], Box[B], Determinisitic [D], Stochastic [S])\n Variable (Discrete [D], Continuous [C], Mixed [M])\n Gradient Available (True [G] or False [N])" ,
                                                    font = "Calibri 9"
                                                    )
        self.attribute_description_label.place(x= 450, rely = 0.478)


        self.post_normal_all_button = ttk.Button(master=self.master,
                                                text = "Post-Normalize Selected",
                                                width = 20,
                                                state = "normal",
                                                command = self.post_normal_all_function)

        self.make_meta_experiment = ttk.Button(master=self.master,
                                                text = "Create Problem-Solver Group from Selected",
                                                width = 35,
                                                state = "normal",
                                                command = self.make_meta_experiment_func) 
                
        self.pickle_file_pathname_label = tk.Label(master=self.master,
                                                    text = "File Selected:",
                                                    font = "Calibri 13")

        self.pickle_file_pathname_show = tk.Label(master=self.master,
                                                    text = "No File Selected!",
                                                    font = "Calibri 12 italic",
                                                    foreground = "red",
                                                    wraplength = "500")


        self.style = ttk.Style()
        self.style.configure("Bold.TLabel", font = ("Calibri",15,"bold"))
        self.label_Workspace = ttk.Label(master = self.master, text = "Workspace", style="Bold.TLabel")
        self.queue_label_frame = ttk.LabelFrame(master=self.master, labelwidget= self.label_Workspace)

        self.queue_canvas = tk.Canvas(master=self.queue_label_frame, borderwidth=0)

        self.queue_frame = ttk.Frame(master=self.queue_canvas)
        self.vert_scroll_bar = Scrollbar(self.queue_label_frame, orient="vertical", command=self.queue_canvas.yview)
        self.horiz_scroll_bar = Scrollbar(self.queue_label_frame, orient="horizontal", command=self.queue_canvas.xview)
        self.queue_canvas.configure(xscrollcommand=self.horiz_scroll_bar.set, yscrollcommand=self.vert_scroll_bar.set)

        self.vert_scroll_bar.pack(side="right", fill="y")
        self.horiz_scroll_bar.pack(side="bottom", fill="x")

        self.queue_canvas.pack(side="left", fill="both", expand=True)
        self.queue_canvas.create_window((0,0), window=self.queue_frame, anchor="nw",
                                  tags="self.queue_frame")

        self.queue_frame.bind("<Configure>", self.onFrameConfigure_queue)

        self.notebook = ttk.Notebook(master=self.queue_frame)
        self.notebook.pack(fill="both")

        self.tab_one = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_one, text="Queue of Problem-Solver Pairs")

        self.tab_one.grid_rowconfigure(0)
        

        self.heading_list = ["Selected","Pair #", "Problem", "Solver", "Macroreps", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_one.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_one, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)

        self.tab_two = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_two, text="Queue of Problem-Solver Groups")
        self.tab_two.grid_rowconfigure(0)
        self.heading_list = ["Problems", "Solvers", "Macroreps", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_two.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_two, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)

        self.tab_three = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_three, text="Post-Normalize by Problem")
        self.tab_three.grid_rowconfigure(0)
        self.heading_list = ["Problem", "Solvers", "Selected", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_three.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_three, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)

        def on_tab_change(event):
            tab = event.widget.tab('current')['text']
            if tab == 'Post-Normalize by Problem':
                self.post_norm_setup()
                self.post_normal_all_button.place(x=10,rely=.92)
            else:
                self.post_normal_all_button.place_forget()
            if tab == 'Queue of Problem-Solver Pairs':
                # My code starts here 
                # Make meta experiment button wider & releative x
                self.make_meta_experiment.place(relx=.02,rely=.92, width= 300)
                # My code ends here
            else:
                self.make_meta_experiment.place_forget()



        self.notebook.bind('<<NotebookTabChanged>>', on_tab_change)

        self.instruction_label.place(relx=.4, y=0)

        self.solver_label.place(relx=.01, rely=.1)
        self.solver_menu.place(relx=.1, rely=.1 )


        self.problem_label.place(relx=.3, rely=.1)
        self.problem_menu.place(relx=.4, rely=.1)

        #self.macro_label.place(relx=.7, rely=.1)
        self.macro_entry.place(relx=.89, rely=.1, width=100)

        self.macro_definition.place(relx=.73, rely=.05)
        self.macro_definition_label.place(relx=.7, rely=.1)

        #self.macro_definition_label.bind("<Enter>",self.on_enter)
        #self.macro_definition_label.bind("<Leave>",self.on_leave)

        self.or_label.place(x=215, rely=.06)
        self.crossdesign_button.place(x=255, rely=.06, width=220)
        
        

        y_place = .06
        self.pickle_file_load_button.place(x=10, rely=y_place, width=195)
        self.or_label2.place(x=480, rely=.06)
        # self.or_label22.place(x=435, rely=.06)

        self.queue_label_frame.place(x=10, rely=.56, relheight=.35, relwidth=.99)
        # self.post_normal_all_button.place(x=400,rely=.95)

        self.frame.pack(fill='both')

        # uncomment this to test hover

        # self.l1 = tk.Button(self.master, text="Hover over me")
        # self.l2 = tk.Label(self.master, text="", width=40)
        # self.l1.place(x=10,y=0)
        # self.l2.place(x=10,y=20)

        # self.l1.bind("<Enter>", self.on_enter)
        # self.l1.bind("<Leave>", self.on_leave)
    #def on_enter(self, event):
        # self.l2(text="Hover Works :)")
    #def on_leave(self, enter):
        #self.l2.configure(text="")

    #def on_enter(self, event):
        #self.macro_definition.configure(text="Definition of MacroReplication")

    #def on_leave(self, enter):
        #self.macro_definition.configure(text="")

    def show_problem_factors(self, *args):
        # if args and len(args) == 2:
        #     print("ARGS: ", args[1])
        #("arg length:", len(args))

        self.problem_factors_list = []
        self.problem_factors_types = []

        self.factor_label_frame_problem = ttk.LabelFrame(master=self.master, text="Problem Factors")

        self.factor_canvas_problem = tk.Canvas(master=self.factor_label_frame_problem, borderwidth=0)

        self.factor_frame_problem = ttk.Frame(master=self.factor_canvas_problem)
        self.vert_scroll_bar_factor_problem = Scrollbar(self.factor_label_frame_problem, orient="vertical", command=self.factor_canvas_problem.yview)
        self.horiz_scroll_bar_factor_problem = Scrollbar(self.factor_label_frame_problem, orient="horizontal", command=self.factor_canvas_problem.xview)
        self.factor_canvas_problem.configure(xscrollcommand=self.horiz_scroll_bar_factor_problem.set, yscrollcommand=self.vert_scroll_bar_factor_problem.set)

        self.vert_scroll_bar_factor_problem.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_problem.pack(side="bottom", fill="x")

        self.factor_canvas_problem.pack(side="left", fill="both", expand=True)
        self.factor_canvas_problem.create_window((0,0), window=self.factor_frame_problem, anchor="nw",
                                  tags="self.factor_frame_problem")

        self.factor_frame_problem.bind("<Configure>", self.onFrameConfigure_factor_problem)

        self.factor_notebook_problem = ttk.Notebook(master=self.factor_frame_problem)
        self.factor_notebook_problem.pack(fill="both")

        self.factor_tab_one_problem = tk.Frame(master=self.factor_notebook_problem)

        self.factor_notebook_problem.add(self.factor_tab_one_problem, text=str(self.problem_var.get()) + " Factors")

        self.factor_tab_one_problem.grid_rowconfigure(0)

        self.factor_heading_list_problem = ["Description", "Input"]

        for heading in self.factor_heading_list_problem:
            self.factor_tab_one_problem.grid_columnconfigure(self.factor_heading_list_problem.index(heading))
            label_problem = tk.Label(master=self.factor_tab_one_problem, text=heading, font="Calibri 14 bold")
            label_problem.grid(row=0, column=self.factor_heading_list_problem.index(heading), padx=10, pady=3)

        
        self.problem_object = problem_unabbreviated_directory[self.problem_var.get()]
        
        count_factors_problem = 1
        
        if args and len(args) == 2 and args[0] == True:
            oldname = args[1][3][1]
            
        else:
            problem_object = problem_unabbreviated_directory[self.problem_var.get()]
            oldname = problem_object().name
            

        self.save_label_problem = tk.Label(master=self.factor_tab_one_problem,
                                            text = "save problem as",
                                            font = "Calibri 13")

        self.save_var_problem = tk.StringVar(self.factor_tab_one_problem)
        self.save_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.save_var_problem, justify = tk.LEFT, width=15)
        
        self.save_entry_problem.insert(index=tk.END, string=oldname)

        self.save_label_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
        self.save_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

        self.problem_factors_list.append(self.save_var_problem)
        self.problem_factors_types.append(str)
        
        count_factors_problem += 1
        
        for num, factor_type in enumerate(self.problem_object().specifications, start=0):
            #(factor_type, len(self.problem_object().specifications[factor_type]['default']) )

            self.dictionary_size_problem = len(self.problem_object().specifications[factor_type])
            datatype = self.problem_object().specifications[factor_type].get("datatype")
            description = self.problem_object().specifications[factor_type].get("description")
            default = self.problem_object().specifications[factor_type].get("default")

            if datatype != bool:


                self.int_float_description_problem = tk.Label(master=self.factor_tab_one_problem,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)

                self.int_float_var_problem = tk.StringVar(self.factor_tab_one_problem)
                self.int_float_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.int_float_var_problem, justify = tk.LEFT, width=15)
                if args and len(args) == 2 and args[0] == True:
                    self.int_float_entry_problem.insert(index=tk.END, string=str(args[1][3][0][factor_type]))
                elif datatype == tuple and len(default) == 1:
                    #(factor_type, len(self.problem_object().specifications[factor_type]['default']) )
                    # self.int_float_entry_problem.insert(index=tk.END, string=str(self.problem_object().specifications[factor_type].get("default")))
                    self.int_float_entry_problem.insert(index=tk.END, string=str(default[0]))
                else:
                    self.int_float_entry_problem.insert(index=tk.END, string=str(default))

                self.int_float_description_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
                self.int_float_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

                self.problem_factors_list.append(self.int_float_var_problem)
                if datatype != tuple:
                    self.problem_factors_types.append(datatype)
                else:
                    self.problem_factors_types.append(str)

                count_factors_problem += 1


            if datatype == bool:

                self.boolean_description_problem = tk.Label(master=self.factor_tab_one_problem,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)
                self.boolean_var_problem = tk.BooleanVar(self.factor_tab_one_problem, value = bool(default))
                self.boolean_menu_problem = tk.Checkbutton(self.factor_tab_one_problem, variable=self.boolean_var_problem.get(), onvalue=True, offvalue=False)
                self.boolean_description_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
                self.boolean_menu_problem.grid(row=count_factors_problem, column=1, sticky='nsew')
                self.problem_factors_list.append(self.boolean_var_problem)
                self.problem_factors_types.append(datatype)

                count_factors_problem += 1

        #self.factor_label_frame_problem.place(x=400, y=70, height=300, width=475)
        self.factor_label_frame_problem.place(relx=.35, rely=.15, relheight=.33, relwidth=.34)

        # Switching from Problems to Oracles

        self.oracle_factors_list = []
        self.oracle_factors_types = []


        ## Rina Adding After this 
        problem = str(self.problem_var.get())  
        self.oracle = model_problem_unabbreviated_directory[problem] # returns model string
        self.oracle_object = model_directory[self.oracle]
        ##self.oracle = problem.split("-") 
        ##self.oracle = self.oracle[0] 
        ##self.oracle_object = model_directory[self.oracle] 
        
        ## Stop adding for Rina  
    
        self.factor_label_frame_oracle = ttk.LabelFrame(master=self.master, text="Model Factors")

        self.factor_canvas_oracle = tk.Canvas(master=self.factor_label_frame_oracle, borderwidth=0)

        self.factor_frame_oracle = ttk.Frame(master=self.factor_canvas_oracle)
        self.vert_scroll_bar_factor_oracle = Scrollbar(self.factor_label_frame_oracle, orient="vertical", command=self.factor_canvas_oracle.yview)
        self.horiz_scroll_bar_factor_oracle = Scrollbar(self.factor_label_frame_oracle, orient="horizontal", command=self.factor_canvas_oracle.xview)
        self.factor_canvas_oracle.configure(xscrollcommand=self.horiz_scroll_bar_factor_oracle.set, yscrollcommand=self.vert_scroll_bar_factor_oracle.set)

        self.vert_scroll_bar_factor_oracle.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_oracle.pack(side="bottom", fill="x")

        self.factor_canvas_oracle.pack(side="left", fill="both", expand=True)
        self.factor_canvas_oracle.create_window((0,0), window=self.factor_frame_oracle, anchor="nw",
                                  tags="self.factor_frame_oracle")

        self.factor_frame_oracle.bind("<Configure>", self.onFrameConfigure_factor_oracle)

        self.factor_notebook_oracle = ttk.Notebook(master=self.factor_frame_oracle)
        self.factor_notebook_oracle.pack(fill="both")

        self.factor_tab_one_oracle = tk.Frame(master=self.factor_notebook_oracle)

        self.factor_notebook_oracle.add(self.factor_tab_one_oracle, text=str(self.oracle + " Factors"))

        self.factor_tab_one_oracle.grid_rowconfigure(0)

        self.factor_heading_list_oracle = ["Description", "Input"]

        for heading in self.factor_heading_list_oracle:
            self.factor_tab_one_oracle.grid_columnconfigure(self.factor_heading_list_oracle.index(heading))
            label_oracle = tk.Label(master=self.factor_tab_one_oracle, text=heading, font="Calibri 14 bold")
            label_oracle.grid(row=0, column=self.factor_heading_list_oracle.index(heading), padx=10, pady=3)


        count_factors_oracle = 1
        for factor_type in self.oracle_object().specifications:

            self.dictionary_size_oracle = len(self.oracle_object().specifications[factor_type])
            datatype = self.oracle_object().specifications[factor_type].get("datatype") 
            description = self.oracle_object().specifications[factor_type].get("description") 
            default = self.oracle_object().specifications[factor_type].get("default") 

            if datatype!= bool:

                #("yes?")
                self.int_float_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)

                self.int_float_var_oracle = tk.StringVar(self.factor_tab_one_oracle)
                self.int_float_entry_oracle = ttk.Entry(master=self.factor_tab_one_oracle, textvariable = self.int_float_var_oracle, justify = tk.LEFT, width = 15)

                if args and len(args) == 2 and args[0] == True:
                    self.int_float_entry_oracle.insert(index=tk.END, string=str(args[1][4][0][factor_type]))
                else:
                    self.int_float_entry_oracle.insert(index=tk.END, string=str(default))

                self.int_float_description_oracle.grid(row=count_factors_oracle, column=0, sticky='nsew')
                self.int_float_entry_oracle.grid(row=count_factors_oracle, column=1, sticky='nsew')

                self.oracle_factors_list.append(self.int_float_var_oracle)
                if datatype != tuple:
                    self.oracle_factors_types.append(datatype)
                else:
                    self.oracle_factors_types.append(str)

                count_factors_oracle += 1


            if datatype == bool:
                self.boolean_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)
                self.boolean_var_oracle = tk.BooleanVar(self.factor_tab_one_oracle, value = bool(default))
                self.boolean_menu_oracle = tk.Checkbutton(self.factor_tab_one_oracle, variable=self.boolean_var_oracle.get(), onvalue=True, offvalue=False)
                self.boolean_description_oracle.grid(row=count_factors_oracle, column=0, sticky='nsew')
                self.boolean_menu_oracle.grid(row=count_factors_oracle, column=1, sticky='nsew')
                self.oracle_factors_list.append(self.boolean_var_oracle)
                self.oracle_factors_types.append(datatype)

                count_factors_oracle += 1

        self.factor_label_frame_oracle.place(relx=.7, rely=.15, relheight=.33, relwidth=.3)
        if str(self.solver_var.get()) != "Solver":
            self.add_button.place(x=10, rely=.48, width=200, height=30)

    def show_solver_factors(self, *args):
        
        if args and len(args) == 3 and args[2] == False:
            pass
        else:
            self.update_problem_list_compatability()

        self.solver_factors_list = []
        self.solver_factors_types = []
    
        self.factor_label_frame_solver = ttk.LabelFrame(master=self.master, text="Solver Factors")

        self.factor_canvas_solver = tk.Canvas(master=self.factor_label_frame_solver, borderwidth=0)

        self.factor_frame_solver = ttk.Frame(master=self.factor_canvas_solver)
        self.vert_scroll_bar_factor_solver = Scrollbar(self.factor_label_frame_solver, orient="vertical", command=self.factor_canvas_solver.yview)
        self.horiz_scroll_bar_factor_solver = Scrollbar(self.factor_label_frame_solver, orient="horizontal", command=self.factor_canvas_solver.xview)
        self.factor_canvas_solver.configure(xscrollcommand=self.horiz_scroll_bar_factor_solver.set, yscrollcommand=self.vert_scroll_bar_factor_solver.set)

        self.vert_scroll_bar_factor_solver.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_solver.pack(side="bottom", fill="x")

        self.factor_canvas_solver.pack(side="left", fill="both", expand=True)
        self.factor_canvas_solver.create_window((0,0), window=self.factor_frame_solver, anchor="nw",
                                  tags="self.factor_frame_solver")

        self.factor_frame_solver.bind("<Configure>", self.onFrameConfigure_factor_solver)

        self.factor_notebook_solver = ttk.Notebook(master=self.factor_frame_solver)
        self.factor_notebook_solver.pack(fill="both")

        self.factor_tab_one_solver = tk.Frame(master=self.factor_notebook_solver)

        self.factor_notebook_solver.add(self.factor_tab_one_solver, text=str(self.solver_var.get()) + " Factors")

        self.factor_tab_one_solver.grid_rowconfigure(0)

        self.factor_heading_list_solver = ["Description", "Input"]

        for heading in self.factor_heading_list_solver:
            self.factor_tab_one_solver.grid_columnconfigure(self.factor_heading_list_solver.index(heading))
            label = tk.Label(master=self.factor_tab_one_solver, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.factor_heading_list_solver.index(heading), padx=10, pady=3)

        self.solver_object = solver_unabbreviated_directory[self.solver_var.get()]

        count_factors_solver = 1
        
        self.save_label_solver = tk.Label(master=self.factor_tab_one_solver,
                                            text = "save solver as",
                                            font = "Calibri 13")

                                  
        if args and len(args) == 3 and args[0] == True:
            oldname = args[1][5][1]
            
        else:
            solver_object = solver_unabbreviated_directory[self.solver_var.get()]
            oldname = solver_object().name
            

        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.save_var_solver, justify = tk.LEFT, width=15)
        

        self.save_entry_solver.insert(index=tk.END, string=oldname)

        self.save_label_solver.grid(row=count_factors_solver, column=0, sticky='nsew')
        self.save_entry_solver.grid(row=count_factors_solver, column=1, sticky='nsew')

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)
        
        count_factors_solver += 1
        
        for factor_type in self.solver_object().specifications:
            #("size of dictionary", len(self.solver_object().specifications[factor_type]))
            #("first", factor_type)
            #("second", self.solver_object().specifications[factor_type].get("description"))
            #("third", self.solver_object().specifications[factor_type].get("datatype"))
            #("fourth", self.solver_object().specifications[factor_type].get("default"))

            self.dictionary_size_solver = len(self.solver_object().specifications[factor_type])
            datatype = self.solver_object().specifications[factor_type].get("datatype")
            description = self.solver_object().specifications[factor_type].get("description")
            default = self.solver_object().specifications[factor_type].get("default")
            if datatype != bool:

                self.int_float_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)

                self.int_float_var = tk.StringVar(self.factor_tab_one_solver)
                self.int_float_entry = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.int_float_var, justify = tk.LEFT, width=15)
                
                if args and len(args) == 3 and args[0] == True:
                    self.int_float_entry.insert(index=tk.END, string=str(args[1][5][0][factor_type]))
                else:
                    self.int_float_entry.insert(index=tk.END, string=str(default))

                self.int_float_description.grid(row=count_factors_solver, column=0, sticky='nsew')
                self.int_float_entry.grid(row=count_factors_solver, column=1, sticky='nsew')
                self.solver_factors_list.append(self.int_float_var)

                
                
                if datatype != tuple:
                    self.solver_factors_types.append(datatype)
                else:
                    self.solver_factors_types.append(str)

                count_factors_solver += 1


            if datatype == bool:

                self.boolean_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)
                self.boolean_var = tk.BooleanVar(self.factor_tab_one_solver, value = bool(default))
                self.boolean_menu = tk.Checkbutton(self.factor_tab_one_solver, variable=self.boolean_var.get(), onvalue=True, offvalue=False)
                self.boolean_description.grid(row=count_factors_solver, column=0, sticky='nsew')
                self.boolean_menu.grid(row=count_factors_solver, column=1, sticky='nsew')
                self.solver_factors_list.append(self.boolean_var)
                self.solver_factors_types.append(datatype)

                count_factors_solver += 1
        
        # self.factor_label_frame_problem.place(relx=.32, y=70, height=150, relwidth=.34)
        self.factor_label_frame_solver.place(x=10, rely=.15, relheight=.33, relwidth=.34)
        if str(self.problem_var.get()) != "Problem":
            self.add_button.place(x=10, rely=.48, width=200, height=30)
    
    #Creates a function that checks the compatibility of the solver picked with the list of problems and adds
    #the compatible problems to a new list 
    def update_problem_list_compatability(self):

        if self.solver_var.get() != "Solver":
            self.problem_menu.destroy()
            temp_problem_list = []
            
            for problem in problem_unabbreviated_directory:

                temp_problem = problem_unabbreviated_directory[problem] # problem object
                temp_problem_name = temp_problem().name
                
                temp_solver = solver_unabbreviated_directory[self.solver_var.get()]
                temp_solver_name = temp_solver().name

                temp_experiment = ProblemSolver(solver_name=temp_solver_name, problem_name=temp_problem_name)
                comp = temp_experiment.check_compatibility()

                if comp == "":
                    temp_problem_list.append(problem)

            # from experiments.inputs.all_factors.py:
            self.problem_list = temp_problem_list
            # stays the same, has to change into a special type of variable via tkinter function
            self.problem_var = tk.StringVar(master=self.master)
            # sets the default OptionMenu value

            # creates drop down menu, for tkinter, it is called "OptionMenu"
            self.problem_menu = ttk.OptionMenu(self.master, self.problem_var, "Problem", *self.problem_list, command=self.show_problem_factors)
            self.problem_menu.place(relx=.4, rely=.1)

    def clearRow_function(self, integer):

        for widget in self.widget_list[integer-1]:
            widget.grid_remove()

        self.experiment_master_list.pop(integer-1)
        self.experiment_object_list.pop(integer-1)
        self.widget_list.pop(integer-1)

        self.check_box_list[integer-1].grid_remove()

        self.check_box_list.pop(integer -1)
        self.check_box_list_var.pop(integer -1)



        # if (integer - 1) in self.normalize_list:
        #     self.normalize_list.remove(integer - 1)
        # for i in range(len(self.normalize_list)):
        #     if i < self.normalize_list[i]:
        #         self.normalize_list[i] = self.normalize_list[i] - 1

        for row_of_widgets in self.widget_list:
            row_index = self.widget_list.index(row_of_widgets)
            row_of_widgets[7]["text"] = str(row_index+1)

            run_button_added = row_of_widgets[3]
            text_on_run = run_button_added["text"]
            split_text = text_on_run.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            # new_text = " ".join(split_text)
            # run_button_added["text"] = new_text
            run_button_added["command"] = partial(self.run_row_function, row_index+1)

            row_of_widgets[3] = run_button_added

            viewEdit_button_added = row_of_widgets[4]
            text_on_viewEdit = viewEdit_button_added["text"]
            split_text = text_on_viewEdit.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            # new_text = " ".join(split_text)
            # viewEdit_button_added["text"] = new_text
            viewEdit_button_added["command"] = partial(self.viewEdit_function, row_index+1)

            row_of_widgets[4] = viewEdit_button_added

            clear_button_added = row_of_widgets[5]
            text_on_clear = clear_button_added["text"]
            split_text = text_on_clear.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            # new_text = " ".join(split_text)
            # clear_button_added["text"] = new_text
            clear_button_added["command"] = partial(self.clearRow_function, row_index+1)

            row_of_widgets[5] = clear_button_added

            postprocess_button_added = row_of_widgets[6]
            postprocess_button_added["command"] = partial(self.post_rep_function, row_index+1)
            
            row_of_widgets[6] = postprocess_button_added

            current_check_box = self.check_box_list[row_index]
            current_check_box.grid(row =(row_index+1), column=0, sticky='nsew', padx=10, pady=3)
            row_of_widgets[7].grid(row= (row_index+1), column=1, sticky='nsew', padx=10, pady=3)
            row_of_widgets[0].grid(row= (row_index+1), column=2, sticky='nsew', padx=10, pady=3)
            row_of_widgets[1].grid(row= (row_index+1), column=3, sticky='nsew', padx=10, pady=3)
            row_of_widgets[2].grid(row= (row_index+1), column=4, sticky='nsew', padx=10, pady=3)
            row_of_widgets[3].grid(row= (row_index+1), column=5, sticky='nsew', padx=10, pady=3)
            row_of_widgets[4].grid(row= (row_index+1), column=6, sticky='nsew', padx=10, pady=3)
            row_of_widgets[5].grid(row= (row_index+1), column=7, sticky='nsew', padx=10, pady=3)
            row_of_widgets[6].grid(row= (row_index+1), column=8, sticky='nsew', padx=10, pady=3)
            



        self.count_experiment_queue = len(self.widget_list) + 1

    def clear_meta_function(self, integer):
        for widget in self.widget_meta_list[integer-1]:
            widget.grid_remove()

        self.meta_experiment_master_list.pop(integer-1)

        self.widget_meta_list.pop(integer-1)

        for row_of_widgets in self.widget_meta_list:
            row_index = self.widget_meta_list.index(row_of_widgets)

            run_button_added = row_of_widgets[3]
            text_on_run = run_button_added["text"]
            split_text = text_on_run.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            new_text = " ".join(split_text)
            run_button_added["text"] = new_text
            run_button_added["command"] = partial(self.run_meta_function, row_index+1)
            row_of_widgets[3] = run_button_added


            clear_button_added = row_of_widgets[4]
            text_on_clear = clear_button_added["text"]
            split_text = text_on_clear.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            new_text = " ".join(split_text)
            clear_button_added["text"] = new_text
            clear_button_added["command"] = partial(self.clear_meta_function, row_index+1)
            row_of_widgets[4] = clear_button_added

            postprocess_button_added = row_of_widgets[5]
            postprocess_button_added["command"] = partial(self.post_rep_meta_function, row_index+1)
            row_of_widgets[5] = postprocess_button_added

            plot_button_added = row_of_widgets[6]
            plot_button_added["command"] = partial(self.plot_meta_function,row_index+1)
            row_of_widgets[6] = plot_button_added

            view_button_added = row_of_widgets[7]
            view_button_added["command"] = partial(self.view_meta_function,row_index+1)
            row_of_widgets[7] = view_button_added


            row_of_widgets[0].grid(row= (row_index+1), column=0, sticky='nsew', padx=10, pady=3)
            row_of_widgets[1].grid(row= (row_index+1), column=1, sticky='nsew', padx=10, pady=3)
            row_of_widgets[2].grid(row= (row_index+1), column=2, sticky='nsew', padx=10, pady=3)
            row_of_widgets[3].grid(row= (row_index+1), column=3, sticky='nsew', padx=10, pady=3)
            row_of_widgets[4].grid(row= (row_index+1), column=4, sticky='nsew', padx=10, pady=3)
            row_of_widgets[5].grid(row= (row_index+1), column=5, sticky='nsew', padx=10, pady=3)
            row_of_widgets[6].grid(row= (row_index+1), column=6, sticky='nsew', padx=10, pady=3)
            row_of_widgets[7].grid(row= (row_index+1), column=6, sticky='nsew', padx=10, pady=3)
            
        # self.count_meta_experiment_queue = len(self.widget_meta_list) + 1
        self.count_meta_experiment_queue = self.count_meta_experiment_queue - 1


        # resets problem_var to default value
        self.problem_var.set("Problem")
        # resets solver_var to default value
        self.solver_var.set("Solver")

    def viewEdit_function(self, integer):
        row_index = integer

        current_experiment = self.experiment_object_list[row_index-1]
        #(current_experiment)
        current_experiment_arguments = self.experiment_master_list[row_index-1]

        
        self.problem_var.set(current_experiment_arguments[0])
        #self.problem_var.set(problem_solver_abbreviated_name_to_unabbreviated(current_experiment_arguments[0], problem_directory, problem_unabbreviated_directory))
        
        self.solver_var.set(current_experiment_arguments[1])
        #self.solver_var.set(problem_solver_abbreviated_name_to_unabbreviated(current_experiment_arguments[1], solver_directory, solver_unabbreviated_directory))'
        
        self.macro_var.set(current_experiment_arguments[2])
        self.show_problem_factors(True, current_experiment_arguments)
        #print(" self.show_problem_factors", self.show_problem_factors(True, current_experiment_arguments))
        # self.my_experiment[1][3][1]
        self.show_solver_factors(True, current_experiment_arguments, False)
        #print("self.show_solver_factors", self. show_solver_factors(True, current_experiment_arguments))
        viewEdit_button_added = self.widget_list[row_index-1][5]
        viewEdit_button_added["text"] = "Save Changes"
        viewEdit_button_added["command"] = partial(self.save_edit_function, row_index)
        viewEdit_button_added.grid(row= (row_index), column=5, sticky='nsew', padx=10, pady=3)

    def clear_queue(self):

        # for row in self.widget_list:
        #     for widget in row:
        #         widget.grid_remove()
        for row in range(len(self.widget_list),0,-1):
            self.clearRow_function(row)


        self.experiment_master_list.clear()
        self.experiment_object_list.clear()
        self.widget_list.clear()

    def add_experiment(self, *args):

        if len(args) == 1 and isinstance(args[0], int) :
            place = args[0] - 1
        else:
            place = len(self.experiment_object_list)

        if (self.problem_var.get() in problem_unabbreviated_directory and self.solver_var.get() in solver_unabbreviated_directory and self.macro_entry.get().isnumeric() != False):
            # creates blank list to store selections
            self.selected = []
            # grabs problem_var (whatever is selected our of OptionMenu)
            self.selected.append(self.problem_var.get())
            # grabs solver_var (" ")
            self.selected.append(self.solver_var.get())
            # grabs macro_entry
            self.selected.append(int(self.macro_entry.get()))
            # grabs problem factors & problem rename
            problem_factors = self.confirm_problem_factors()
            self.selected.append(problem_factors)
            # grabs oracle factors
            oracle_factors = self.confirm_oracle_factors()
            self.selected.append(oracle_factors)
            # grabs solver factors & solver rename
            solver_factors = self.confirm_solver_factors()
            self.selected.append(solver_factors)

            self.macro_reps = self.selected[2]
            self.solver_name = self.selected[1]
            self.problem_name = self.selected[0]



            # macro_entry is a positive integer
            if int(self.macro_entry.get()) != 0:
                # resets current entry from index 0 to length of entry
                self.macro_entry.delete(0, len(self.macro_entry.get()))
                # resets macro_entry textbox
                self.macro_entry.insert(index=tk.END, string="10")

                # complete experiment with given arguments
                self.solver_dictionary_rename = self.selected[5]
                self.solver_rename = self.solver_dictionary_rename[1]
                self.solver_factors = self.solver_dictionary_rename[0]

                self.oracle_factors = self.selected[4]
                self.oracle_factors = self.oracle_factors[0]

                self.problem_dictionary_rename = self.selected[3]
                self.problem_rename = self.problem_dictionary_rename[1]
                self.problem_factors = self.problem_dictionary_rename[0]

                self.macro_reps = self.selected[2]
                self.solver_name = self.selected[1]
                self.problem_name = self.selected[0]

                
                
                solver_object,self.solver_name = problem_solver_unabbreviated_to_object(self.solver_name,solver_unabbreviated_directory)
                problem_object, self.problem_name = problem_solver_unabbreviated_to_object(self.problem_name,problem_unabbreviated_directory)
                

                # self.selected[0] = self.problem_name

                self.my_experiment = ProblemSolver(solver_name=self.solver_name, problem_name=self.problem_name, solver_rename=self.solver_rename, problem_rename=self.problem_rename, solver_fixed_factors=self.solver_factors, problem_fixed_factors=self.problem_factors, model_fixed_factors=self.oracle_factors)
                # print("type", type(self.selected[2]))
                self.my_experiment.n_macroreps = self.selected[2]
                self.my_experiment.post_norm_ready = False

                compatibility_result = self.my_experiment.check_compatibility()
                for exp in self.experiment_object_list:
                    if exp.problem.name == self.my_experiment.problem.name and exp.solver.name == self.my_experiment.solver.name:
                        if exp.problem != self.my_experiment.problem:
                            message = "Please Save the Problem for Unique Factors with a Unique Name"
                            tk.messagebox.showerror(title="Error Window", message=message)
                            return False


                if compatibility_result == "":
                    self.experiment_object_list.insert(place,self.my_experiment)
                    self.experiment_master_list.insert(place,self.selected)
                    #this option list doesnt autoupdate - not sure why but this will force it to update
                    self.experiment_master_list[place][5][0]['crn_across_solns'] = self.boolean_var.get()

                    self.rows = 5

                    self.problem_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[3][1],
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.problem_added.grid(row=self.count_experiment_queue, column=2, sticky='nsew', padx=10, pady=3)

                    self.checkbox_select_var = tk.BooleanVar(self.tab_one, value = False)
                    self.checkbox_select = tk.Checkbutton(master=self.tab_one,text="", state = "normal", variable =self.checkbox_select_var )
                    self.checkbox_select.deselect()
                    self.checkbox_select.grid(row=self.count_experiment_queue, column=0, sticky='nsew', padx=10, pady=3)

                    self.exp_num = tk.Label(master=self.tab_one,
                                                    text = str(self.count_experiment_queue),
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.exp_num.grid(row=self.count_experiment_queue, column=1, sticky='nsew', padx=10, pady=3)

                    self.solver_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[5][1],
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.solver_added.grid(row=self.count_experiment_queue, column=3, sticky='nsew', padx=10, pady=3)

                    self.macros_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[2],
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.macros_added.grid(row=self.count_experiment_queue, column=4, sticky='nsew', padx=10, pady=3)

                    self.run_button_added = ttk.Button(master=self.tab_one,
                                                        text="Run" ,
                                                        command= partial(self.run_row_function, self.count_experiment_queue))
                    self.run_button_added.grid(row=self.count_experiment_queue, column=5, sticky='nsew', padx=10, pady=3)

                    self.viewEdit_button_added = ttk.Button(master=self.tab_one,
                                                        text="View / Edit" ,
                                                        command= partial(self.viewEdit_function, self.count_experiment_queue))
                    self.viewEdit_button_added.grid(row=self.count_experiment_queue, column=6, sticky='nsew', padx=10, pady=3)

                    self.clear_button_added = ttk.Button(master=self.tab_one,
                                                        text="Remove" ,
                                                        command= partial(self.clearRow_function, self.count_experiment_queue))
                    self.clear_button_added.grid(row=self.count_experiment_queue, column=7, sticky='nsew', padx=10, pady=3)

                    self.postprocess_button_added = ttk.Button(master=self.tab_one,
                                                        text="Post-Process",
                                                        command= partial(self.post_rep_function, self.count_experiment_queue),
                                                        state = "disabled")
                    self.postprocess_button_added.grid(row=self.count_experiment_queue, column=8, sticky='nsew', padx=10, pady=3)

                    self.widget_row = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.viewEdit_button_added, self.clear_button_added, self.postprocess_button_added, self.exp_num]
                    self.check_box_list.append(self.checkbox_select)
                    self.check_box_list_var.append(self.checkbox_select_var)

                    self.widget_list.insert(place,self.widget_row)

                    # separator = ttk.Separator(master=self.tab_one, orient='horizontal')

                    # separator.place(x=0.1, y=self.prev, relwidth=1)
                    # self.prev += 32

                    self.count_experiment_queue += 1


                else:
                    tk.messagebox.showerror(title="Error Window", message=compatibility_result)
                    self.selected.clear()

            else:
                # reset macro_entry to "10"
                self.macro_entry.delete(0, len(self.macro_entry.get()))
                # resets macro_entry textbox
                self.macro_entry.insert(index=tk.END, string="10")

                message = "Please enter a postivie (non zero) integer for the number of Macroreplications, example: 10"
                tk.messagebox.showerror(title="Error Window", message=message)

            #s selected (list) in console/terminal
            #("it works", self.experiment_master_list)
            self.notebook.select(self.tab_one)
            return self.experiment_master_list

        # problem selected, but solver NOT selected
        elif self.problem_var.get() in problem_unabbreviated_directory and self.solver_var.get() not in solver_unabbreviated_directory:
            message = "You have not selected a Solver!"
            tk.messagebox.showerror(title="Error Window", message=message)

        # problem NOT selected, but solver selected
        elif self.problem_var.get() not in problem_unabbreviated_directory and self.solver_var.get() in solver_unabbreviated_directory:
            message = "You have not selected a Problem!"
            tk.messagebox.showerror(title="Error Window", message=message)

        # macro_entry not numeric or negative
        elif self.macro_entry.get().isnumeric() == False:
            # reset macro_entry to "10"
            self.macro_entry.delete(0, len(self.macro_entry.get()))
            # resets macro_entry textbox
            self.macro_entry.insert(index=tk.END, string="10")

            message = "Please enter a positive (non zero) integer for the number of Macroreplications, example: 10"
            tk.messagebox.showerror(title="Error Window", message=message)

        # neither problem nor solver selected
        else:
            # reset problem_var
            self.problem_var.set("Problem")
            # reset solver_var
            self.solver_var.set("Solver")

            # reset macro_entry to "10"
            self.macro_entry.delete(0, len(self.macro_entry.get()))
            # resets macro_entry textbox
            self.macro_entry.insert(index=tk.END, string="10")

            message = "You have not selected all required fields, check for '*' near input boxes."
            tk.messagebox.showerror(title="Error Window", message=message)

    def confirm_problem_factors(self):
        self.problem_factors_return = []
        self.problem_factors_dictionary = dict()

        keys = list(self.problem_object().specifications.keys())
        #("keys ->", keys)
        #("self.problem_factors_types -> ", self.problem_factors_types)

        for problem_factor in self.problem_factors_list:
           #(problem_factor.get() + " " + str(type(problem_factor.get())))
            index = self.problem_factors_list.index(problem_factor)

            #(problem_factor.get())
            if index == 0:
                if problem_factor.get()  == self.problem_var.get():
                    # self.problem_object().specifications[factor_type].get("default")
                    #self.problem_factors_return.append(None)
                    self.problem_factors_return.append(problem_factor.get())
                else:
                    self.problem_factors_return.append(problem_factor.get())
                    # self.problem_factors_dictionary["rename"] = problem_factor.get()
                    
            if index > 0:
                #(self.problem_factors_types[index])
                #datatype = self.problem_factors_types[index]

                # if the data type is tuple update data
                #self.problem_factors_dictionary[keys[index]] = datatype(nextVal)
                #(ast.literal_eval(problem_factor.get()) , keys[index])
                if keys[index-1] == 'initial_solution' and type(ast.literal_eval(problem_factor.get())) == int:
                    t = (ast.literal_eval(problem_factor.get()),)
                    #(t)
                    self.problem_factors_dictionary[keys[index-1]] = t
                else:
                    self.problem_factors_dictionary[keys[index-1]] = ast.literal_eval(problem_factor.get())
                #("datatype of factor -> ", type(datatype(problem_factor.get())))
            

        self.problem_factors_return.insert(0, self.problem_factors_dictionary)
        return self.problem_factors_return

    def confirm_oracle_factors(self):
        self.oracle_factors_return = []
        self.oracle_factors_dictionary = dict()

        keys = list(self.oracle_object().specifications.keys())
        #("keys ->", keys)
        #("self.oracle_factors_types -> ", self.oracle_factors_types)

        keys = list(self.oracle_object().specifications.keys())

        for oracle_factor in self.oracle_factors_list:
            index = self.oracle_factors_list.index(oracle_factor)
            self.oracle_factors_dictionary[keys[index]] = oracle_factor.get()
            #(self.oracle_factors_types[index])

            datatype = self.oracle_factors_types[index]
            if (str(datatype) == "<class 'list'>"):
                newList = ast.literal_eval(oracle_factor.get())

                self.oracle_factors_dictionary[keys[index]] = newList
            else:
                self.oracle_factors_dictionary[keys[index]] = datatype(oracle_factor.get())
            #(str(datatype(oracle_factor.get())) + " " + str(datatype))
            #("datatype of factor -> ", type(datatype(oracle_factor.get())))

        self.oracle_factors_return.append(self.oracle_factors_dictionary)
        return self.oracle_factors_return

    def confirm_solver_factors(self):
        self.solver_factors_return = []
        self.solver_factors_dictionary = dict()

        keys = list(self.solver_object().specifications.keys())
        #("keys ->", keys)
        #("self.solver_factors_types -> ", self.solver_factors_types)

        for solver_factor in self.solver_factors_list:
            index = self.solver_factors_list.index(solver_factor)
            #(solver_factor.get())
            if index == 0:
                if solver_factor.get() == self.solver_var.get():
                    #self.solver_factors_return.append(None)
                    self.solver_factors_return.append(solver_factor.get())
                else:
                    self.solver_factors_return.append(solver_factor.get())
                    # self.solver_factors_dictionary["rename"] = solver_factor.get()
            if index > 0:
                #(self.solver_factors_types[index])
                datatype = self.solver_factors_types[index]
                self.solver_factors_dictionary[keys[index-1]] = datatype(solver_factor.get())
                #("datatype of factor -> ", type(datatype(solver_factor.get())))
            

        self.solver_factors_return.insert(0, self.solver_factors_dictionary)
        return self.solver_factors_return

    def onFrameConfigure_queue(self, event):
        self.queue_canvas.configure(scrollregion=self.queue_canvas.bbox("all"))

    def onFrameConfigure_factor_problem(self, event):
        self.factor_canvas_problem.configure(scrollregion=self.factor_canvas_problem.bbox("all"))

    def onFrameConfigure_factor_solver(self, event):
        self.factor_canvas_solver.configure(scrollregion=self.factor_canvas_solver.bbox("all"))

    def onFrameConfigure_factor_oracle(self, event):
        self.factor_canvas_oracle.configure(scrollregion=self.factor_canvas_oracle.bbox("all"))

    def save_edit_function(self, integer):

        row_index = integer
        self.experiment_master_list[row_index-1]
        self.experiment_master_list[row_index-1][5][0]['crn_across_solns'] = self.boolean_var.get()


        if self.add_experiment(row_index):
            self.clearRow_function(row_index + 1)

             # resets problem_var to default value
            self.problem_var.set("Problem")
            # resets solver_var to default value
            self.solver_var.set("Solver")

            self.factor_label_frame_problem.destroy()
            self.factor_label_frame_oracle.destroy()
            self.factor_label_frame_solver.destroy()

    def select_pickle_file_fuction(self, *args):
        filename = filedialog.askopenfilename(parent = self.master,
                                            initialdir = "./",
                                            title = "Select Pickle File",
                                            # filetypes = (("Pickle files", "*.pickle;*.pck;*.pcl;*.pkl;*.db")
                                            #              ,("Python files", "*.py"),("All files", "*.*") )
                                                           )
        if filename != "":
            # filename_short_list = filename.split("/")
            # filename_short = filename_short_list[len(filename_short_list)-1]
            self.pickle_file_pathname_show["text"] = filename
            self.pickle_file_pathname_show["foreground"] = "blue"
            # self.pickle_file_pathname_show.place(x=950, y=400)
        # else:
        #     message = "You attempted to select a file but failed, please try again if necessary"
        #     tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)

    def load_pickle_file_function(self):
        self.select_pickle_file_fuction()

        filename = self.pickle_file_pathname_show["text"]
        acceptable_types = ["pickle", "pck", "pcl", "pkl", "db"]

        if filename != "No file selected":
            filetype = filename.split(".")
            filetype = filetype[len(filetype)-1]
            if filetype in acceptable_types:
                experiment_pathname = filename[filename.index("experiments/outputs/"):]

                pickle_file = experiment_pathname
                infile = open(pickle_file,'rb')
                new_dict = pickle.load(infile)
                infile.close()

                self.my_experiment = new_dict
                compatibility_result = self.my_experiment.check_compatibility()
                place = len(self.experiment_object_list)
                self.my_experiment.post_norm_ready = True

                if compatibility_result == "":
                    self.experiment_object_list.insert(place,self.my_experiment)

                    # filler in master list so that placement stays correct
                    self.experiment_master_list.insert(place,None)

                    self.rows = 5

                    self.checkbox_select_var = tk.BooleanVar(self.tab_one, value = False)
                    self.checkbox_select = tk.Checkbutton(master=self.tab_one,text="", state = "normal", variable =self.checkbox_select_var )
                    self.checkbox_select.deselect()
                    self.checkbox_select.grid(row=self.count_experiment_queue, column=0, sticky='nsew', padx=10, pady=3)

                    self.exp_num = tk.Label(master=self.tab_one,
                                                    text=str(self.count_experiment_queue),
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.exp_num.grid(row=self.count_experiment_queue, column=1, sticky='nsew', padx=10, pady=3)

                    self.problem_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.problem.name,
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.problem_added.grid(row=self.count_experiment_queue, column=2, sticky='nsew', padx=10, pady=3)

                    self.solver_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.solver.name,
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.solver_added.grid(row=self.count_experiment_queue, column=3, sticky='nsew', padx=10, pady=3)

                    self.macros_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.n_macroreps,
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.macros_added.grid(row=self.count_experiment_queue, column=4, sticky='nsew', padx=10, pady=3)

                    self.run_button_added = ttk.Button(master=self.tab_one,
                                                        text="Run",
                                                        command= partial(self.run_row_function, self.count_experiment_queue))
                    self.run_button_added.grid(row=self.count_experiment_queue, column=5, sticky='nsew', padx=10, pady=3)

                    self.viewEdit_button_added = ttk.Button(master=self.tab_one,
                                                        text="View / Edit" ,
                                                        command= partial(self.viewEdit_function, self.count_experiment_queue))
                    self.viewEdit_button_added.grid(row=self.count_experiment_queue, column=6, sticky='nsew', padx=10, pady=3)

                    self.clear_button_added = ttk.Button(master=self.tab_one,
                                                        text="Remove  " ,
                                                        command= partial(self.clearRow_function, self.count_experiment_queue))
                    self.clear_button_added.grid(row=self.count_experiment_queue, column=7, sticky='nsew', padx=10, pady=3)

                    self.postprocess_button_added = ttk.Button(master=self.tab_one,
                                                        text="Post-Process",
                                                        command= partial(self.post_rep_function, self.count_experiment_queue),
                                                        state = "disabled")
                    self.postprocess_button_added.grid(row=self.count_experiment_queue, column=8, sticky='nsew', padx=10, pady=3)


                    self.widget_row = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.viewEdit_button_added, self.clear_button_added, self.postprocess_button_added, self.exp_num]
                    self.widget_list.insert(place,self.widget_row)
                    self.check_box_list.append(self.checkbox_select)
                    self.check_box_list_var.append(self.checkbox_select_var)
    
                    row_of_widgets = self.widget_list[len(self.widget_list) - 1]
                    if self.my_experiment.check_run() == True:
                        run_button = row_of_widgets[3]
                        run_button["state"] = "disabled"
                        run_button["text"] = "Run Complete"
                        run_button = row_of_widgets[4]
                        run_button["state"] = "disabled"
                        run_button = row_of_widgets[6]
                        run_button["state"] = "normal"
                        self.my_experiment.post_norm_ready = False
                        if self.my_experiment.check_postreplicate():
                            self.experiment_object_list[place].post_norm_ready = True
                            self.widget_list[place][6]["text"] = "Post-Processing Complete"
                            self.widget_list[place][6]["state"] = "disabled"

                        # separator = ttk.Separator(master=self.tab_one, orient='horizontal')

                        # separator.place(x=0.1, y=self.prev, relwidth=1)
                        # self.prev += 32

                    self.count_experiment_queue += 1
                    if self.notebook.index('current') == 2:
                        self.post_norm_setup()

            else:
                message = f"You have loaded a file, but {filetype} files are not acceptable!\nPlease try again."
                tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)
        # else:
        #     message = "You are attempting to load a file, but haven't selected one yet.\nPlease select a file first."
        #     tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)

    def run_row_function(self, integer):
        # stringtuple[1:-1].split(separator=",")
        row_index = integer - 1

        # run_button = row_of_widgets[3]
        self.widget_list[row_index][3]["state"] = "disabled"
        self.widget_list[row_index][3]["text"] = "Run Complete"
        self.widget_list[row_index][4]["state"] = "disabled"
        self.widget_list[row_index][6]["state"] = "normal"
        # run_button["state"] = "disabled"
        # run_button = row_of_widgets[4]
        # run_button["state"] = "disabled"
        # row_of_widgets[6]["state"] = "normal"
        #run_button.grid(row=integer, column=3, sticky='nsew', padx=10, pady=3)

        # widget_row = [row_of_widgets[0], row_of_widgets[1], row_of_widgets[2], row_of_widgets[3], run_button, row_of_widgets[4], row_of_widgets[5], row_of_widgets[6],row_of_widgets[7] ]
        # self.widget_list[row_index] = widget_row

        self.my_experiment = self.experiment_object_list[row_index]

        self.selected = self.experiment_master_list[row_index]
        self.macro_reps = self.selected[2]
        self.my_experiment.run(n_macroreps=self.macro_reps)
        

    def post_rep_function(self, integer):
        row_index = integer - 1
        self.my_experiment = self.experiment_object_list[row_index]
        self.selected = self.experiment_object_list[row_index]
        self.post_rep_function_row_index = integer
        # calls postprocessing window


        self.postrep_window = tk.Tk()
        self.postrep_window.geometry("600x400")
        self.postrep_window.title("Post-Processing Page")
        self.app = Post_Processing_Window(self.postrep_window, self.my_experiment, self.selected, self)

    def post_process_disable_button(self, meta=False):
        if meta:
            row_index = self.post_rep_function_row_index - 1
            self.widget_meta_list[row_index][5]["text"] = "Post-Processed & Post-Normalized"
            self.widget_meta_list[row_index][5]["state"] = "disabled"
            self.widget_meta_list[row_index][6]["state"] = "normal"
            # self.normalize_button_added["state"] = "normal"
        else:
            row_index = self.post_rep_function_row_index - 1
            self.experiment_object_list[row_index].post_norm_ready = True
            self.widget_list[row_index][6]["text"] = "Post-Processing Complete"
            self.widget_list[row_index][6]["state"] = "disabled"
            # self.widget_list[row_index][7]["state"] = "normal"
    
    def checkbox_function2(self, exp, rowNum):
        newlist = sorted(self.experiment_object_list, key=lambda x: x.problem.name)
        prob_name = newlist[rowNum].problem.name
        if rowNum in self.normalize_list2:
            self.normalize_list2.remove(rowNum)
            self.post_norm_exp_list.remove(exp)

            if len(self.normalize_list2) == 0:
                for i in self.widget_norm_list:
                    i[2]["state"] = "normal"
        else:
            self.normalize_list2.append(rowNum)
            self.post_norm_exp_list.append(exp)
            for i in self.widget_norm_list:
                if i[0]["text"] != prob_name:
                    i[2]["state"] = "disable"

    def crossdesign_function(self):
        # self.crossdesign_window = tk.Tk()
        self.crossdesign_window = tk.Toplevel(self.master)
        self.crossdesign_window.geometry("650x850")
        self.crossdesign_window.title("Cross-Design Problem-Solver Group")
        self.cross_app = Cross_Design_Window(self.crossdesign_window, self)
        
    # My code starts here
    # Open data farming window
    def datafarming_function(self):
        self.datafarming_window = tk.Toplevel(self.master)
        self.datafarming_window.geometry("650x850")
        self.datafarming_window.title("Data Farming")
        self.datafarming_app = Data_Farming_Window(self.datafarming_window, self)
        
    # My code ends here

    def add_meta_exp_to_frame(self, n_macroreps=None, input_meta_experiment=None):
        if n_macroreps == None and input_meta_experiment != None:
            self.cross_app = Cross_Design_Window(master = None, main_widow = None, forced_creation = True)
            self.cross_app.crossdesign_MetaExperiment = input_meta_experiment
            self.meta_experiment_macro_reps.append("mixed")
            text_macros_added = "mixed"
        elif n_macroreps != None and input_meta_experiment == None:
            self.meta_experiment_macro_reps.append(int(n_macroreps.get()))
            text_macros_added = n_macroreps.get()

        row_num = self.count_meta_experiment_queue + 1

        self.macros_added = tk.Label(master=self.tab_two,
                                        text= text_macros_added,
                                        font = "Calibri 12",
                                        justify="center")
        self.macros_added.grid(row=row_num, column=2, sticky='nsew', padx=10, pady=3)

        
        self.problem_added = tk.Label(master=self.tab_two,
                                                    text=self.cross_app.crossdesign_MetaExperiment.problem_names,
                                                    font = "Calibri 12",
                                                    justify="center")
        self.problem_added.grid(row=row_num, column=0, sticky='nsew', padx=10, pady=3)

        self.solver_added = tk.Label(master=self.tab_two,
                                        text=self.cross_app.crossdesign_MetaExperiment.solver_names,
                                        font = "Calibri 12",
                                        justify="center")
        self.solver_added.grid(row=row_num, column=1, sticky='nsew', padx=10, pady=3)

        

        self.run_button_added = ttk.Button(master=self.tab_two,
                                            text="Run" ,
                                            command = partial(self.run_meta_function,row_num))
        self.run_button_added.grid(row=row_num, column=3, sticky='nsew', padx=10, pady=3)

        self.clear_button_added = ttk.Button(master=self.tab_two,
                                            text="Remove",
                                            command= partial(self.clear_meta_function,row_num))
        self.clear_button_added.grid(row=row_num, column=4, sticky='nsew', padx=10, pady=3)

        self.postprocess_button_added = ttk.Button(master=self.tab_two,
                                            text="Post-Process and Post-Normalize",
                                            command = partial(self.post_rep_meta_function,row_num),
                                            state = "disabled")
        self.postprocess_button_added.grid(row=row_num, column=5, sticky='nsew', padx=10, pady=3)

        self.plot_button_added = ttk.Button(master=self.tab_two,
                                            text="Plot",
                                            command = partial(self.plot_meta_function,row_num),
                                            state = "disabled")
        self.plot_button_added.grid(row=row_num, column=6, sticky='nsew', padx=10, pady=3)

        self.view_button_added = ttk.Button(master=self.tab_two,
                                            text="View Problem-Solver Group",
                                            command = partial(self.view_meta_function,row_num))
        self.view_button_added.grid(row=row_num, column=7, sticky='nsew', padx=10, pady=3)

        

        # self.select_checkbox = tk.Checkbutton(self.tab_one,text="",state="disabled",command=partial(self.checkbox_function, self.count_experiment_queue - 1))
        # self.select_checkbox.grid(row=self.count_experiment_queue, column=7, sticky='nsew', padx=10, pady=3)

        self.widget_row_meta = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.clear_button_added, self.postprocess_button_added, self.plot_button_added, self.view_button_added]
        self.widget_meta_list.insert(row_num-1,self.widget_row_meta)
        self.meta_experiment_master_list.insert(row_num-1,self.cross_app.crossdesign_MetaExperiment)
        # self.select_checkbox.deselect()

        self.count_meta_experiment_queue += 1
        self.notebook.select(self.tab_two)

    def plot_meta_function(self, integer):
        row_index = integer - 1
        self.my_experiment = self.meta_experiment_master_list[row_index]
        #(self.my_experiment.experiments)
        exps = []
        for ex in self.my_experiment.experiments:
            for e in ex:
                exps.append(e)

        self.postrep_window = tk.Toplevel()
        self.postrep_window.geometry("1000x800")
        self.postrep_window.title("Plotting Page")
        Plot_Window(self.postrep_window,self, experiment_list = exps, metaList = self.my_experiment)

    def run_meta_function(self, integer):      
        row_index = integer - 1
        self.widget_meta_list[row_index][5]["state"] = "normal"
        self.widget_meta_list[row_index][3]["state"] = "disabled"


        self.my_experiment = self.meta_experiment_master_list[row_index]
        #self.macro_reps = self.selected[2]
        self.macro_reps =  self.meta_experiment_macro_reps[row_index]

        #(self.my_experiment.n_solvers)
        #(self.my_experiment.n_problems)
        #(self.macro_reps)

        
        if self.macro_reps == "mixed":
            ask_for_macro_rep = simpledialog.askinteger("Macroreplication", "To make a Problem-Solver Group a common macroreplication is needed:")
            self.my_experiment.run(n_macroreps=ask_for_macro_rep)
        else:
            self.my_experiment.run(n_macroreps=int(self.macro_reps))

    def post_rep_meta_function(self, integer):
        row_index = integer - 1
        self.selected = self.meta_experiment_master_list[row_index]
        #(self.selected)
        self.post_rep_function_row_index = integer
        # calls postprocessing window
        self.postrep_window = tk.Tk()
        self.postrep_window.geometry("500x450")
        self.postrep_window.title("Post-Processing and Post-Normalization Page")
        self.app = Post_Processing_Window(self.postrep_window, self.selected, self.selected, self, True)

    def progress_bar_test(self):
        root = tk.Tk()
        progress = ttk.Progressbar(root, orient = 'horizontal', length = 100, mode = 'determinate')
        progress['value'] = 20
        root.update_idletasks()
        time.sleep(1)

        progress['value'] = 40
        root.update_idletasks()
        time.sleep(1)

        progress['value'] = 50
        root.update_idletasks()
        time.sleep(1)

        progress['value'] = 60
        root.update_idletasks()
        time.sleep(1)

        progress['value'] = 80
        root.update_idletasks()
        time.sleep(1)
        progress['value'] = 100

        progress.pack(pady = 10)

    def post_norm_setup(self):

        newlist = sorted(self.experiment_object_list, key=lambda x: x.problem.name)
        for widget in self.tab_three.winfo_children():
            widget.destroy()

        self.heading_list = ["Problem", "Solvers", "Selected", "", "", "", "",""]
        for heading in self.heading_list:
            self.tab_three.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_three, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)

        self.widget_norm_list = []
        self.normalize_list2 = []
        self.post_norm_exp_list = []

        for i,exp in enumerate(newlist):
            if exp.post_norm_ready:
                row_num = i + 1
                self.problem_added = tk.Label(master=self.tab_three,
                                                            text=exp.problem.name,
                                                            font = "Calibri 12",
                                                            justify="center")
                self.problem_added.grid(row=row_num, column=0, sticky='nsew', padx=10, pady=3)

                self.solver_added = tk.Label(master=self.tab_three,
                                                text=exp.solver.name,
                                                font = "Calibri 12",
                                                justify="center")
                self.solver_added.grid(row=row_num, column=1, sticky='nsew', padx=10, pady=3)

                self.select_checkbox = tk.Checkbutton(self.tab_three,text="",command=partial(self.checkbox_function2, exp, row_num-1))
                self.select_checkbox.grid(row=row_num, column=2, sticky='nsew', padx=10, pady=3)
                self.select_checkbox.deselect()

                self.widget_norm_list.append([self.problem_added, self.solver_added, self.select_checkbox])

    def post_normal_all_function(self):
        self.postrep_window = tk.Toplevel()
        self.postrep_window.geometry("610x350")
        self.postrep_window.title("Post-Normalization Page")
        self.app = Post_Normal_Window(self.postrep_window, self.post_norm_exp_list, self)
        # post_normalize(self.post_norm_exp_list, n_postreps_init_opt, crn_across_init_opt=True, proxy_init_val=None, proxy_opt_val=None, proxy_opt_x=None)

    def post_norm_return_func(self):
        #('IN post_process_disable_button ', self.post_rep_function_row_index)
        # print("youve returned")
        pass
    
    def make_meta_experiment_func(self):
        self.list_checked_experiments = []
        self.list_unique_solver = []
        self.list_unique_problems = []
        self.list_missing_experiments  = []
        

        message2 = "There are experiments missing, would you like to add them?"
        response = tk.messagebox.askyesno(title = "Make ProblemsSolvers Experiemnts",message = message2)

        if response == True:
            for index, checkbox in enumerate(self.check_box_list_var):
                if checkbox.get() == True:
                    index = self.check_box_list_var.index(checkbox)
                    experiment_checked = self.experiment_object_list[index] ## Is this right?
                    self.list_checked_experiments.append(experiment_checked)
                    # print("checkbox",checkbox.get())
                    # print("experiment_checked:", experiment_checked )
                    # Making the checkbox in the Queue of Porblem-Solver Groups disabled
                    check_box_object = self.check_box_list[index]
                    check_box_object["state"] = "disabled"
            self.list_unique_solver,self.list_unique_problems,self.list_missing_experiments = find_missing_experiments(self.list_checked_experiments)
            self.meta_experiment_created = make_full_metaexperiment(self.list_checked_experiments,self.list_unique_solver,self.list_unique_problems,self.list_missing_experiments)
            
            self.add_meta_exp_to_frame(n_macroreps = None, input_meta_experiment=self.meta_experiment_created)
            self.meta_experiment_problem_solver_list(self.meta_experiment_created)
            self.meta_experiment_master_list.append(self.meta_experiment_created)

    def meta_experiment_problem_solver_list(self, metaExperiment):
        self.list_meta_experiment_problems = []
        self.list_meta_experiment_solvers = []
        
        self.list_meta_experiment_problems = metaExperiment.problem_names
        # print("self.list_meta_experiment_problems", self.list_meta_experiment_problems)
        self.list_meta_experiment_solvers = metaExperiment.solver_names
        # print("self.list_meta_experiment_solvers", self.list_meta_experiment_solvers)

    def view_meta_function(self, row_num):
        self.factor_label_frame_solvers.destroy()
        self.factor_label_frame_oracle.destroy()
        self.factor_label_frame_problems.destroy()
        
        row_index = row_num -1
        self.problem_menu.destroy()
        self.problem_label.destroy()
        self.solver_menu.destroy()
        self.solver_label.destroy()

        self.problem_label2 = tk.Label(master=self.master, 
                            text = "Group Problem(s):*",
                            font = "Calibri 13")
        self.problem_var2 = tk.StringVar(master=self.master)
        
        self.problem_menu2 = ttk.OptionMenu(self.master, self.problem_var2, "Problem", *self.list_meta_experiment_problems, command=partial(self.show_problem_factors2, row_index))

        self.problem_label2.place(relx=.35, rely=.1)
        self.problem_menu2.place(relx=.45, rely=.1)
        self.solver_label2 = tk.Label(master=self.master, 
                            text = "Group Solver(s):*",
                
                            font = "Calibri 13")
        self.solver_var2 = tk.StringVar(master=self.master)
        self.solver_menu2 = ttk.OptionMenu(self.master, self.solver_var2, "Solver", *self.list_meta_experiment_solvers, command=partial(self.show_solver_factors2,row_index))

        self.solver_label2.place(relx=.01, rely=.1)
        self.solver_menu2.place(relx=.1, rely=.1 )

        view_button_added = self.widget_meta_list[row_index][7]
        view_button_added["text"] = "Exit View Problem-Solver Group"
        view_button_added["command"] = partial(self.exit_meta_view, row_num)
        view_button_added.grid(row= (row_num), column=7, sticky='nsew', padx=10, pady=3)

        self.add_button["state"] = "disabled"

        for i in range(self.count_meta_experiment_queue):
            self.clear_button_added = self.widget_meta_list[i][4]
            self.clear_button_added["state"] = "disabled"
            
            self.run_button = self.widget_meta_list[i][3]
            self.run_button["state"] = "disabled"

            if i != (row_index):
                view_button_added = self.widget_meta_list[i][7]
                view_button_added["state"] = "disabled"

        for i in range(self.count_experiment_queue-1):
            # print("VALUE OF I",i)
            self.run_button_added = self.widget_list[i][3]
            self.run_button_added["state"] = "disabled"

            self.viewEdit_button_added = self.widget_list[i][4]
            self.viewEdit_button_added["state"] = "disabled"

            self.clear_button_added = self.widget_list[i][5]
            self.clear_button_added["state"] = "disabled"

        self.pickle_file_load_button["state"] = "disabled"
        self.crossdesign_button["state"] = "disabled"
        self.macro_entry["state"] = "disabled"
        

               

    def exit_meta_view(self, row_num):
        row_index= row_num -1 
        self.add_button["state"] = "normal"
        self.problem_menu2.destroy()
        self.problem_label2.destroy()
        self.solver_menu2.destroy()
        self.solver_label2.destroy()
        self.factor_label_frame_solver.destroy()
        self.factor_label_frame_oracle.destroy()
        self.factor_label_frame_problem.destroy()
        self.problem_label = tk.Label(master=self.master, # window label is used in
                        text = "Select Problem:",
                        font = "Calibri 13")
        self.problem_var = tk.StringVar(master=self.master)
        self.problem_menu = ttk.OptionMenu(self.master, self.problem_var, "Problem", *self.problem_list, command=self.show_problem_factors)

        self.problem_label.place(relx=.3, rely=.1)
        self.problem_menu.place(relx=.4, rely=.1)
        self.solver_label = tk.Label(master=self.master, # window label is used in
                            text = "Select Solver(s):*",
                            font = "Calibri 13")
        self.solver_var = tk.StringVar(master=self.master)
        self.solver_menu = ttk.OptionMenu(self.master, self.solver_var, "Solver", *self.solver_list, command=self.show_solver_factors)
        
        self.solver_label.place(relx=.01, rely=.1)
        self.solver_menu.place(relx=.1, rely=.1 )

        view_button_added = self.widget_meta_list[row_index][7]
        view_button_added["text"] = "View Problem-Solver Group"
        view_button_added["command"] = partial(self.view_meta_function, row_num)
        view_button_added.grid(row= (row_num), column=7, sticky='nsew', padx=10, pady=3)

        for i in range(self.count_meta_experiment_queue):
            self.clear_button_added = self.widget_meta_list[i][4]
            self.clear_button_added["state"] = "normal"

            self.run_button = self.widget_meta_list[i][3]
            self.run_button["state"] = "normal"

            if i != (row_index):
                view_button_added = self.widget_meta_list[i][7]
                view_button_added["state"] = "normal"

        for i in range(self.count_experiment_queue -1):
            self.run_button_added = self.widget_list[i][3]
            self.run_button_added["state"] = "normal"

            self.viewEdit_button_added = self.widget_list[i][4]
            self.viewEdit_button_added["state"] = "normal"

            self.clear_button_added = self.widget_list[i][5]
            self.clear_button_added["state"] = "normal"

        self.pickle_file_load_button["state"] = "normal"
        self.crossdesign_button["state"] = "normal"
        self.macro_entry["state"] = "normal"
        

    def show_solver_factors2(self, row_index, *args):
        self.factor_label_frame_solver.destroy()
        
        self.solver_factors_list = []
        self.solver_factors_types = []

        self.factor_label_frame_solver = ttk.LabelFrame(master=self.master, text="Solver Factors")

        self.factor_canvas_solver = tk.Canvas(master=self.factor_label_frame_solver, borderwidth=0)

        self.factor_frame_solver = ttk.Frame(master=self.factor_canvas_solver)
        self.vert_scroll_bar_factor_solver = Scrollbar(self.factor_label_frame_solver, orient="vertical", command=self.factor_canvas_solver.yview)
        self.horiz_scroll_bar_factor_solver = Scrollbar(self.factor_label_frame_solver, orient="horizontal", command=self.factor_canvas_solver.xview)
        self.factor_canvas_solver.configure(xscrollcommand=self.horiz_scroll_bar_factor_solver.set, yscrollcommand=self.vert_scroll_bar_factor_solver.set)

        self.vert_scroll_bar_factor_solver.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_solver.pack(side="bottom", fill="x")

        self.factor_canvas_solver.pack(side="left", fill="both", expand=True)
        self.factor_canvas_solver.create_window((0,0), window=self.factor_frame_solver, anchor="nw",
                                  tags="self.factor_frame_solver")

        self.factor_frame_solver.bind("<Configure>", self.onFrameConfigure_factor_solver)

        self.factor_notebook_solver = ttk.Notebook(master=self.factor_frame_solver)
        self.factor_notebook_solver.pack(fill="both")

        self.factor_tab_one_solver = tk.Frame(master=self.factor_notebook_solver)

        self.factor_notebook_solver.add(self.factor_tab_one_solver, text=str(self.solver_var2.get()) + " Factors")

        self.factor_tab_one_solver.grid_rowconfigure(0)

        self.factor_heading_list_solver = ["Description", "Input"]

        for heading in self.factor_heading_list_solver:
            self.factor_tab_one_solver.grid_columnconfigure(self.factor_heading_list_solver.index(heading))
            label = tk.Label(master=self.factor_tab_one_solver, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.factor_heading_list_solver.index(heading), padx=10, pady=3)

        metaExperiment = self.meta_experiment_master_list[row_index]
        solver_name = self.solver_var2.get()
        solver_index = metaExperiment.solver_names.index(str(solver_name))
        self.solver_object = metaExperiment.solvers[solver_index]

        metaExperiment = self.meta_experiment_master_list[row_index]
        solver_name = self.solver_var2.get()
        solver_index = metaExperiment.solver_names.index(str(solver_name))
        self.custom_solver_object = metaExperiment.solvers[solver_index]
        # explanation: https://stackoverflow.com/questions/5924879/how-to-create-a-new-instance-from-a-class-object-in-python
        default_solver_class = self.custom_solver_object.__class__
        self.default_solver_object = default_solver_class()

        count_factors_solver = 1
        

        self.save_label_solver = tk.Label(master=self.factor_tab_one_solver,
                                            text = "save solver as",
                                            font = "Calibri 13")

        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.save_var_solver, justify = tk.LEFT, width=15)
        

        self.save_entry_solver.insert(index=tk.END, string=solver_name)
        self.save_entry_solver["state"] = "disabled"
        self.save_label_solver.grid(row=count_factors_solver, column=0, sticky='nsew')
        self.save_entry_solver.grid(row=count_factors_solver, column=1, sticky='nsew')

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)
        
        count_factors_solver += 1
        
        for factor_type in self.default_solver_object.specifications:
    
            self.dictionary_size_solver = len(self.default_solver_object.specifications[factor_type])
            datatype = self.default_solver_object.specifications[factor_type].get("datatype")
            description = self.default_solver_object.specifications[factor_type].get("description")
            default = self.default_solver_object.specifications[factor_type].get("default")

            if datatype != bool:

                self.int_float_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)

                self.int_float_var = tk.StringVar(self.factor_tab_one_solver)
                self.int_float_entry = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.int_float_var, justify = tk.LEFT, width=15)
                self.int_float_entry.insert(index=tk.END, string=str(self.custom_solver_object.factors[factor_type]))
                self.int_float_entry["state"] = "disabled"
                self.int_float_description.grid(row=count_factors_solver, column=0, sticky='nsew')
                self.int_float_entry.grid(row=count_factors_solver, column=1, sticky='nsew')
                self.solver_factors_list.append(self.int_float_var)

                if datatype != tuple:
                    self.solver_factors_types.append(datatype)
                else:
                    self.solver_factors_types.append(str)

                count_factors_solver += 1

            if datatype == bool:
                self.boolean_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)

                self.boolean_var = tk.BooleanVar(self.factor_tab_one_solver, value = bool(default))
                self.boolean_menu = tk.Checkbutton(self.factor_tab_one_solver, variable=self.boolean_var, onvalue=True, offvalue=False)
                
                # self.boolean_menu.configure(state = "disabled")
                self.boolean_description.grid(row=count_factors_solver, column=0, sticky='nsew')
                self.boolean_menu.grid(row=count_factors_solver, column=1, sticky='nsew')
                self.solver_factors_list.append(self.boolean_var)
                self.solver_factors_types.append(datatype)

                count_factors_solver += 1

        
        
        self.factor_label_frame_solver.place(x=10, rely=.15, relheight=.33, relwidth=.34)
        if str(self.problem_var.get()) != "Problem":
            self.add_button.place(x=10, rely=.48, width=200, height=30)
    
    def show_problem_factors2(self,row_index, *args):
        self.factor_label_frame_problem.destroy()
        self.factor_label_frame_oracle.destroy()
        self.problem_factors_list = []
        self.problem_factors_types = []

        self.factor_label_frame_problem = ttk.LabelFrame(master=self.master, text="Problem Factors")

        self.factor_canvas_problem = tk.Canvas(master=self.factor_label_frame_problem, borderwidth=0)

        self.factor_frame_problem = ttk.Frame(master=self.factor_canvas_problem)
        self.vert_scroll_bar_factor_problem = Scrollbar(self.factor_label_frame_problem, orient="vertical", command=self.factor_canvas_problem.yview)
        self.horiz_scroll_bar_factor_problem = Scrollbar(self.factor_label_frame_problem, orient="horizontal", command=self.factor_canvas_problem.xview)
        self.factor_canvas_problem.configure(xscrollcommand=self.horiz_scroll_bar_factor_problem.set, yscrollcommand=self.vert_scroll_bar_factor_problem.set)

        self.vert_scroll_bar_factor_problem.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_problem.pack(side="bottom", fill="x")

        self.factor_canvas_problem.pack(side="left", fill="both", expand=True)
        self.factor_canvas_problem.create_window((0,0), window=self.factor_frame_problem, anchor="nw",
                                  tags="self.factor_frame_problem")

        self.factor_frame_problem.bind("<Configure>", self.onFrameConfigure_factor_problem)

        self.factor_notebook_problem = ttk.Notebook(master=self.factor_frame_problem)
        self.factor_notebook_problem.pack(fill="both")

        self.factor_tab_one_problem = tk.Frame(master=self.factor_notebook_problem)

        self.factor_notebook_problem.add(self.factor_tab_one_problem, text=str(self.problem_var2.get()) + " Factors")

        self.factor_tab_one_problem.grid_rowconfigure(0)

        self.factor_heading_list_problem = ["Description", "Input"]

        for heading in self.factor_heading_list_problem:
            self.factor_tab_one_problem.grid_columnconfigure(self.factor_heading_list_problem.index(heading))
            label_problem = tk.Label(master=self.factor_tab_one_problem, text=heading, font="Calibri 14 bold")
            label_problem.grid(row=0, column=self.factor_heading_list_problem.index(heading), padx=10, pady=3)

        metaExperiment = self.meta_experiment_master_list[row_index]
        problem_name = self.problem_var2.get()
        problem_index = metaExperiment.problem_names.index(str(problem_name))
        self.custom_problem_object = metaExperiment.problems[problem_index]
        # explanation: https://stackoverflow.com/questions/5924879/how-to-create-a-new-instance-from-a-class-object-in-python
        default_problem_class = self.custom_problem_object.__class__
        self.default_problem_object = default_problem_class()

        count_factors_problem = 1
        
        self.save_label_problem = tk.Label(master=self.factor_tab_one_problem,
                                            text = "save problem as",
                                            font = "Calibri 13")

        self.save_var_problem = tk.StringVar(self.factor_tab_one_problem)
        self.save_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.save_var_problem, justify = tk.LEFT, width = 15)
        
        self.save_entry_problem.insert(index=tk.END, string=problem_name)
        self.save_entry_problem["state"] = "disabled"
        self.save_label_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
        self.save_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

        self.problem_factors_list.append(self.save_var_problem)
        self.problem_factors_types.append(str)
        
        count_factors_problem += 1          

        for num, factor_type in enumerate(self.default_problem_object.specifications, start=0):
            self.dictionary_size_problem = len(self.default_problem_object.specifications[factor_type])
            datatype = self.default_problem_object.specifications[factor_type].get("datatype")
            description= self.default_problem_object.specifications[factor_type].get("description")
            default = self.default_problem_object.specifications[factor_type]['default']
            
            if datatype != bool:


                self.int_float_description_problem = tk.Label(master=self.factor_tab_one_problem,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)

                self.int_float_var_problem = tk.StringVar(self.factor_tab_one_problem)
                self.int_float_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.int_float_var_problem, justify = tk.LEFT, width = 15)
                if datatype == tuple and len(default) == 1:
                    self.int_float_entry_problem.insert(index=tk.END, string=str(self.custom_problem_object.factors[factor_type][0]))
                else:
                    self.int_float_entry_problem.insert(index=tk.END, string=str(self.custom_problem_object.factors[factor_type]))

                self.int_float_entry_problem["state"] = "disabled"
                self.int_float_description_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
                self.int_float_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

                self.problem_factors_list.append(self.int_float_var_problem)
                datatype = self.default_problem_object.specifications[factor_type].get("datatype")
                
                if datatype != tuple:
                    self.problem_factors_types.append(datatype)
                else:
                    self.problem_factors_types.append(str)

                count_factors_problem += 1


            if datatype == bool:

                self.boolean_description_problem = tk.Label(master=self.factor_tab_one_problem,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)
                self.boolean_var_problem = tk.BooleanVar(self.factor_tab_one_problem, value = bool(default))
                self.boolean_menu_problem = tk.Checkbutton(self.factor_tab_one_problem, variable=self.boolean_var_problem, onvalue=True, offvalue=False)
                self.boolean_description_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
                self.boolean_menu_problem.grid(row=count_factors_problem, column=1, sticky='nsew')
                
                self.problem_factors_list.append(self.boolean_var_problem)
                self.problem_factors_types.append(datatype)

                count_factors_problem += 1

        self.factor_label_frame_problem.place(relx=.35, rely=.15, relheight=.33, relwidth=.34)

        # Switching from Problems to Oracles

        self.oracle_factors_list = []
        self.oracle_factors_types = []
    
        self.factor_label_frame_oracle = ttk.LabelFrame(master=self.master, text="Model Factors")

        self.factor_canvas_oracle = tk.Canvas(master=self.factor_label_frame_oracle, borderwidth=0)

        self.factor_frame_oracle = ttk.Frame(master=self.factor_canvas_oracle)
        self.vert_scroll_bar_factor_oracle = Scrollbar(self.factor_label_frame_oracle, orient="vertical", command=self.factor_canvas_oracle.yview)
        self.horiz_scroll_bar_factor_oracle = Scrollbar(self.factor_label_frame_oracle, orient="horizontal", command=self.factor_canvas_oracle.xview)
        self.factor_canvas_oracle.configure(xscrollcommand=self.horiz_scroll_bar_factor_oracle.set, yscrollcommand=self.vert_scroll_bar_factor_oracle.set)

        self.vert_scroll_bar_factor_oracle.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_oracle.pack(side="bottom", fill="x")

        self.factor_canvas_oracle.pack(side="left", fill="both", expand=True)
        self.factor_canvas_oracle.create_window((0,0), window=self.factor_frame_oracle, anchor="nw",
                                  tags="self.factor_frame_oracle")

        self.factor_frame_oracle.bind("<Configure>", self.onFrameConfigure_factor_oracle)

        self.factor_notebook_oracle = ttk.Notebook(master=self.factor_frame_oracle)
        self.factor_notebook_oracle.pack(fill="both")

        self.factor_tab_one_oracle = tk.Frame(master=self.factor_notebook_oracle)

        self.factor_notebook_oracle.add(self.factor_tab_one_oracle, text=str(self.oracle+ " Factors"))

        self.factor_tab_one_oracle.grid_rowconfigure(0)

        self.factor_heading_list_oracle = ["Description", "Input"]

        for heading in self.factor_heading_list_oracle:
            self.factor_tab_one_oracle.grid_columnconfigure(self.factor_heading_list_oracle.index(heading))
            label_oracle = tk.Label(master=self.factor_tab_one_oracle, text=heading, font="Calibri 14 bold")
            label_oracle.grid(row=0, column=self.factor_heading_list_oracle.index(heading), padx=10, pady=3)

        self.default_oracle_object = self.default_problem_object.model
        self.custom_oracle_object = self.custom_problem_object.model

        count_factors_oracle = 1
        for factor_type in self.default_oracle_object.specifications:

            self.dictionary_size_oracle = len(self.default_oracle_object.specifications[factor_type])
            datatype = self.default_oracle_object.specifications[factor_type].get("datatype")
            description = self.default_oracle_object.specifications[factor_type].get("description")
            default = self.default_oracle_object.specifications[factor_type].get("default")

            if datatype != bool:

                #("yes?")
                self.int_float_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)
                self.int_float_var_oracle = tk.StringVar(self.factor_tab_one_oracle)
                self.int_float_entry_oracle = ttk.Entry(master=self.factor_tab_one_oracle, textvariable = self.int_float_var_oracle, justify = tk.LEFT, width = 15)
                self.int_float_entry_oracle.insert(index=tk.END, string=str(self.custom_oracle_object.factors[factor_type]))
                self.int_float_entry_oracle["state"] = "disabled"
                self.int_float_description_oracle.grid(row=count_factors_oracle, column=0, sticky='nsew')
                self.int_float_entry_oracle.grid(row=count_factors_oracle, column=1, sticky='nsew')

                self.oracle_factors_list.append(self.int_float_var_oracle)

                if datatype != tuple:
                    self.oracle_factors_types.append(datatype)
                else:
                    self.oracle_factors_types.append(str)

                count_factors_oracle += 1


            if datatype == bool:

                #("yes!")
                self.boolean_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)
                self.boolean_var_oracle = tk.BooleanVar(self.factor_tab_one_oracle, value = bool(default))
                self.boolean_menu_oracle = tk.Checkbutton(self.factor_tab_one_oracle, variable=self.boolean_var_oracle, onvalue=True, offvalue=False)
                self.boolean_description_oracle.grid(row=count_factors_oracle, column=0, sticky='nsew')
                self.boolean_menu_oracle.grid(row=count_factors_oracle, column=1, sticky='nsew')
                self.oracle_factors_list.append(self.boolean_var_oracle)
                self.oracle_factors_types.append(datatype)

                count_factors_oracle += 1


        self.factor_label_frame_oracle.place(relx=.7, rely=.15, relheight=.33, relwidth=.3)
        if str(self.solver_var.get()) != "Solver":
            self.add_button.place(x=10, rely=.48, width=200, height=30)

    def show_solver_factors(self, *args):
        
        if args and len(args) == 3 and args[2] == False:
            pass
        else:
            self.update_problem_list_compatability()

        self.solver_factors_list = []
        self.solver_factors_types = []

        self.factor_label_frame_solver = ttk.LabelFrame(master=self.master, text="Solver Factors")

        self.factor_canvas_solver = tk.Canvas(master=self.factor_label_frame_solver, borderwidth=0)

        self.factor_frame_solver = ttk.Frame(master=self.factor_canvas_solver)
        self.vert_scroll_bar_factor_solver = Scrollbar(self.factor_label_frame_solver, orient="vertical", command=self.factor_canvas_solver.yview)
        self.horiz_scroll_bar_factor_solver = Scrollbar(self.factor_label_frame_solver, orient="horizontal", command=self.factor_canvas_solver.xview)
        self.factor_canvas_solver.configure(xscrollcommand=self.horiz_scroll_bar_factor_solver.set, yscrollcommand=self.vert_scroll_bar_factor_solver.set)

        self.vert_scroll_bar_factor_solver.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_solver.pack(side="bottom", fill="x")

        self.factor_canvas_solver.pack(side="left", fill="both", expand=True)
        self.factor_canvas_solver.create_window((0,0), window=self.factor_frame_solver, anchor="nw",
                                  tags="self.factor_frame_solver")

        self.factor_frame_solver.bind("<Configure>", self.onFrameConfigure_factor_solver)

        self.factor_notebook_solver = ttk.Notebook(master=self.factor_frame_solver)
        self.factor_notebook_solver.pack(fill="both")

        self.factor_tab_one_solver = tk.Frame(master=self.factor_notebook_solver)

        self.factor_notebook_solver.add(self.factor_tab_one_solver, text=str(self.solver_var.get()) + " Factors")

        self.factor_tab_one_solver.grid_rowconfigure(0)

        self.factor_heading_list_solver = ["Description", "Input"]

        for heading in self.factor_heading_list_solver:
            self.factor_tab_one_solver.grid_columnconfigure(self.factor_heading_list_solver.index(heading))
            label = tk.Label(master=self.factor_tab_one_solver, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.factor_heading_list_solver.index(heading), padx=10, pady=3)

        self.solver_object = solver_unabbreviated_directory[self.solver_var.get()]

        count_factors_solver = 1
        
        

        self.save_label_solver = tk.Label(master=self.factor_tab_one_solver,
                                            text = "save solver as",
                                            font = "Calibri 13")

                                  
        if args and len(args) == 3 and args[0] == True:
            oldname = args[1][5][1]
            
        else:
            solver_object = solver_unabbreviated_directory[self.solver_var.get()]
            oldname = solver_object().name
            

        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.save_var_solver, justify = tk.LEFT, width=15)
        

        self.save_entry_solver.insert(index=tk.END, string=oldname)

        self.save_label_solver.grid(row=count_factors_solver, column=0, sticky='nsew')
        self.save_entry_solver.grid(row=count_factors_solver, column=1, sticky='nsew')

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)
        
        count_factors_solver += 1
        
        for factor_type in self.solver_object().specifications:
            #("size of dictionary", len(self.solver_object().specifications[factor_type]))
            #("first", factor_type)
            #("second", self.solver_object().specifications[factor_type].get("description"))
            #("third", self.solver_object().specifications[factor_type].get("datatype"))
            #("fourth", self.solver_object().specifications[factor_type].get("default"))

            self.dictionary_size_solver = len(self.solver_object().specifications[factor_type])
            datatype = self.solver_object().specifications[factor_type].get("datatype")
            description = self.solver_object().specifications[factor_type].get("description")
            default = self.solver_object().specifications[factor_type].get("default")

            if datatype != bool:

                self.int_float_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)

                self.int_float_var = tk.StringVar(self.factor_tab_one_solver)
                self.int_float_entry = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.int_float_var, justify = tk.LEFT, width=15)
                
                if args and len(args) == 3 and args[0] == True:
                    self.int_float_entry.insert(index=tk.END, string=str(args[1][5][0][factor_type]))
                else:
                    self.int_float_entry.insert(index=tk.END, string=str(default))

                self.int_float_description.grid(row=count_factors_solver, column=0, sticky='nsew')
                self.int_float_entry.grid(row=count_factors_solver, column=1, sticky='nsew')
                self.solver_factors_list.append(self.int_float_var)

                if datatype != tuple:
                    self.solver_factors_types.append(datatype)
                else:
                    self.solver_factors_types.append(str)

                count_factors_solver += 1


            if datatype == bool:

                self.boolean_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(description),
                                                    font = "Calibri 13",
                                                    wraplength=150)
                self.boolean_var = tk.BooleanVar(self.factor_tab_one_solver, value = bool(default))
                self.boolean_menu = tk.Checkbutton(self.factor_tab_one_solver, variable=self.boolean_var, onvalue=True, offvalue=False)
                self.boolean_description.grid(row=count_factors_solver, column=0, sticky='nsew')
                self.boolean_menu.grid(row=count_factors_solver, column=1, sticky='nsew')
                self.solver_factors_list.append(self.boolean_var)
                self.solver_factors_types.append(datatype)

                count_factors_solver += 1

        
        # self.factor_label_frame_problem.place(relx=.32, y=70, height=150, relwidth=.34)
        self.factor_label_frame_solver.place(x=10, rely=.15, relheight=.33, relwidth=.34)
        if str(self.problem_var.get()) != "Problem":
            self.add_button.place(x=10, rely=.48, width=200, height=30)
                 
# My code starts here
# Create data farming window class
class Data_Farming_Window():
    def __init__(self, master, main_widow, forced_creation = False):
        if not forced_creation:
            self.master = master
            self.main_window = main_widow
            self.master.grid_rowconfigure(0, weight=0)
            self.master.grid_rowconfigure(1, weight=0)
            self.master.grid_rowconfigure(2, weight=0)
            self.master.grid_rowconfigure(3, weight=0)
            self.master.grid_rowconfigure(4, weight=0)
            self.master.grid_rowconfigure(5, weight=0)
            self.master.grid_rowconfigure(6, weight=0)
            self.master.grid_rowconfigure(7, weight=0)
            self.master.grid_columnconfigure(0, weight=1)
            self.master.grid_columnconfigure(1, weight=1)
            self.master.grid_columnconfigure(2, weight=1)
            self.master.grid_columnconfigure(3, weight=1)
            self.master.grid_columnconfigure(4, weight=1)
            
            # Intitialize frames so prevous entries can be deleted
            self.design_frame = tk.Frame(master = self.master)
            self.design_frame.grid(row = 5, column = 0)
            
            self.create_design_frame = tk.Frame(master = self.master)
            self.run_frame = tk.Frame(master = self.master)
            self.factor_canvas = tk.Canvas (master = self.master)
            self.factors_frame = tk.Frame( master = self.factor_canvas)
            
            
            # Initial variable values
            self.factor_que_length = 1
            self.default_values_list = []
            self.checkstate_list=[]
            self.min_list = []
            self.max_list = []
            self.dec_list = []
            
            
            
            # Create main window title
            self.title_frame = tk.Frame(master=self.master)
            self.title_frame.grid_rowconfigure(0, weight=1)
            self.title_frame.grid_columnconfigure(0, weight=1)
            self.title_frame.grid( row=0, column = 0, sticky = tk.N)
            self.datafarming_title_label = tk.Label(master=self.title_frame,
                                                    text = "Model Data Farming",
                                                    font = "Calibri 15 bold")
            self.datafarming_title_label.grid( row = 0, column = 0) 
            
            # Create model selection drop down menu
            self.model_list = model_unabbreviated_directory
            self.modelselect_frame = tk.Frame(master=self.master)
            self.modelselect_frame.grid_rowconfigure(0, weight=1)
            self.modelselect_frame.grid_rowconfigure(1, weight=1)
            self.modelselect_frame.grid_columnconfigure(0, weight=1)
            self.modelselect_frame.grid_columnconfigure(1, weight=1)
            self.modelselect_frame.grid_columnconfigure(2, weight=1)
            self.modelselect_frame.grid_columnconfigure(3, weight=1)
            self.modelselect_frame.grid_columnconfigure(4, weight=1)
            
            self.modelselect_frame.grid( row =2, column = 0, sticky = tk.W )
            self.model_label = tk.Label(master=self.modelselect_frame, # window label is used in
                            text = "Select Model:",
                            font = "Calibri 13",
                            width = 20)
            self.model_label.grid( row = 0, column = 0, sticky = tk.W)       
            self.model_var = tk.StringVar()
            self.model_menu = ttk.OptionMenu(self.modelselect_frame, self.model_var, "Model", 
                                             *self.model_list, command= self.show_model_factors)
            self.model_menu.grid( row = 0, column = 1, sticky = tk.W)
            
            # Create load design button
            
            self.or_label = tk.Label(master = self.modelselect_frame,
                                     text = "OR",
                                     font = "Calibri 13",
                                     width = 20)
            self.or_label.grid( row = 0, column = 2, sticky = tk.W)
            
            self.load_design_button = tk.Button( master = self.modelselect_frame, text = 'Load Design CSV',
                                                width = 20, command = self.load_design)
            self.load_design_button.grid( row = 0, column = 3, sticky = tk.W)
            
            
    
            
            
    def load_design(self):
        
    
        #Clear previous selections
        for widget in self.factors_frame.winfo_children():
            widget.destroy()
            
        # Delete previous design tree
        for widget in self.create_design_frame.winfo_children():
            widget.destroy()
        
        for widget in self.run_frame.winfo_children():
            widget.destroy()
            
        for widget in self.design_frame.winfo_children():
            widget.destroy()
            
        
        # Initialize frame canvas
        self.factor_canvas = tk.Canvas (master = self.master)
        self.factor_canvas.grid_rowconfigure(0, weight = 1)
        self.factor_canvas.grid_columnconfigure(0, weight = 1)
        self.factor_canvas.grid( row = 4, column = 0, sticky = 'nsew')
        
        self.factors_title_frame = tk.Frame(master = self.master)
        self.factors_title_frame.grid( row = 3, column = 0, sticky = tk.N + tk.W)
        self.factors_title_frame.grid_rowconfigure(0, weight = 0)
        self.factors_title_frame.grid_columnconfigure(0, weight =1)
        self.factors_title_frame.grid_columnconfigure(1, weight =1)
        self.factors_title_frame.grid_columnconfigure(2, weight =1)
        
        self.factors_frame = tk.Frame( master = self.factor_canvas)
        self.factors_frame.grid( row = 0, column = 0, sticky = tk.W + tk.N)
        self.factors_frame.grid_rowconfigure(0, weight =1)
        self.factors_frame.grid_columnconfigure(0, weight =1)
        self.factors_frame.grid_columnconfigure(1, weight =1)
        self.factors_frame.grid_columnconfigure(2, weight =1)
        self.factors_frame.grid_columnconfigure(3, weight =1)
        
        
        # Create column for model factor names
        self.headername_label = tk.Label(master = self.factors_frame, text = 'Default Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W, padx = 10)
        
        # Create column for factor type
        self.headertype_label = tk.Label(master = self.factors_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 20, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 1, sticky = tk.N + tk.W)
        
        #Values to help with formatting
        entry_width = 20
        
        # List to hold default values
        self.default_values_list = []
        self.fixed_str = {}
        
        # Create column for factor default values
        self.headerdefault_label = tk.Label(master = self.factors_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 2, sticky = tk.N + tk.W)
            
        # Name of design csv file
        self.design_csv_name = filedialog.askopenfilename()
        
        #Specify what model is being used
        split_name = self.design_csv_name.split("experiments/")
        split_name = split_name[1].split('_')
        self.selected_model = split_name[0]
     
       
        self.model_object = model_unabbreviated_directory[self.selected_model]() #Eventually allow selection here
        
        
        
        
        self.model_var.set(self.selected_model)
        
        
        #Determine factors not included in design
 
        with open( self.design_csv_name, 'r') as design_file:
            reader = csv.reader(design_file)
            self.all_factor_headers = next(reader)[1:]
         
        self.default_factor_list = [] 
        for model_factor in self.model_object.specifications:
            if model_factor not in self.all_factor_headers:
                self.default_factor_list.append(model_factor)
       
        # Number of all factors in model
        num_all_factors = len(self.all_factor_headers)
        # Number of vaired factors in experiment        
        num_factors = num_all_factors - len(self.default_factor_list)
        # Factor headers dictionary to be used in run function
        self.factor_headers = self.all_factor_headers[: num_factors]
        print('factor headers', self.factor_headers)
        #print( 'num factors', num_factors)
        
        # Determine values of default factors
        default_list = []
        with open( self.design_csv_name, 'r') as design_file:
            reader = csv.reader(design_file)
            # skip header row
            next(reader)
            # Read only default factor values
            default_list = next(reader)[num_factors + 1:]
            #print('default_list', default_list)
        

        # Allow user to change default values
        factor_idx = 0
        for  factor in self.default_factor_list:
            
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
            self.factor_description = self.model_object.specifications[factor].get("description")
            self.factor_default = default_list[factor_idx]
            
            
            self.factors_frame.grid_rowconfigure(self.factor_que_length, weight =1)
            
            if self.factor_datatype == int:
                self.str_type = 'int'
            elif self.factor_datatype == float:
                self.str_type = 'float'
            elif self.factor_datatype == list:
                self.str_type = 'list'
            elif self.factor_datatype == tuple:
                self.str_type = 'tuple'
          
           
            # Add label for factor names
            self.factorname_label = tk.Label (master = self.factors_frame, text = f"{factor} - {self.factor_description}", font = "Calibri 13", width = 40, anchor = 'w')
            self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W, padx = 10)
            
            # Add label for factor type
            self.factortype_label = tk.Label (master = self.factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
            self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
            
            #Add entry box for default value
            default_len = len(str(self.factor_default))
            if default_len > entry_width:
                entry_width = default_len
                if default_len > 150:
                    entry_width = 150
            self.default_value= tk.StringVar()
            self.default_entry = tk.Entry( master = self.factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
            self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
            #Display original default value
            self.default_entry.insert(0, str(self.factor_default))
            self.default_values_list.append(self.default_value)
            
            
            self.factor_que_length += 1
            factor_idx += 1
            
        #Create change defaults button
        self.change_def_frame = tk.Frame( master = self.master)
        self.change_def_frame.grid(row = 5, column = 0)
        self.change_def_button = tk.Button(master = self.change_def_frame, text = 'Change Experiment Defaults', font = "Calibri 13",
                                                width = 30, command = self.update_defaults_button)
        self.change_def_button.grid( row = 0, column = 0)
            
           
        
        # Run to store default values
        self.update_defaults()
        
     
        
        # Create design text file to be used in experiment "model_factors_design"
        with open( "./data_farming_experiments/model_factors_design.txt", 'w', encoding="utf-8") as design_file:
            design_file.write("")
        with open(self.design_csv_name, 'r') as design_csv:
            reader = csv.reader(design_csv)
            next(reader)
            for row in design_csv:
                data_insert = ""
                # values for all factors
                factor_list = row.split(',')
                # values only for vaired factors in experiment
                design_list = factor_list[1: num_factors + 1]
                
                
                for factor in design_list:
                    data_insert += str(factor) + "\t"
                                       
                data_insert = data_insert[:-1]
                with open( "./data_farming_experiments/model_factors_design.txt", 'a', encoding="utf-8" ) as design_file:
                    design_file.write(data_insert + "\n")
                    
                    
                
            
            
                  
    def update_defaults(self):
        
        
        
        # Get default user values
        self.fixed_factors = {}
        default_csv_insert = []
        
        # List of values entered by user
        self.default_values = [self.default_value.get() for self.default_value in self.default_values_list]
        factor_index = 0
        for factor in self.default_factor_list:
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
            current_def_val = self.default_values[factor_index]
            if self.factor_datatype == float:
                self.fixed_factors[factor] = float(current_def_val)
            
            elif self.factor_datatype == int:
                self.fixed_factors[factor] = int(current_def_val)
            
            if self.factor_datatype == list:
                self.fixed_factors[factor] = ast.literal_eval(current_def_val)
        
            elif self.factor_datatype == tuple:
           
                tuple_str = tuple(current_def_val[1:-1].split(","))
                self.fixed_factors[factor] = tuple(float(s) for s in tuple_str)
            
            default_csv_insert.append(self.fixed_factors[factor])    
            factor_index += 1
    
        
        self.display_design_tree()
        

    # Display Model Factors
    def show_model_factors(self,*args ):
        
        
    
        self.factor_canvas.destroy()
        
        # Initialize frame canvas
        self.factor_canvas = tk.Canvas (master = self.master)
        self.factor_canvas.grid_rowconfigure(0, weight = 1)
        self.factor_canvas.grid_columnconfigure(0, weight = 1)
        self.factor_canvas.grid( row = 4, column = 0, sticky = 'nsew')
        self.factors_frame = tk.Frame( master = self.factor_canvas)
        self.factor_canvas.create_window((0, 0), window=self.factors_frame, anchor="nw")
        
        self.factors_frame.grid_rowconfigure(self.factor_que_length + 1, weight =1)
        

        
        self.factors_title_frame = tk.Frame(master = self.master)
        self.factors_title_frame.grid( row = 3, column = 0, sticky = 'nsew')
        self.factors_title_frame.grid_rowconfigure(0, weight = 0)
        self.factors_title_frame.grid_columnconfigure(0, weight =0)
        self.factors_title_frame.grid_columnconfigure(1, weight =0)
        self.factors_title_frame.grid_columnconfigure(2, weight =0)
        self.factors_title_frame.grid_columnconfigure(3, weight =0)
        self.factors_title_frame.grid_columnconfigure(4, weight =0)
        self.factors_title_frame.grid_columnconfigure(5, weight =0)
        self.factors_title_frame.grid_columnconfigure(6, weight =0)
        self.factors_title_frame.grid_columnconfigure(7, weight =0)
        
        #self.factors_frame = tk.Frame( master = self.factor_canvas)
        self.factors_frame.grid( row = 0, column = 0, sticky = 'nsew')
        self.factors_frame.grid_rowconfigure(0, weight =0)
        self.factors_frame.grid_columnconfigure(0, weight =0)
        self.factors_frame.grid_columnconfigure(1, weight =0)
        self.factors_frame.grid_columnconfigure(2, weight =0)
        self.factors_frame.grid_columnconfigure(3, weight =0)
        self.factors_frame.grid_columnconfigure(4, weight =0)
        self.factors_frame.grid_columnconfigure(5, weight =0)
        self.factors_frame.grid_columnconfigure(6, weight =0)
        self.factors_frame.grid_columnconfigure(7, weight =0)
        
      
  
        #Clear previous selections
        for widget in self.factors_frame.winfo_children():
            widget.destroy()
            
            
        # Delete previous design tree
        for widget in self.create_design_frame.winfo_children():
            widget.destroy()
        
        for widget in self.run_frame.winfo_children():
            widget.destroy()
            
        for widget in self.design_frame.winfo_children():
            widget.destroy()
        
        # Widget lists
        self.default_widgets = {}
        self.check_widgets = {}
        self.min_widgets = {}
        self.max_widgets = {}
        self.dec_widgets = {}
        self.cat_widgets = {}
        
               
        # Initial variable values
        self.factor_que_length = 1
        self.default_values_list = []
        self.checkstate_list=[]
        self.min_list = []
        self.max_list = []
        self.dec_list = []
        self.cat_checklist = []
        
        #Values to help with formatting
        entry_width = 20
        
        # Create column for model factor names
        self.headername_label = tk.Label(master = self.factors_frame, text = 'Model Factors', font = "Calibri 13 bold", width = 10, anchor = 'w')
        self.headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W, padx = 10)
        
        # Create column for factor type
        self.headertype_label = tk.Label(master = self.factors_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 10, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 1, sticky = tk.N + tk.W)
        
        
        # Create column for factor default values
        self.headerdefault_label = tk.Label(master = self.factors_frame, text = 'Default Value', font = "Calibri 13 bold", width = 15 )
        self.headerdefault_label.grid(row = 0, column = 2, sticky = tk.N + tk.W)
        
        # Create column for factor check box
        self.headercheck_label = tk.Label(master = self.factors_frame, text = 'Include in Experiment', font = "Calibri 13 bold", width = 20 )
        self.headercheck_label.grid(row = 0, column = 3, sticky = tk.N + tk.W)
        
        # Create header for experiment options
        self.headercheck_label = tk.Label(master = self.factors_frame, text = 'Experiment Options', font = "Calibri 13 bold", width = 60 )
        self.headercheck_label.grid(row = 0, column = 4, columnspan = 3)
        
    
        
       
        
        
        # Get model selected from drop down
        self.selected_model = self.model_var.get()
        

        
    
                
        
        # Get model infor from dictionary
        self.model_object = self.model_list[self.selected_model]()
        
        for  factor in self.model_object.specifications:
            
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
            self.factor_description = self.model_object.specifications[factor].get("description")
            self.factor_default = self.model_object.specifications[factor].get("default")
            
            
            #Values to help with formatting
            entry_width = 10
            
            self.factors_frame.grid_rowconfigure(self.factor_que_length, weight =1)
            
            # Add label for factor names
            self.factorname_label = tk.Label (master = self.factors_frame, text = f"{factor} - {self.factor_description}", font = "Calibri 13", width = 40, anchor = 'w')
            self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W, padx = 10)
        
            
            
            if self.factor_datatype == float:
            
                self.factors_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'float'
             
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.factors_frame, text = self.str_type, font = "Calibri 13", width = 10, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                
                # Add check box
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.factors_frame, variable = self.checkstate,
                                               command = self.include_factor, width = 5)
                self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
                
                # Add entry box for min val
                self.min_frame = tk.Frame (master = self.factors_frame)
                self.min_frame.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W )
                
                self.min_label = tk.Label(master = self.min_frame, text = 'Min Value', font = "Calibri 13", width = 10 )
                self.min_label.grid( row = 0, column = 0)
                self.min_val = tk.StringVar()
                self.min_entry = tk.Entry( master = self.min_frame, width = 10, textvariable = self.min_val, justify = 'right')
                self.min_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.min_list.append(self.min_val)    
                
                self.min_widgets[factor] = self.min_entry
                
                self.min_entry.configure(state = 'disabled')
                
                # Add entry box for max val
                self.max_frame = tk.Frame (master = self.factors_frame)
                self.max_frame.grid( row = self.factor_que_length, column = 5, sticky = tk.N + tk.W )
                
                self.max_label = tk.Label(master = self.max_frame, text = 'Max Value', font = "Calibri 13", width = 10 )
                self.max_label.grid( row = 0, column = 0) 
                
                self.max_val = tk.StringVar()
                self.max_entry = tk.Entry( master = self.max_frame, width = 10, textvariable = self.max_val, justify = 'right')
                self.max_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
               
                self.max_list.append(self.max_val)    
                
                self.max_widgets[factor] = self.max_entry
                
                self.max_entry.configure(state = 'disabled')
                
                # Add entry box for editable decimals
                self.dec_frame = tk.Frame (master = self.factors_frame)
                self.dec_frame.grid( row = self.factor_que_length, column = 6, sticky = tk.N + tk.W )
                
                self.dec_label = tk.Label(master = self.dec_frame, text = '# Decimals', font = "Calibri 13", width = 10 )
                self.dec_label.grid( row = 0, column = 0) 
                
                self.dec_val = tk.StringVar()
                self.dec_entry = tk.Entry( master = self.dec_frame, width = 10, textvariable = self.dec_val, justify = 'right')
                self.dec_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.dec_list.append(self.dec_val)  
                
                self.dec_widgets[factor] = self.dec_entry
                
                self.dec_entry.configure(state = 'disabled')
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == int:
            
                self.factors_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'int'
    
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.factors_frame, text = self.str_type, font = "Calibri 13", width = 10, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.factors_frame, variable = self.checkstate,
                                               command = self.include_factor)
                self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
                
                # Add entry box for min val
                self.min_frame = tk.Frame (master = self.factors_frame)
                self.min_frame.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W )
                
                self.min_label = tk.Label(master = self.min_frame, text = 'Min Value', font = "Calibri 13", width = 10 )
                self.min_label.grid( row = 0, column = 0)
                self.min_val = tk.StringVar()
                self.min_entry = tk.Entry( master = self.min_frame, width = 10, textvariable = self.min_val, justify = 'right')
                self.min_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.min_list.append(self.min_val)    
                
                self.min_widgets[factor] = self.min_entry
                
                self.min_entry.configure(state = 'disabled')
                
                # Add entry box for max val
                self.max_frame = tk.Frame (master = self.factors_frame)
                self.max_frame.grid( row = self.factor_que_length, column = 5, sticky = tk.N + tk.W )
                
                self.max_label = tk.Label(master = self.max_frame, text = 'Max Value', font = "Calibri 13", width = 10 )
                self.max_label.grid( row = 0, column = 0) 
                
                self.max_val = tk.StringVar()
                self.max_entry = tk.Entry( master = self.max_frame, width = 10, textvariable = self.max_val, justify = 'right')
                self.max_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
               
                self.max_list.append(self.max_val)    
                
                self.max_widgets[factor] = self.max_entry
                
                self.max_entry.configure(state = 'disabled')
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == list:
                
                self.factors_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'list'
               
                # Add label for factor names
                # self.factorname_label = tk.Label (master = self.factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                # self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.factors_frame, text = self.str_type, font = "Calibri 13", width = 10, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                #Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 25:
                        entry_width = 25
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.default_values_list.append(self.default_value)
                
                
                
      
                
                # Add checkbox (currently not visible)
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.factors_frame, variable = self.checkstate,
                                               command = self.include_factor)
                #self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
            
                self.factor_que_length += 1
                
            elif self.factor_datatype == tuple:
                
                self.factors_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'tuple'
               
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.factors_frame, text = self.str_type, font = "Calibri 13", width = 10, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 25:
                        entry_width = 25
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.default_values_list.append(self.default_value)
                
             
                
                # Add checkbox (currently not visible)
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.factors_frame, variable = self.checkstate,
                                               command = self.include_factor)
                #self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
               
                self.factor_que_length += 1

        

        
        # Design type selection menu
        self.design_frame = tk.Frame(master = self.master)
        self.design_frame.grid(row = 5, column = 0)
        self.design_type_label = tk.Label (master = self.design_frame, text = 'Select Design Type', font = "Calibri 13", width = 20)
        self.design_type_label.grid( row = 0, column = 0)
        
        self.design_types_list = ['NOLH']
        self.design_var = tk.StringVar()
        self.design_type_menu = ttk.OptionMenu(self.design_frame, self.design_var, 'Design Type', *self.design_types_list, command = self.enable_stacks)
        self.design_type_menu.grid(row = 0, column = 1, padx = 30)
        
        #Stack selection menu
        self.stack_label = tk.Label (master = self.design_frame, text = "Select Number of Stacks", font = "Calibri 13", width = 20)
        self.stack_label.grid( row =1, column = 0)
        self.stack_list = ['1', '2', '3']
        self.stack_var = tk.StringVar()
        self.stack_menu = ttk.OptionMenu(self.design_frame, self.stack_var, 'Stacks', *self.stack_list, command = self.get_design_pts)
        self.stack_menu.grid( row = 1, column = 1)
        self.stack_menu.configure(state = 'disabled')
        
        # # Design pts label
        # self.design_pts_title = tk.Label (master = self.design_frame, text = 'Design Points', font = "Calibri 13", width = 50)
        # self.design_pts_title.grid( row = 0, column = 2)
        
        # Create design button
        self.create_design_button = tk.Button(master = self.design_frame, text = 'Create Design', font = "Calibri 13", command = self.create_design , width = 20)
        self.create_design_button.grid( row = 0, column = 3)
        self.create_design_button.configure(state = 'disabled')
        
 
    def onFrameConfigure_factor(self, *args):
        self.factor_canvas.configure(scrollregion=self.factor_canvas.bbox('all'))
        
 
    # Used to display the design tree for both created and loaded designs
    def display_design_tree(self):
        
        

        # Get list of factor names from csv file
        with open( self.design_csv_name, 'r') as design_file:
            reader = csv.reader(design_file)
            self.all_factor_headers = next(reader)[1:]
       
        #Get list of default factor names 
        default_factor_names = [] 
        for factor in self.all_factor_headers:
            if factor not in self.model_object.specifications:
                default_factor_names.append(factor)
        # number of variable factors in experiment
        self.num_factors = len(self.all_factor_headers) - len(default_factor_names)
       
        
        #Initialize design tree
        self.create_design_frame = tk.Frame(master = self.master)
        self.create_design_frame.grid( row = 6, column = 0)
        self.create_design_frame.grid_rowconfigure( 0, weight = 0)
        self.create_design_frame.grid_rowconfigure( 1, weight = 1)
        self.create_design_frame.grid_columnconfigure( 0, weight = 1)
        self.create_design_frame.grid_columnconfigure( 1, weight = 1)
        
        self.create_design_label = tk.Label( master = self.create_design_frame, text = 'Generated Designs', font = "Calibri 13 bold", width = 50)
        self.create_design_label.grid(row = 0, column = 0, sticky = tk.W)
   
        self.design_tree = ttk.Treeview( master = self.create_design_frame)
        self.design_tree.grid(row = 1, column = 0, sticky = 'nsew', padx = 10)
        
        # Create headers for each factor 
        self.design_tree['columns'] = self.all_factor_headers
        self.design_tree.column("#0", width=80, anchor = 'e' )
        for factor in self.all_factor_headers:
            self.design_tree.column(factor, width=250, anchor = 'e') 
            self.design_tree.heading( factor, text = factor)
       
        
        self.style = ttk.Style()
        self.style.configure("Treeview.Heading", font=('Calibri', 13, 'bold'))
        self.style.configure("Treeview", foreground="black", font = ('Calibri', 13))
        self.design_tree.heading( '#0', text = 'Design #' )
        
        # Read design file and output row plus default values to design tree
        with open( self.design_csv_name, 'r') as design_file:
            reader = csv.reader(design_file)
            #skip header row
            next(reader)
            # used to number design rows
            row_index = 0
            for row in reader:
                current_design_row = row[1:]
                data_insert = current_design_row 
                self.design_tree.insert("", 'end', text = row_index, values = data_insert, tag = 'style')
                row_index += 1
                
            # Create a horizontal scrollbar
        xscrollbar = ttk.Scrollbar(master = self.create_design_frame, orient="horizontal", command= self.design_tree.xview)
        xscrollbar.grid(row = 2, column = 0, sticky = 'nsew')
        yscrollbar = ttk.Scrollbar(master = self.create_design_frame, orient="vertical", command= self.design_tree.yview)
        yscrollbar.grid(row = 1, column = 1, sticky = 'nsew')
        
        # Configure the Treeview to use the horizontal scrollbar
        self.design_tree.configure(xscrollcommand=xscrollbar.set)
        self.design_tree.configure(yscrollcommand=yscrollbar.set)
        
        # Create buttons to run experiment
        self.run_frame = tk.Frame(master = self.master)
        self.run_frame.grid(row = 7, column = 0)
        self.run_frame.grid_columnconfigure(0, weight = 1)
        self.run_frame.grid_columnconfigure(1, weight = 1)
        self.run_frame.grid_columnconfigure(2, weight = 1)
        self.run_frame.grid_rowconfigure(0, weight = 0)
        self.run_frame.grid_rowconfigure(1, weight = 0)
  
        
        self.rep_label = tk.Label(master = self.run_frame, text = 'Replications', font = 'Calibri 13', width = 20)
        self.rep_label.grid( row = 0, column = 0, sticky = tk.W)
        self.rep_var = tk.StringVar()
        self.rep_entry = tk.Entry( master = self.run_frame, textvariable = self.rep_var, width = 10)
        self.rep_entry.grid( row = 0, column = 1, sticky = tk.W)
        
        self.crn_label = tk.Label(master = self.run_frame, text = 'CRN', font = 'Calibri 13', width = 20)
        self.crn_label.grid( row = 1, column = 0, sticky = tk.W)
        self.crn_var = tk.StringVar()
        self.crn_option = ttk.OptionMenu( self.run_frame, self.crn_var,'Yes', 'Yes', 'No')
        self.crn_option.grid(row = 1, column =1, sticky = tk.W)
        
      
        
        self.run_button = tk.Button(master = self.run_frame, text = 'Run All', font = 'Calibri 13', width = 20, command = self.run_experiment)
        self.run_button.grid( row = 0, column = 2, sticky = tk.E, padx = 30)
        
  
        
        
    def enable_stacks(self, *args):
        self.stack_menu.configure(state = 'normal')
    
    def get_design_pts(self, *args):
        self.design_pts = self.stack_var.get() + " test "
        
        #self.design_pts_label = tk.Label( master = self.design_frame, text = self.design_pts, font = "Calibri 13", width = 50)
        #self.design_pts_label.grid( row = 1, column =2)
        
        self.create_design_button.configure(state = 'normal')
        
        
        
    def create_design(self, *args):
       
        self.create_design_frame = tk.Frame(master = self.master)
        self.create_design_frame.grid( row = 6, column = 0)
        self.create_design_frame.grid_rowconfigure( 0, weight = 0)
        self.create_design_frame.grid_rowconfigure( 1, weight = 1)
        self.create_design_frame.grid_columnconfigure( 0, weight = 1)
    
        
        #Export design factors
        
        self.default_values = [self.default_value.get() for self.default_value in self.default_values_list]
        self.check_values = [self.checkstate.get() for self.checkstate in self.checkstate_list]
        self.min_values = [self.min_val.get() for self.min_val in self.min_list]
        self.max_values = [self.max_val.get() for self.max_val in self.max_list]
        self.dec_values = [self.dec_val.get() for self.dec_val in self.dec_list]
        self.fixed_factors = {}
        self.factor_index = 0
        self.maxmin_index = 0
        self.dec_index = 0
        # List used for parsing design file
        self.factor_headers = [] 
        
        #Dictionary used for tree view display of fixed factors
        self.fixed_str = {}
        
       
      
        # Write model factors file
        
        with open("./data_farming_experiments/model_factors.txt", "w") as self.model_design_factors:
            self.model_design_factors.write("")
         
        # List to hold names of all factors part of model to be displayed in csv
        self.factor_names = []
        self.def_factor_names = []
        # Get experiment information    
        for  factor in self.model_object.specifications:
            
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
            self.factor_description = self.model_object.specifications[factor].get("description")
           
            
            
            
            self.factor_default = self.default_values[self.factor_index]
            self.factor_include = self.check_values[self.factor_index]
            
            
            
            if self.factor_include == True:
                
                # Add factor to list of design factors in order that will be varied
                self.factor_headers.append(factor)
                # Factor names in csv are unedited
                self.factor_names.append(factor)
            
                if self.factor_datatype == float or self.factor_datatype == int:
                    self.factor_min = str(self.min_values[self.maxmin_index])
                    self.factor_max = str(self.max_values[self.maxmin_index])
                    self.maxmin_index += 1
                    
                    if self.factor_datatype == float:
                        self.factor_dec = str(self.dec_values[self.dec_index])
                        self.dec_index += 1
                        
                    elif self.factor_datatype == int:
                        self.factor_dec = '0'
                        
                    self.data_insert = self.factor_min + ' ' + self.factor_max + ' ' + self.factor_dec
                    with open("./data_farming_experiments/model_factors.txt", "a") as self.model_design_factors:
                        self.model_design_factors.write(self.data_insert + '\n') 
                        
                
                          
                   
            
            # Include fixed default values in design
            if self.factor_include == False:
                
                # Factor names in csv have "(default)" appended to end
                self.def_factor_names.append(factor + ' (default)')
                
                # Values to be placed in tree view of design
                self.fixed_str[factor] = self.factor_default
               
                if self.factor_datatype == float or self.factor_datatype == int:
                    # self.factor_default = str(self.default_values[self.factor_index])
                    # self.data_insert = self.factor_default + ' ' + self.factor_default + ' 0'
                    self.maxmin_index += 1
               
                # Add default values to exeriment and set to correct datatype
                if self.factor_datatype == float:
                    self.fixed_factors[factor] = float(self.factor_default)
                    self.dec_index += 1
                    
                elif self.factor_datatype == int:
                    self.fixed_factors[factor] = int(self.factor_default)
                    
            if self.factor_datatype == list:
                self.fixed_factors[factor] = ast.literal_eval(self.factor_default)
                
            elif self.factor_datatype == tuple:
                   
                tuple_str = tuple(self.factor_default[1:-1].split(","))
                self.fixed_factors[factor] = tuple(float(s) for s in tuple_str)
   
            self.factor_index += 1
            
        
        #Create design file
        model_name = self.model_object.name
        model_fixed_factors = self.fixed_factors
        
        self.model = model_directory[model_name](fixed_factors=model_fixed_factors)
        
        factor_settings_filename = 'model_factors'
       
        
        # for  factor in self.model_object.specifications:
            
        # # Specify the names of the model factors (in order) that will be varied.
        
        #     # factor_headers.append(factor)
            
        
        # Number of stacks specified by user
        num_stack = self.stack_var.get()
        # Create model factor design from .txt file of factor settings.
        # Hard-coded for a single-stack NOLHS.
        #command = "stack_nolhs.rb -s 1 model_factor_settings.txt > outputs.txt"
        #command = f"stack_nolhs.rb -s 2 ./data_farming_experiments/{factor_settings_filename}.txt > ./data_farming_experiments/{factor_settings_filename}_design.txt"
        command = "stack_nolhs.rb -s " + num_stack + f" ./data_farming_experiments/{factor_settings_filename}.txt > ./data_farming_experiments/{factor_settings_filename}_design.txt"
       
        os.system(command) 
        # Append design to base filename.
        design_filename = f"{factor_settings_filename}_design"
        # Read in design matrix from .txt file. Result is a pandas DataFrame.
        design_table = pd.read_csv(f"./data_farming_experiments/{design_filename}.txt", header=None, delimiter="\t", encoding="utf-8")
        # Count number of design_points.
        self.n_design_pts = len(design_table)
        # Create all design points.
        self.design = []
     
        
        #Time stamp for file name
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        # Create design csv file with headers
        self.design_csv_name= "./data_farming_experiments/" + self.selected_model + "_design_" + timestamp + ".csv"
      
        with open(self.design_csv_name, "w", newline='') as self.model_design_csv:
            writer = csv.writer(self.model_design_csv)
            writer.writerow(['Design #'] + self.factor_names + self.def_factor_names)
        


        # Get default values for factors not included in experiment
        default_list = []
        for factor in self.fixed_str:
            default_list.append(self.fixed_str[factor])
            
        for dp_index in range(self.n_design_pts):
            current_factor_designs = []
            for factor_idx in range(len(self.factor_headers)):
                current_factor_designs.append(design_table[factor_idx][dp_index])
            data_insert = current_factor_designs + default_list   
                
             
            
            
            # Write design points to csv file
            with open(self.design_csv_name, "a", newline='') as self.model_design_csv:
                writer = csv.writer(self.model_design_csv)
                writer.writerow([dp_index] + data_insert)
                
        # Pop up message that csv design file has been created
        tk.messagebox.showinfo("Information", "Design file " + self.design_csv_name + " has been created.")    
    
    
        # Display Design Values
        self.display_design_tree() 
        
        
        
            
    def update_defaults_button(self):
        
        # Get default user values
        self.fixed_factors = {}
        default_csv_insert = []
        
        # List of values entered by user
        self.default_values = [self.default_value.get() for self.default_value in self.default_values_list]
        factor_index = 0
        for factor in self.default_factor_list:
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
            current_def_val = self.default_values[factor_index]
            if self.factor_datatype == float:
                self.fixed_factors[factor] = float(current_def_val)
            
            elif self.factor_datatype == int:
                self.fixed_factors[factor] = int(current_def_val)
            
            if self.factor_datatype == list:
                self.fixed_factors[factor] = ast.literal_eval(current_def_val)
        
            elif self.factor_datatype == tuple:
           
                tuple_str = tuple(current_def_val[1:-1].split(","))
                self.fixed_factors[factor] = tuple(float(s) for s in tuple_str)
            
            default_csv_insert.append(self.fixed_factors[factor])    
            factor_index += 1
    
        
        self.display_design_tree()
        
        tk.messagebox.showinfo("Defaults Updated", "Defaults have been successfully updated.")   
        
    def run_experiment(self, *args):
        
        #Name of model used for default save file
        model_save_name = self.selected_model
        
        #Time stamp for file name
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        # Ask user for file save location
        save_path = filedialog.asksaveasfilename(initialfile = self.selected_model + "_datafarming_experiment_" + timestamp)
        
        
               
        # Specify the name of the model as it appears in directory.py
        model_name = self.model_object.name
        
        
       
        # factor_headers = [] 
        
        # for  factor in self.model_object.specifications:
        # # Specify the names of the model factors (in order) that will be varied.
        
        #     factor_headers.append(factor)

        # If creating the design, provide the name of a .txt file containing
        # the following:
        #    - one row corresponding to each model factor being varied
        #    - three columns:
        #         - first column: lower bound for factor value
        #         - second column: upper bound for factor value
        #         - third column: (integer) number of digits for discretizing values
        #                         (e.g., 0 corresponds to integral values for the factor)
        #factor_settings_filename = "model_factors"
        factor_settings_filename = None

        # OR, if the design has been created, provide the name of a .text file
        # containing the following:
        #    - one row corresponding to each design point
        #    - the number of columns equal to the number of factors being varied
        #    - each value in the table gives the value of the factor (col index)
        #      for the design point (row index)
        # E.g., design_filename = "model_factor_settings_design"
        #design_filename = None
        design_filename = "model_factors_design"

        # Specify a common number of replications to run of the model at each
        # design point.
        n_reps = int(self.rep_var.get())

        # Specify whether to use common random numbers across different versions
        # of the model.
        if self.crn_var.get() == 'Yes':
            crn_across_design_pts = True
        else:
            crn_across_design_pts = False

        # Specify filename for outputs.
        #output_filename = "test_experiment"
        output_filename = save_path

        # No code beyond this point needs to be edited.

        # Create DataFarmingExperiment object.
        myexperiment = DataFarmingExperiment(model_name=model_name,
                                             factor_settings_filename=factor_settings_filename,
                                             factor_headers=self.factor_headers,
                                             design_filename=design_filename,
                                             model_fixed_factors = self.fixed_factors
                                             )

        
        
        # Run replications and print results to file.
        myexperiment.run(n_reps=n_reps, crn_across_design_pts=crn_across_design_pts)
        myexperiment.print_to_csv(csv_filename=output_filename)
        
        # run confirmation message
        tk.messagebox.showinfo("Run Completed", f"Experiment Completed. Output file can be found at {output_filename}")  

            
   
    def include_factor(self, *args):

        self.check_values = [self.checkstate.get() for self.checkstate in self.checkstate_list]
        self.check_index = 0
        self.cat_index = 0
    
        # If checkbox to include in experiment checked, enable experiment option buttons
        for factor in self.model_object.specifications:
                  
            # Get current checksate values from include experiment column
            self.current_checkstate = self.check_values[self.check_index]
            # Cross check factor type
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
            self.factor_description = self.model_object.specifications[factor].get("description")
            self.factor_default = self.model_object.specifications[factor].get("default")
            
            # Disable / enable experiment option widgets depending on factor type
            if self.factor_datatype == float or self.factor_datatype == int:
                self.current_min_entry = self.min_widgets[factor]
                self.current_max_entry = self.max_widgets[factor]               
                
                             
                if self.current_checkstate == True:
                    self.current_min_entry.configure(state = 'normal')
                    self.current_max_entry.configure(state = 'normal')
                    
                elif self.current_checkstate == False:
                    #Empty current entries
                    self.current_min_entry.delete(0, tk.END)
                    self.current_max_entry.delete(0, tk.END)
                   
                    
                    self.current_min_entry.configure(state = 'disabled')
                    self.current_max_entry.configure(state = 'disabled')
                                      
            if self.factor_datatype == float:              
                self.current_dec_entry = self.dec_widgets[factor]
                
                if self.current_checkstate == True:
                    self.current_dec_entry.configure(state = 'normal')
                    
                elif self.current_checkstate == False:
                    self.current_dec_entry.delete(0, tk.END)
                    self.current_dec_entry.configure(state = 'disabled')
                    
            self.check_index += 1     
                  


class Solver_Datafarming_Window(tk.Toplevel):
    def __init__(self, master):
        
        self.master = master
        self.master.grid_rowconfigure(0, weight = 0)
        self.master.grid_rowconfigure(1, weight = 0)
        self.master.grid_rowconfigure(2, weight = 1)
        self.master.grid_rowconfigure(3, weight = 1)
        self.master.grid_rowconfigure(4, weight = 0)
        self.master.grid_rowconfigure(5, weight = 1)
        self.master.grid_rowconfigure(6, weight = 0)
        self.master.grid_rowconfigure(7, weight = 1)
        self.master.grid_rowconfigure(8, weight = 1)
        self.master.grid_rowconfigure(9, weight = 1)
        self.master.grid_rowconfigure(10, weight = 1)
        self.master.grid_columnconfigure(0, weight = 1)
        
        #Initalize frames
        self.title_frame = tk.Frame(master = self.master)
        self.title_frame.grid(row = 0, column = 0)
        self.title_frame.grid_rowconfigure(0, weight = 0)
        self.title_frame.grid_columnconfigure(0, weight = 1)
        
        self.solver_select_frame = tk.Frame(master = self.master)
        self.solver_select_frame.grid(row = 1, column =0)
        self.solver_select_frame.grid_rowconfigure(0, weight = 0)
        self.solver_select_frame.grid_columnconfigure(0, weight = 1)
        self.solver_select_frame.grid_columnconfigure(1, weight = 1)
        self.solver_select_frame.grid_columnconfigure(2, weight = 1)
        self.solver_select_frame.grid_columnconfigure(3, weight = 1)
        
        self.solver_frame = tk.Frame(master = self.master)
        self.solver_frame.grid(row = 2, column = 0)
        
        # frames created here so previous selections can be cleared
        self.design_frame = tk.Frame(master = self.master)
        self.problem_model_factors_frame = tk.Frame(master = self.master)
        self.problem_select_frame = tk.Frame(master = self.master)
        self.experiment_frame = tk.Frame(master = self.master)
        self.problem_frame = tk.Frame(master = self.problem_model_factors_frame)
        self.model_frame = tk.Frame(master = self.problem_model_factors_frame)
        self.notebook_frame = tk.Frame (master = self.master)
        self.design_view_frame = tk.Frame(master = self.master)
        self.create_pair_frame = tk.Frame(master = self.master)
        
        
        
        
        # Window title 
        self.title_label = tk.Label( master = self.title_frame, text = 'Solver Data Farming', width = 50,
                                    font = 'Calibir 15 bold')
        self.title_label.grid( row = 0, column = 0)
        
        # Option menu to select solver
        self.solver_select_label = tk.Label( master = self.solver_select_frame, text = 'Select Solver:', width = 20,
                                    font = 'Calibir 13')
        self.solver_select_label.grid( row = 0, column = 0)
        
        # Variable to store selected solver
        self.solver_var = tk.StringVar()
        
        #Directory of solver names
        self.solver_list = solver_unabbreviated_directory
        
        self.solver_select_menu = ttk.OptionMenu(self.solver_select_frame, self.solver_var, 'Solver', *self.solver_list, command = self.show_solver_factors)
        self.solver_select_menu.grid(row = 0, column = 1)
        
        # Load design selection
        self.load_design_label =  tk.Label( master = self.solver_select_frame, text = ' OR ', width = 20,
                                    font = 'Calibir 13')
        self.load_design_label.grid( row = 0, column = 2)
        self.load_design_button = tk.Button(master = self.solver_select_frame, text = 'Load Solver Design', font = 'Calibir 11',
                                            width = 20, command = self.load_solver_design)
        self.load_design_button.grid( row = 0, column = 3)
        
        #Dictonaries to hold experiment info
        self.experiment_pairs = {} #Dictionary to hold all experiment/ problem pairs, contains list of experiment, problem name, problem factors, then model factors
        self.select_pair_vars ={}
        self.macro_rep_vars = {}
        self.post_rep_vars= {}
        self.norm_rep_vars = {}
        self.run_buttons = {}
        self.post_buttons = {}
        self.norm_buttons = {}
        self.save_buttons = {}
        
        
        
    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()
    
    def load_solver_design(self):
    
        #Clear previous selections
        self.clear_frame(frame = self.solver_frame) #Clear previous solver selections
        self.clear_frame(frame = self.design_frame) # Clear design selections
        self.clear_frame(frame = self.problem_model_factors_frame) # Clear problem and model factor selections
        self.clear_frame(frame = self.problem_select_frame) # Clear problem selection widgets
        self.clear_frame(frame = self.design_view_frame) # Clear design tree
        self.clear_frame(self.create_pair_frame)
            
        
        # Initialize frames
        self.solver_frame.grid(row = 2, column = 0)
        self.solver_frame.grid_rowconfigure(0, weight =1)
        self.solver_frame.grid_columnconfigure(0, weight =1)
        self.solver_frame.grid_columnconfigure(1, weight =1)
        self.solver_frame.grid_columnconfigure(2, weight =1)
        self.solver_frame.grid_columnconfigure(3, weight =1)
        
        self.loaded_design = True # Design was loaded by user

        # Create column for model factor names
        self.headername_label = tk.Label(master = self.solver_frame, text = 'Default Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W)
        
        # Create column for factor type
        self.headertype_label = tk.Label(master = self.solver_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 20, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 1, sticky = tk.N + tk.W)
       
        
        # List to hold default values
        self.default_values_list = []
        self.fixed_str = {}
        
        # Create column for factor default values
        self.headerdefault_label = tk.Label(master = self.solver_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 2, sticky = tk.N + tk.W)
            
        # Name of design csv file
        self.csv_filename = filedialog.askopenfilename()
        
        # convert loaded design to data frame
        self.design_table = pd.read_csv(self.csv_filename, index_col=False)
        
        # Get design information from table
        self.solver_name = self.design_table.at[1, 'Solver Name']
        self.design_type = self.design_table.at[1, 'Design Type']
        self.n_stacks = self.design_table.at[1, 'Number Stacks']
        
        
        # determine what factors are included in design
        self.factor_status = {} #dictionary that contains true/false for wheither factor is in design
        for col in self.design_table.columns[1:-3]: # col correspond to factor names, exclude index and information cols
            factor_set = set(self.design_table[col])
            
            if len(factor_set) > 1:
                design_factor = True
            else:
                design_factor = False
                
            self.factor_status[col] = design_factor
            
            
            
    
            
        # get default values for fixed factors
        self.default_factors = {} #contains only factors not in design, factor default vals input as str
        for factor in self.factor_status:
            if self.factor_status[factor] == False:
                self.default_factors[factor] = self.design_table.at[1, factor]
                    
  
        print(self.default_factors)
        self.solver_class = solver_directory[self.solver_name]
        self.solver_object = self.solver_class()

        #Display model name
        self.solver_name_label = tk.Label( master = self.solver_select_frame, text = 'Selected Solver: ' + self.solver_name, font = "Calibri 13", width = 40  )
        self.solver_name_label.grid( row = 0, column = 4, sticky = tk.W)
        
        # Allow user to change default values
        factor_idx = 0
        self.factor_que_length = 1
        for  factor in self.default_factors:
            
            self.factor_datatype = self.solver_object.specifications[factor].get("datatype")
            self.factor_description = self.solver_object.specifications[factor].get("description")
            self.factor_default = self.default_factors[factor]
            
            
            self.solver_frame.grid_rowconfigure(self.factor_que_length, weight =1)
            
            # Convert datatype to string for display
            if self.factor_datatype == int:
                self.str_type = 'int'
            elif self.factor_datatype == float:
                self.str_type = 'float'
            elif self.factor_datatype == bool:
                self.str_type = 'bool'
       
            # Add label for factor names
            self.factorname_label = tk.Label (master = self.solver_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
            self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
            
            # Add label for factor type
            self.factortype_label = tk.Label (master = self.solver_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
            self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
            
            #Add entry box for default value for int and float
            self.default_value= tk.StringVar()
            if self.factor_datatype == int or self.factor_datatype == float:
                self.default_entry = tk.Entry( master = self.solver_frame, width = 10, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.default_values_list.append(self.default_value)
                
            # Add option menu for bool factors
            elif self.factor_datatype == bool:
                self.default_value.set(self.factor_default) #Set default bool option
                self.bool_menu = ttk.OptionMenu(self.solver_frame, self.default_value, self.factor_default, 'TRUE', 'FALSE')
                self.bool_menu.grid( row = self.factor_que_length, column = 2, sticky = tk.N + tk.W)
                self.default_values_list.append(self.default_value)
 
            self.factor_que_length += 1
            factor_idx += 1
     
                    
        self.show_design_options() # run function to design creation options
        self.display_design_tree()


    def show_solver_factors(self, *args):
        
        
        #Initalize frames
        self.solver_frame.grid_rowconfigure(0, weight = 0)
        self.solver_frame.grid_columnconfigure(0, weight = 1)
        self.solver_frame.grid_columnconfigure(1, weight = 1)
        self.solver_frame.grid_columnconfigure(2, weight = 1)
        self.solver_frame.grid_columnconfigure(3, weight = 1)
        self.solver_frame.grid_columnconfigure(4, weight = 1)
        self.solver_frame.grid_columnconfigure(5, weight = 1)
        self.solver_frame.grid_columnconfigure(6, weight = 1)
        
       
        # Clear previous selections    
        self.clear_frame(frame = self.solver_frame) #Clear previous solver selections
        self.clear_frame(frame = self.design_frame) # Clear design selections
        self.clear_frame(frame = self.problem_model_factors_frame) # Clear problem and model factor selections
        self.clear_frame(frame = self.problem_select_frame) # Clear problem selection widgets
        self.clear_frame(frame = self.design_view_frame) # Clear design tree
        self.clear_frame(self.create_pair_frame)
        
        self.loaded_design = False # design was not loaded by user
        
        # Create column for solver factor names
        self.headername_label = tk.Label(master = self.solver_frame, text = 'Solver Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W)
        

        
        # Create column for factor type
        self.headertype_label = tk.Label(master = self.solver_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 20, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 1, sticky = tk.N + tk.W)
        
        
        # Create column for factor default values
        self.headerdefault_label = tk.Label(master = self.solver_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 2, sticky = tk.N + tk.W)
        
        # Create column for factor check box
        self.headercheck_label = tk.Label(master = self.solver_frame, text = 'Include in Experiment', font = "Calibri 13 bold", width = 20 )
        self.headercheck_label.grid(row = 0, column = 3, sticky = tk.N + tk.W)
        
        # Create header for experiment options
        self.headercheck_label = tk.Label(master = self.solver_frame, text = 'Experiment Options', font = "Calibri 13 bold", width = 50 )
        self.headercheck_label.grid(row = 0, column = 4, columnspan = 3)
        
        # Get solver info from dictionary
        self.selected_solver = self.solver_var.get()
        self.solver_object = self.solver_list[self.selected_solver]()
        self.solver_name = self.solver_object.name # name of solver used for save files
        
        
        entry_width = 10
        
        # Widget lists
        self.default_widgets = {}
        self.check_widgets = {}
        self.min_widgets = {}
        self.max_widgets = {}
        self.dec_widgets = {}
        self.description_buttons = {}
        
        
               
        # Initial variable values
        self.factor_que_length = 1
        self.default_values_list = []
        self.checkstate_list=[]
        self.min_list = []
        self.max_list = []
        self.dec_list = []
        # self.descriptions = {} #used for description pop ups
        
       
        for  factor in self.solver_object.specifications:
            
            self.factor_datatype = self.solver_object.specifications[factor].get("datatype")
            factor_description = self.solver_object.specifications[factor].get("description")
            self.factor_default = self.solver_object.specifications[factor].get("default")
            
            # self.descriptions[factor] = factor_description
            
            
            # Add label for factor names
            display_name = f"{factor} - {factor_description}"
            self.factorname_label = tk.Label (master = self.solver_frame, text = display_name, font = "Calibri 13", width = 80, anchor = 'w')
            self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
            
            
            
            
            if self.factor_datatype == float:
            
                self.solver_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'float'
    
                # # Add label for factor names
                # self.factorname_label = tk.Label (master = self.solver_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                # self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
               
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.solver_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.solver_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                
                # Add check box
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.solver_frame, variable = self.checkstate,
                                               command = self.include_factor)
                self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
                
                # Add entry box for min val
                self.min_frame = tk.Frame (master = self.solver_frame)
                self.min_frame.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W )
                
                self.min_label = tk.Label(master = self.min_frame, text = 'Min Value', font = "Calibri 13", width = 10 )
                self.min_label.grid( row = 0, column = 0)
                self.min_val = tk.StringVar()
                self.min_entry = tk.Entry( master = self.min_frame, width = 10, textvariable = self.min_val, justify = 'right')
                self.min_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.min_list.append(self.min_val)    
                
                self.min_widgets[factor] = self.min_entry
                
                self.min_entry.configure(state = 'disabled')
                
                # Add entry box for max val
                self.max_frame = tk.Frame (master = self.solver_frame)
                self.max_frame.grid( row = self.factor_que_length, column = 5, sticky = tk.N + tk.W )
                
                self.max_label = tk.Label(master = self.max_frame, text = 'Max Value', font = "Calibri 13", width = 10 )
                self.max_label.grid( row = 0, column = 0) 
                
                self.max_val = tk.StringVar()
                self.max_entry = tk.Entry( master = self.max_frame, width = 10, textvariable = self.max_val, justify = 'right')
                self.max_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
               
                self.max_list.append(self.max_val)    
                
                self.max_widgets[factor] = self.max_entry
                
                self.max_entry.configure(state = 'disabled')
                
                # Add entry box for editable decimals
                self.dec_frame = tk.Frame (master = self.solver_frame)
                self.dec_frame.grid( row = self.factor_que_length, column = 6, sticky = tk.N + tk.W )
                
                self.dec_label = tk.Label(master = self.dec_frame, text = '# Decimals', font = "Calibri 13", width = 10 )
                self.dec_label.grid( row = 0, column = 0) 
                
                self.dec_val = tk.StringVar()
                self.dec_entry = tk.Entry( master = self.dec_frame, width = 10, textvariable = self.dec_val, justify = 'right')
                self.dec_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.dec_list.append(self.dec_val)  
                
                self.dec_widgets[factor] = self.dec_entry
                
                self.dec_entry.configure(state = 'disabled')
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == int:
            
                self.solver_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'int'
    
                # # Add label for factor names
                # self.factorname_label = tk.Label (master = self.solver_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                # self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.solver_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.solver_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.solver_frame, variable = self.checkstate,
                                               command = self.include_factor)
                self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
                
                # Add entry box for min val
                self.min_frame = tk.Frame (master = self.solver_frame)
                self.min_frame.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W )
                
                self.min_label = tk.Label(master = self.min_frame, text = 'Min Value', font = "Calibri 13", width = 10 )
                self.min_label.grid( row = 0, column = 0)
                self.min_val = tk.StringVar()
                self.min_entry = tk.Entry( master = self.min_frame, width = 10, textvariable = self.min_val, justify = 'right')
                self.min_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.min_list.append(self.min_val)    
                
                self.min_widgets[factor] = self.min_entry
                
                self.min_entry.configure(state = 'disabled')
                
                # Add entry box for max val
                self.max_frame = tk.Frame (master = self.solver_frame)
                self.max_frame.grid( row = self.factor_que_length, column = 5, sticky = tk.N + tk.W )
                
                self.max_label = tk.Label(master = self.max_frame, text = 'Max Value', font = "Calibri 13", width = 10 )
                self.max_label.grid( row = 0, column = 0) 
                
                self.max_val = tk.StringVar()
                self.max_entry = tk.Entry( master = self.max_frame, width = 10, textvariable = self.max_val, justify = 'right')
                self.max_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
               
                self.max_list.append(self.max_val)    
                
                self.max_widgets[factor] = self.max_entry
                
                self.max_entry.configure(state = 'disabled') 
                
                self.factor_que_length += 1
                
            elif self.factor_datatype == bool:
            
                self.solver_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'bool'
    
                # # Add label for factor names
                # self.factorname_label = tk.Label (master = self.solver_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                # self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.solver_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add option menu for true/false
                self.default_value = tk.StringVar() #Variable to store selected bool factor state
                self.default_value.set('TRUE') #Set default bool option
                self.bool_menu = ttk.OptionMenu(self.solver_frame, self.default_value, 'TRUE', 'TRUE', 'FALSE')
                self.bool_menu.grid( row = self.factor_que_length, column = 2, sticky = tk.N + tk.W)
                
                
                
                # Default value if not included in design
                self.default_values_list.append(self.default_value)
                
                    
                # Add checkbox 
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.solver_frame, variable = self.checkstate,
                                               command = self.include_factor)
                self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
                
                self.factor_que_length += 1
                
        # # Button for factor description
        # self.description_button = tk.Button(master = self.solver_frame, text = 'Factor Details', font = 'Calibri 11', width = 15, command = lambda: self.show_description_window(solver = self.solver_object ))
        # self.description_button.grid(row = self.factor_que_length + 1, column = 0)
        self.show_design_options() # run function to show design creation options
        # # self.bind_factor_descriptions()
    
    # def bind_factor_descriptions(self):
    #     for factor in self.description_buttons:
    #         print('factor', factor)
    #         self.description_buttons[factor].configure(command = lambda: self.show_description(factor))
            
        
        
        
    def show_design_options(self):
        # Design type selection menu
        self.design_frame.grid(row = 3, column = 0)
        self.design_frame.grid_rowconfigure(0, weight = 0)
        self.design_frame.grid_columnconfigure(0, weight = 1)
        self.design_frame.grid_columnconfigure(1, weight = 1)
        self.design_frame.grid_columnconfigure(2, weight = 1)
        self.design_frame.grid_columnconfigure(3, weight = 1)
        self.design_frame.grid_columnconfigure(4, weight = 1)
        
        # Input options from loaded designs
        if self.loaded_design == True:
            stack_display = self.n_stacks # same num of stacks as original loaded design
            design_display = self.design_type
        else:
            stack_display = 'Stacks'
            design_display = 'Design Type'
        
        self.design_type_label = tk.Label (master = self.design_frame, text = 'Select Design Type:', font = "Calibri 13", width = 20)
        self.design_type_label.grid( row = 0, column = 0)
        
        self.design_types_list = ['nolhs']
        self.design_var = tk.StringVar()
        self.design_type_menu = ttk.OptionMenu(self.design_frame, self.design_var, design_display, *self.design_types_list, command = self.enable_stacks)
        self.design_type_menu.grid(row = 0, column = 1)
        if self.loaded_design == True:
            self.design_type_menu.configure(state = 'disabled')
        
        #Stack selection menu
        self.stack_label = tk.Label (master = self.design_frame, text = "Select Number of Stacks:", font = "Calibri 13", width = 20)
        self.stack_label.grid( row =1, column = 0)
        self.stack_list = ['1', '2', '3']
        self.stack_var = tk.StringVar()
        self.stack_menu = ttk.OptionMenu(self.design_frame, self.stack_var, stack_display, *self.stack_list, command = self.enable_create_design_button)
        self.stack_menu.grid( row = 1, column = 1)
        self.stack_menu.configure(state = 'disabled')
        
        # Name of design file entry
        self.design_filename_label = tk.Label (master = self.design_frame, text = "Name of design:", font = "Calibri 13", width = 20)
        self.design_filename_label.grid( row = 0, column = 2)
        self.design_filename_var = tk.StringVar() # variable to hold user specification of desing file name
        #timestamp = time.strftime("%Y%m%d%H%M%S")
        self.design_filename_var.set(self.solver_name)
        self.design_filename_entry = tk.Entry(master = self.design_frame,  width = 40, textvariable = self.design_filename_var, justify = 'right' )
        self.design_filename_entry.grid( row = 0, column = 3)
        
        
        # Create design button 
        if self.loaded_design == False:
            self.create_design_button = tk.Button(master = self.design_frame, text = 'Create Design', font = "Calibri 13", command = self.create_design , width = 20)
            self.create_design_button.grid( row = 0, column = 4)
            
        # Modify design button for loaded designs
        if self.loaded_design == True:
            
            self.mod_design_button = tk.Button(master = self.design_frame, text = 'Modify Design', font = "Calibri 13", command = self.mod_design , width = 20)
            self.mod_design_button.grid( row = 0, column = 4)
            
            self.continue_design_button = tk.Button(master = self.design_frame, text = 'Continue w/o Changes', font = "Calibri 13", command = self.con_design , width = 25)
            self.continue_design_button.grid( row = 1, column = 4)
            
    def mod_design(self):
        
        self.default_values = [self.default_value.get() for self.default_value in self.default_values_list] # default value of each factor
        factor_index = 0
        for factor in self.default_factors:
            #self.default_values = [self.default_value.get() for self.default_value in self.default_values_list] # default value of each factor
            new_val = self.default_values[factor_index]
            self.design_table[factor] = new_val
            factor_index += 1
            
        self.design_filename = self.design_filename_var.get() # name of design file specified by user
        
        self.csv_filename = f'./data_farming_experiments/{self.design_filename}_design.csv'
        
        self.design_table.to_csv(self.csv_filename, index = False)

        self.experiment = DataFarmingMetaExperiment(csv_filename = self.csv_filename)
        self.select_problem()
        self.display_design_tree()
        
                                                    
        
        
        
    def con_design(self):
        self.experiment = DataFarmingMetaExperiment(csv_filename = self.csv_filename)
        self.select_problem()
                                                       
        
    
    
    def select_problem(self):
       
        # Problem selection frame
        self.problem_select_frame.grid( row = 5, column = 0)
        self.problem_select_frame.grid_rowconfigure(0, weight = 1)
        self.problem_select_frame.grid_columnconfigure(0, weight = 1)
        self.problem_select_frame.grid_columnconfigure(1, weight = 1)
        self.problem_select_frame.grid_columnconfigure(2, weight = 1)
        
        
        self.update_problem_list_compatability() #Check compatibility of solver problems, returns self.problem_list for option menu
        
        # Option menu to select problem
        self.problem_select_label = tk.Label( master = self.problem_select_frame, text = 'Select Problem:', width = 20,
                                    font = 'Calibir 13')
        self.problem_select_label.grid( row = 0, column = 0)
        
        self.problem_var = tk.StringVar() # Variable to store selected problem
          
        
        self.problem_select_menu = ttk.OptionMenu(self.problem_select_frame, self.problem_var, 'Problem', *self.problem_list, command = self.show_problem_factors)
        self.problem_select_menu.grid(row = 0, column = 1)   
        
        # Display model name
        self.model_name = ""
        self.model_label = tk.Label(master = self.problem_select_frame, text = 'Model: ' + self.model_name, font = "Calibri 13")
        self.model_label.grid(row = 0, column = 2)
        
      
    def enable_stacks(self, *args):
        if self.loaded_design == False: #cannot change stacks for loaded design
            self.stack_menu.configure(state = 'normal')
        
    def enable_create_design_button(self, *args):
        self.create_design_button.configure(state = 'normal')
        
        

    
    
    def create_design(self, *args):

        
        #Export design factors

        
        self.n_stacks = self.stack_var.get() # user input for num stacks
        self.design_type = self.design_var.get() #user input for design type
        
        # self.problem_fixed_factors = {} # holds fixed factors for problem to be used in design
        # self.model_fixed_factors ={} # holds fixed factors of model to be used in design
        self.fixed_factors = {} # holds fixed factors of solver to be used in design
        self.factor_index = 0
         
        
        # List to hold names of all factors part of model to be displayed in csv
        self.factor_names = [] # names of solver factors included in experiment
        self.def_factor_names = [] # names of default and cross design solver factors
        self.problem_factor_names = [] # names of all problem factors
        self.model_factor_names = [] # names of all model factors
        
 
      
        # Write solver factors file
        
        self.design_filename = self.design_filename_var.get() # name of design file specified by user
        
        with open(f"./data_farming_experiments/{self.design_filename}.txt", "w") as self.solver_design_factors:
            self.solver_design_factors.write("")
            
        
       
            
        if self.loaded_design == False: # get factor settings for created design (not loaded)
        
            # List used for parsing design file
            self.factor_headers = [] 
           
            # Lists that hold factor information set by user
            self.check_values = [self.checkstate.get() for self.checkstate in self.checkstate_list] # checkstate of each factor
            self.default_values = [self.default_value.get() for self.default_value in self.default_values_list] # default value of each factor
            self.min_values = [self.min_val.get() for self.min_val in self.min_list] # max value of each int & float factor
            self.max_values = [self.max_val.get() for self.max_val in self.max_list] # min value of each int & float factor
            self.dec_values = [self.dec_val.get() for self.dec_val in self.dec_list] # dec value of each float factor
            
            
            # values to index through factors
            self.factor_index = 0
            self.maxmin_index = 0
            self.dec_index = 0
            
            #Dictionary used for tree view display of fixed factors
            self.solver_fixed_str = {}
            self.def_factors = [] # list of factors not included in design
            
           
            # Get solver experiment information    
            for  factor in self.solver_object.specifications:
                
                self.factor_datatype = self.solver_object.specifications[factor].get("datatype")
               
                self.factor_default = self.default_values[self.factor_index]
                self.factor_include = self.check_values[self.factor_index]
                
                
               
               
                if self.factor_include == True:
                    
                    # # Add factor to list of design factors in order that will be varied
                    # self.factor_headers.append(factor)
                    # Factor names in csv are unedited
                    # self.factor_names.append(factor)
                
                    if self.factor_datatype == float or self.factor_datatype == int:
                        # Add factor to list of design factors in order that will be varied
                        self.factor_headers.append(factor)
                        # Factor names in csv are unedited
                        self.factor_names.append(factor)
                        
                        self.factor_min = str(self.min_values[self.maxmin_index])
                        self.factor_max = str(self.max_values[self.maxmin_index])
                        self.maxmin_index += 1
                        
                        if self.factor_datatype == float:
                            self.factor_dec = str(self.dec_values[self.dec_index])
                            self.dec_index += 1
                            
                        elif self.factor_datatype == int:
                            self.factor_dec = '0'
                            
                        self.data_insert = self.factor_min + ' ' + self.factor_max + ' ' + self.factor_dec
                        with open(f"./data_farming_experiments/{self.design_filename}.txt", "a") as self.solver_design_factors:
                            self.solver_design_factors.write(self.data_insert + '\n')  
                            
                    elif self.factor_datatype == bool: 
                            
                            factor_options = [True,False] # list of values factor can take, temp hard coded for true/false
                            self.cross_design_factors[factor] = factor_options #add factor to cross design dictionary
                            display = ""
                            for opt in factor_options:
                                opt_str = str(opt)
                                display += opt_str + '/'
                            self.solver_fixed_str[factor] = display[:-1] # add cross design factor to fixed str for design table
                            self.def_factor_names.append(factor + ' (cross)') # list of factor names in cross design
                       
                
                # Include fixed default values in design
                if self.factor_include == False:
                    
                    # Factor names in csv have "(default)" appended to end
                    self.def_factors.append(factor)
                    self.def_factor_names.append(factor + ' (fixed)')
                    
                    # Values to be placed in tree view of design
                    self.solver_fixed_str[factor] = self.factor_default
                   
                    if self.factor_datatype == float or self.factor_datatype == int:
                        self.maxmin_index += 1
                   
                    # Add default values to exeriment and set to correct datatype
                    if self.factor_datatype == float:
                        self.fixed_factors[factor] = float(self.factor_default)
                        self.dec_index += 1
                        
                    elif self.factor_datatype == int:
                        self.fixed_factors[factor] = int(self.factor_default)
                        
                # bool values currently not able to be included in design
                if self.factor_datatype == bool:
                    if self.factor_default == 'TRUE':
                        bool_val = bool(1)
                    else:
                        bool_val = bool(0) 
                    self.fixed_factors[factor] = bool_val
                    
       
                self.factor_index += 1
                
            # Create design csv file with headers
            self.all_factor_headers = self.factor_names +  self.def_factor_names # combine factor names
                
            # Create solver factor design from .txt file of factor settings.
            # Hard-coded for NOLHS.
            self.design_filename = self.design_filename_var.get() # name of design file specified by user
            
            #self.filename = 'solver_factors' # base for all design file names, temp, turn into ask dialog
            
            if self.loaded_design == False:
                self.csv_filename = f'./data_farming_experiments/{self.design_filename}_design.csv' # used to display design tree
            
            self.experiment = DataFarmingMetaExperiment(solver_name = self.solver_name,
                                                           solver_factor_headers = self.factor_names,
                                                           n_stacks = self.n_stacks,
                                                           design_type = self.design_type,
                                                           solver_factor_settings_filename = self.design_filename,
                                                           design_filename= None,
                                                           solver_fixed_factors = self.fixed_factors,
                                                           cross_design_factors = self.cross_design_factors,
                                                           csv_filename = None
                                                           )
            
           
                       
 
        self.display_design_tree() # show created design
        self.select_problem() # show problem selection menu 
       
      
        
    # Used to display the design tree for both created and loaded designs
    def display_design_tree(self):
      
        #Initialize design tree
        self.design_view_frame.grid( row = 4, column = 0, padx = 10)
        self.design_view_frame.grid_rowconfigure( 0, weight = 0)
        self.design_view_frame.grid_rowconfigure( 1, weight = 1)
        self.design_view_frame.grid_columnconfigure( 0, weight = 1)
        self.design_view_frame.grid_columnconfigure( 1, weight = 1)
        
      
   
        self.design_tree = ttk.Treeview( master = self.design_view_frame)
        self.design_tree.grid(row = 1, column = 0, sticky = 'nsew')
      
       
        
        self.style = ttk.Style()
        self.style.configure("Treeview.Heading", font=('Calibri', 13, 'bold'))
        self.style.configure("Treeview", foreground="black", font = ('Calibri', 13))
        self.design_tree.heading( '#0', text = 'Design #' )
        
        # Get design point values from csv
        design_table = pd.read_csv(self.csv_filename, index_col="Design #")
        num_dp = len(design_table) #used for label
        self.create_design_label = tk.Label( master = self.design_view_frame, text = f'Total Design Points: {num_dp}', font = "Calibri 13 bold", width = 50)
        self.create_design_label.grid(row = 0, column = 0, sticky = tk.W)
        
        # Enter design values into treeview
        self.design_tree['columns'] = tuple(design_table.columns)[:-3]
       
        
        
        
        for column in design_table.columns[:-3]:
            self.design_tree.heading( column, text = column)
            self.design_tree.column(column, width = 100)
            
        for  index, row in design_table.iterrows():
            print('row', row)
            
            self.design_tree.insert("", index, text = index, values = tuple(row)[:-3])
        
  
       
      
            # Create a horizontal scrollbar
        xscrollbar = ttk.Scrollbar(master = self.design_view_frame, orient="horizontal", command= self.design_tree.xview)
        xscrollbar.grid(row = 2, column = 0, sticky = 'nsew')
        
        # Configure the Treeview to use the horizontal scrollbar
        self.design_tree.configure(xscrollcommand=xscrollbar.set) 
    def show_problem_factors(self, *args):
        
        # self.show_experiment_options() # show options for experiment creation
        
        #Initalize frames
        self.problem_model_factors_frame.grid(row = 6, column = 0)
        self.problem_model_factors_frame.grid_rowconfigure(0, weight = 1)
        self.problem_model_factors_frame.grid_columnconfigure(0, weight = 1)
        self.problem_model_factors_frame.grid_columnconfigure(1, weight = 1)
        self.problem_model_factors_frame.grid_columnconfigure(2, weight = 1)
        
        self.create_pair_frame.grid( row = 7, column = 0)
        self.create_pair_frame.grid_rowconfigure(0, weight = 1)
        self.create_pair_frame.grid_columnconfigure(0, weight = 1)
        self.create_pair_frame.grid_columnconfigure(1, weight = 1)
        self.create_pair_frame.grid_columnconfigure(2, weight = 1)
       
        
        #Clear previous selections
        self.clear_frame(self.problem_model_factors_frame) # clear problem-model factor selections
        self.clear_frame(self.create_pair_frame) # clear create pair buttons
        self.model_label.destroy() #remove old model label
        
            
        self.problem_frame = tk.Frame(master = self.problem_model_factors_frame)
        self.model_frame = tk.Frame(master = self.problem_model_factors_frame)
         
         
        self.problem_frame.grid(row = 1, column = 0)
        self.problem_frame.grid_rowconfigure(0, weight = 0)
        self.problem_frame.grid_columnconfigure(0, weight = 1)
        self.problem_frame.grid_columnconfigure(1, weight = 1)
        self.problem_frame.grid_columnconfigure(2, weight = 1)
        self.problem_frame.grid_columnconfigure(3, weight = 1)
        self.problem_frame.grid_columnconfigure(4, weight = 1)
        self.problem_frame.grid_columnconfigure(5, weight = 1)
        self.problem_frame.grid_columnconfigure(6, weight = 1)
         
        self.model_frame.grid(row = 0, column = 1 )
        self.model_frame.grid_rowconfigure(0, weight = 0)
        self.model_frame.grid_columnconfigure(0, weight = 1)
        self.model_frame.grid_columnconfigure(1, weight = 1)
        self.model_frame.grid_columnconfigure(2, weight = 1)
        
        
        


        # Create column for problem factor names
        self.headername_label = tk.Label(master = self.problem_model_factors_frame, text = 'Problem Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W)
        
        # Create column for problem factor type
        self.headertype_label = tk.Label(master = self.problem_model_factors_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 15, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 1, sticky = tk.N + tk.W)
        
        
        # Create column for problem factor default values
        self.headerdefault_label = tk.Label(master = self.problem_model_factors_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 2, sticky = tk.N + tk.W)
        
        # Create column for model factor names
        self.headername_label = tk.Label(master = self.problem_model_factors_frame, text = 'Model Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 3, sticky = tk.N + tk.W)
        
        # Create column for model factor type
        self.headertype_label = tk.Label(master = self.problem_model_factors_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 20, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 4, sticky = tk.N + tk.W)
        
        
        # Create column for model factor default values
        self.headerdefault_label = tk.Label(master = self.problem_model_factors_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 5, sticky = tk.N + tk.W)
        
      
        entry_width = 10
        
        # Widget lists
        self.default_widgets = {}
        
        # Initial variable values
        self.factor_que_length = 1
        self.problem_default_values_list = []
        self.model_default_values_list = []
        
        # Get problem info from dictionary
        self.selected_problem = self.problem_var.get()
        self.problem_object = problem_unabbreviated_directory[self.selected_problem]()
        self.problem_name = self.problem_object.name # name of problem used for save files
   
        # Get model info from dictonary
        self.model_problem_dict = model_problem_class_directory # directory that relates problem name to model class
        self.model_object = self.model_problem_dict[self.selected_problem]()
        self.model_name = self.model_object.name # name of model that relates to problem 
        
              
        # Display model name
        self.model_label = tk.Label(master = self.problem_select_frame, text = 'Model: ' + self.model_name, font = "Calibri 13")
        self.model_label.grid(row = 0, column = 2)
        
        
        
        for  factor in self.problem_object.specifications:
            
            self.factor_datatype = self.problem_object.specifications[factor].get("datatype")
            self.factor_description = self.problem_object.specifications[factor].get("description")
            self.factor_default = self.problem_object.specifications[factor].get("default")
            
            self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
        
            
            
            if self.factor_datatype == float:
            
                self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'float'
    
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
               
                
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.problem_default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
               
                self.factor_que_length += 1
            
            elif self.factor_datatype == int:
            
                self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'int'
    
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.problem_default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == list:
                
                self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'list'
               
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                #Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.problem_default_values_list.append(self.default_value)
                
            
                self.factor_que_length += 1
                
            elif self.factor_datatype == tuple:
                
                self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'tuple'
               
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.problem_default_values_list.append(self.default_value)
                
                self.factor_que_length += 1
                
        self.factor_que_length = 1
                
        for  factor in self.model_object.specifications:
            
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
            self.factor_description = self.model_object.specifications[factor].get("description")
            self.factor_default = self.model_object.specifications[factor].get("default")
            
            
            #Values to help with formatting
            entry_width = 10
            
            self.model_frame.grid_rowconfigure(self.factor_que_length, weight =1)
        
            
            
            if self.factor_datatype == float:
                
                self.str_type = 'float'
    
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 3, sticky = tk.N + tk.W)

                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =5, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.model_default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
          
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == int:
            
                self.model_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'int'
    
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 3, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =5, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.model_default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == list:
                
                self.model_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'list'
               
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 3, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W)
                
                #Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =5, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.model_default_values_list.append(self.default_value)
                
            
                self.factor_que_length += 1
                
            elif self.factor_datatype == tuple:
                
                self.model_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'tuple'
               
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 3, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =5, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.model_default_values_list.append(self.default_value)
                
               
                self.factor_que_length += 1  
                
        # Save pair name
        self.pair_name_label = tk.Label(master = self.create_pair_frame, text = 'Save pair as:' , width = 20, font = 'Calibri 13')
        self.pair_name_label.grid(row = 0, column = 0)
        self.pair_name_var = tk.StringVar() #store name of design-problem pair
        self.pair_name_var.set(self.solver_name + '_' + self.problem_name)
        self.pair_name_entry = tk.Entry( master = self.create_pair_frame, width = 20, textvariable = self.pair_name_var, justify = 'right')
        self.pair_name_entry.grid(row = 0, column = 1)
        
        # Create design-problem pair button
        self.add_pair_button = tk.Button(master = self.create_pair_frame, text = 'Add Design-Problem Pair', font = 'Calibri 13',
                                         width = 20, command = self.add_pair)
        self.add_pair_button.grid(row = 0, column = 2)
        
       
        
        
        
                
    def add_pair(self):
        
        # Experiment frames
        self.experiment_frame.grid( row = 8, column = 0)
        self.experiment_frame.grid_rowconfigure(0, weight = 1)
        self.experiment_frame.grid_columnconfigure(0, weight = 1)
        self.experiment_frame.grid_columnconfigure(1, weight = 1)
        self.experiment_frame.grid_columnconfigure(2, weight = 1)
        self.experiment_frame.grid_columnconfigure(3, weight = 1)
        self.experiment_frame.grid_columnconfigure(4, weight = 1)
        self.experiment_frame.grid_columnconfigure(5, weight = 1)
        self.experiment_frame.grid_columnconfigure(6, weight = 1)
        self.experiment_frame.grid_columnconfigure(7, weight = 1)
        self.experiment_frame.grid_columnconfigure(8, weight = 1)
        self.experiment_frame.grid_columnconfigure(9, weight = 1)
        self.experiment_frame.grid_columnconfigure(10, weight = 1)
        self.experiment_frame.grid_columnconfigure(11, weight = 1)
        self.experiment_frame.grid_columnconfigure(12, weight = 1)
            
        
        if len(self.experiment_pairs) == 0: # for first added pair create header
            
            self.pair_name_header = tk.Label( master = self.experiment_frame, text = 'Generated Pairs', width = 20, font = 'Calibri 13 bold')
            self.pair_name_header.grid(row = 0, column = 1)
            
            self.mac_rep_label = tk.Label(master = self.experiment_frame, text = '# Macro Replications', font = "Calibri 13 bold", width = 20)
            self.mac_rep_label.grid(row = 0, column = 2)
            
            self.post_rep_label = tk.Label(master = self.experiment_frame, text = '# Post Replications', font = "Calibri 13 bold", width = 20)
            self.post_rep_label.grid(row = 0, column = 4)
            
            self.crn_budget_label = tk.Label(master = self.experiment_frame, text = 'CRN Across Budget?', font = "Calibri 13 bold", width = 20)
            self.crn_budget_label.grid(row = 0, column = 5)
            
            self.crn_macro_label = tk.Label(master = self.experiment_frame, text = 'CRN Across Macro?', font = "Calibri 13 bold", width = 20)
            self.crn_macro_label.grid(row = 0, column = 6)
            
            self.norm_rep_label = tk.Label(master = self.experiment_frame, text = '# Normalization Replications', font = "Calibri 13 bold", width = 0)
            self.norm_rep_label.grid(row = 0, column = 8)
            
            self.crn_macro_label = tk.Label(master = self.experiment_frame, text = 'CRN Across Initial Option?', font = "Calibri 13 bold", width = 25)
            self.crn_macro_label.grid(row = 0, column = 9)
            

        self.problem_default_values = [self.default_value.get() for self.default_value in self.problem_default_values_list]
        self.model_default_values = [self.default_value.get() for self.default_value in self.model_default_values_list]
        
        self.problem_fixed_factors = {} # holds fixed factors for problem to be used in design
        self.model_fixed_factors ={} # holds fixed factors of model to be used in design
        self.factor_index = 0
         
        
      
        
        # Get problem default factor information    
        for  factor in self.problem_object.specifications:
            
            self.factor_datatype = self.problem_object.specifications[factor].get("datatype")
      
            self.factor_default = self.problem_default_values[self.factor_index]
            print("default", self.factor_default)
            # self.problem_factor_names.append( factor + ' (fixed)')
            
            # # Values to be placed in tree view of design
            # self.problem_fixed_str[factor] = self.factor_default
            
          
           
            # Add default values to exeriment and set to correct datatype
            if self.factor_datatype == float:
                self.problem_fixed_factors[factor] = float(self.factor_default)
                
            if self.factor_datatype == int:
                self.problem_fixed_factors[factor] = int(self.factor_default)
                
            if self.factor_datatype == list:
                self.problem_fixed_factors[factor] = ast.literal_eval(self.factor_default)
                
            if self.factor_datatype == tuple:
                last_val = self.factor_default[-2]
                tuple_str = self.factor_default[1:-1].split(",")
                print('tuple string', )
                # determine if last tuple value is empty
                if last_val != ",":
                    self.problem_fixed_factors[factor] = tuple(float(s) for s in tuple_str)
                else: 
                    tuple_exclude_last = tuple_str[:-1]
                    float_tuple = [float(s) for s in tuple_exclude_last]
                    self.problem_fixed_factors[factor] = tuple(float_tuple)
                    print( 'final tuple', tuple(float_tuple) )
                    
   
            self.factor_index += 1
        
        # Get model default factor information
        self.factor_index = 0
        for  factor in self.model_object.specifications:
            
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
      
            self.factor_default = self.model_default_values[self.factor_index]
            
            # self.model_factor_names.append( factor + ' (fixed)')
            
            # # Values to be placed in tree view of design
            # self.model_fixed_str[factor] = self.factor_default
         
            # Add default values to exeriment and set to correct datatype
            if self.factor_datatype == float:
                self.model_fixed_factors[factor] = float(self.factor_default)
               
            if self.factor_datatype == int:
                self.model_fixed_factors[factor] = int(self.factor_default)
                
            if self.factor_datatype == list:
                self.model_fixed_factors[factor] = ast.literal_eval(self.factor_default)
            
            if self.factor_datatype == tuple:
                   
                tuple_str = tuple(self.factor_default[1:-1].split(","))
                self.model_fixed_factors[factor] = tuple(float(s) for s in tuple_str)
       
            self.factor_index += 1
        
        # #Specify the name of the solver as it appears in directory.py
        # # solver_name = "RNDSRCH"
        # solver_name = self.solver_object.name # name of solver selected
       

        # # Specify the names of the model factors (in order) that will be varied.
        # # solver_factor_headers = ["sample_size"]
        # solver_factor_headers = self.factor_names 
        
        

        # # Specify the name of the problem as it appears in directory.py
        # # problem_name = "FACSIZE-2"
        #problem_name = self.problem_object.name
        
        # cross_design_factors = self.cross_design_factors # factors included in cross design

        
        # solver_factor_settings_filename = None

      
        # design_filename = 'solver_factors_design'

        # OPTIONAL: Provide additional overrides for default solver/problem/model factors.
        # If empty, default factor settings are used.
        # solver_fixed_factors = self.fixed_factors
        # print('solver_fixed_factors', self.fixed_factors)
        # problem_fixed_factors = self.problem_fixed_factors
        # print('problem fixed factors', problem_fixed_factors)
        # model_fixed_factors = self.model_fixed_factors
        # print('model fixed factors', model_fixed_factors)
   

        # No code beyond this point needs to be edited.

        # Create DataFarmingExperiment object.
        # self.myDFMetaExperiment = DataFarmingMetaExperiment(solver_name=solver_name,
        #                                                problem_name=problem_name,
        #                                                solver_factor_headers=solver_factor_headers,
        #                                                solver_factor_settings_filename=solver_factor_settings_filename,
        #                                                design_filename=design_filename,
        #                                                solver_fixed_factors=solver_fixed_factors,
        #                                                problem_fixed_factors=problem_fixed_factors,
        #                                                model_fixed_factors=model_fixed_factors,
        #                                                cross_design_factors = cross_design_factors
        #                                                )
        pair_name = self.pair_name_var.get() #name of problem pair
        self.experiment_pairs[pair_name] = [self.experiment, self.problem_object.name, self.problem_fixed_factors, self.model_fixed_factors] # Add experiment to dictionary by pair name
        
        self.show_experiment_options(pair_name = pair_name)


        
    def show_experiment_options(self, pair_name):
        # Show experiment options
        current_row = len(self.experiment_pairs)
        self.experiment_frame.grid_rowconfigure(current_row, weight = 1)

       
        
        # Pair name label
        self.pair_name_label = tk.Label(master = self.experiment_frame, text = pair_name, font = 'Calibri 11', width = 20)
        self.pair_name_label.grid( row = current_row, column = 1)
        
       # number macro replications
        self.mac_rep_var = tk.StringVar() # variable to store number of macro replication of experiment
        self.macro_rep_vars[pair_name] = self.mac_rep_var # store to macro list for all experiments
        
        self.mac_rep_entry = tk.Entry(master = self.experiment_frame, width = 10, textvariable = self.mac_rep_var, justify = 'right')
        self.mac_rep_entry.grid(row = current_row, column = 2)
        
        
        # Run experiment button
        self.run_exp_button = tk.Button( master = self.experiment_frame, width = 10, text = 'Run', font = "Calibri 11", command = lambda: self.run_experiment(pair_name = pair_name))
        self.run_exp_button.grid( row = current_row, column = 3)
        self.run_buttons[pair_name] = self.run_exp_button # store to run button list for all experiments
        
        # Number post replications
        self.post_rep_var = tk.StringVar() # variable to store number of post replications of experiment
        self.post_rep_vars[pair_name] = self.post_rep_var # store to post rep list for all experiments
        
        self.post_rep_entry = tk.Entry(master = self.experiment_frame, width = 10, textvariable = self.post_rep_var, justify = 'right')
        self.post_rep_entry.grid(row = current_row, column = 4) 
        
        # CRN across budget for post rep
        crn_budget_var = tk.StringVar() # variable to hold true/false state of crn across budget
        self.crn_budget_menu = ttk.OptionMenu(self.experiment_frame, crn_budget_var, 'True', *['True', 'False'])
        self.crn_budget_menu.grid(row = current_row, column = 5)
        
        # CRN across macro for post rep
        crn_macro_var = tk.StringVar() # variable to hold true/false state of crn across macro reps
        self.crn_macro_menu = ttk.OptionMenu(self.experiment_frame, crn_macro_var, 'False', *['True', 'False'])
        self.crn_macro_menu.grid(row = current_row, column = 6)
        
        # Post process button experiment button
        self.post_button = tk.Button( master = self.experiment_frame, width = 10, text = 'Post-Process', font = "Calibri 11", command = lambda: self.post_replicate(pair_name =pair_name, 
                                                                                                                                                                    crn_across_budget=crn_budget_var.get(),
                                                                                                                                                                    crn_across_macro=crn_macro_var.get()))
        self.post_button.grid( row = current_row, column = 7)
        self.post_buttons[pair_name] = self.post_button # store to post button list for all experiments
        self.post_button.configure( state = 'disabled')
        
        
        # Number normalization replications
       
        self.norm_rep_var = tk.StringVar() # variable to store number of normalization replications of experiment
        self.norm_rep_vars[pair_name] = self.norm_rep_var # store to norm rep list for all experiments
        
        self.norm_rep_entry = tk.Entry(master = self.experiment_frame, width = 10, textvariable = self.norm_rep_var, justify = 'right')
        self.norm_rep_entry.grid(row = current_row, column = 8) 
        
        # CRN across init option
        crn_init_var = tk.StringVar() # variable to hold true/false state of crn across initial option
        self.crn_init_menu = ttk.OptionMenu(self.experiment_frame, crn_init_var, 'True', *['True', 'False'])
        self.crn_init_menu.grid(row = current_row, column = 9)
        
        
        # Normalize experiment button
        self.norm_button = tk.Button( master = self.experiment_frame, width = 10, text = 'Normalize', font = "Calibri 11", command = lambda: self.post_normalize(pair_name = pair_name,
                                                                                                                                                                 crn_across_init = crn_init_var.get()))
        self.norm_button.grid( row = current_row, column = 10)
        self.norm_buttons[pair_name] = self.norm_button # store to norm button list for all experiments
        self.norm_button.configure( state = 'disabled')
        
        # Save experiment button
        self.save_button = tk.Button( master = self.experiment_frame, width = 10, text = 'Save', font = "Calibri 11", command = lambda: self.save_results(pair_name = pair_name))
                                                                                                                                           
        self.save_button.grid( row = current_row, column = 11)
        self.save_buttons[pair_name] = self.save_button # store to save button list for all experiments
        self.save_button.configure( state = 'disabled')



    def run_experiment(self, pair_name):
        
        
        current_exp = self.experiment_pairs[pair_name][0] # experiment is first element in stored list
        n_macroreps = int(self.macro_rep_vars[pair_name].get())
        
        problem_name = self.experiment_pairs[pair_name][1]
        problem_fixed_factors = self.experiment_pairs[pair_name][2]
        model_fixed_factors = self.experiment_pairs[pair_name][3]


        # Run macroreplications at each design point.
        current_exp.run(n_macroreps=n_macroreps, 
                        problem_name=problem_name, 
                        problem_fixed_factors=problem_fixed_factors, 
                        model_fixed_factors=model_fixed_factors 
                        )

      
    
        self.run_buttons[pair_name].configure( bg = 'dark gray') # change color of button
        
        self.post_buttons[pair_name].configure(state = 'normal')
        self.norm_buttons[pair_name].configure( state = 'disabled')
        self.save_buttons[pair_name].configure( state = 'disabled')
    
    
   
    def post_replicate(self, pair_name, crn_across_budget, crn_across_macro):
        
        
        current_exp = self.experiment_pairs[pair_name][0]
       
        # Specify the number of postreplications to take at each recommended solution
        # from each macroreplication at each design point.
        # self.postreps = self.post_rep_var.get() # number of post reps specified by user (string)
        n_postreps = int(self.post_rep_vars[pair_name].get())
        # n_postreps = int(self.postreps)
        
        # Specify the CRN control for postreplications.   
        if crn_across_budget == 'True':
            crn_across_budget = True 
        else:
            crn_across_budget = False
            
        if crn_across_macro == 'True':
            crn_across_macroreps = True 
        else:
            crn_across_macroreps = False 
        
        
        # Postprocess the experimental results from each design point.
        current_exp.post_replicate(n_postreps=n_postreps,
                                          crn_across_budget=crn_across_budget,
                                          crn_across_macroreps=crn_across_macroreps
                                          )
        
        self.post_buttons[pair_name].configure( bg = 'dark gray') # change color of button
        
        self.norm_buttons[pair_name].configure(state = 'normal')
        self.save_buttons[pair_name].configure( state = 'disabled')
        
        
   
    def post_normalize(self, pair_name, crn_across_init):
 
        
        current_exp = self.experiment_pairs[pair_name][0]
        #Determine CRN
        if crn_across_init == 'True':
            crn_across_init_opt = True 
        else:
            crn_across_init_opt = False
        
        # Specify the number of postreplications to take at x0 and x*.
        n_postreps_init_opt = int(self.norm_rep_vars[pair_name].get())
        # self.normreps = self.norm_rep_var.get()
        # n_postreps_init_opt = int(self.normreps)
        
        current_exp.post_normalize(n_postreps_init_opt=n_postreps_init_opt,
                                          crn_across_init_opt=crn_across_init_opt
                                          )
        
        self.norm_buttons[pair_name].configure( bg = 'dark gray') # change color of button
        
        self.save_buttons[pair_name].configure( state = 'normal')
        

    def save_results(self, pair_name):
        
        
        current_exp = self.experiment_pairs[pair_name][0]
        
        # Save experiment results file name
        export_csv_filename =  filedialog.asksaveasfilename(initialfile = pair_name + "_datafarming_experiment" )
        
        # Compute the performance metrics at each design point and print to csv.
        current_exp.report_statistics(solve_tols=[0.05, 0.10, 0.20, 0.50], csv_filename = export_csv_filename)
        
        self.save_buttons[pair_name].configure( bg = 'dark gray') # change color of button
        
        
    def update_problem_list_compatability(self):
        
        temp_problem_list = []
        temp_solver_name = self.solver_name
        
        for problem in problem_unabbreviated_directory:

            temp_problem = problem_unabbreviated_directory[problem] # problem object
            temp_problem_name = temp_problem().name

            temp_experiment = ProblemSolver(solver_name=temp_solver_name, problem_name=temp_problem_name)
            comp = temp_experiment.check_compatibility()

            if comp == "":
                temp_problem_list.append(problem)

        # from experiments.inputs.all_factors.py:
        self.problem_list = temp_problem_list # list of problems used for option menu 

           
    
    def include_factor(self, *args):

        self.check_values = [self.checkstate.get() for self.checkstate in self.checkstate_list]
        self.check_index = 0
        self.cat_index = 0
        self.cross_design_factors = {} # Dictionary to hold cross design factors and lists containing possible factor values
    
        # If checkbox to include in experiment checked, enable experiment option buttons
        for factor in self.solver_object.specifications:
                  
            # Get current checksate values from include experiment column
            self.current_checkstate = self.check_values[self.check_index]
            # Cross check factor type
            self.factor_datatype = self.solver_object.specifications[factor].get("datatype")
            
            # Disable / enable experiment option widgets depending on factor type
            if self.factor_datatype == float or self.factor_datatype == int:
                self.current_min_entry = self.min_widgets[factor]
                self.current_max_entry = self.max_widgets[factor]               
                
                             
                if self.current_checkstate == True:
                    self.current_min_entry.configure(state = 'normal')
                    self.current_max_entry.configure(state = 'normal')
                    
                elif self.current_checkstate == False:
                    #Empty current entries
                    self.current_min_entry.delete(0, tk.END)
                    self.current_max_entry.delete(0, tk.END)
                   
                    
                    self.current_min_entry.configure(state = 'disabled')
                    self.current_max_entry.configure(state = 'disabled')
                                      
            if self.factor_datatype == float:              
                self.current_dec_entry = self.dec_widgets[factor]
                
                if self.current_checkstate == True:
                    self.current_dec_entry.configure(state = 'normal')
                    
                elif self.current_checkstate == False:
                    self.current_dec_entry.delete(0, tk.END)
                    self.current_dec_entry.configure(state = 'disabled')
                    
          
            self.check_index += 1                     
    
            
            
        
        
    
# My code ends here

class Cross_Design_Window():

    def __init__(self, master, main_widow, forced_creation = False):
        if not forced_creation:
            self.master = master
            self.main_window = main_widow

            self.crossdesign_title_label = tk.Label(master=self.master,
                                                    text = "Create Cross-Design Problem-Solver Group",
                                                    font = "Calibri 13 bold")
            self.crossdesign_title_label.place(x=10, y=25)

            self.crossdesign_problem_label = tk.Label(master=self.master,
                                                        text = "Select Problems:",
                                                        font = "Calibri 13")
            self.crossdesign_problem_label.place(x=190, y=55)

            self.crossdesign_solver_label = tk.Label(master=self.master,
                                                        text = "Select Solvers:",
                                                        font = "Calibri 13")
            self.crossdesign_solver_label.place(x=10, y=55)

            self.crossdesign_checkbox_problem_list = []
            self.crossdesign_checkbox_problem_names = []
            self.crossdesign_checkbox_solver_list = []
            self.crossdesign_checkbox_solver_names = []

            solver_cnt = 0
            
            for solver in solver_unabbreviated_directory:
                self.crossdesign_solver_checkbox_var = tk.BooleanVar(self.master, value=False)
                self.crossdesign_solver_checkbox = tk.Checkbutton(master=self.master,
                                                                text = solver,
                                                                variable = self.crossdesign_solver_checkbox_var)
                self.crossdesign_solver_checkbox.place(x=10, y=85+(25*solver_cnt))

                self.crossdesign_checkbox_solver_list.append(self.crossdesign_solver_checkbox_var)
                self.crossdesign_checkbox_solver_names.append(solver)

                solver_cnt += 1

            problem_cnt = 0
            for problem in problem_unabbreviated_directory:
                self.crossdesign_problem_checkbox_var = tk.BooleanVar(self.master, value=False)
                self.crossdesign_problem_checkbox = tk.Checkbutton(master=self.master,
                                                    text = problem,
                                                    variable = self.crossdesign_problem_checkbox_var)
                self.crossdesign_problem_checkbox.place(x=190, y=85+(25*problem_cnt))

                self.crossdesign_checkbox_problem_list.append(self.crossdesign_problem_checkbox_var)
                self.crossdesign_checkbox_problem_names.append(problem)

                problem_cnt += 1

            

            if problem_cnt < solver_cnt:
                solver_cnt += 1
                self.crossdesign_macro_label = tk.Label(master=self.master,
                                                        text = "Number of Macroreplications:",
                                                        font = "Calibri 13")
                self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

                self.crossdesign_macro_var = tk.StringVar(self.master)
                self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT, width=15)
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")
                self.crossdesign_macro_entry.place(x=15, y=105+(25*solver_cnt))

                self.crossdesign_button = ttk.Button(master=self.master,
                                                text = "Add Cross-Design Problem-Solver Group",
                                                width = 65,
                                                command = self.confirm_cross_design_function)
                self.crossdesign_button.place(x=15, y=135+(25*solver_cnt))

            if problem_cnt > solver_cnt:
                problem_cnt += 1

                self.crossdesign_macro_label = tk.Label(master=self.master,
                                                        text = "Number of Macroreplications:",
                                                        font = "Calibri 13")
                self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

                self.crossdesign_macro_var = tk.StringVar(self.master)
                self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT, width=15)
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")

                self.crossdesign_macro_entry.place(x=15, y=105+(25*problem_cnt))

                self.crossdesign_button = ttk.Button(master=self.master,
                                                text = "Add Cross-Design Problem-Solver Group",
                                                width = 45,
                                                command = self.confirm_cross_design_function)
                self.crossdesign_button.place(x=15, y=135+(25*problem_cnt))

            if problem_cnt == solver_cnt:
                problem_cnt += 1

                self.crossdesign_macro_label = tk.Label(master=self.master,
                                                        text = "Number of Macroreplications:",
                                                        font = "Calibri 13")
                self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

                self.crossdesign_macro_var = tk.StringVar(self.master)
                self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT, width=15)
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")
                self.crossdesign_macro_entry.place(x=15, y=105+(25*problem_cnt))

                self.crossdesign_button = ttk.Button(master=self.master,
                                                text = "Add Cross-Design Problem-Solver Group",
                                                width = 30,
                                                command = self.confirm_cross_design_function)
                self.crossdesign_button.place(x=15, y=135+(25*problem_cnt))
            else:
                # print("forced creation of cross design window class")
                pass

    def confirm_cross_design_function(self):
        solver_names_list = list(solver_directory.keys())
        problem_names_list = list(problem_directory.keys())
        problem_list = []
        solver_list = []

        for checkbox in self.crossdesign_checkbox_solver_list:
            if checkbox.get() == True:
                #(self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)] + " was selected (solver)")
                #solver_list.append(solver_directory[self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)]])
                solver_list.append(solver_names_list[self.crossdesign_checkbox_solver_list.index(checkbox)])
                
        for checkbox in self.crossdesign_checkbox_problem_list:
            if checkbox.get() == True:
                #(self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)] + " was selected (problem)")
                #problem_list.append(problem_directory[self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)]])
                problem_list.append(problem_names_list[self.crossdesign_checkbox_problem_list.index(checkbox)])

        

        # Solver can handle upto deterministic constraints, but problem has stochastic constraints.
        stochastic = ["FACSIZE-1","FACSIZE-2","RMITD-1"]
        if len(solver_list) == 0 or len(problem_list) == 0:
            self.crossdesign_warning = tk.Label(master=self.master,
                                                text = "Select at least one solver and one problem",
                                                font = "Calibri 13 bold",
                                                wraplength=300)
            self.crossdesign_warning.place(x=10, y=345)
            return

        if "ASTRODF" in solver_list and any(item in stochastic for item in problem_list) :
            self.crossdesign_warning = tk.Label(master=self.master,
                                                text = "ASTRODF can handle upto deterministic constraints, but problem has stochastic constraints",
                                                font = "Calibri 13 bold",
                                                wraplength=300)
            self.crossdesign_warning.place(x=10, y=345)
            return
        # macro_reps = self.crossdesign_macro_var.get()
        #(solver_list, problem_list)
        # self.crossdesign_ProblemsSolvers = ProblemsSolvers(solver_names=solver_list, problem_names=problem_list, fixed_factors_filename="all_factors")
        self.crossdesign_MetaExperiment = ProblemsSolvers(solver_names=solver_list, problem_names=problem_list)

        # if self.count_meta_experiment_queue == 0:
        #     self.create_meta_exp_frame()
        self.master.destroy()
        Experiment_Window.add_meta_exp_to_frame(self.main_window, self.crossdesign_macro_var)

        return self.crossdesign_MetaExperiment

        #(self.crossdesign_MetaExperiment)

    def test_function(self, *args):
        # print("test function connected")
        pass

    def get_crossdesign_MetaExperiment(self):
        return self.crossdesign_MetaExperiment

class Post_Processing_Window():
    """
    Postprocessing Page of the GUI

    Arguments
    ----------
    master : tk.Tk
        Tkinter window created from Experiment_Window.run_single_function
    myexperiment : object(Experiment)
        Experiment object created in Experiment_Window.run_single_function
    experiment_list : list
        List of experiment object arguments
    """
    def __init__(self, master, myexperiment, experiment_list, main_window, meta = False):

        self.meta = meta
        self.main_window = main_window
        self.master = master
        self.my_experiment = myexperiment
        #("my exp post pro ", experiment_list)
        self.selected = experiment_list

        self.frame = tk.Frame(self.master)

        self.title = tk.Label(master = self.master,
                                text = "Welcome to the Post-Processing Page",
                                font = "Calibri 15 bold",justify="center")
        if self.meta:
            self.title = tk.Label(master = self.master,
                                text = "Welcome to the Post-Processing \nand Post-Normalization Page",
                                font = "Calibri 15 bold",justify="center")

        self.n_postreps_label = tk.Label(master = self.master,
                                    text = "Number of Postreplications at each Recommended Solution:",
                                    font = "Calibri 13",
                                    wraplength = "250")

        self.n_postreps_var = tk.StringVar(self.master)
        self.n_postreps_entry = ttk.Entry(master=self.master, textvariable = self.n_postreps_var, justify = tk.LEFT, width=15)
        self.n_postreps_entry.insert(index=tk.END, string="100")


        self.crn_across_budget_label = tk.Label(master=self.master,
                                    text = "Use CRN for Postreplications at Solutions Recommended at Different Times?",
                                    font = "Calibri 13",
                                    wraplength = "250")

        self.crn_across_budget_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_budget_var = tk.StringVar(self.master)
        # sets the default OptionMenu selection
        # self.crn_across_budget_var.set("True")
        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.crn_across_budget_menu = ttk.OptionMenu(self.master, self.crn_across_budget_var, "True", *self.crn_across_budget_list)

        self.crn_across_macroreps_label = tk.Label(master=self.master,
                                        text = "Use CRN for Postreplications at Solutions Recommended on Different Macroreplications?",
                                        font = "Calibri 13",
                                        wraplength = "325")

        self.crn_across_macroreps_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_macroreps_var = tk.StringVar(self.master)

        self.crn_across_macroreps_menu = ttk.OptionMenu(self.master, self.crn_across_macroreps_var, "False", *self.crn_across_macroreps_list)

        self.crn_norm_budget_label = tk.Label(master=self.master,
                                    text = "Use CRN for Postreplications at x\u2080 and x\u002A?",
                                    font = "Calibri 13",
                                    wraplength = "325")
        self.crn_norm_across_macroreps_var = tk.StringVar(self.master)
        self.crn_norm_across_macroreps_menu = ttk.OptionMenu(self.master, self.crn_norm_across_macroreps_var, "True", *self.crn_across_macroreps_list)

        self.n_norm_label = tk.Label(master = self.master,
                                    text = "Post-Normalization Parameters",
                                    font = "Calibri 14 bold",
                                    wraplength = "300")

        self.n_proc_label = tk.Label(master = self.master,
                                    text = "Post-Processing Parameters",
                                    font = "Calibri 14 bold",
                                    wraplength = "300")

        self.n_norm_ostreps_label = tk.Label(master = self.master,
                                    text = "Number of Postreplications at x\u2080 and x\u002A:",
                                    font = "Calibri 13",
                                    wraplength = "300")

        self.n_norm_postreps_var = tk.StringVar(self.master)
        self.n_norm_postreps_entry = ttk.Entry(master=self.master, textvariable = self.n_norm_postreps_var, justify = tk.LEFT, width=15)
        self.n_norm_postreps_entry.insert(index=tk.END, string="200")

        self.post_processing_run_label = tk.Label(master=self.master, # window label is used for
                        text = "Complete Post-Processing of the Problem-Solver Pairs:",
                        font = "Calibri 13",
                        wraplength = "250")

        if self.meta:
            self.post_processing_run_label = tk.Label(master=self.master, # window label is used for
                            text = "Complete Post-Processing and Post-Normalization of the Problem-Solver Pair(s)",
                            font = "Calibri 13",
                            wraplength = "300")

        self.post_processing_run_button = ttk.Button(master=self.master, # window button is used in
                        # aesthetic of button and specific formatting options
                        text = "Post-Process",
                        width = 15, # width of button
                        command = self.post_processing_run_function) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click


        self.title.place(x=145, y=15)

        if not self.meta:
            self.n_postreps_label.place(x=10, y=55)
            self.n_postreps_entry.place(x=300, y=55)

            self.crn_across_budget_label.place(x=10, y=105)
            self.crn_across_budget_menu.place(x=345, y=105)

            self.crn_across_macroreps_label.place(x=10, y=160)
            self.crn_across_macroreps_menu.place(x=345, y=160)

            self.post_processing_run_label.place(x=10, y=233)
            self.post_processing_run_button.place(x=310, y=237)
        else:
            self.n_proc_label.place(x=15, y=65)

            self.n_postreps_label.place(x=10, y=105)
            self.n_postreps_entry.place(x=300, y=105)

            self.crn_across_budget_label.place(x=10, y=155)
            self.crn_across_budget_menu.place(x=300, y=155)

            self.crn_across_macroreps_label.place(x=10, y=205)
            self.crn_across_macroreps_menu.place(x=300, y=205)

            self.n_norm_label.place(x=15, y=265)

            self.crn_norm_budget_label.place(x=10,y=305)
            self.crn_norm_across_macroreps_menu.place(x=300,y=305)

            self.n_norm_ostreps_label.place(x=10, y=355)
            self.n_norm_postreps_entry.place(x=300,y=355)

            self.post_processing_run_label.place(x=10, y=405)
            self.post_processing_run_button.place(x=300, y=405)

        self.frame.pack(side="top", fill="both", expand=True)
        self.run_all = all

    def post_processing_run_function(self):

        self.experiment_list = []
        # self.experiment_list = [self.selected[3], self.selected[4], self.selected[2]]

        # if self.n_postreps_entry.get().isnumeric() != False and self.n_postreps_init_opt_entry.get().isnumeric() != False and self.crn_across_budget_var.get() in self.crn_across_budget_list and self.crn_across_macroreps_var.get() in self.crn_across_macroreps_list:
        if self.n_postreps_entry.get().isnumeric() != False and self.crn_across_budget_var.get() in self.crn_across_budget_list and self.crn_across_macroreps_var.get() in self.crn_across_macroreps_list and (self.meta == True and self.n_norm_postreps_entry.get().isnumeric() or self.meta == False):
            self.experiment_list.append(int(self.n_postreps_entry.get()))
            # self.experiment_list.append(int(self.n_postreps_init_opt_entry.get()))

            # actually adding a boolean value to the list instead of a string
            if self.crn_across_budget_var.get()=="True":
                self.experiment_list.append(True)
            else:
                self.experiment_list.append(False)

            if self.crn_across_macroreps_var.get()=="True":
                self.experiment_list.append(True)
            else:
                self.experiment_list.append(False)

            norm = False
            if self.crn_norm_across_macroreps_var.get() == "True":
                norm = True
            # reset n_postreps_entry
            self.n_postreps_entry.delete(0, len(self.n_postreps_entry.get()))
            self.n_postreps_entry.insert(index=tk.END, string="100")

            # reset crn_across_budget_bar
            self.crn_across_budget_var.set("True")

            # reset crn_across_macroreps_var
            self.crn_across_macroreps_var.set("False")

            self.n_postreps = self.experiment_list[0] # int
            # print("self.n_prostreps", type(self.n_postreps))
            # self.n_postreps_init_opt = self.experiment_list[4] # int
            self.crn_across_budget = self.experiment_list[1] # boolean
            # print("self.n_prostreps", type(self.n_postreps))
            self.crn_across_macroreps = self.experiment_list[2] # boolean

            # print("This is the experiment object", self.my_experiment)
            # print("This is the problem name: ", self.my_experiment.problem.name)
            # print("This is the solver name: ", self.my_experiment.solver.name)
            # print("This is the experiment list", self.selected)
            # print ("This is experiment_list ", self.experiment_list)
            # self, n_postreps, crn_across_budget=True, crn_across_macroreps=False
            self.my_experiment.post_replicate(self.n_postreps, self.crn_across_budget, self.crn_across_macroreps)

            if self.meta:
                self.my_experiment.post_normalize(n_postreps_init_opt=int(self.n_norm_postreps_entry.get()), crn_across_init_opt=norm)

            #(self.experiment_list)
            self.master.destroy()
            self.post_processed_bool = True
            Experiment_Window.post_process_disable_button(self.main_window,self.meta)



            return self.experiment_list

        elif self.n_postreps_entry.get().isnumeric() == False:
            message = "Please enter a valid value for the number of postreplications at each recommended solution."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_entry.delete(0, len(self.n_postreps_entry.get()))
            self.n_postreps_entry.insert(index=tk.END, string="100")

        elif self.crn_across_macroreps_var.get() not in self.crn_across_macroreps_list:
            message = "Please answer the following question: 'Use CRN for postreplications at Solutions Recommended at Different Times?' with True or False."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.crn_across_budget_var.set("----")

        elif self.crn_across_budget_var.get() not in self.crn_across_budget_list:
            message = "Please answer the following question: 'Use CRN for Postreplications at Solutions Recommended on Different Macroreplications?' with True or False."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.crn_across_macroreps_var.set("----")

        else:
            message = "You have not selected all required field! Check for '*' signs near required input boxes."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_init_opt_entry.delete(0, len(self.n_postreps_init_opt_entry.get()))
            self.n_postreps_init_opt_entry.insert(index=tk.END, string="6")

            self.crn_across_budget_var.set("True")

            self.crn_across_macroreps_var.set("False")

    def test_function2(self, *args):
        print("connection enabled")

class Post_Normal_Window():
    """
    Post-Normalization Page of the GUI

    Arguments
    ----------
    master : tk.Tk
        Tkinter window created from Experiment_Window.run_single_function
    myexperiment : object(Experiment)
        Experiment object created in Experiment_Window.run_single_function
    experiment_list : list
        List of experiment object arguments
    """
    def __init__(self, master, experiment_list, main_window, meta = False):
        self.post_norm_exp_list = experiment_list
        self.meta = meta
        self.main_window = main_window
        self.master = master
        self.optimal_var = tk.StringVar(master=self.master)
        self.initial_var = tk.StringVar(master=self.master)
        self.check_var = tk.IntVar(master=self.master)
        self.init_var = tk.StringVar(self.master)
        self.proxy_var = tk.StringVar(self.master)
        self.proxy_sol = tk.StringVar(self.master)

        self.all_solvers = []
        for solvers in self.post_norm_exp_list:
            if solvers.solver.name not in self.all_solvers:
                self.all_solvers.append(solvers.solver.name)

        #("my exp post pro ", experiment_list)
        self.selected = experiment_list

        self.frame = tk.Frame(self.master)
        top_lab = "Welcome to the Post-Normalization Page for " + self.post_norm_exp_list[0].problem.name + " \n with Solvers:"
        if self.post_norm_exp_list[0].problem.minmax[0] == 1:
            minmax = "max"
        else:
            minmax = "min"

        opt = "unknown"
        if self.post_norm_exp_list[0].problem.optimal_solution != None:
            if len(self.post_norm_exp_list[0].problem.optimal_solution) == 1:
                opt = str(self.post_norm_exp_list[0].problem.optimal_solution[0])
            else:
                opt = str(self.post_norm_exp_list[0].problem.optimal_solution)


        for solv in self.all_solvers:
            top_lab = top_lab + " " + solv

        self.title = tk.Label(master = self.master,
                                text = top_lab,
                                font = "Calibri 15 bold",
                                justify="center")
        initsol = self.post_norm_exp_list[0].problem.factors['initial_solution']
        if len(initsol) == 1:
            initsol = str(initsol[0])
        else:
            initsol = str(initsol)

        self.n_init_label = tk.Label(master = self.master,
                                text = "The Initial Solution, x\u2080, is " + initsol +".",
                                font = "Calibri 13",
                                wraplength = "400")

        self.n_opt_label = tk.Label(master = self.master,
                                text = "The Optimal Solution, x\u002A, is " + opt +  " for this " + minmax + "imization Problem. \nIf the Proxy Optimal Value or the Proxy Optimal Solution is unspecified, SimOpt uses the best Solution found in the selected Problem-Solver Pair experiments as the Proxy Optimal Solution.",
                                font = "Calibri 13",
                                wraplength = "600",
                                justify="left")

        self.n_optimal_label = tk.Label(master = self.master,
                                text = "Optimal Solution (optional):",
                                font = "Calibri 13",
                                wraplength = "250")
        self.n_proxy_val_label = tk.Label(master = self.master,
                                text = "Insert Proxy Optimal Value, f(x\u002A):",
                                font = "Calibri 13",
                                wraplength = "250")
        self.n_proxy_sol_label = tk.Label(master = self.master,
                                text = "Insert Proxy Optimal Solution, x\u002A:",
                                font = "Calibri 13",
                                wraplength = "250")


        # t = ["x","f(x)"]
        self.n_proxy_sol_entry = ttk.Entry(master=self.master, textvariable = self.proxy_sol, justify = tk.LEFT, width=8)
        self.n_proxy_val_entry = ttk.Entry(master=self.master, textvariable = self.proxy_var, justify = tk.LEFT, width=8)
        self.n_initial_entry = ttk.Entry(master=self.master, textvariable = self.init_var, justify = tk.LEFT, width=10)

        self.n_crn_label = tk.Label(master = self.master,
                                text = "CRN for x\u2080 and Optimal x\u002A?",
                                font = "Calibri 13",
                                wraplength = "310")
        self.n_crn_checkbox = tk.Checkbutton(self.master,text="",variable=self.check_var)


        self.n_postreps_init_opt_label = tk.Label(master = self.master,
                                text = "Number of Post-Normalizations at x\u2080 and x\u002A:",
                                font = "Calibri 13",
                                wraplength = "310")

        self.n_postreps_init_opt_var = tk.StringVar(self.master)
        self.n_postreps_init_opt_entry = ttk.Entry(master=self.master, textvariable = self.n_postreps_init_opt_var, justify = tk.LEFT, width=15)
        self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

        self.post_processing_run_label = tk.Label(master=self.master, # window label is used for
                        text = "Click to Post-Normalize the Problem-Solver Pairs",
                        font = "Calibri 13",
                        wraplength = "300")

        self.post_processing_run_button = ttk.Button(master=self.master, # window button is used in
                        # aesthetic of button and specific formatting options
                        text = "Post-Normalize",
                        width = 15, # width of button
                        command = self.post_norm_run_function) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click


        self.title.place(x=75, y=15)

        self.n_init_label.place(x=10, y=70)

        self.n_opt_label.place(x=10,y=90)

        # self.n_proxy_label.place(x=10, y=200)
        self.n_proxy_val_label.place(x=10,y=190)
        self.n_proxy_sol_label.place(x=325,y=190)
        self.n_proxy_val_entry.place(x=220, y=190)
        self.n_proxy_sol_entry.place(x=530, y=190)

        self.n_crn_label.place(x=10, y=230)
        self.n_crn_checkbox.place(x=325, y=230)
        #default to selected
        self.n_crn_checkbox.select()

        self.n_postreps_init_opt_label.place(x=10, y=270)
        self.n_postreps_init_opt_entry.place(x=325, y=270)


        self.post_processing_run_label.place(x=10, y=310)
        self.post_processing_run_button.place(x=325, y=310)

        self.frame.pack(side="top", fill="both", expand=True)

    def post_norm_run_function(self):

        self.experiment_list = []

        # if self.n_postreps_entry.get().isnumeric() != False and self.n_postreps_init_opt_entry.get().isnumeric() != False and self.crn_across_budget_var.get() in self.crn_across_budget_list and self.crn_across_macroreps_var.get() in self.crn_across_macroreps_list:
        if self.n_postreps_init_opt_entry.get().isnumeric() != False :
            n_postreps_init_opt = int(self.n_postreps_init_opt_entry.get())
            crn = self.check_var.get()
            proxy_val = None
            proxy_sol = None
            if self.proxy_sol.get() != "":
                proxy_sol = ast.literal_eval(self.proxy_sol.get())
            if self.proxy_var.get() != "":
                proxy_val = ast.literal_eval(self.proxy_var.get())
            post_normalize(self.post_norm_exp_list, n_postreps_init_opt, crn_across_init_opt=crn, proxy_init_val=None, proxy_opt_val=proxy_val, proxy_opt_x=proxy_sol)
            # self.master.destroy()
            self.post_processed_bool = True

            self.postrep_window = tk.Toplevel()
            self.postrep_window.geometry("1000x800")
            self.postrep_window.title("Plotting Page")
            self.master.destroy()
            Plot_Window(self.postrep_window, self.main_window, experiment_list = self.post_norm_exp_list)

            return

        elif self.n_postreps_init_opt_entry.get().isnumeric() == False:
            message = "Please enter a valid value for the number of postreplications at each recommended solution."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_entry.delete(0, len(self.n_postreps_entry.get()))
            self.n_postreps_entry.insert(index=tk.END, string="100")

    def test_function2(self, *args):
        print("connection enabled")

class Plot_Window():
        """
        Plot Window Page of the GUI

        Arguments
        ----------
        master : tk.Tk
            Tkinter window created from Experiment_Window.run_single_function
        myexperiment : object(Experiment)
            Experiment object created in Experiment_Window.run_single_function
        experiment_list : list
            List of experiment object arguments
        """
        def __init__(self, master, main_window, experiment_list = None, metaList = None):

            self.metaList = metaList
            self.master = master
            self.experiment_list = experiment_list
            self.main_window = main_window
            self.plot_types_inputs = ["cdf_solvability", "quantile_solvability","diff_cdf_solvability","diff_quantile_solvability"]
            self.plot_type_names = ["All Progress Curves", "Mean Progress Curve", "Quantile Progress Curve", "Solve time CDF", "Area Scatter Plot", "CDF Solvability","Quantile Solvability","CDF Difference Plot", "Quantile Difference Plot", "Terminal Progress Plot", "Terminal Scatter Plot"]
            self.num_plots = 0
            self.plot_exp_list = []
            self.plot_type_list = []
            self.checkbox_list = []
            self.plot_CI_list = []
            self.plot_param_list = []
            self.all_path_names = []
            self.bad_label = None
            self.plot_var = tk.StringVar(master=self.master)

            self.params = [tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master)]

            self.problem_menu = Listbox(self.master, selectmode = MULTIPLE, exportselection = False, width=10, height=6)
            self.solver_menu = Listbox(self.master, selectmode = MULTIPLE, exportselection = False, width=10, height=6)


            self.all_problems = []
            i = 0

            # Creating a list of problems from the experiment list
            for problem in self.experiment_list:
                if problem.problem.name not in self.all_problems:
                    self.all_problems.append(problem.problem.name)
                    self.problem_menu.insert(i,problem.problem.name)
                    i += 1



            #("solvers:",self.all_solvers)
            if self.metaList != None:
                i = 0
            # Getting the names for the solvers from the metalist and add it to the solver menu 
                for name in self.metaList.solver_names:
                    self.solver_menu.insert(i,name)
                    i += 1
            else:
                self.all_solvers = []
                i = 0
             # Getting the solvers from the experiment list and add it to the solver menu 
                for solvers in self.experiment_list:
                    if solvers.solver.name not in self.all_solvers:
                        self.all_solvers.append(solvers.solver.name)
                        self.solver_menu.insert(i,solvers.solver.name)
                        i += 1
            #("exp:",self.experiment_list[0].solver_names)

            self.solver_menu.bind('<<ListboxSelect>>', self.solver_select_function)

            self.instruction_label = tk.Label(master=self.master, # window label is used in
                            text = "Welcome to the Plotting Page of SimOpt \n Select Problems and Solvers to Plot",
                            font = "Calibri 15 bold",justify="center")

            self.problem_label = tk.Label(master=self.master, # window label is used in
                            text = "Select Problem(s):*",
                            font = "Calibri 13")
            self.plot_label = tk.Label(master=self.master, # window label is used in
                            text = "Select Plot Type:*",
                            font = "Calibri 13")

            # from experiments.inputs.all_factors.py:
            self.problem_list = problem_unabbreviated_directory
            # stays the same, has to change into a special type of variable via tkinter function
            self.problem_var = tk.StringVar(master=self.master)


            # self.problem_menu = tk.Listbox(self.master, self.problem_var, "Problem", *self.all_problems, command=self.experiment_list[0].problem.name)
            self.plot_menu = ttk.OptionMenu(self.master, self.plot_var, "Plot", *self.plot_type_names, command=partial(self.get_parameters_and_settings, self.plot_var))
            self.solver_label = tk.Label(master=self.master, # window label is used in
                            text = "Select Solver(s):*",
                            font = "Calibri 13")

            # from experiments.inputs.all_factors.py:
            self.solver_list = solver_unabbreviated_directory
            # stays the same, has to change into a special type of variable via tkinter function
            self.solver_var = tk.StringVar(master=self.master)

            self.add_button = ttk.Button(master=self.master,
                                        text = "Add",
                                        width = 15,
                                        command=self.add_plot)


            self.post_normal_all_button = ttk.Button(master=self.master,
                                                    text = "See All Plots",
                                                    width = 20,
                                                    state = "normal",
                                                    command = self.plot_button)


            self.style = ttk.Style()
            self.style.configure("Bold.TLabel", font = ("Calibri",15,"bold"))
            Label = ttk.Label(master = self.master, text ="Plotting Workspace", style="Bold.TLabel")
            
            self.queue_label_frame = ttk.LabelFrame(master=self.master, labelwidget = Label)

            self.queue_canvas = tk.Canvas(master=self.queue_label_frame, borderwidth=0)

            self.queue_frame = ttk.Frame(master=self.queue_canvas)
            self.vert_scroll_bar = Scrollbar(self.queue_label_frame, orient="vertical", command=self.queue_canvas.yview)
            self.horiz_scroll_bar = Scrollbar(self.queue_label_frame, orient="horizontal", command=self.queue_canvas.xview)
            self.queue_canvas.configure(xscrollcommand=self.horiz_scroll_bar.set, yscrollcommand=self.vert_scroll_bar.set)

            self.vert_scroll_bar.pack(side="right", fill="y")
            self.horiz_scroll_bar.pack(side="bottom", fill="x")
    
            self.queue_canvas.pack(side="left", fill="both", expand=True)
            self.queue_canvas.create_window((0,0), window=self.queue_frame, anchor="nw",
                                    tags="self.queue_frame")

            self.notebook = ttk.Notebook(master=self.queue_frame)
            self.notebook.pack(fill="both")
            self.tab_one = tk.Frame(master=self.notebook)
            self.notebook.add(self.tab_one, text="Problem-Solver Pairs to Plots")
            self.tab_one.grid_rowconfigure(0)

            self.heading_list = ["Problem", "Solver", "Plot Type", "Remove Row", "View Plot", "Parameters", "PNG File Path"]

            for heading in self.heading_list:
                self.tab_one.grid_columnconfigure(self.heading_list.index(heading))
                label = tk.Label(master=self.tab_one, text=heading, font="Calibri 14 bold")
                label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)


            self.instruction_label.place(relx=.3, y=0)

            self.problem_label.place(x=10, rely=.08)
            self.problem_menu.place(x=10, rely=.11, relwidth=.3)

            self.solver_label.place(x=10, rely=.25)
            self.solver_menu.place(x=10, rely=.28, relwidth=.3)

            self.plot_label.place(relx=.4, rely=.08)
            self.plot_menu.place(relx=.55, rely=.08)

            self.add_button.place(relx=.45, rely=.45)

            separator = ttk.Separator(master=self.master, orient='horizontal')
            separator.place(relx=0.35, rely=.08, relheight=.4)

            self.post_normal_all_button.place(relx=.01,rely=.92)

            # self.queue_label_frame.place(x=10, rely=.7, relheight=.3, relwidth=1)
            self.queue_label_frame.place(x=10, rely=.56, relheight=.35, relwidth=.99)

            self.param_label = []
            self.param_entry = []
            self.factor_label_frame_problem = None

            self.CI_label_frame = ttk.LabelFrame(master=self.master, text="Plot Settings and Parameters")
            self.CI_canvas = tk.Canvas(master=self.CI_label_frame, borderwidth=0)
            self.CI_frame = ttk.Frame(master=self.CI_canvas)

            self.CI_canvas.pack(side="left", fill="both", expand=True)
            self.CI_canvas.create_window((0,0), window=self.CI_frame, anchor="nw",
                                    tags="self.queue_frame")

            self.CI_label_frame.place(relx=.4, rely=.15, relheight=.2, relwidth=.3)

            self.settings_label_frame = ttk.LabelFrame(master=self.master, text="Error Estimation Setting and Parameters")
            self.settings_canvas = tk.Canvas(master=self.settings_label_frame, borderwidth=0)
            self.settings_frame = ttk.Frame(master=self.settings_canvas)

            self.settings_canvas.pack(side="left", fill="both", expand=True)
            self.settings_canvas.create_window((0,0), window=self.settings_frame, anchor="nw",
                                    tags="self.queue_frame")
            self.settings_canvas.grid_rowconfigure(0)
            self.settings_label_frame.place(relx=.65, rely=.15, relheight=.2, relwidth=.3)
           
            """
            # Confidence Interval Checkbox
            entry1 = tk.Checkbutton(self.settings_canvas, variable=self.params[0], onvalue=True, offvalue=False)
            entry1.select()
            # entry1 = ttk.OptionMenu(self.settings_canvas, self.params[0], "True", *tf_list)
            label1 = tk.Label(master=self.settings_canvas, text="Confidence Intervals", font="Calibri 14")
            label1.grid(row=0, column=0, padx=10, pady=3)
            entry1.grid(row=0, column=1, padx=10, pady=3)

            # Plot Together Checkbox
            entry = tk.Checkbutton(self.settings_canvas, variable=self.params[1], onvalue=True, offvalue=False)
            entry.select()
            # Creates the Check Mark that checks whether the plots will be plot together
            label = tk.Label(master=self.settings_canvas, text="Plot Together", font="Calibri 14")
            label.grid(row=1, column=0, padx=10, pady=3)
            entry.grid(row=1, column=1, padx=10, pady=3)

            entry2 = tk.Checkbutton(self.settings_canvas, variable=self.params[2], onvalue=True, offvalue=False)
            entry2.select()
            label2 = tk.Label(master=self.settings_canvas, text="Print Max HW", font="Calibri 14")
            label2.grid(row=2, column=0, padx=10, pady=3)
            entry2.grid(row=2, column=1, padx=10, pady=3)
            """

        def add_plot(self):
            self.plot_exp_list = []
                            
            solverList = ""
            # Appends experiment that is part of the experiment list if it matches what was chosen in the solver menu
            for i in self.solver_menu.curselection():
                solverList = solverList + self.solver_menu.get(i) + " "
                for  j in self.problem_menu.curselection():
                    problemList = ""
                    if self.metaList != None: 
                        for metaexp in self.metaList.experiments:
                            for exp in metaexp:
                                if exp.solver.name == self.solver_menu.get(i) and exp.problem.name == self.problem_menu.get(j):
                                    self.plot_exp_list.append(exp)
                    else:
                        for exp in self.experiment_list:
                            if exp.solver.name == self.solver_menu.get(i) and exp.problem.name == self.problem_menu.get(j):
                                self.plot_exp_list.append(exp)
                    problemList = problemList + self.problem_menu.get(j) + " "
            
            plotType = str(self.plot_var.get())
            if len(self.plot_exp_list) == 0 or str(plotType) == "Plot":
                txt = "At least 1 Problem, 1 Solver, and 1 Plot Type must be selected."
                self.bad_label = tk.Label(master=self.master,text=txt,font = "Calibri 12",justify="center")
                self.bad_label.place(relx=.45, rely=.5)
                return
            elif self.bad_label != None:
                self.bad_label.destroy()
                self.bad_label = None
            
            self.plot_type_list.append(plotType)
            
            param_value_list = []
            for t in self.params:
                new_value = ""
                if t.get() == True:
                    new_value = True
                elif t.get() == False:
                    new_value = False
                elif t.get() != "":
                    try:
                        new_value = float(t.get())
                    except ValueError:
                        new_value = t.get()
                param_value_list.append(new_value)
            

            exp_list = self.plot_exp_list
            if self.metaList != None: 
                list_exp_list = self.metaList.experiments
            else:
                list_exp_list = [[exp] for exp in exp_list]
            
            if self.plot_type_list[-1] == "All Progress Curves":
                path_name = plot_progress_curves(exp_list, plot_type = "all", normalize = bool(param_value_list[1]), all_in_one = bool(param_value_list[0]))
                param_list = {"normalize":bool(param_value_list[1])}
            if self.plot_type_list[-1] == "Mean Progress Curve":
                path_name = plot_progress_curves(exp_list, plot_type = "mean", normalize = bool(param_value_list[3]), all_in_one = bool(param_value_list[1]), plot_CIs = bool(param_value_list[0]), print_max_hw = bool(param_value_list[2]), n_bootstraps = int(param_value_list[4]), conf_level = param_value_list[5])
                param_list = {"plot CIs":bool(param_value_list[0]), "print max hw":bool(param_value_list[2]), "normalize":bool(param_value_list[3]),"n_bootstraps":int(param_value_list[4]), "conf_level":param_value_list[5]}
            elif self.plot_type_list[-1] == "Quantile Progress Curve":
                path_name = plot_progress_curves(exp_list, plot_type = "quantile",  beta = param_value_list[3], normalize = bool(param_value_list[4]), plot_CIs = bool(param_value_list[0]), all_in_one = bool(param_value_list[1]), print_max_hw = bool(param_value_list[2]),n_bootstraps = int(param_value_list[5]), conf_level = param_value_list[6] )
                param_list = {"plot CIs":bool(param_value_list[0]), "print max hw":bool(param_value_list[2]), "normalize":bool(param_value_list[4]), "beta":param_value_list[3],"n_bootstraps":int(param_value_list[5]), "conf_level":param_value_list[6]}
            elif self.plot_type_list[-1] == "Solve time CDF":
                path_name = plot_solvability_cdfs(exp_list, solve_tol = param_value_list[2], plot_CIs = bool(param_value_list[0]), print_max_hw = bool(param_value_list[1]), n_bootstraps = int(param_value_list[3]), conf_level = param_value_list[4] )
                param_list = {"plot CIs":bool(param_value_list[0]), "print max hw":bool(param_value_list[1]), "solve tol":param_value_list[2],"n_bootstraps":int(param_value_list[3]), "conf_level":param_value_list[4]}
            elif self.plot_type_list[-1] == "Area Scatter Plot":
                path_name = plot_area_scatterplots(list_exp_list, plot_CIs = bool(param_value_list[0]), print_max_hw = bool(param_value_list[1]), n_bootstraps = int(param_value_list[2]), conf_level = param_value_list[3] )
                param_list = {"plot CIs":bool(param_value_list[0]), "print max hw":bool(param_value_list[1]), "n_bootstraps":int(param_value_list[2]), "conf_level":param_value_list[3]}
            elif self.plot_type_list[-1] == "CDF Solvability":
                path_name = plot_solvability_profiles(list_exp_list, plot_type = "cdf_solvability", plot_CIs = bool(param_value_list[0]), print_max_hw = bool(param_value_list[1]), solve_tol = param_value_list[2], ref_solver = None, n_bootstraps = int(param_value_list[3]), conf_level = param_value_list[4] )
                param_list = {"plot CIs":bool(param_value_list[0]), "print max hw":bool(param_value_list[1]), "solve tol":param_value_list[2],"n_bootstraps":int(param_value_list[3]), "conf_level":param_value_list[4]}
            elif self.plot_type_list[-1] == "Quantile Solvability":
                path_name = plot_solvability_profiles(list_exp_list, plot_type = "quantile_solvability", plot_CIs = bool(param_value_list[0]), print_max_hw = bool(param_value_list[1]), solve_tol = param_value_list[2], beta = param_value_list[3], ref_solver = None, n_bootstraps = int(param_value_list[4]), conf_level = param_value_list[5] )
                param_list = {"plot CIs":bool(param_value_list[0]), "print max hw":bool(param_value_list[1]), "solve tol": param_value_list[2], "beta":param_value_list[3], "n_bootstraps":int(param_value_list[4]), "conf_level":param_value_list[5]}
            elif self.plot_type_list[-1] == "CDF Difference Plot":
                path_name = plot_solvability_profiles(list_exp_list, plot_type = "diff_cdf_solvability", plot_CIs = bool(param_value_list[0]), print_max_hw = bool(param_value_list[1]), solve_tol = param_value_list[2], ref_solver = param_value_list[3], n_bootstraps = int(param_value_list[4]), conf_level = param_value_list[5] )
                param_list = {"plot CIs":bool(param_value_list[0]), "print max hw":bool(param_value_list[1]), "solve tol":param_value_list[2], "ref solver":param_value_list[3], "n_bootstraps":int(param_value_list[4]), "conf_level":param_value_list[5]}
            elif self.plot_type_list[-1] == "Quantile Difference Plot":
                path_name = plot_solvability_profiles(list_exp_list, plot_type = "diff_quantile_solvability", plot_CIs = bool(param_value_list[0]), print_max_hw = bool(param_value_list[1]), solve_tol = param_value_list[2], beta = param_value_list[3], ref_solver = param_value_list[4], n_bootstraps = int(param_value_list[5]), conf_level = param_value_list[6] )
                param_list = {"plot CIs":bool(param_value_list[0]), "print max hw":bool(param_value_list[1]), "solve tol":param_value_list[2],"ref solver":param_value_list[4], "beta":param_value_list[3],"n_bootstraps":int(param_value_list[5]), "conf_level":param_value_list[6]}
            elif self.plot_type_list[-1] == "Terminal Progress Plot":
                path_name = plot_terminal_progress(exp_list, plot_type = param_value_list[1], normalize = bool(param_value_list[2]), all_in_one = bool(param_value_list[0]))
                param_list = {"normalize":bool(param_value_list[2])}
            elif self.plot_type_list[-1] == "Terminal Scatter Plot":
                path_name = plot_terminal_scatterplots(list_exp_list, all_in_one = bool(param_value_list[0]))
                param_list = {}
            else:
                print(f"{self.plot_type_list[-1]} is the plot_type_list")



            for i, new_plot in enumerate(path_name):
                place = self.num_plots + 1
                if len(path_name) == 1:
                    prob_text = solverList
                else:
                    prob_text = self.solver_menu.get(i)

                self.problem_button_added = tk.Label(master=self.tab_one,
                                                        text=problemList,
                                                        font = "Calibri 12",
                                                        justify="center")
                self.problem_button_added.grid(row=place, column=0, sticky='nsew', padx=10, pady=3)

                self.solver_button_added = tk.Label(master=self.tab_one,
                                                        text=prob_text,
                                                        font = "Calibri 12",
                                                        justify="center")
                self.solver_button_added.grid(row=place, column=1, sticky='nsew', padx=10, pady=3)

                self.plot_type_button_added = tk.Label(master=self.tab_one,
                                                        text=plotType,
                                                        font = "Calibri 12",
                                                        justify="center")
                self.plot_type_button_added.grid(row=place, column=2, sticky='nsew', padx=10, pady=3)

                param_text = ""
                for key, item in param_list.items():
                    param_text = param_text + key + ": " + str(item) + ", "
                param_text = param_text[:len(param_text)-2]

                self.params_label_added = tk.Label(master=self.tab_one,
                                                        text=param_text,
                                                        font = "Calibri 12",
                                                        justify="center")
                self.params_label_added.grid(row=place, column=5, sticky='nsew', padx=10, pady=3)

                # TODO: remove plot does not work
                self.clear_plot = tk.Button(master=self.tab_one,
                                                        text="Remove",
                                                        font = "Calibri 12",
                                                        justify="center",
                                                        command=partial(self.clear_row, place-1))
                self.clear_plot.grid(row=place, column=3, sticky='nsew', padx=10, pady=3)

                self.view_plot = tk.Button(master=self.tab_one,
                                                        text="View Plot",
                                                        font = "Calibri 12",
                                                        justify="center",
                                                        command=partial(self.view_one_pot, new_plot))
                self.view_plot.grid(row=place, column=4, sticky='nsew', padx=10, pady=3)

                self.plot_path = tk.Label(master=self.tab_one,
                                                        text=new_plot,
                                                        font = "Calibri 12",
                                                        justify="center")
                self.plot_path.grid(row=place, column=6, sticky='nsew', padx=10, pady=3)
                # self.view_plot.pack()
                self.changeOnHover(self.view_plot, "red", "yellow")
                self.all_path_names.append(new_plot)
                # print("all_path_names",self.all_path_names)
                self.num_plots += 1

        def changeOnHover(self, button, colorOnHover, colorOnLeave):
            # adjusting backgroung of the widget
            # background on entering widget
            button.bind("<Enter>", func=lambda e: button.config(
                background=colorOnHover))

            # background color on leving widget
            button.bind("<Leave>", func=lambda e: button.config(
                background=colorOnLeave))

        def solver_select_function(self,a):
            # if user clicks plot type then a solver, this is update parameters
            if self.plot_var.get() != "Plot" and self.plot_var.get() != "":
                self.get_parameters_and_settings(0, self.plot_var.get())

        def get_parameters_and_settings(self,a, plot_choice):
            # ref solver needs to a drop down of solvers that is selected in the problem
            # numbers between 0 and 1
            # checkbox for normalize
            # move CI to parameters
            # checkbox with print_max_hw checkbox
            # remove CI from experiment box

            # beta=0.50, normalize=True
            if plot_choice == "All Progress Curves":
                param_list = {'normalize':True}
            elif plot_choice == "Mean Progress Curve":
                param_list = {'normalize':True, 'n_bootstraps': 100, 'conf_level':0.95}
            elif plot_choice == "Quantile Progress Curve":
                param_list = {'beta':0.50, 'normalize':True, 'n_bootstraps': 100, 'conf_level':0.95}
            elif plot_choice == "Solve time CDF":
                param_list = {'solve_tol':0.1, 'n_bootstraps':100, 'conf_level':0.95}
            elif plot_choice == "Area Scatter Plot":
                param_list = { 'n_bootstraps':100, 'conf_level':0.95}
            elif plot_choice == "CDF Solvability":
                param_list = {'solve_tol':0.1, 'n_bootstraps':100, 'conf_level':0.95}
            elif plot_choice == "Quantile Solvability":
                param_list = {'solve_tol':0.1, 'beta':0.5, 'n_bootstraps':100, 'conf_level':0.95}
            elif plot_choice == "CDF Difference Plot":
                param_list = {'solve_tol':0.1, 'ref_solver':None, 'n_bootstraps':100, 'conf_level':0.95}
            elif plot_choice == "Quantile Difference Plot":
                param_list = {'solve_tol':0.1, 'beta':0.5, 'ref_solver':None, 'n_bootstraps':100, 'conf_level':0.95}
            elif plot_choice == "Terminal Progress Plot":
                param_list = {'plot type': "violin", 'normalize': True}
            elif plot_choice == "Terminal Scatter Plot":
                param_list = {}
            else:
                print("invalid plot?")
            self.param_list = param_list


            # self.params = [tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master)]

            self.CI_label_frame.destroy()
            self.CI_label_frame = ttk.LabelFrame(master=self.master, text="Plot Settings and Parameters")
            self.CI_canvas = tk.Canvas(master=self.CI_label_frame, borderwidth=0)
            self.CI_frame = ttk.Frame(master=self.CI_canvas)

            self.CI_canvas.pack(side="left", fill="both", expand=True)
            self.CI_canvas.create_window((0,0), window=self.CI_frame, anchor="nw",
                                    tags="self.queue_frame")
            self.CI_canvas.grid_rowconfigure(0)

            self.CI_label_frame.place(relx=.4, rely=.15, relheight=.3, relwidth=.25)
            
            self.settings_label_frame.destroy()
            self.settings_label_frame = ttk.LabelFrame(master=self.master, text="Error Estimation Settings and Parameters")
            self.settings_canvas = tk.Canvas(master=self.settings_label_frame, borderwidth=0)
            self.settings_frame = ttk.Frame(master=self.settings_canvas)

            self.settings_canvas.pack(side="left", fill="both", expand=True)
            self.settings_canvas.create_window((0,0), window=self.settings_frame, anchor="nw",
                                    tags="self.queue_frame")
            self.settings_canvas.grid_rowconfigure(0)

            self.settings_label_frame.place(relx=.65, rely=.15, relheight=.3, relwidth=.3)

            bp_list = ['violin','box']
            self.solvers_names = []
            for i in self.solver_menu.curselection():
                self.solvers_names.append(self.solver_menu.get(i))

            
            # Plot Settings
            i = 0 
            if plot_choice == "Mean Progress Curve" or plot_choice == "Quantile Progress Curve" or plot_choice ==  "Solve time CDF" or plot_choice =="Area Scatter Plot" or plot_choice == "CDF Solvability" or plot_choice == "Quantile Solvability" or plot_choice == "CDF Difference Plot" or plot_choice == "Quantile Difference Plot":
                # Confidence Intervals
                entry1 = tk.Checkbutton(self.settings_canvas, variable=self.params[i], onvalue=True, offvalue=False)
                entry1.select()
                # entry1 = ttk.OptionMenu(self.settings_canvas, self.params[0], "True", *tf_list)
                label1 = tk.Label(master=self.settings_canvas, text="Show Confidence Intervals", font="Calibri 13", wraplength="150")
                label1.grid(row=0, column=0, padx=10, pady=3)
                entry1.grid(row=0, column=1, padx=10, pady=3)
                i += 1

            if plot_choice == "All Progress Curves" or plot_choice == "Mean Progress Curve" or plot_choice == "Quantile Progress Curve" or plot_choice == "Terminal Progress Plot" or plot_choice == "Terminal Scatter Plot":
                # Plot Together Checkbox
                entry = tk.Checkbutton(self.CI_canvas, variable=self.params[i], onvalue=True, offvalue=False)
                entry.select()
                # Creates the Check Mark that checks whether the plots will be plot together
                label = tk.Label(self.CI_canvas, text="Plot Together", font="Calibri 13", wraplength="150")
                label.grid(row=i, column=0, padx=10, pady=3)
                entry.grid(row=i, column=1, padx=10, pady=3) 
                i += 1
            
            if plot_choice == "Mean Progress Curve" or plot_choice == "Quantile Progress Curve" or plot_choice ==  "Solve time CDF" or plot_choice =="Area Scatter Plot" or plot_choice == "CDF Solvability" or plot_choice == "Quantile Solvability" or plot_choice == "CDF Difference Plot" or plot_choice == "Quantile Difference Plot":
                # Show Print Max Halfwidth
                entry2 = tk.Checkbutton(self.settings_canvas, variable=self.params[i], onvalue=True, offvalue=False)
                entry2.select()
                label2 = tk.Label(master=self.settings_canvas, text="Show Max Halfwidth", font="Calibri 13", wraplength="150")
                label2.grid(row=1, column=0, padx=10, pady=3)
                entry2.grid(row=1, column=1, padx=10, pady=3)
                i += 1
            
            for param, param_val in param_list.items():
                if param == 'normalize':
                    entry = tk.Checkbutton(master=self.CI_canvas, variable=self.params[i], onvalue=True, offvalue=False)
                    entry.select()
                    label = tk.Label(master=self.CI_canvas, text="Normalize by Relative Optimality Gap", font="Calibri 13", wraplength="150")
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                elif param == 'ref_solver':
                    label = tk.Label(master=self.CI_canvas, text="Select Solver", font="Calibri 13")
                    if len(self.solvers_names) != 0:
                        label = tk.Label(master=self.CI_canvas, text="Benchmark Solver", font="Calibri 13", wraplength="150")
                        entry = ttk.OptionMenu(self.CI_canvas, self.params[i], self.solvers_names[0], *self.solvers_names)
                        entry.grid(row=i, column=1, padx=10, pady=3)
                    label.grid(row=i, column=0, padx=10, pady=3)
                elif param == 'solve_tol':
                    label = tk.Label(master=self.CI_canvas, text="Optimality Gap Threshold", font="Calibri 13", wraplength="150")
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry = ttk.Entry(master=self.CI_canvas, textvariable = self.params[i], justify = tk.LEFT, width=15)
                    if param_val is not None:
                        entry.delete(0, 'end')
                        entry.insert(index=tk.END, string=param_val)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                elif param == 'beta':
                    label = tk.Label(master=self.CI_canvas, text="Quantile Probability", font="Calibri 13", wraplength="150")
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry = ttk.Entry(master=self.CI_canvas, textvariable = self.params[i], justify = tk.LEFT, width=15)
                    if param_val is not None:
                        entry.delete(0, 'end')
                        entry.insert(index=tk.END, string=param_val)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                elif param == 'plot type':
                    label = tk.Label(master=self.CI_canvas, text="Type of Terminal Progress Plot", font="Calibri 13", wraplength="150")
                    entry = ttk.OptionMenu(self.CI_canvas, self.params[i], "violin",*bp_list)
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                elif param == 'n_bootstraps':
                    label = tk.Label(master=self.settings_canvas, text="Number of Bootstraps", font="Calibri 13", wraplength="150")
                    label.grid(row=3, column=0, padx=10, pady=3)
                    entry = ttk.Entry(master=self.settings_canvas, textvariable = self.params[i], justify = tk.LEFT, width=15)
                    if param_val is not None:
                        entry.delete(0, 'end')
                        entry.insert(index=tk.END, string=param_val)
                    entry.grid(row=3, column=1, padx=10, pady=3)
                elif param == 'conf_level':
                    label = tk.Label(master=self.settings_canvas, text="Confidence Level", font="Calibri 13", wraplength="150")
                    label.grid(row=2, column=0, padx=10, pady=3)
                    entry = ttk.Entry(master=self.settings_canvas, textvariable = self.params[i], justify = tk.LEFT, width=15)
                    if param_val is not None:
                        entry.delete(0, 'end')
                        entry.insert(index=tk.END, string=param_val)
                    entry.grid(row=2, column=1, padx=10, pady=3)
                else:
                    label = tk.Label(master=self.CI_canvas, text=param, font="Calibri 13")
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry = ttk.Entry(master=self.CI_canvas, textvariable = self.params[i], justify = tk.LEFT, width=15)
                    if param_val is not None:
                        entry.delete(0, 'end')
                        entry.insert(index=tk.END, string=param_val)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                i += 1

             

            
        

        def clear_row(self, place):
            self.plot_CI_list.pop(place)
            self.plot_exp_list.pop(place)
            print("Clear")

        def plot_button(self):
            self.postrep_window = tk.Toplevel()
            self.postrep_window.geometry("1000x600")
            self.postrep_window.title("Plotting Page")
            # have one plot and have arrow to scroll through each
            # one plot solver per row
            # view individual plot options
            # hover over for image
            # Plot individually vs together
            # all plots
            # https://www.tutorialspoint.com/python/tk_place.htm
            # widget.place(relx = percent of x, rely = percent of y)
            ro = 0
            c = 0
            
            for i, path_name in enumerate(self.all_path_names):

                width = 350
                height = 350
                img = Image.open(path_name)
                img = img.resize((width,height), Image.LANCZOS)
                img =  ImageTk.PhotoImage(img)
                # img = tk.PhotoImage(file=path_name)

                # img = img.resize(200,200)
                self.panel = tk.Label(
                    self.postrep_window,
                    image=img
                )
                self.panel.photo = img
                self.panel.grid(row=ro,column=c)
                c += 1
                if c == 3:
                    c = 0
                    ro += 1

                # panel.place(x=10,y=0)

        def view_one_pot(self, path_name):
            self.postrep_window = tk.Toplevel()
            self.postrep_window.geometry("400x400")
            self.postrep_window.title("View One Plot")

            ro = 0
            c = 0

            width = 400
            height = 400
            print("This is path",path_name)
            img = Image.open(path_name)

            img = img.resize((width,height), Image.LANCZOS)
            img =  ImageTk.PhotoImage(img)
            # img = tk.PhotoImage(file=path_name)

            # img = img.resize(200,200)
            self.panel = tk.Label(
                self.postrep_window,
                image=img
            )
            self.panel.photo = img
            self.panel.grid(row=ro,column=c)
            c += 1
            if c == 4:
                c = 0
                ro += 1
        

def problem_solver_unabbreviated_to_object(problem_or_solver,unabbreviated_dictionary):
    if problem_or_solver in unabbreviated_dictionary.keys():
        problem_or_solver_object = unabbreviated_dictionary[problem_or_solver]
        return problem_or_solver_object, problem_or_solver_object().name

    else:
        print(f"{problem_or_solver} not found in {unabbreviated_dictionary}")

def problem_solver_abbreviated_name_to_unabbreviated(problem_or_solver, abbreviated_dictionary, unabbreviated_dictionary):
    if problem_or_solver in abbreviated_dictionary.keys():
        problem_or_solver_object = abbreviated_dictionary[problem_or_solver]
        for key, value in unabbreviated_dictionary.items():
            if problem_or_solver_object == value:
                problem_or_solver_unabbreviated_name = key
        return problem_or_solver_unabbreviated_name

    else:
        print(f"{problem_or_solver} not found in {abbreviated_dictionary}")
def main():
    root = tk.Tk()
    root.title("SimOpt Library Graphical User Interface")
    root.geometry("600x700")
    root.pack_propagate(False)

    #app = Experiment_Window(root)
    app = Main_Menu_Window(root)
    root.mainloop()

if __name__ == '__main__':
    main()
