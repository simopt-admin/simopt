from os import path
from random import expovariate
import tkinter as tk
from tkinter import ttk, Scrollbar, filedialog
from timeit import timeit
from functools import partial
from tkinter.constants import FALSE, MULTIPLE, S
from matplotlib.colors import Normalize
from numpy import e
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from PIL import ImageTk, Image 

from directory import problem_directory
from directory import solver_directory
from directory import oracle_directory
from wrapper_base import Experiment, MetaExperiment
import wrapper_base
import pickle
from tkinter import messagebox
from tkinter import Listbox
import ast
from PIL import ImageTk

class Experiment_Window(tk.Tk):
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
            connected to : self.clear_button_added <- ttk.Button, within self.add_function
    clear_queue(self) : clears entire experiment queue and resets all lists containing experiment data
            connected to : self.clear_queue_button <- ttk.Button
    add_function(self) : adds function to experiment queue
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
        
        self.instruction_label = tk.Label(master=self.master, # window label is used in
                            text = "Welcome to SimOpt \n Please complete the fields below to run your experiment: \n Please note: '*' are required fields",
                            font = "Calibri 15 bold")
        
        self.problem_label = tk.Label(master=self.master, # window label is used in
                        text = "Please select the type of Problem:*",
                        font = "Calibri 11 bold")
        
        # from experiments.inputs.all_factors.py:
        self.problem_list = problem_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.problem_var = tk.StringVar(master=self.master)
        # sets the default OptionMenu value
        # self.problem_var.set("----")
        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.problem_menu = ttk.OptionMenu(self.master, self.problem_var, "Problem", *self.problem_list, command=self.show_problem_factors)

        self.solver_label = tk.Label(master=self.master, # window label is used in
                        text = "Please select the type of Solver:*",
                        font = "Calibri 11 bold")

        # from experiments.inputs.all_factors.py:
        self.solver_list = solver_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.solver_var = tk.StringVar(master=self.master)
        # sets the default OptionMenu value
        # self.solver_var.set("----")
        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.solver_menu = ttk.OptionMenu(self.master, self.solver_var, "Solver", *self.solver_list, command=self.show_solver_factors)       

        self.macro_label = ttk.Label(master=self.master,
                        text = "Number of Macro Replications:*",
                        font = "Calibri 11 bold")

        self.macro_var = tk.StringVar(self.master)
        self.macro_entry = ttk.Entry(master=self.master, textvariable = self.macro_var, justify = tk.LEFT)
        self.macro_entry.insert(index=tk.END, string="10")
        
        
        self.run_button = ttk.Button(master=self.master, # window button is used in
                        # aesthetic of button and specific formatting options
                        text = "Run", 
                        width = 10, # width of button
                        command = self.run_single_function) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click)

        self.add_button = ttk.Button(master=self.master,
                                    text = "Add Experiment",
                                    width = 15,
                                    command=self.add_function)

        self.clear_queue_button = ttk.Button(master=self.master,
                                    text = "Clear All Experiments",
                                    width = 15,
                                    command = self.clear_queue)#(self.experiment_added, self.problem_added, self.solver_added, self.macros_added, self.run_button_added))

        self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Cross-Design Experiment",
                                            width = 20,
                                            command = self.crossdesign_function)

        self.pickle_file_select_label = ttk.Label(master=self.master,
                                                text = "Select a pickle file to access: ",
                                                font = "Calibri 11 bold",
                                                wraplength = "250")

        self.pickle_file_select_button = ttk.Button(master=self.master,
                                                    text = "Browse Files",
                                                    width = 15,
                                                    command = self.select_pickle_file_fuction)

        self.pickle_file_load_button = ttk.Button(master=self.master,
                                                text = "Load File",
                                                width = 15,
                                                command = self.load_pickle_file_function)
        
        # self.post_process_all_button = ttk.Button(master=self.master,
        #                                         text = "Post Process All",
        #                                         width = 15,
        #                                         command = self.post_rep_all_function)

        self.post_normal_all_button = ttk.Button(master=self.master,
                                                text = "Post Normalize Selected",
                                                width = 20,
                                                state = "normal",
                                                command = self.post_normal_all_function)

        self.pickle_file_pathname_label = ttk.Label(master=self.master,
                                                    text = "File Selected:",
                                                    font = "Calibri 11 bold")

        self.pickle_file_pathname_show = ttk.Label(master=self.master,
                                                    text = "No file selected",
                                                    font = "Calibri 11 italic",
                                                    foreground = "red",
                                                    wraplength = "500")


        self.queue_label_frame = ttk.Labelframe(master=self.master, text="Experiment")

        self.queue_canvas = tk.Canvas(master=self.queue_label_frame, borderwidth=0)

        self.queue_frame = ttk.Frame(master=self.queue_canvas)
        self.vert_scroll_bar = Scrollbar(self.queue_label_frame, orient="vertical", command=self.queue_canvas.yview)
        self.queue_canvas.configure(yscrollcommand=self.vert_scroll_bar.set)

        self.horiz_scroll_bar = Scrollbar(self.queue_label_frame, orient="horizontal", command=self.queue_canvas.xview)
        self.queue_canvas.configure(xscrollcommand=self.horiz_scroll_bar.set)

        self.vert_scroll_bar.pack(side="right", fill="y")
        self.horiz_scroll_bar.pack(side="bottom", fill="x")

        self.queue_canvas.pack(side="left", fill="both", expand=True)
        self.queue_canvas.create_window((0,0), window=self.queue_frame, anchor="nw",
                                  tags="self.queue_frame")
        
        self.queue_frame.bind("<Configure>", self.onFrameConfigure_queue)

        self.notebook = ttk.Notebook(master=self.queue_frame)
        self.notebook.pack(fill="both")

        def on_tab_change(event):
            tab = event.widget.tab('current')['text']
            if tab == 'Post Normalize by Problem':
                self.post_norm_setup()
                
        self.notebook.bind('<<NotebookTabChanged>>', on_tab_change)

        self.tab_one = tk.Frame(master=self.notebook)

        self.notebook.add(self.tab_one, text="Queue of Experiments")

        self.tab_one.grid_rowconfigure(0)

        self.heading_list = ["Problem", "Solver", "Macro Reps", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_one.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_one, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=5, pady=3)

        self.tab_two = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_two, text="Queue of Meta Experiments")
        self.tab_two.grid_rowconfigure(0)
        self.heading_list = ["Problems", "Solvers", "Macro Reps", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_two.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_two, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=5, pady=3)

        self.tab_three = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_three, text="Post Normalize by Problem")
        self.tab_three.grid_rowconfigure(0)
        self.heading_list = ["Problem", "Solvers", "Select", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_three.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_three, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=5, pady=3)

        self.instruction_label.place(x=0, y=0)

        self.problem_label.place(x=0, y=85)
        self.problem_menu.place(x=225, y=85)

        self.solver_label.place(x=0, y=165)
        self.solver_menu.place(x=225, y=165)

        self.macro_label.place(x=0, y=245)
        self.macro_entry.place(x=225, y=245)

        self.run_button.place(x=5, y=285)
        self.crossdesign_button.place(x=175, y=285)
        self.add_button.place(x=5, y=325)
        self.clear_queue_button.place(x=175, y=325)

        self.pickle_file_select_label.place(x=850, y=375)
        self.pickle_file_select_button.place(x=1040, y=375)
        self.pickle_file_load_button.place(x=1150, y=375)
        self.pickle_file_pathname_label.place(x=850, y=400)
        self.pickle_file_pathname_show.place(x=950, y=400)
        # self.post_process_all_button.place(x=5,y=800)
        self.post_normal_all_button.place(x=250,y=800)

        self.queue_label_frame.place(x=0, y=375, height=400, width=800)

        self.frame.pack(fill='both')

    def show_problem_factors(self, *args):
        # if args and len(args) == 2:
        #     print("ARGS: ", args[1])
        # print("arg length:", len(args))

        self.problem_factors_list = []
        self.problem_factors_types = []

        self.factor_label_frame_problem = ttk.Labelframe(master=self.master, text="Problem Factors")

        self.factor_canvas_problem = tk.Canvas(master=self.factor_label_frame_problem, borderwidth=0)

        self.factor_frame_problem = ttk.Frame(master=self.factor_canvas_problem)
        self.vert_scroll_bar_factor_problem = Scrollbar(self.factor_label_frame_problem, orient="vertical", command=self.factor_canvas_problem.yview)
        self.factor_canvas_problem.configure(yscrollcommand=self.vert_scroll_bar_factor_problem.set)

        self.horiz_scroll_bar_factor_problem = Scrollbar(self.factor_label_frame_problem, orient="horizontal", command=self.factor_canvas_problem.xview)
        self.factor_canvas_problem.configure(xscrollcommand=self.horiz_scroll_bar_factor_problem.set)

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
            label_problem.grid(row=0, column=self.factor_heading_list_problem.index(heading), padx=5, pady=3)

        self.problem_object = problem_directory[self.problem_var.get()]

        count_factors_problem = 1
        for num, factor_type in enumerate(self.problem_object().specifications, start=0):

            self.dictionary_size_problem = len(self.problem_object().specifications[factor_type])

            if self.problem_object().specifications[factor_type].get("datatype") != bool:
                self.int_float_description_problem = tk.Label(master=self.factor_tab_one_problem,
                                                    text = str(self.problem_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 11 bold")

                self.int_float_var_problem = tk.StringVar(self.factor_tab_one_problem)
                self.int_float_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.int_float_var_problem, justify = tk.LEFT)
                if args and len(args) == 2 and args[0] == True:
                    self.int_float_entry_problem.insert(index=tk.END, string=str(args[1][3][0][factor_type]))
                else:
                    self.int_float_entry_problem.insert(index=tk.END, string=str(self.problem_object().specifications[factor_type].get("default")))

                self.int_float_description_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
                self.int_float_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

                datatype = self.problem_object().specifications[factor_type].get("datatype")
                if datatype != tuple:    
                    self.problem_factors_types.append(datatype)
                else:
                    self.problem_factors_types.append(str)
                # self.int_float_var_problem = datatype(self.int_float_var_problem)
                # print(datatype)
                # print("datatype of var ", type(self.int_float_var_problem))

                self.problem_factors_list.append(self.int_float_var_problem)
                
                count_factors_problem += 1


            if self.problem_object().specifications[factor_type].get("datatype") == bool:
                self.boolean_description_problem = tk.Label(master=self.factor_tab_one_problem,
                                                    text = str(self.problem_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 11 bold")

                self.boolean_list_problem = ["True", "False"]
                self.boolean_var_problem = tk.StringVar(self.factor_tab_one_problem)

                self.boolean_menu_problem = ttk.OptionMenu(self.factor_tab_one_problem, self.boolean_var_problem, str(self.problem_object().specifications[factor_type].get("default")), *self.boolean_list)

                # self.boolean_datatype_problem = tk.Label(master=self.factor_tab_one,
                #                                     text = str(self.problem_object().specifications[factor_type].get("datatype")),
                #                                     font = "Calibri 11 bold")

                self.boolean_description_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
                self.boolean_menu_problem.grid(row=count_factors_problem, column=1, sticky='nsew')
                # self.boolean_datatype_problem.grid(row=count_factors, column=2, sticky='nsew')

                datatype = self.problem_object().specifications[factor_type].get("datatype")

                self.problem_factors_list.append(self.boolean_var_problem)
                self.problem_factors_types.append(datatype)

                count_factors_problem += 1

        count_factors_problem += 1
        if args and len(args) == 2 and args[0] == True:
            oldname = args[1][3][1]
        else:
            oldname = self.problem_var.get()

        self.save_label_problem = tk.Label(master=self.factor_tab_one_problem,
                                            text = "Save Problem As",
                                            font = "Calibri 11 bold")

        self.save_var_problem = tk.StringVar(self.factor_tab_one_problem)
        self.save_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.save_var_problem, justify = tk.LEFT)
        self.save_entry_problem.insert(index=tk.END, string=oldname)

        self.save_label_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
        self.save_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

        self.problem_factors_list.append(self.save_var_problem)
        self.problem_factors_types.append(str)

        self.factor_label_frame_problem.place(x=400, y=70, height=300, width=475)

        # Switching from Problems to Oracles

        self.oracle_factors_list = []
        self.oracle_factors_types = []

        problem = str(self.problem_var.get())
        self.oracle = problem.split("-")
        self.oracle = self.oracle[0]
        self.oracle_object = oracle_directory[self.oracle]

        self.factor_label_frame_oracle = ttk.Labelframe(master=self.master, text="Oracle Factors")

        self.factor_canvas_oracle = tk.Canvas(master=self.factor_label_frame_oracle, borderwidth=0)

        self.factor_frame_oracle = ttk.Frame(master=self.factor_canvas_oracle)
        self.vert_scroll_bar_factor_oracle = Scrollbar(self.factor_label_frame_oracle, orient="vertical", command=self.factor_canvas_oracle.yview)
        self.factor_canvas_oracle.configure(yscrollcommand=self.vert_scroll_bar_factor_oracle.set)

        self.horiz_scroll_bar_factor_oracle = Scrollbar(self.factor_label_frame_oracle, orient="horizontal", command=self.factor_canvas_oracle.xview)
        self.factor_canvas_oracle.configure(xscrollcommand=self.horiz_scroll_bar_factor_oracle.set)

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
            label_oracle.grid(row=0, column=self.factor_heading_list_oracle.index(heading), padx=5, pady=3)


        count_factors_oracle = 1
        for factor_type in self.oracle_object().specifications:

            self.dictionary_size_oracle = len(self.oracle_object().specifications[factor_type])

            if self.oracle_object().specifications[factor_type].get("datatype") != bool:

                # print("yes?")
                self.int_float_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(self.oracle_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 11 bold")

                self.int_float_var_oracle = tk.StringVar(self.factor_tab_one_oracle)
                self.int_float_entry_oracle = ttk.Entry(master=self.factor_tab_one_oracle, textvariable = self.int_float_var_oracle, justify = tk.LEFT, width = "50")
                
                if args and len(args) == 2 and args[0] == True:
                    self.int_float_entry_oracle.insert(index=tk.END, string=str(args[1][4][0][factor_type]))
                else:
                    self.int_float_entry_oracle.insert(index=tk.END, string=str(self.oracle_object().specifications[factor_type].get("default")))

                self.int_float_description_oracle.grid(row=count_factors_oracle, column=0, sticky='nsew')
                self.int_float_entry_oracle.grid(row=count_factors_oracle, column=1, sticky='nsew')

                self.oracle_factors_list.append(self.int_float_var_oracle)

                datatype = self.oracle_object().specifications[factor_type].get("datatype")
                if datatype != tuple:    
                    self.oracle_factors_types.append(datatype)
                else:
                    self.oracle_factors_types.append(str)

                count_factors_oracle += 1


            if self.oracle_object().specifications[factor_type].get("datatype") == bool:

                # print("yes!")
                self.boolean_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(self.oracle_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 11 bold")

                self.boolean_list_oracle = ["True", "False"]
                self.boolean_var_oracle = tk.StringVar(self.factor_tab_one_oracle)

                self.boolean_menu_oracle = ttk.OptionMenu(self.factor_tab_one_oracle, self.boolean_var_oracle, str(self.oracle_object().specifications[factor_type].get("default")), *self.boolean_list)

                # self.boolean_datatype_oracle = tk.Label(master=self.factor_tab_one,
                #                                     text = str(self.oracle_object().specifications[factor_type].get("datatype")),
                #                                     font = "Calibri 11 bold")

                self.boolean_description_oracle.grid(row=count_factors_oracle, column=0, sticky='nsew')
                self.boolean_menu_oracle.grid(row=count_factors_oracle, column=1, sticky='nsew')
                # self.boolean_datatype_oracle.grid(row=count_factors, column=2, sticky='nsew')

                self.oracle_factors_list.append(self.boolean_var_oracle)

                datatype = self.oracle_object().specifications[factor_type].get("datatype")
                self.oracle_factors_types.append(datatype)

                count_factors_oracle += 1

        # print(self.oracle_factors_list)
        # self.factor_label_frame_oracle.place(x=900, y=70, height=300, width=600)

    def show_solver_factors(self, *args):

        self.solver_factors_list = []
        self.solver_factors_types = []

        self.factor_label_frame_solver = ttk.Labelframe(master=self.master, text="Solver Factors")

        self.factor_canvas_solver = tk.Canvas(master=self.factor_label_frame_solver, borderwidth=0)

        self.factor_frame_solver = ttk.Frame(master=self.factor_canvas_solver)
        self.vert_scroll_bar_factor_solver = Scrollbar(self.factor_label_frame_solver, orient="vertical", command=self.factor_canvas_solver.yview)
        self.factor_canvas_solver.configure(yscrollcommand=self.vert_scroll_bar_factor_solver.set)

        self.horiz_scroll_bar_factor_solver = Scrollbar(self.factor_label_frame_solver, orient="horizontal", command=self.factor_canvas_solver.xview)
        self.factor_canvas_solver.configure(xscrollcommand=self.horiz_scroll_bar_factor_solver.set)

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
            label.grid(row=0, column=self.factor_heading_list_solver.index(heading), padx=5, pady=3)

        self.solver_object = solver_directory[self.solver_var.get()]

        count_factors_solver = 1
        for factor_type in self.solver_object().specifications:
            # print("size of dictionary", len(self.solver_object().specifications[factor_type]))
            # print("first", factor_type)
            # print("second", self.solver_object().specifications[factor_type].get("description"))
            # print("third", self.solver_object().specifications[factor_type].get("datatype"))    
            # print("fourth", self.solver_object().specifications[factor_type].get("default"))   

            self.dictionary_size = len(self.solver_object().specifications[factor_type])

            if self.solver_object().specifications[factor_type].get("datatype") != bool:

                self.int_float_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(self.solver_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 11 bold")

                self.int_float_var = tk.StringVar(self.factor_tab_one_solver)
                self.int_float_entry = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.int_float_var, justify = tk.LEFT)
                if args and len(args) == 2 and args[0] == True:

                    self.int_float_entry.insert(index=tk.END, string=str(args[1][5][0][factor_type]))
                else:
                    self.int_float_entry.insert(index=tk.END, string=str(self.solver_object().specifications[factor_type].get("default")))

                # self.int_float_datatype = tk.Label(master=self.factor_tab_one,
                #                                     text = str(self.solver_object().specifications[factor_type].get("datatype")),
                #                                     font = "Calibri 11 bold")

                self.int_float_description.grid(row=count_factors_solver, column=0, sticky='nsew')
                self.int_float_entry.grid(row=count_factors_solver, column=1, sticky='nsew')
                # self.int_float_datatype.grid(row=count_factors_solver, column=2, sticky='nsew')
                
                self.solver_factors_list.append(self.int_float_var)

                datatype = self.solver_object().specifications[factor_type].get("datatype")
                if datatype != tuple:    
                    self.solver_factors_types.append(datatype)
                else:
                    self.solver_factors_types.append(str)

                count_factors_solver += 1


            if self.solver_object().specifications[factor_type].get("datatype") == bool:
                
                self.boolean_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(self.solver_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 11 bold")

                self.boolean_list = ["True", "False"]
                self.boolean_var = tk.StringVar(self.factor_tab_one_solver)

               # self.boolean_menu = ttk.OptionMenu(self.factor_tab_one_solver, self.boolean_var, str(self.solver_object().specifications[factor_type].get("default")), *self.boolean_list)
                
                if args and len(args) == 2 and args[0] == True:    
                    self.boolean_menu = ttk.OptionMenu(self.factor_tab_one_solver, self.boolean_var, str(args[1][5][0][factor_type]), *self.boolean_list)
                else:
                    self.boolean_menu = ttk.OptionMenu(self.factor_tab_one_solver, self.boolean_var, str(self.solver_object().specifications[factor_type].get("default")), *self.boolean_list)


                self.boolean_description.grid(row=count_factors_solver, column=0, sticky='nsew')
                self.boolean_menu.grid(row=count_factors_solver, column=1, sticky='nsew')
                # self.boolean_datatype.grid(row=count_factors_solver, column=2, sticky='nsew')

                self.solver_factors_list.append(self.boolean_var)

                datatype = self.solver_object().specifications[factor_type].get("datatype")
                self.solver_factors_types.append(datatype)

                count_factors_solver += 1

        count_factors_solver += 1

        self.save_label_solver = tk.Label(master=self.factor_tab_one_solver,
                                            text = "Save Solver As",
                                            font = "Calibri 11 bold")
        if args and len(args) == 2 and args[0] == True:
            oldname = args[1][5][1]
        else:
            oldname = self.solver_var.get()
        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.save_var_solver, justify = tk.LEFT)
        self.save_entry_solver.insert(index=tk.END, string=oldname)

        self.save_label_solver.grid(row=count_factors_solver, column=0, sticky='nsew')
        self.save_entry_solver.grid(row=count_factors_solver, column=1, sticky='nsew')

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)

        self.factor_label_frame_solver.place(x=900, y=70, height=300, width=500)

    def run_single_function(self):
        if self.problem_var.get() in problem_directory and self.solver_var.get() in solver_directory and self.macro_entry.get().isnumeric() != False:
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

            # resets problem_var to default value
            self.problem_var.set("Problem")
            # resets solver_var to default value
            self.solver_var.set("Solver")

            # macro_entry is a positive integer
            if int(self.macro_entry.get()) != 0:
                # resets current entry from index 0 to length of entry
                self.macro_entry.delete(0, len(self.macro_entry.get()))
                # resets macro_entry textbox
                self.macro_entry.insert(index=tk.END, string="10")

                # complete experiment with given arguments
                self.solver_dictionary_rename = self.selected[5]
                print("solver combined", self.solver_dictionary_rename)
                self.solver_rename = self.solver_dictionary_rename[1]
                self.solver_factors = self.solver_dictionary_rename[0]
                print("solver rename", self.solver_rename)
                print("solver factors", self.solver_factors)

                self.oracle_factors = self.selected[4]
                self.oracle_factors = self.oracle_factors[0]
                print("oracle factors", self.oracle_factors)

                self.problem_dictionary_rename = self.selected[3]
                print("problem combined", self.problem_dictionary_rename)
                self.problem_rename = self.problem_dictionary_rename[1]
                self.problem_factors = self.problem_dictionary_rename[0]
                print("problem rename", self.problem_rename)
                print("problem factors", self.problem_factors)

                self.macro_reps = self.selected[2]
                self.solver_name = self.selected[1]
                self.problem_name = self.selected[0]

                self.my_experiment = Experiment(solver_name=self.solver_name, problem_name=self.problem_name) #, solver_rename=self.solver_rename, problem_rename=self.problem_rename, solver_fixed_factors=self.solver_factors, problem_fixed_factors=self.problem_factors, oracle_fixed_factors=self.oracle_factors)
                compatibility_result = self.my_experiment.check_compatibility()
                if compatibility_result == "":
                    self.experiment_object_list.append(self.my_experiment)
                    self.experiment_master_list.append(self.selected)

                    # tk.messagebox.showinfo(title="Status Update", message="Function will now begin running")

                    self.my_experiment.run(n_macroreps=self.macro_reps)

                    # calls postprocessing window
                    # self.postrep_window = tk.Tk()
                    # self.postrep_window.geometry("1500x1000")
                    # self.postrep_window.title("Post Processing Page")
                    # self.app = Post_Processing_Window(self.postrep_window, self.my_experiment, self.selected)

                    # prints selected (list) in console/terminal
                    # print("it works", self.selected)
                else:
                    tk.messagebox.showerror(title="Error Window", message=compatibility_result)
                    self.selected.clear()

                print(self.selected)
                return self.selected

            else:
                # reset macro_entry to "10"
                self.macro_entry.delete(0, len(self.macro_entry.get()))
                # resets macro_entry textbox
                self.macro_entry.insert(index=tk.END, string="10")

                message = "Please enter a postivie (non zero) integer for the number of Macro Replications, example: 10"
                tk.messagebox.showerror(title="Error Window", message=message)

        # problem selected, but solver NOT selected
        elif self.problem_var.get() in problem_directory and self.solver_var.get() not in solver_directory:
            message = "You have not selected a Solver!"
            tk.messagebox.showerror(title="Error Window", message=message)   

        # problem NOT selected, but solver selected
        elif self.problem_var.get() not in problem_directory and self.solver_var.get() in solver_directory:
            message = "You have not selected a Problem!"
            tk.messagebox.showerror(title="Error Window", message=message)
        
        # macro_entry not numeric or negative
        elif self.macro_entry.get().isnumeric() == False:
            # reset macro_entry to "10"
            self.macro_entry.delete(0, len(self.macro_entry.get()))
            # resets macro_entry textbox
            self.macro_entry.insert(index=tk.END, string="10")

            message = "Please enter a positive (non zero) integer for the number of Macro Replications, example: 10"
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

    def clearRow_function(self, integer):
        
        for widget in self.widget_list[integer-1]:
            widget.grid_remove()

        self.experiment_master_list.pop(integer-1)      
        self.experiment_object_list.pop(integer-1)
        self.widget_list.pop(integer-1)

        # if (integer - 1) in self.normalize_list:
        #     self.normalize_list.remove(integer - 1)
        # for i in range(len(self.normalize_list)):
        #     if i < self.normalize_list[i]:
        #         self.normalize_list[i] = self.normalize_list[i] - 1

        for row_of_widgets in self.widget_list:
            row_index = self.widget_list.index(row_of_widgets)

            run_button_added = row_of_widgets[3]
            text_on_run = run_button_added["text"]
            split_text = text_on_run.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            new_text = " ".join(split_text)
            run_button_added["text"] = new_text
            run_button_added["command"] = partial(self.run_row_function, row_index+1)

            row_of_widgets[3] = run_button_added

            viewEdit_button_added = row_of_widgets[4]
            text_on_viewEdit = viewEdit_button_added["text"]
            split_text = text_on_viewEdit.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            new_text = " ".join(split_text)
            viewEdit_button_added["text"] = new_text
            viewEdit_button_added["command"] = partial(self.viewEdit_function, row_index+1)

            row_of_widgets[4] = viewEdit_button_added

            clear_button_added = row_of_widgets[5]
            text_on_clear = clear_button_added["text"]
            split_text = text_on_clear.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            new_text = " ".join(split_text)
            clear_button_added["text"] = new_text
            clear_button_added["command"] = partial(self.clearRow_function, row_index+1)   

            row_of_widgets[5] = clear_button_added

            postprocess_button_added = row_of_widgets[6]
            postprocess_button_added["command"] = partial(self.post_rep_function, row_index+1)   

            row_of_widgets[6] = postprocess_button_added

            row_of_widgets[0].grid(row= (row_index+1), column=0, sticky='nsew', padx=5, pady=3)
            row_of_widgets[1].grid(row= (row_index+1), column=1, sticky='nsew', padx=5, pady=3)
            row_of_widgets[2].grid(row= (row_index+1), column=2, sticky='nsew', padx=5, pady=3)
            row_of_widgets[3].grid(row= (row_index+1), column=3, sticky='nsew', padx=5, pady=3)
            row_of_widgets[4].grid(row= (row_index+1), column=4, sticky='nsew', padx=5, pady=3)
            row_of_widgets[5].grid(row= (row_index+1), column=5, sticky='nsew', padx=5, pady=3)
            row_of_widgets[6].grid(row= (row_index+1), column=6, sticky='nsew', padx=5, pady=3)
            # row_of_widgets[7].grid(row= (row_index+1), column=7, sticky='nsew', padx=5, pady=3)

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

            row_of_widgets[0].grid(row= (row_index+1), column=0, sticky='nsew', padx=5, pady=3)
            row_of_widgets[1].grid(row= (row_index+1), column=1, sticky='nsew', padx=5, pady=3)
            row_of_widgets[2].grid(row= (row_index+1), column=2, sticky='nsew', padx=5, pady=3)
            row_of_widgets[3].grid(row= (row_index+1), column=3, sticky='nsew', padx=5, pady=3)
            row_of_widgets[4].grid(row= (row_index+1), column=4, sticky='nsew', padx=5, pady=3)
            row_of_widgets[5].grid(row= (row_index+1), column=5, sticky='nsew', padx=5, pady=3)

        self.count_meta_experiment_queue = len(self.widget_meta_list) + 1
        
        # resets problem_var to default value
        self.problem_var.set("Problem")
        # resets solver_var to default value
        self.solver_var.set("Solver")

    def viewEdit_function(self, integer):
        row_index = integer

        current_experiment = self.experiment_object_list[row_index-1]
        #print(current_experiment)
        current_experiment_arguments = self.experiment_master_list[row_index-1]

        self.problem_var.set(current_experiment_arguments[0])
        self.solver_var.set(current_experiment_arguments[1])
        self.macro_var.set(current_experiment_arguments[2])
        self.show_problem_factors(True, current_experiment_arguments)
        # self.my_experiment[1][3][1]
        self.show_solver_factors(True, current_experiment_arguments)

        viewEdit_button_added = self.widget_list[row_index-1][4]
        viewEdit_button_added["text"] = "Save Changes"
        viewEdit_button_added["command"] = partial(self.save_edit_function, row_index)
        viewEdit_button_added.grid(row= (row_index), column=4, sticky='nsew', padx=5, pady=3)

    def clear_queue(self):
        
        # for row in self.widget_list:
        #     for widget in row:
        #         widget.grid_remove()
        for row in range(len(self.widget_list),0,-1):
            self.clearRow_function(row)


        self.experiment_master_list.clear()
        self.experiment_object_list.clear()
        self.widget_list.clear()

    def add_function(self, *args):

        if len(args) == 1 and isinstance(args[0], int) :
            place = args[0] - 1 
        else:
            place = len(self.experiment_object_list)
                    
        if (self.problem_var.get() in problem_directory and self.solver_var.get() in solver_directory and self.macro_entry.get().isnumeric() != False):
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
                
                self.my_experiment = Experiment(solver_name=self.solver_name, problem_name=self.problem_name, solver_rename=self.solver_rename, problem_rename=self.problem_rename, solver_fixed_factors=self.solver_factors, problem_fixed_factors=self.problem_factors, oracle_fixed_factors=self.oracle_factors)
                self.my_experiment.n_macroreps = self.selected[2]
                self.my_experiment.post_norm_ready = False

                compatibility_result = self.my_experiment.check_compatibility()
                for exp in self.experiment_object_list:
                    if exp.problem.name == self.my_experiment.problem.name:
                        if exp.problem != self.my_experiment.problem:
                            message = "Please Rename Problem with Unique Name, for Unique Factors"
                            tk.messagebox.showerror(title="Error Window", message=message)
                            return False
                    # if exp.solver.name == self.my_experiment.solver.name:
                    #     if exp.solver != self.my_experiment.solver:
                    #         message = "Please Rename Solver with Unique Name, for Unique Factors"
                    #         tk.messagebox.showerror(title="Error Window", message=message)
                    #         return
                
                if compatibility_result == "":
                    self.experiment_object_list.insert(place,self.my_experiment)
                    self.experiment_master_list.insert(place,self.selected)
                    #this option list doesnt autoupdate - not sure why but this will force it to update
                    self.experiment_master_list[place][5][0]['crn_across_solns'] = self.boolean_var.get()
                    
                    self.rows = 5
                    
                    self.problem_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[3][1],
                                                    font = "Calibri 10",
                                                    justify="center")
                    self.problem_added.grid(row=self.count_experiment_queue, column=0, sticky='nsew', padx=5, pady=3)

                    self.solver_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[5][1],
                                                    font = "Calibri 10",
                                                    justify="center")
                    self.solver_added.grid(row=self.count_experiment_queue, column=1, sticky='nsew', padx=5, pady=3)

                    self.macros_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[2],
                                                    font = "Calibri 10",
                                                    justify="center")
                    self.macros_added.grid(row=self.count_experiment_queue, column=2, sticky='nsew', padx=5, pady=3)

                    self.run_button_added = ttk.Button(master=self.tab_one,
                                                        text="Run Exp. " + str(self.count_experiment_queue),
                                                        command= partial(self.run_row_function, self.count_experiment_queue))
                    self.run_button_added.grid(row=self.count_experiment_queue, column=3, sticky='nsew', padx=5, pady=3)

                    self.viewEdit_button_added = ttk.Button(master=self.tab_one,
                                                        text="View / Edit Exp. " + str(self.count_experiment_queue),
                                                        command= partial(self.viewEdit_function, self.count_experiment_queue))
                    self.viewEdit_button_added.grid(row=self.count_experiment_queue, column=4, sticky='nsew', padx=5, pady=3)

                    self.clear_button_added = ttk.Button(master=self.tab_one,
                                                        text="Clear Exp. " + str(self.count_experiment_queue),
                                                        command= partial(self.clearRow_function, self.count_experiment_queue))
                    self.clear_button_added.grid(row=self.count_experiment_queue, column=5, sticky='nsew', padx=5, pady=3)

                    self.postprocess_button_added = ttk.Button(master=self.tab_one,
                                                        text="Post Process Function",
                                                        command= partial(self.post_rep_function, self.count_experiment_queue),
                                                        state = "disabled")
                    self.postprocess_button_added.grid(row=self.count_experiment_queue, column=6, sticky='nsew', padx=5, pady=3)
                    
                    self.widget_row = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.viewEdit_button_added, self.clear_button_added, self.postprocess_button_added]
                    self.widget_list.insert(place,self.widget_row)

                    self.count_experiment_queue += 1


                else:
                    tk.messagebox.showerror(title="Error Window", message=compatibility_result)
                    self.selected.clear()

            else:
                # reset macro_entry to "10"
                self.macro_entry.delete(0, len(self.macro_entry.get()))
                # resets macro_entry textbox
                self.macro_entry.insert(index=tk.END, string="10")

                message = "Please enter a postivie (non zero) integer for the number of Macro Replications, example: 10"
                tk.messagebox.showerror(title="Error Window", message=message)

            # prints selected (list) in console/terminal
            # print("it works", self.experiment_master_list)
            self.notebook.select(self.tab_one)
            return self.experiment_master_list

        # problem selected, but solver NOT selected
        elif self.problem_var.get() in problem_directory and self.solver_var.get() not in solver_directory:
            message = "You have not selected a Solver!"
            tk.messagebox.showerror(title="Error Window", message=message)
        
        # problem NOT selected, but solver selected
        elif self.problem_var.get() not in problem_directory and self.solver_var.get() in solver_directory:
            message = "You have not selected a Problem!"
            tk.messagebox.showerror(title="Error Window", message=message)
        
        # macro_entry not numeric or negative
        elif self.macro_entry.get().isnumeric() == False:
            # reset macro_entry to "10"
            self.macro_entry.delete(0, len(self.macro_entry.get()))
            # resets macro_entry textbox
            self.macro_entry.insert(index=tk.END, string="10")

            message = "Please enter a positive (non zero) integer for the number of Macro Replications, example: 10"
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
        #print("keys ->", keys)
        #print("self.problem_factors_types -> ", self.problem_factors_types)

        for problem_factor in self.problem_factors_list:
           # print(problem_factor.get() + " " + str(type(problem_factor.get())))
            index = self.problem_factors_list.index(problem_factor)
            
            #print(problem_factor.get())
            if index < len(keys):
                #print(self.problem_factors_types[index])
                #datatype = self.problem_factors_types[index]
                
                # if the data type is tuple update data
                #self.problem_factors_dictionary[keys[index]] = datatype(nextVal)
                self.problem_factors_dictionary[keys[index]] = ast.literal_eval(problem_factor.get()) 
                #print("datatype of factor -> ", type(datatype(problem_factor.get())))
            if index == len(keys):
                if problem_factor.get()  == self.problem_var.get():
                    # self.problem_object().specifications[factor_type].get("default")
                    #self.problem_factors_return.append(None)
                    self.problem_factors_return.append(problem_factor.get()) 
                else:
                    self.problem_factors_return.append(problem_factor.get()) 
                    # self.problem_factors_dictionary["rename"] = problem_factor.get()
        
        self.problem_factors_return.insert(0, self.problem_factors_dictionary)
        return self.problem_factors_return

    def confirm_oracle_factors(self):
        self.oracle_factors_return = []
        self.oracle_factors_dictionary = dict()

        keys = list(self.oracle_object().specifications.keys())
        #print("keys ->", keys)
        #print("self.oracle_factors_types -> ", self.oracle_factors_types)

        keys = list(self.oracle_object().specifications.keys())

        for oracle_factor in self.oracle_factors_list:
            index = self.oracle_factors_list.index(oracle_factor)
            self.oracle_factors_dictionary[keys[index]] = oracle_factor.get()
            #print(self.oracle_factors_types[index])
            
            datatype = self.oracle_factors_types[index]
            if (str(datatype) == "<class 'list'>"):
                newList = ast.literal_eval(oracle_factor.get())
                
                self.oracle_factors_dictionary[keys[index]] = newList
            else:
                self.oracle_factors_dictionary[keys[index]] = datatype(oracle_factor.get())
            #print(str(datatype(oracle_factor.get())) + " " + str(datatype))
            #print("datatype of factor -> ", type(datatype(oracle_factor.get())))
        
        self.oracle_factors_return.append(self.oracle_factors_dictionary)
        return self.oracle_factors_return

    def confirm_solver_factors(self):
        self.solver_factors_return = []
        self.solver_factors_dictionary = dict()

        keys = list(self.solver_object().specifications.keys())
        #print("keys ->", keys)
        #print("self.solver_factors_types -> ", self.solver_factors_types)

        for solver_factor in self.solver_factors_list:
            index = self.solver_factors_list.index(solver_factor)
            #print(solver_factor.get())
            if index < len(keys):
                #print(self.solver_factors_types[index])
                datatype = self.solver_factors_types[index]
                self.solver_factors_dictionary[keys[index]] = datatype(solver_factor.get())
                #print("datatype of factor -> ", type(datatype(solver_factor.get())))
            if index == len(keys):
                if solver_factor.get() == self.solver_var.get():
                    #self.solver_factors_return.append(None)
                    self.solver_factors_return.append(solver_factor.get())
                else:
                    self.solver_factors_return.append(solver_factor.get())
                    # self.solver_factors_dictionary["rename"] = solver_factor.get()
        
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

    def test_function(self, integer):
        
        # row_index = integer
        # self.experiment_master_list[row_index-1]
        # self.experiment_master_list[row_index-1][5][0]['crn_across_solns'] = self.boolean_var.get()
        # current_experiment_arguments = self.experiment_master_list[row_index-1][5]
        # integer = integer
        print(F"test function connected to the number {integer}")
    
    def save_edit_function(self, integer):
        
        row_index = integer
        self.experiment_master_list[row_index-1]
        self.experiment_master_list[row_index-1][5][0]['crn_across_solns'] = self.boolean_var.get()
        
        if self.add_function(row_index):
            self.clearRow_function(row_index + 1)

            self.factor_label_frame_problem.destroy()
            self.factor_label_frame_oracle.destroy()
            self.factor_label_frame_solver.destroy()
     
    def select_pickle_file_fuction(self, *args):
        filename = filedialog.askopenfilename(parent = self.master,
                                            initialdir = "./",
                                            title = "Select a Pickle File",
                                            # filetypes = (("Pickle files", "*.pickle;*.pck;*.pcl;*.pkl;*.db")
                                            #              ,("Python files", "*.py"),("All files", "*.*") )
                                                           )
        if filename != "":
            # filename_short_list = filename.split("/")
            # filename_short = filename_short_list[len(filename_short_list)-1]
            self.pickle_file_pathname_show["text"] = filename
            self.pickle_file_pathname_show["foreground"] = "blue"
            self.pickle_file_pathname_show.place(x=950, y=400)
        else:
            message = "You attempted to select a file but failed, please try again if necessary"
            tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)
 
    def load_pickle_file_function(self):
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
                    self.problem_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.problem.name,
                                                    font = "Calibri 10",
                                                    justify="center")
                    self.problem_added.grid(row=self.count_experiment_queue, column=0, sticky='nsew', padx=5, pady=3)

                    self.solver_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.solver.name,
                                                    font = "Calibri 10",
                                                    justify="center")
                    self.solver_added.grid(row=self.count_experiment_queue, column=1, sticky='nsew', padx=5, pady=3)

                    self.macros_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.n_macroreps,
                                                    font = "Calibri 10",
                                                    justify="center")
                    self.macros_added.grid(row=self.count_experiment_queue, column=2, sticky='nsew', padx=5, pady=3)

                    self.run_button_added = ttk.Button(master=self.tab_one,
                                                        text="Run Exp. " + str(self.count_experiment_queue),
                                                        command= partial(self.run_row_function, self.count_experiment_queue))
                    self.run_button_added.grid(row=self.count_experiment_queue, column=3, sticky='nsew', padx=5, pady=3)

                    self.viewEdit_button_added = ttk.Button(master=self.tab_one,
                                                        text="View / Edit Exp. " + str(self.count_experiment_queue),
                                                        command= partial(self.viewEdit_function, self.count_experiment_queue))
                    self.viewEdit_button_added.grid(row=self.count_experiment_queue, column=4, sticky='nsew', padx=5, pady=3)

                    self.clear_button_added = ttk.Button(master=self.tab_one,
                                                        text="Clear Exp. " + str(self.count_experiment_queue),
                                                        command= partial(self.clearRow_function, self.count_experiment_queue))
                    self.clear_button_added.grid(row=self.count_experiment_queue, column=5, sticky='nsew', padx=5, pady=3)

                    self.postprocess_button_added = ttk.Button(master=self.tab_one,
                                                        text="Post Process Function",
                                                        command= partial(self.post_rep_function, self.count_experiment_queue),
                                                        state = "disabled")
                    self.postprocess_button_added.grid(row=self.count_experiment_queue, column=6, sticky='nsew', padx=5, pady=3)
                    
                    
                    self.widget_row = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.viewEdit_button_added, self.clear_button_added, self.postprocess_button_added]
                    self.widget_list.insert(place,self.widget_row)

                    row_of_widgets = self.widget_list[len(self.widget_list) - 1]
                    if self.my_experiment.check_run() == True:
                        run_button = row_of_widgets[3]
                        run_button["state"] = "disabled"
                        run_button = row_of_widgets[4]
                        run_button["state"] = "disabled"
                        run_button = row_of_widgets[6]
                        run_button["state"] = "normal"
                        self.my_experiment.post_norm_ready = False
                        if self.my_experiment.check_postreplicate():
                            self.experiment_object_list[place].post_norm_ready = True
                            self.widget_list[place][6]["text"] = "Done Post Processing"
                            self.widget_list[place][6]["state"] = "disabled"

                    self.count_experiment_queue += 1
                    if self.notebook.index('current') == 2:
                        self.post_norm_setup()

            else:
                message = f"You have loaded a file, but {filetype} files are not acceptable!\nPlease try again."
                tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)
        else:
            message = "You are attempting to load a file, but haven't selected one yet.\nPlease select a file first."
            tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)

    def run_row_function(self, integer):
        # stringtuple[1:-1].split(separator=",")
        row_index = integer - 1
        row_of_widgets = self.widget_list[row_index]

        # run_button = row_of_widgets[3]
        self.widget_list[row_index][3]["state"] = "disabled"
        self.widget_list[row_index][4]["state"] = "disabled"
        self.widget_list[row_index][6]["state"] = "normal"
        # run_button["state"] = "disabled"
        # run_button = row_of_widgets[4]
        # run_button["state"] = "disabled"
        # row_of_widgets[6]["state"] = "normal"
        # print(run_button["text"], run_button["state"])
        #run_button.grid(row=integer, column=3, sticky='nsew', padx=5, pady=3)

        # widget_row = [row_of_widgets[0], row_of_widgets[1], row_of_widgets[2], row_of_widgets[3], run_button, row_of_widgets[4], row_of_widgets[5], row_of_widgets[6],row_of_widgets[7] ]
        # self.widget_list[row_index] = widget_row

        self.my_experiment = self.experiment_object_list[row_index]

        self.selected = self.experiment_master_list[row_index]
        self.macro_reps = self.selected[2]

        self.my_experiment.run(n_macroreps=self.macro_reps)

    def post_rep_function(self, integer):
        row_index = integer - 1
        self.selected = self.experiment_object_list[row_index]
        self.post_rep_function_row_index = integer
        # calls postprocessing window
        self.postrep_window = tk.Tk()
        self.postrep_window.geometry("1000x600")
        self.postrep_window.title("Post Processing Page")
        self.app = Post_Processing_Window(self.postrep_window, self.my_experiment, self.selected, self)

    def post_process_disable_button(self, meta=False):
        # print('IN post_process_disable_button ', self.post_rep_function_row_index)
        if meta:
            row_index = self.post_rep_function_row_index - 1
            self.widget_meta_list[row_index][5]["text"] = "Done Post Processing"
            self.widget_meta_list[row_index][5]["state"] = "disabled"
        else:
            row_index = self.post_rep_function_row_index - 1
            self.experiment_object_list[row_index].post_norm_ready = True
            self.widget_list[row_index][6]["text"] = "Done Post Processing"
            self.widget_list[row_index][6]["state"] = "disabled"
            # self.widget_list[row_index][7]["state"] = "normal"
        #print(self.widget_list[row_index])
    
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
        self.crossdesign_window.geometry("600x600")
        self.crossdesign_window.title("Cross-Design Experiment")
        self.cross_app = Cross_Design_Window(self.crossdesign_window, self)

    def add_meta_exp_to_frame(self):
        row_num = self.count_meta_experiment_queue + 1
        self.problem_added = tk.Label(master=self.tab_two,
                                                    text=self.cross_app.crossdesign_MetaExperiment.problem_names,
                                                    font = "Calibri 10",
                                                    justify="center")
        self.problem_added.grid(row=row_num, column=0, sticky='nsew', padx=5, pady=3)

        self.solver_added = tk.Label(master=self.tab_two,
                                        text=self.cross_app.crossdesign_MetaExperiment.solver_names,
                                        font = "Calibri 10",
                                        justify="center")
        self.solver_added.grid(row=row_num, column=1, sticky='nsew', padx=5, pady=3)

        self.macros_added = tk.Label(master=self.tab_two,
                                        text="10",
                                        font = "Calibri 10",
                                        justify="center")
        self.macros_added.grid(row=row_num, column=2, sticky='nsew', padx=5, pady=3)

        self.run_button_added = ttk.Button(master=self.tab_two,
                                            text="Run Exp. " + str(row_num),
                                            command = partial(self.run_meta_function,row_num))
        self.run_button_added.grid(row=row_num, column=3, sticky='nsew', padx=5, pady=3)

        self.clear_button_added = ttk.Button(master=self.tab_two,
                                            text="Clear Exp. " + str(row_num),
                                            command= partial(self.clear_meta_function,row_num))
        self.clear_button_added.grid(row=row_num, column=4, sticky='nsew', padx=5, pady=3)

        self.postprocess_button_added = ttk.Button(master=self.tab_two,
                                            text="Post Process Function",
                                            command = partial(self.post_rep_meta_function,row_num),
                                            state = "disabled")
        self.postprocess_button_added.grid(row=row_num, column=5, sticky='nsew', padx=5, pady=3)
        
        
        # self.select_checkbox = tk.Checkbutton(self.tab_one,text="",state="disabled",command=partial(self.checkbox_function, self.count_experiment_queue - 1))
        # self.select_checkbox.grid(row=self.count_experiment_queue, column=7, sticky='nsew', padx=5, pady=3)
        
        self.widget_row_meta = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.clear_button_added, self.postprocess_button_added]
        self.widget_meta_list.insert(row_num-1,self.widget_row_meta)
        self.meta_experiment_master_list.insert(row_num-1,self.cross_app.crossdesign_MetaExperiment)
        # self.select_checkbox.deselect()

        self.count_meta_experiment_queue += 1
        self.notebook.select(self.tab_two)
    
    def run_meta_function(self, integer):
        row_index = integer - 1
        self.widget_meta_list[row_index][5]["state"] = "normal"
        self.widget_meta_list[row_index][3]["state"] = "disabled"


        self.my_experiment = self.meta_experiment_master_list[row_index]
        # self.macro_reps = self.selected[2]
        self.macro_reps = 10

        print(self.my_experiment.n_solvers)
        print(self.my_experiment.n_problems)
        print(self.macro_reps)

        self.my_experiment.run(n_macroreps=self.macro_reps)

    def post_rep_meta_function(self, integer):
        row_index = integer - 1
        self.selected = self.meta_experiment_master_list[row_index]
        # print(self.selected)
        self.post_rep_function_row_index = integer
        # calls postprocessing window
        self.postrep_window = tk.Tk()
        self.postrep_window.geometry("1000x600")
        self.postrep_window.title("Post Processing Page")
        self.app = Post_Processing_Window(self.postrep_window, self.my_experiment, self.selected, self, True)

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

        self.heading_list = ["Problem", "Solvers", "Select", "", "", "", "",""]
        for heading in self.heading_list:
            self.tab_three.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_three, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=5, pady=3)

        self.widget_norm_list = []
        self.normalize_list2 = []
        self.post_norm_exp_list = []

        for i,exp in enumerate(newlist):
            if exp.post_norm_ready:
                row_num = i + 1
                self.problem_added = tk.Label(master=self.tab_three,
                                                            text=exp.problem.name,
                                                            font = "Calibri 10",
                                                            justify="center")
                self.problem_added.grid(row=row_num, column=0, sticky='nsew', padx=5, pady=3)

                self.solver_added = tk.Label(master=self.tab_three,
                                                text=exp.solver.name,
                                                font = "Calibri 10",
                                                justify="center")
                self.solver_added.grid(row=row_num, column=1, sticky='nsew', padx=5, pady=3)

                self.select_checkbox = tk.Checkbutton(self.tab_three,text="",command=partial(self.checkbox_function2, exp, row_num-1))
                self.select_checkbox.grid(row=row_num, column=2, sticky='nsew', padx=5, pady=3)
                self.select_checkbox.deselect()

                self.widget_norm_list.append([self.problem_added, self.solver_added, self.select_checkbox])
    
    def post_normal_all_function(self):
        self.postrep_window = tk.Toplevel()
        self.postrep_window.geometry("1000x600")
        self.postrep_window.title("Post Processing Page")
        self.app = Post_Normal_Window(self.postrep_window, self.post_norm_exp_list, self)
        # wrapper_base.post_normalize(self.post_norm_exp_list, n_postreps_init_opt, crn_across_init_opt=True, proxy_init_val=None, proxy_opt_val=None, proxy_opt_x=None)
    
    def post_norm_return_func(self):
        # print('IN post_process_disable_button ', self.post_rep_function_row_index)
        print("youve returned")
        
    
    
class Cross_Design_Window():
    def __init__(self, master, main_widow):

        self.master = master
        self.main_window = main_widow

        self.crossdesign_title_label = ttk.Label(master=self.master,
                                                text = "Create a Cross-Design Experiment",
                                                font = "Calibri 13 bold")
        self.crossdesign_title_label.place(x=5, y=25)
        
        self.crossdesign_problem_label = ttk.Label(master=self.master,
                                                    text = "Select Problems:",
                                                    font = "Calibri 11 bold")
        self.crossdesign_problem_label.place(x=5, y=55)

        self.crossdesign_solver_label = ttk.Label(master=self.master,
                                                    text = "Select Solvers:",
                                                    font = "Calibri 11 bold")
        self.crossdesign_solver_label.place(x=145, y=55)

        self.crossdesign_checkbox_problem_list = []
        self.crossdesign_checkbox_problem_names = []
        self.crossdesign_checkbox_solver_list = [] 
        self.crossdesign_checkbox_solver_names = []

        problem_cnt = 0
        for problem in problem_directory:
            self.crossdesign_problem_checkbox_var = tk.BooleanVar(self.master, value=False)
            self.crossdesign_problem_checkbox = tk.Checkbutton(master=self.master,
                                                text = problem,
                                                variable = self.crossdesign_problem_checkbox_var)
            self.crossdesign_problem_checkbox.place(x=5, y=85+(25*problem_cnt))

            self.crossdesign_checkbox_problem_list.append(self.crossdesign_problem_checkbox_var)
            self.crossdesign_checkbox_problem_names.append(problem)
            
            problem_cnt += 1
        
        solver_cnt = 0
        for solver in solver_directory:
            self.crossdesign_solver_checkbox_var = tk.BooleanVar(self.master, value=False)
            self.crossdesign_solver_checkbox = tk.Checkbutton(master=self.master,
                                                            text = solver,
                                                            variable = self.crossdesign_solver_checkbox_var)
            self.crossdesign_solver_checkbox.place(x=145, y=85+(25*solver_cnt))

            self.crossdesign_checkbox_solver_list.append(self.crossdesign_solver_checkbox_var)
            self.crossdesign_checkbox_solver_names.append(solver)

            solver_cnt += 1

        if problem_cnt < solver_cnt:
            solver_cnt += 1
            self.crossdesign_macro_label = ttk.Label(master=self.master,
                                                    text = "Number of Macro Replications:",
                                                    font = "Calibri 11 bold")
            self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

            self.crossdesign_macro_var = tk.StringVar(self.master)
            self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT)
            self.crossdesign_macro_entry.insert(index=tk.END, string="10")
            self.crossdesign_macro_entry.place(x=15, y=105+(25*solver_cnt))

            self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Confirm Cross-Design Experiment",
                                            width = 30,
                                            command = self.confirm_cross_design_function)
            self.crossdesign_button.place(x=15, y=135+(25*solver_cnt))

        if problem_cnt > solver_cnt:
            problem_cnt += 1

            self.crossdesign_macro_label = ttk.Label(master=self.master,
                                                    text = "Number of Macro Replications:",
                                                    font = "Calibri 11 bold")
            self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

            self.crossdesign_macro_var = tk.StringVar(self.master)
            self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT)
            self.crossdesign_macro_entry.insert(index=tk.END, string="10")
            
            self.crossdesign_macro_entry.place(x=15, y=105+(25*problem_cnt))

            self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Confirm Cross-Design Experiment",
                                            width = 30,
                                            command = self.confirm_cross_design_function)
            self.crossdesign_button.place(x=15, y=135+(25*problem_cnt))

        if problem_cnt == solver_cnt:
            problem_cnt += 1
            print("problem == solver")

            self.crossdesign_macro_label = ttk.Label(master=self.master,
                                                    text = "Number of Macro Replications:",
                                                    font = "Calibri 11 bold")
            self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

            self.crossdesign_macro_var = tk.StringVar(self.master)
            self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT)
            self.crossdesign_macro_entry.insert(index=tk.END, string="10")
            self.crossdesign_macro_entry.place(x=15, y=105+(25*problem_cnt))

            self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Confirm Cross-Design Experiment",
                                            width = 30,
                                            command = self.confirm_cross_design_function)
            self.crossdesign_button.place(x=15, y=135+(25*problem_cnt))

    def confirm_cross_design_function(self):
        solver_names_list = ["ASTRODF","RNDSRCH","SANE"]
        problem_names_list = ["CNTNEWS-1","MM1-1","FACSIZE-1","FACSIZE-2","RMITD-1","SSCONT-1"]
        problem_list = []
        solver_list = []

        for checkbox in self.crossdesign_checkbox_problem_list:
            if checkbox.get() == True:
                #print(self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)] + " was selected (problem)")
                #problem_list.append(problem_directory[self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)]])
                problem_list.append(problem_names_list[self.crossdesign_checkbox_problem_list.index(checkbox)])

        for checkbox in self.crossdesign_checkbox_solver_list:
            if checkbox.get() == True:
                #print(self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)] + " was selected (solver)")
                #solver_list.append(solver_directory[self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)]])
                solver_list.append(solver_names_list[self.crossdesign_checkbox_solver_list.index(checkbox)])
        
        # macro_reps = self.crossdesign_macro_var.get()

        self.crossdesign_MetaExperiment = MetaExperiment(solver_names=solver_list, problem_names=problem_list, fixed_factors_filename="all_factors")
        
        # if self.count_meta_experiment_queue == 0:
        #     self.create_meta_exp_frame()
        self.master.destroy()
        Experiment_Window.add_meta_exp_to_frame( self.main_window)

        return self.crossdesign_MetaExperiment
        
        # print(self.crossdesign_MetaExperiment)

    def test_function(self, *args):
        print("test function connected")
    
    def get_crossdesign_MetaExperiment(self):
        return self.crossdesign_MetaExperiment

class Post_Processing_Window():
    """
    Post Processing Page of the GUI

    Arguments
    ----------
    master : tk.Tk
        Tkinter window created from Experiment_Window.run_single_function
    myexperiment : object(Experiment)
        Experiment object created in Experiment_Window.run_single_function
    experiment_list : list
        List of experiment object arguments
    """
    def __init__(self, master, myexperiment, experiment_list, main_window,meta=False):

        self.meta = meta
        self.main_window = main_window
        self.master = master
        self.my_experiment = myexperiment
        #print("my exp post pro ", experiment_list)
        self.selected = experiment_list

        self.frame = tk.Frame(self.master)

        self.title = ttk.Label(master = self.master,
                                text = "Welcome to the Post-Processing Page",
                                font = "Calibri 15 bold")

        self.n_postreps_label = ttk.Label(master = self.master,
                                    text = "Number of postreplications to take at each recommended solution:",
                                    font = "Calibri 11 bold",
                                    wraplength = "250")

        self.n_postreps_var = tk.StringVar(self.master)
        self.n_postreps_entry = ttk.Entry(master=self.master, textvariable = self.n_postreps_var, justify = tk.LEFT)
        self.n_postreps_entry.insert(index=tk.END, string="100")


        self.crn_across_budget_label = ttk.Label(master=self.master,
                                    text = "Use CRN for post-replications at solutions recommended at different times?",
                                    font = "Calibri 11 bold",
                                    wraplength = "250")

        self.crn_across_budget_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_budget_var = tk.StringVar(self.master)
        # sets the default OptionMenu selection
        # self.crn_across_budget_var.set("True")
        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.crn_across_budget_menu = ttk.OptionMenu(self.master, self.crn_across_budget_var, "True", *self.crn_across_budget_list)

        self.crn_across_macroreps_label = ttk.Label(master=self.master,
                                        text = "Use CRN for post-replications at solutions recommended on different macroreplications?",
                                        font = "Calibri 11 bold",
                                        wraplength = "250")

        self.crn_across_macroreps_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_macroreps_var = tk.StringVar(self.master)

        self.crn_across_macroreps_menu = ttk.OptionMenu(self.master, self.crn_across_macroreps_var, "False", *self.crn_across_macroreps_list)

        self.post_processing_run_label = ttk.Label(master=self.master, # window label is used for
                        text = "Finish Post-Replication of Experiment",
                        font = "Calibri 11 bold",
                        wraplength = "250")

        self.post_processing_run_button = ttk.Button(master=self.master, # window button is used in
                        # aesthetic of button and specific formatting options
                        text = "Post Process", 
                        width = 15, # width of button
                        command = self.post_processing_run_function) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click


        self.title.place(x=15, y=15)

        self.n_postreps_label.place(x=0, y=95)
        self.n_postreps_entry.place(x=255, y=95)

        # self.n_postreps_init_opt_label.place(x=0, y=180)
        # self.n_postreps_init_opt_entry.place(x=255, y=180)

        self.crn_across_budget_label.place(x=0, y=180)
        self.crn_across_budget_menu.place(x=255, y=180)

        self.crn_across_macroreps_label.place(x=0, y=275)
        self.crn_across_macroreps_menu.place(x=255, y=275)

        self.post_processing_run_label.place(x=0, y=350)
        self.post_processing_run_button.place(x=255, y=350)        

        self.frame.pack(side="top", fill="both", expand=True)
        self.run_all = all

    def post_processing_run_function(self):

        self.experiment_list = []
        # self.experiment_list = [self.selected[3], self.selected[4], self.selected[2]]
        
        # if self.n_postreps_entry.get().isnumeric() != False and self.n_postreps_init_opt_entry.get().isnumeric() != False and self.crn_across_budget_var.get() in self.crn_across_budget_list and self.crn_across_macroreps_var.get() in self.crn_across_macroreps_list:
        if self.n_postreps_entry.get().isnumeric() != False and self.crn_across_budget_var.get() in self.crn_across_budget_list and self.crn_across_macroreps_var.get() in self.crn_across_macroreps_list:
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
                        
            # reset n_postreps_entry
            self.n_postreps_entry.delete(0, len(self.n_postreps_entry.get()))
            self.n_postreps_entry.insert(index=tk.END, string="100")

            # reset n_postreps_init_opt_entry
            # self.n_postreps_init_opt_entry.delete(0, len(self.n_postreps_init_opt_entry.get()))
            # self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

            # reset crn_across_budget_bar
            self.crn_across_budget_var.set("True")

            # reset crn_across_macroreps_var 
            self.crn_across_macroreps_var.set("False")

            self.n_postreps = self.experiment_list[0] # int
            # self.n_postreps_init_opt = self.experiment_list[4] # int
            self.crn_across_budget = self.experiment_list[1] # boolean
            self.crn_across_macroreps = self.experiment_list[2] # boolean

            
            # self, n_postreps, crn_across_budget=True, crn_across_macroreps=False
            self.my_experiment.post_replicate(self.n_postreps, self.crn_across_budget, self.crn_across_macroreps)

            # print(self.experiment_list)
            self.master.destroy()
            self.post_processed_bool = True

            Experiment_Window.post_process_disable_button(self.main_window,self.meta)

            return self.experiment_list

        elif self.n_postreps_entry.get().isnumeric() == False:
            message = "Please enter a valid value for the number of post replications to take at each recommended solution."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_entry.delete(0, len(self.n_postreps_entry.get()))
            self.n_postreps_entry.insert(index=tk.END, string="100")

        # elif self.n_postreps_init_opt_entry.get().isnumeric() == False:
        #     message = "Please enter a valid value for the number of post repliactions at the initial x\u2070 and optimal x\u002A."
        #     tk.messagebox.showerror(title="Error Window", message=message)

        #     self.n_postreps_init_opt_entry.delete(0, len(self.n_postreps_init_opt_entry.get()))
        #     self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

        elif self.crn_across_macroreps_var.get() not in self.crn_across_macroreps_list:
            message = "Please answer the following question: 'Use CRN for post-replications at solutions recommended at different times?' with True or False."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.crn_across_budget_var.set("----")

        elif self.crn_across_budget_var.get() not in self.crn_across_budget_list:
            message = "Please answer the following question: 'Use CRN for post-replications at solutions recommended on different macroreplications?' with True or False."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.crn_across_macroreps_var.set("----")

        else:
            message = "You have not selected all required fields, check for '*' near input boxes."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_init_opt_entry.delete(0, len(self.n_postreps_init_opt_entry.get()))
            self.n_postreps_init_opt_entry.insert(index=tk.END, string="6")

            self.crn_across_budget_var.set("True")

            self.crn_across_macroreps_var.set("False")
        
    def select_pickle_file_fuction(self, *args):
        filename = filedialog.askopenfilename(parent = self.master,
                                            initialdir = "./",
                                            title = "Select a Pickle File",
                                            # filetypes = (("Pickle files", "*.pickle;*.pck;*.pcl;*.pkl;*.db")
                                            #              ,("Python files", "*.py")
                                            #              ,("All files", "*.*") )
                                                         )
        if filename != "":
            self.pickle_file_pathname_show["text"] = filename
            self.pickle_file_pathname_show["foreground"] = "blue"
            self.pickle_file_pathname_show.place(x=100, y=555)
        else:
            message = "You attempted to select a file but failed, please try again if necessary"
            tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)

    def test_function2(self, *args):
        print("connection enabled")

class Post_Normal_Window():
    """
    Post Normalize Page of the GUI

    Arguments
    ----------
    master : tk.Tk
        Tkinter window created from Experiment_Window.run_single_function
    myexperiment : object(Experiment)
        Experiment object created in Experiment_Window.run_single_function
    experiment_list : list
        List of experiment object arguments
    """
    def __init__(self, master, experiment_list, main_window, meta=False):
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
        print(self.all_solvers)

        #print("my exp post pro ", experiment_list)
        self.selected = experiment_list

        self.frame = tk.Frame(self.master)
        top_lab = "Welcome to the Post-Normalization Page for " + self.post_norm_exp_list[0].problem.name + " \n With Solvers"
        if self.post_norm_exp_list[0].problem.minmax[0] == 1:
            minmax = "max"
        else:
            minmax = "min"
        for solv in self.all_solvers:
            top_lab = top_lab + ", " + solv 

        self.title = ttk.Label(master = self.master,
                                text = top_lab,
                                font = "Calibri 15 bold",
                                background="#fff")

        self.n_init_label = ttk.Label(master = self.master,
                                text = "The Selected Initial Solution, x, is " + str(self.post_norm_exp_list[0].x0),
                                font = "Calibri 11 bold",
                                wraplength = "400")

        self.n_opt_label = ttk.Label(master = self.master,
                                text = "The Optimal Solution, x*, is " + str(self.post_norm_exp_list[0].xstar) + " for this " + minmax + ". \nIf the proxy optimal value and proxy optimal solution are unspecified simopt will chose the best solution ",
                                font = "Calibri 11 bold",
                                wraplength = "600")

        self.n_initial_label = ttk.Label(master = self.master,
                                text = "Initial Function Value, f(x) (optional)",
                                font = "Calibri 11 bold",
                                wraplength = "250")
        self.n_optimal_label = ttk.Label(master = self.master,
                                text = "Optimal Solution (opitional)",
                                font = "Calibri 11 bold",
                                wraplength = "250")
        self.n_proxy_val_label = ttk.Label(master = self.master,
                                text = "Proxy Optimal Value, f(x)",
                                font = "Calibri 11 bold",
                                wraplength = "250")
        self.n_proxy_sol_label = ttk.Label(master = self.master,
                                text = "Proxy Optimal Solution, x",
                                font = "Calibri 11 bold",
                                wraplength = "250")
        
        
        t = ["x","f(x)"]
        self.n_proxy_sol_entry = ttk.Entry(master=self.master, textvariable = self.proxy_sol, justify = tk.LEFT, width=10)
        self.n_proxy_val_entry = ttk.Entry(master=self.master, textvariable = self.proxy_var, justify = tk.LEFT, width=10)
        self.n_initial_entry = ttk.Entry(master=self.master, textvariable = self.init_var, justify = tk.LEFT)

        self.n_crn_label = ttk.Label(master = self.master,
                                text = "CRN for x\u2070 and optimal x\u002A?",
                                font = "Calibri 11 bold",
                                wraplength = "250")
        self.n_crn_checkbox = tk.Checkbutton(self.master,text="",variable=self.check_var)


        self.n_postreps_init_opt_label = ttk.Label(master = self.master,
                                text = "Number of post-normalizations to take at initial x\u2070 and optimal x\u002A:",
                                font = "Calibri 11 bold",
                                wraplength = "250")

        self.n_postreps_init_opt_var = tk.StringVar(self.master)
        self.n_postreps_init_opt_entry = ttk.Entry(master=self.master, textvariable = self.n_postreps_init_opt_var, justify = tk.LEFT)
        self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")
        # self.n_initial_entry.insert(index=tk.END, string=str(self.post_norm_exp_list[0].x0) )
        if self.post_norm_exp_list[0].xstar:
            self.n_proxy_sol_entry.insert(index=tk.END, string=str(self.post_norm_exp_list[0].xstar))
        # if self.post_norm_exp_list[0].xstar:
        #     self.n_proxy_sol_entry.insert(index=tk.END, string=str(self.post_norm_exp_list[0].xstar[0]))


        self.post_processing_run_label = ttk.Label(master=self.master, # window label is used for
                        text = "Finish Post-Normalization of Experiment",
                        font = "Calibri 11 bold",
                        wraplength = "250")

        self.post_processing_run_button = ttk.Button(master=self.master, # window button is used in
                        # aesthetic of button and specific formatting options
                        text = "Post Normalize", 
                        width = 15, # width of button
                        command = self.post_norm_run_function) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click
        # self.macro_var = tk.StringVar(self.master)
        # self.macro_entry = ttk.Entry(master=self.master, textvariable = self.macro_var, justify = tk.LEFT)

        self.title.place(x=15, y=15)

        self.n_init_label.place(x=0, y=70)
        
        # self.n_initial_label.place(x=0, y=95)
        # self.n_initial_entry.place(x=255, y=95)

        self.n_opt_label.place(x=0,y=150)

        # self.n_proxy_label.place(x=0, y=200)
        self.n_proxy_val_label.place(x=0,y=250)
        self.n_proxy_sol_label.place(x=300,y=250)
        self.n_proxy_val_entry.place(x=150, y=250)
        self.n_proxy_sol_entry.place(x=470, y=250)

        self.n_crn_label.place(x=0, y=300)
        self.n_crn_checkbox.place(x=255, y=300)
        #default to selected
        self.n_crn_checkbox.select()

        self.n_postreps_init_opt_label.place(x=0, y=400)
        self.n_postreps_init_opt_entry.place(x=255, y=400)
        

        self.post_processing_run_label.place(x=0, y=500)
        self.post_processing_run_button.place(x=255, y=500)        

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
            print(proxy_val,proxy_sol)
            wrapper_base.post_normalize(self.post_norm_exp_list, n_postreps_init_opt, crn_across_init_opt=crn, proxy_init_val=None, proxy_opt_val=proxy_val, proxy_opt_x=proxy_sol)
            # self.master.destroy()
            self.post_processed_bool = True

            self.postrep_window = tk.Toplevel()
            self.postrep_window.geometry("1000x800")
            self.postrep_window.title("Post Processing Page")
            self.master.destroy()
            Plot_Window(self.postrep_window, self.post_norm_exp_list, self.main_window)

            return 

        elif self.n_postreps_init_opt_entry.get().isnumeric() == False:
            message = "Please enter a valid value for the number of post replications to take at each recommended solution."
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
        def __init__(self, master, experiment_list, main_window, meta=False):
            self.master = master
            self.experiment_list = experiment_list
            self.main_window = main_window
            self.plot_types_inputs = ["cdf_solvability", "quantile_solvability","diff_cdf_solvability","diff_quantile_solvability"]
            self.plot_type_names = ["Mean Progress Curve", "Quatile Progress Curve", "Solve time cdf", "Scatter Plot", "cdf Solvability","Quantile Solvability","cdf Difference Plot", "Quanitle Difference Plot"]
            self.num_plots = 0
            self.plot_exp_list = []
            self.plot_type_list = []

            self.problem_menu = Listbox(self.master,selectmode = "multiple",exportselection=False,width=10,height=6)
            self.solver_menu = Listbox(self.master,selectmode = "multiple",exportselection=False,width=10,height=6)
            

            self.all_problems = []
            i = 0
            for problem in self.experiment_list:
                if problem.problem.name not in self.all_problems:
                    self.all_problems.append(problem.problem.name)
                    self.problem_menu.insert(i,problem.problem.name)
                    i += 1

            self.all_solvers = []
            i = 0
            for solvers in self.experiment_list:
                if solvers.solver.name not in self.all_solvers:
                    self.all_solvers.append(solvers.solver.name)
                    self.solver_menu.insert(i,solvers.solver.name)
                    i += 1

            self.instruction_label = tk.Label(master=self.master, # window label is used in
                            text = "Welcome to the Plotting Page of SimOpt \n Select which Problem and Solvers to Plot",
                            font = "Calibri 15 bold")
        
            self.problem_label = tk.Label(master=self.master, # window label is used in
                            text = "Please Select the Problem:*",
                            font = "Calibri 11 bold")
            self.plot_label = tk.Label(master=self.master, # window label is used in
                            text = "Please Select Plot Type",
                            font = "Calibri 11 bold")
            
            # from experiments.inputs.all_factors.py:
            self.problem_list = problem_directory
            # stays the same, has to change into a special type of variable via tkinter function
            self.problem_var = tk.StringVar(master=self.master)
            self.plot_var = tk.StringVar(master=self.master)

            # self.problem_menu = tk.Listbox(self.master, self.problem_var, "Problem", *self.all_problems, command=self.experiment_list[0].problem.name)
            self.plot_menu = ttk.OptionMenu(self.master, self.plot_var, "Plot", *self.plot_type_names)
            
            self.solver_label = tk.Label(master=self.master, # window label is used in
                            text = "Please select the Solver:*",
                            font = "Calibri 11 bold")

            # from experiments.inputs.all_factors.py:
            self.solver_list = solver_directory
            # stays the same, has to change into a special type of variable via tkinter function
            self.solver_var = tk.StringVar(master=self.master)
        
            # creates drop down menu, for tkinter, it is called "OptionMenu"
            # self.solver_menu = ttk.OptionMenu(self.master, self.solver_var, "Solver", *self.all_solvers)       
            
            self.add_button = ttk.Button(master=self.master,
                                        text = "Add",
                                        width = 15,
                                        command=self.add_plot)


            self.post_normal_all_button = ttk.Button(master=self.master,
                                                    text = "Plot",
                                                    width = 20,
                                                    state = "normal",
                                                    command = self.plot_button)


            self.queue_label_frame = ttk.Labelframe(master=self.master, text="Experiment")

            self.queue_canvas = tk.Canvas(master=self.queue_label_frame, borderwidth=0)

            self.queue_frame = ttk.Frame(master=self.queue_canvas)
            self.vert_scroll_bar = Scrollbar(self.queue_label_frame, orient="vertical", command=self.queue_canvas.yview)
            self.queue_canvas.configure(yscrollcommand=self.vert_scroll_bar.set)

            self.horiz_scroll_bar = Scrollbar(self.queue_label_frame, orient="horizontal", command=self.queue_canvas.xview)
            self.queue_canvas.configure(xscrollcommand=self.horiz_scroll_bar.set)

            self.vert_scroll_bar.pack(side="right", fill="y")
            self.horiz_scroll_bar.pack(side="bottom", fill="x")

            self.queue_canvas.pack(side="left", fill="both", expand=True)
            self.queue_canvas.create_window((0,0), window=self.queue_frame, anchor="nw",
                                    tags="self.queue_frame")
            
            # self.queue_frame.bind("<Configure>", self.onFrameConfigure_queue)

            self.notebook = ttk.Notebook(master=self.queue_frame)
            self.notebook.pack(fill="both")

            self.tab_one = tk.Frame(master=self.notebook)

            # 
            self.notebook.add(self.tab_one, text="Experiments to Plot")

            self.tab_one.grid_rowconfigure(0)

            self.heading_list = ["Problem", "Solver", "Plot Type", "Parameters", "Confidence Intervals"]

            for heading in self.heading_list:
                self.tab_one.grid_columnconfigure(self.heading_list.index(heading))
                label = tk.Label(master=self.tab_one, text=heading, font="Calibri 14 bold")
                label.grid(row=0, column=self.heading_list.index(heading), padx=5, pady=3)


            self.instruction_label.place(x=0, y=0)

            self.problem_label.place(x=0, y=85)
            self.problem_menu.place(x=200, y=85)

            self.solver_label.place(x=400, y=85)
            self.solver_menu.place(x=600, y=85)

            self.plot_label.place(x=0, y=250)
            self.plot_menu.place(x=200, y=250)

            self.add_button.place(x=5, y=310)

            self.post_normal_all_button.place(x=270,y=700)

            self.queue_label_frame.place(x=0, y=400, height=250, width=700)
            self.checkbox_list = []
            # self.frame.pack(fill='both')
        
        def test_funct(self):
            for i in self.solver_menu.curselection():
                print(self.solver_menu.get(i))
                
        
        def add_plot(self):
            place = self.num_plots + 1

            solverList = ""
            self.solvers = []
            for i in self.solver_menu.curselection():
                solverList = solverList + self.solver_menu.get(i) + " "
                for exp in self.experiment_list:
                    if exp.solver.name == self.solver_menu.get(i):
                        self.solvers.append(exp)
            
            problemList = ""
            probs = []
            
            for i in self.problem_menu.curselection():
                problemList = problemList + self.problem_menu.get(i) + " "
                for exp in self.experiment_list:
                    if exp.problem.name == self.problem_menu.get(i):
                        probs.append(exp)
            
            self.plot_exp_list.append(self.solvers)
            # print(probs,solvers)
            plotType = str(self.plot_var.get())
            self.plot_type_list.append(plotType)
            self.plot_params = {"cdf_solvability":[{"beta":.5},{"normalize":True},{"all_in_one":True}], "quantile_solvability":0,"diff_cdf_solvability":0,"diff_quantile_solvability":0}


            self.problem_button_added = tk.Label(master=self.tab_one,
                                                    text=problemList,
                                                    font = "Calibri 10",
                                                    justify="center")
            self.problem_button_added.grid(row=place, column=0, sticky='nsew', padx=5, pady=3)

            self.solver_button_added = tk.Label(master=self.tab_one,
                                                    text=solverList,
                                                    font = "Calibri 10",
                                                    justify="center")
            self.solver_button_added.grid(row=place, column=1, sticky='nsew', padx=5, pady=3)

            self.plot_type_button_added = tk.Label(master=self.tab_one,
                                                    text=plotType,
                                                    font = "Calibri 10",
                                                    justify="center")
            self.plot_type_button_added.grid(row=place, column=2, sticky='nsew', padx=5, pady=3)
            
            self.select_checkbox = tk.Checkbutton(self.tab_one,text="",command=partial(self.checkbox_function, place))
            self.select_checkbox.grid(row=place, column=4, sticky='nsew', padx=5, pady=3)
            self.select_checkbox.select()
            self.checkbox_list.append(True)
            

            self.num_plots += 1

        def checkbox_function(self,place):
            curVal = self.checkbox_list[place - 1]
            if curVal:
                self.checkbox_list[place - 1] = False
            else:
                self.checkbox_list[place - 1] = True

        def plot_button(self):
            self.postrep_window = tk.Toplevel()
            self.postrep_window.geometry("1000x600")
            self.postrep_window.title("Plotting Page")

            
            
            ro = 0
            c = 0
            # print(self.plot_exp_list)
            #  self.plot_types_inputs = ["cdf_solvability", "quantile_solvability","diff_cdf_solvability","diff_quantile_solvability"]
            # self.plot_type_names = ["Mean Progress Curve", "Quatile Progress Curve", "Solve time cdf", "Scatter Plot", "cdf Solvability","Quantile Solvability","cdf Difference Plot", "Quanitle Difference Plot"]
            
            for i,exp in enumerate(self.plot_exp_list):
                print(exp)
                # plotType = self.plot_var
                ci = self.checkbox_list[i]
                if self.plot_type_list[i] == "Mean Progress Curve":
                    path_name = wrapper_base.plot_progress_curves(exp,"mean",plot_CIs=ci,print_max_hw=ci)
                
                elif self.plot_type_list[i] == "Quatile Progress Curve":
                    path_name = wrapper_base.plot_progress_curves(exp,"quantile",plot_CIs=ci,print_max_hw=ci)
                
                elif self.plot_type_list[i] == "Solve time cdf":
                    path_name = wrapper_base.plot_solvability_cdfs(exp,plot_CIs=ci,print_max_hw=ci)
                
                elif self.plot_type_list[i] == "Scatter Plot":
                    path_name = wrapper_base.plot_area_scatterplots(exp,plot_CIs=ci,print_max_hw=ci)
                
                elif self.plot_type_list[i] == "cdf Solvability":
                    path_name = wrapper_base.plot_solvability_profiles(exp,"cdf_solvability",plot_CIs=ci,print_max_hw=ci)
                
                elif self.plot_type_list[i] == "Quantile Solvability":
                    path_name = wrapper_base.plot_solvability_profiles(exp,"quantile_solvability",plot_CIs=ci,print_max_hw=ci)
                
                elif self.plot_type_list[i] == "cdf Difference Plot":
                    path_name = wrapper_base.plot_solvability_profiles(exp,"diff_cdf_solvability",plot_CIs=ci,print_max_hw=ci)
                
                elif self.plot_type_list[i] == "Quanitle Difference Plot":
                    path_name = wrapper_base.plot_solvability_profiles(exp,"diff_quantile_solvability",plot_CIs=ci,print_max_hw=ci)
                
                else:
                    print(self.plot_type_list[i])
                
                width = 200
                height = 200
                img = Image.open(path_name)
                img = img.resize((width,height), Image.ANTIALIAS)
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
                print(ro,c)
                
                # panel.place(x=0,y=0)
            
            i = 0

# If we wanted to have a pop-up help message show if the user hovers over one of the widgets: https://jakirkpatrick.wordpress.com/2012/02/01/making-a-hovering-box-in-tkinter/
# and
# https://stackoverflow.com/questions/20399243/display-message-when-hovering-over-something-with-mouse-cursor-in-python (see the one with 11 upvotes)




def main(): 
    root = tk.Tk()
    root.title("SimOpt Application")
    root.geometry("1200x1000")
    root.pack_propagate(False)

    app = Experiment_Window(root)
    root.mainloop()

if __name__ == '__main__':
    main()
