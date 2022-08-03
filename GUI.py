from os import path
from random import expovariate
import tkinter as tk
from tkinter import Place, ttk, Scrollbar, filedialog
from timeit import timeit
from functools import partial
from tkinter.constants import FALSE, MULTIPLE, S
import time
from PIL import ImageTk, Image

from directory import problem_directory
from directory import solver_directory
from directory import model_directory
from experiment_base import Experiment, MetaExperiment
import experiment_base
import pickle
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

        self.instruction_label = tk.Label(master=self.master, # window label is used in
                            text = "Welcome to SimOpt \n Please Load or Add Your Experiment(s): ",
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
            text = "Select from Below",
            font = "Calibri 12")

        # from experiments.inputs.all_factors.py:
        self.problem_list = problem_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.problem_var = tk.StringVar(master=self.master)
        # sets the default OptionMenu value

        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.problem_menu = ttk.OptionMenu(self.master, self.problem_var, "Problem", *self.problem_list, command=self.show_problem_factors)

        self.solver_label = tk.Label(master=self.master, # window label is used in
                        text = "Select Solver:",
                        font = "Calibri 13")

        # from experiments.inputs.all_factors.py:
        self.solver_list = solver_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.solver_var = tk.StringVar(master=self.master)
        # sets the default OptionMenu value

        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.solver_menu = ttk.OptionMenu(self.master, self.solver_var, "Solver", *self.solver_list, command=self.show_solver_factors)

        self.macro_label = tk.Label(master=self.master,
                        text = "Number of Macroreplications:",
                        font = "Calibri 13")

        self.macro_var = tk.StringVar(self.master)
        self.macro_entry = ttk.Entry(master=self.master, textvariable = self.macro_var, justify = tk.LEFT)
        self.macro_entry.insert(index=tk.END, string="10")

        self.add_button = ttk.Button(master=self.master,
                                    text = "Add Experiment",
                                    width = 15,
                                    command=self.add_function)

        self.clear_queue_button = ttk.Button(master=self.master,
                                    text = "Clear All Experiments",
                                    width = 15,
                                    command = self.clear_queue)#(self.experiment_added, self.problem_added, self.solver_added, self.macros_added, self.run_button_added))

        self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Cross-Design Experiments",
                                            width = 20,
                                            command = self.crossdesign_function)

        self.pickle_file_load_button = ttk.Button(master=self.master,
                                                text = "Load File",
                                                width = 15,
                                                command = self.load_pickle_file_function)


        self.post_normal_all_button = ttk.Button(master=self.master,
                                                text = "Post-Normalize Selected",
                                                width = 20,
                                                state = "normal",
                                                command = self.post_normal_all_function)

        self.pickle_file_pathname_label = tk.Label(master=self.master,
                                                    text = "File Selected:",
                                                    font = "Calibri 13")

        self.pickle_file_pathname_show = tk.Label(master=self.master,
                                                    text = "No File Selected!",
                                                    font = "Calibri 12 italic",
                                                    foreground = "red",
                                                    wraplength = "500")


        self.queue_label_frame = ttk.LabelFrame(master=self.master, text="Experiment")

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


        self.tab_one = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_one, text="Queue of Experiments")

        self.tab_one.grid_rowconfigure(0)

        self.heading_list = ["Exp. Num", "Problem", "Solver", "Macroreps", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_one.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_one, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)

        self.tab_two = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_two, text="Queue of Meta-Experiments")
        self.tab_two.grid_rowconfigure(0)
        self.heading_list = ["Problems", "Solvers", "Macroreps", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_two.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_two, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)

        self.tab_three = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_three, text="Post-Normalize by Problem")
        self.tab_three.grid_rowconfigure(0)
        self.heading_list = ["Problem", "Solvers", "Select", "", "", "", "",""]

        for heading in self.heading_list:
            self.tab_three.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_three, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)

        def on_tab_change(event):
            tab = event.widget.tab('current')['text']
            if tab == 'Post-Normalize by Problem':
                self.post_norm_setup()
                self.post_normal_all_button.place(x=10,rely=.95)
            else:
                self.post_normal_all_button.place_forget()


        self.notebook.bind('<<NotebookTabChanged>>', on_tab_change)

        self.instruction_label.place(relx=.4, y=0)

        self.solver_label.place(relx=.01, rely=.1)
        self.solver_menu.place(relx=.1, rely=.1 )


        self.problem_label.place(relx=.35, rely=.1)
        self.problem_menu.place(relx=.45, rely=.1)

        self.macro_label.place(relx=.7, rely=.1)
        self.macro_entry.place(relx=.87, rely=.1, width=100)

        self.or_label.place(x=185, rely=.06)
        self.crossdesign_button.place(x=215, rely=.06, width=200)

        y_place = .06
        self.pickle_file_load_button.place(x=10, rely=y_place, width=175)
        self.or_label2.place(x=412, rely=.06)
        # self.or_label22.place(x=435, rely=.06)

        self.queue_label_frame.place(x=10, rely=.53, relheight=.39, relwidth=.99)
        # self.post_normal_all_button.place(x=400,rely=.95)

        self.frame.pack(fill='both')

        # uncomment this to test hover

        # self.l1 = tk.Button(self.master, text="Hover over me")
        # self.l2 = tk.Label(self.master, text="", width=40)
        # self.l1.place(x=10,y=0)
        # self.l2.place(x=10,y=20)

        # self.l1.bind("<Enter>", self.on_enter)
        # self.l1.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.l2.configure(text="HOVER WORKS (:")

    def on_leave(self, enter):
        self.l2.configure(text="")

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
            label_problem.grid(row=0, column=self.factor_heading_list_problem.index(heading), padx=10, pady=3)

        self.problem_object = problem_directory[self.problem_var.get()]

        count_factors_problem = 1
        for num, factor_type in enumerate(self.problem_object().specifications, start=0):
            #(factor_type, len(self.problem_object().specifications[factor_type]['default']) )

            self.dictionary_size_problem = len(self.problem_object().specifications[factor_type])

            if self.problem_object().specifications[factor_type].get("datatype") != bool:


                self.int_float_description_problem = tk.Label(master=self.factor_tab_one_problem,
                                                    text = str(self.problem_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 13",
                                                    wraplength=200)

                self.int_float_var_problem = tk.StringVar(self.factor_tab_one_problem)
                self.int_float_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.int_float_var_problem, justify = tk.LEFT)
                if args and len(args) == 2 and args[0] == True:
                    self.int_float_entry_problem.insert(index=tk.END, string=str(args[1][3][0][factor_type]))
                elif self.problem_object().specifications[factor_type].get("datatype") == tuple and len(self.problem_object().specifications[factor_type]['default']) == 1:
                    #(factor_type, len(self.problem_object().specifications[factor_type]['default']) )
                    # self.int_float_entry_problem.insert(index=tk.END, string=str(self.problem_object().specifications[factor_type].get("default")))
                    self.int_float_entry_problem.insert(index=tk.END, string=str(self.problem_object().specifications[factor_type].get("default")[0]))
                else:
                    self.int_float_entry_problem.insert(index=tk.END, string=str(self.problem_object().specifications[factor_type].get("default")))

                self.int_float_description_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
                self.int_float_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

                datatype = self.problem_object().specifications[factor_type].get("datatype")
                if datatype != tuple:
                    self.problem_factors_types.append(datatype)
                else:
                    self.problem_factors_types.append(str)

                self.problem_factors_list.append(self.int_float_var_problem)

                count_factors_problem += 1


            if self.problem_object().specifications[factor_type].get("datatype") == bool:

                self.boolean_description_problem = tk.Label(master=self.factor_tab_one_problem,
                                                    text = str(self.problem_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 13",
                                                    wraplength=200)

                self.boolean_list_problem = ["True", "False"]
                self.boolean_var_problem = tk.StringVar(self.factor_tab_one_problem)

                self.boolean_menu_problem = ttk.OptionMenu(self.factor_tab_one_problem, self.boolean_var_problem, str(self.problem_object().specifications[factor_type].get("default")), *self.boolean_list)

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
                                            font = "Calibri 13")

        self.save_var_problem = tk.StringVar(self.factor_tab_one_problem)
        self.save_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.save_var_problem, justify = tk.LEFT)
        self.save_entry_problem.insert(index=tk.END, string=oldname)

        self.save_label_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
        self.save_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

        self.problem_factors_list.append(self.save_var_problem)
        self.problem_factors_types.append(str)

        #self.factor_label_frame_problem.place(x=400, y=70, height=300, width=475)
        self.factor_label_frame_problem.place(relx=.35, rely=.15, relheight=.33, relwidth=.34)

        # Switching from Problems to Oracles

        self.oracle_factors_list = []
        self.oracle_factors_types = []

        problem = str(self.problem_var.get())
        self.oracle = problem.split("-")
        self.oracle = self.oracle[0]
        self.oracle_object = model_directory[self.oracle]

        self.factor_label_frame_oracle = ttk.LabelFrame(master=self.master, text="Model Factors")

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
            label_oracle.grid(row=0, column=self.factor_heading_list_oracle.index(heading), padx=10, pady=3)


        count_factors_oracle = 1
        for factor_type in self.oracle_object().specifications:

            self.dictionary_size_oracle = len(self.oracle_object().specifications[factor_type])

            if self.oracle_object().specifications[factor_type].get("datatype") != bool:

                #("yes?")
                self.int_float_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(self.oracle_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 13",
                                                    wraplength=200)

                self.int_float_var_oracle = tk.StringVar(self.factor_tab_one_oracle)
                self.int_float_entry_oracle = ttk.Entry(master=self.factor_tab_one_oracle, textvariable = self.int_float_var_oracle, justify = tk.LEFT, width = 20)

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

                #("yes!")
                self.boolean_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(self.oracle_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 13",
                                                    wraplength=200)

                self.boolean_list_oracle = ["True", "False"]
                self.boolean_var_oracle = tk.StringVar(self.factor_tab_one_oracle)

                self.boolean_menu_oracle = ttk.OptionMenu(self.factor_tab_one_oracle, self.boolean_var_oracle, str(self.oracle_object().specifications[factor_type].get("default")), *self.boolean_list)

                # self.boolean_datatype_oracle = tk.Label(master=self.factor_tab_one,
                #                                     text = str(self.oracle_object().specifications[factor_type].get("datatype")),
                #                                     font = "Calibri 13")

                self.boolean_description_oracle.grid(row=count_factors_oracle, column=0, sticky='nsew')
                self.boolean_menu_oracle.grid(row=count_factors_oracle, column=1, sticky='nsew')
                # self.boolean_datatype_oracle.grid(row=count_factors, column=2, sticky='nsew')

                self.oracle_factors_list.append(self.boolean_var_oracle)

                datatype = self.oracle_object().specifications[factor_type].get("datatype")
                self.oracle_factors_types.append(datatype)

                count_factors_oracle += 1

        #(self.oracle_factors_list)
        # relx=.32, rely=.08, relheight=.2, relwidth=.34

        self.factor_label_frame_oracle.place(relx=.7, rely=.15, relheight=.33, relwidth=.3)
        if str(self.solver_var.get()) != "Solver":
            self.add_button.place(x=10, rely=.48, width=200, height=30)

    def show_solver_factors(self, *args):

        self.solver_factors_list = []
        self.solver_factors_types = []

        self.factor_label_frame_solver = ttk.LabelFrame(master=self.master, text="Solver Factors")

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
            label.grid(row=0, column=self.factor_heading_list_solver.index(heading), padx=10, pady=3)

        self.solver_object = solver_directory[self.solver_var.get()]

        count_factors_solver = 1
        for factor_type in self.solver_object().specifications:
            #("size of dictionary", len(self.solver_object().specifications[factor_type]))
            #("first", factor_type)
            #("second", self.solver_object().specifications[factor_type].get("description"))
            #("third", self.solver_object().specifications[factor_type].get("datatype"))
            #("fourth", self.solver_object().specifications[factor_type].get("default"))

            self.dictionary_size = len(self.solver_object().specifications[factor_type])

            if self.solver_object().specifications[factor_type].get("datatype") != bool:

                self.int_float_description = tk.Label(master=self.factor_tab_one_solver,
                                                    text = str(self.solver_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 13",
                                                    wraplength=200)

                self.int_float_var = tk.StringVar(self.factor_tab_one_solver)
                self.int_float_entry = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.int_float_var, justify = tk.LEFT, width=15)
                if args and len(args) == 2 and args[0] == True:

                    self.int_float_entry.insert(index=tk.END, string=str(args[1][5][0][factor_type]))
                else:
                    self.int_float_entry.insert(index=tk.END, string=str(self.solver_object().specifications[factor_type].get("default")))

                # self.int_float_datatype = tk.Label(master=self.factor_tab_one,
                #                                     text = str(self.solver_object().specifications[factor_type].get("datatype")),
                #                                     font = "Calibri 13")

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
                                                    font = "Calibri 13",
                                                    wraplength=200)

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
                                            font = "Calibri 13")
        if args and len(args) == 2 and args[0] == True:
            oldname = args[1][5][1]
        else:
            oldname = self.solver_var.get()
        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.save_var_solver, justify = tk.LEFT, width=15)
        self.save_entry_solver.insert(index=tk.END, string=oldname)

        self.save_label_solver.grid(row=count_factors_solver, column=0, sticky='nsew')
        self.save_entry_solver.grid(row=count_factors_solver, column=1, sticky='nsew')

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)
        # self.factor_label_frame_problem.place(relx=.32, y=70, height=150, relwidth=.34)
        self.factor_label_frame_solver.place(x=10, rely=.15, relheight=.33, relwidth=.34)
        if str(self.problem_var.get()) != "Problem":
            self.add_button.place(x=10, rely=.48, width=200, height=30)

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
            row_of_widgets[7]["text"] = str(row_index+1)

            run_button_added = row_of_widgets[3]
            text_on_run = run_button_added["text"]
            split_text = text_on_run.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            new_text = " ".join(split_text)
            # run_button_added["text"] = new_text
            run_button_added["command"] = partial(self.run_row_function, row_index+1)

            row_of_widgets[3] = run_button_added

            viewEdit_button_added = row_of_widgets[4]
            text_on_viewEdit = viewEdit_button_added["text"]
            split_text = text_on_viewEdit.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            new_text = " ".join(split_text)
            # viewEdit_button_added["text"] = new_text
            viewEdit_button_added["command"] = partial(self.viewEdit_function, row_index+1)

            row_of_widgets[4] = viewEdit_button_added

            clear_button_added = row_of_widgets[5]
            text_on_clear = clear_button_added["text"]
            split_text = text_on_clear.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            new_text = " ".join(split_text)
            # clear_button_added["text"] = new_text
            clear_button_added["command"] = partial(self.clearRow_function, row_index+1)

            row_of_widgets[5] = clear_button_added

            postprocess_button_added = row_of_widgets[6]
            postprocess_button_added["command"] = partial(self.post_rep_function, row_index+1)

            row_of_widgets[6] = postprocess_button_added

            row_of_widgets[7].grid(row= (row_index+1), column=0, sticky='nsew', padx=10, pady=3)
            row_of_widgets[0].grid(row= (row_index+1), column=1, sticky='nsew', padx=10, pady=3)
            row_of_widgets[1].grid(row= (row_index+1), column=2, sticky='nsew', padx=10, pady=3)
            row_of_widgets[2].grid(row= (row_index+1), column=3, sticky='nsew', padx=10, pady=3)
            row_of_widgets[3].grid(row= (row_index+1), column=4, sticky='nsew', padx=10, pady=3)
            row_of_widgets[4].grid(row= (row_index+1), column=5, sticky='nsew', padx=10, pady=3)
            row_of_widgets[5].grid(row= (row_index+1), column=6, sticky='nsew', padx=10, pady=3)
            row_of_widgets[6].grid(row= (row_index+1), column=7, sticky='nsew', padx=10, pady=3)

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

            row_of_widgets[0].grid(row= (row_index+1), column=0, sticky='nsew', padx=10, pady=3)
            row_of_widgets[1].grid(row= (row_index+1), column=1, sticky='nsew', padx=10, pady=3)
            row_of_widgets[2].grid(row= (row_index+1), column=2, sticky='nsew', padx=10, pady=3)
            row_of_widgets[3].grid(row= (row_index+1), column=3, sticky='nsew', padx=10, pady=3)
            row_of_widgets[4].grid(row= (row_index+1), column=4, sticky='nsew', padx=10, pady=3)
            row_of_widgets[5].grid(row= (row_index+1), column=5, sticky='nsew', padx=10, pady=3)
            row_of_widgets[6].grid(row= (row_index+1), column=6, sticky='nsew', padx=10, pady=3)

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
        self.solver_var.set(current_experiment_arguments[1])
        self.macro_var.set(current_experiment_arguments[2])
        self.show_problem_factors(True, current_experiment_arguments)
        # self.my_experiment[1][3][1]
        self.show_solver_factors(True, current_experiment_arguments)

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

                self.my_experiment = Experiment(solver_name=self.solver_name, problem_name=self.problem_name, solver_rename=self.solver_rename, problem_rename=self.problem_rename, solver_fixed_factors=self.solver_factors, problem_fixed_factors=self.problem_factors, model_fixed_factors=self.oracle_factors)
                self.my_experiment.n_macroreps = self.selected[2]
                self.my_experiment.post_norm_ready = False

                compatibility_result = self.my_experiment.check_compatibility()
                for exp in self.experiment_object_list:
                    if exp.problem.name == self.my_experiment.problem.name:
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
                    self.problem_added.grid(row=self.count_experiment_queue, column=1, sticky='nsew', padx=10, pady=3)

                    self.exp_num = tk.Label(master=self.tab_one,
                                                    text = str(self.count_experiment_queue),
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.exp_num.grid(row=self.count_experiment_queue, column=0, sticky='nsew', padx=10, pady=3)

                    self.solver_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[5][1],
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.solver_added.grid(row=self.count_experiment_queue, column=2, sticky='nsew', padx=10, pady=3)

                    self.macros_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[2],
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.macros_added.grid(row=self.count_experiment_queue, column=3, sticky='nsew', padx=10, pady=3)

                    self.run_button_added = ttk.Button(master=self.tab_one,
                                                        text="Run" ,
                                                        command= partial(self.run_row_function, self.count_experiment_queue))
                    self.run_button_added.grid(row=self.count_experiment_queue, column=4, sticky='nsew', padx=10, pady=3)

                    self.viewEdit_button_added = ttk.Button(master=self.tab_one,
                                                        text="View / Edit" ,
                                                        command= partial(self.viewEdit_function, self.count_experiment_queue))
                    self.viewEdit_button_added.grid(row=self.count_experiment_queue, column=5, sticky='nsew', padx=10, pady=3)

                    self.clear_button_added = ttk.Button(master=self.tab_one,
                                                        text="Remove" ,
                                                        command= partial(self.clearRow_function, self.count_experiment_queue))
                    self.clear_button_added.grid(row=self.count_experiment_queue, column=6, sticky='nsew', padx=10, pady=3)

                    self.postprocess_button_added = ttk.Button(master=self.tab_one,
                                                        text="Post-Process",
                                                        command= partial(self.post_rep_function, self.count_experiment_queue),
                                                        state = "disabled")
                    self.postprocess_button_added.grid(row=self.count_experiment_queue, column=7, sticky='nsew', padx=10, pady=3)

                    self.widget_row = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.viewEdit_button_added, self.clear_button_added, self.postprocess_button_added, self.exp_num]
                    self.widget_list.insert(place,self.widget_row)

                    separator = ttk.Separator(master=self.tab_one, orient='horizontal')

                    separator.place(x=0.1, y=self.prev, relwidth=1)
                    self.prev += 30

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
            if index < len(keys):
                #(self.problem_factors_types[index])
                #datatype = self.problem_factors_types[index]

                # if the data type is tuple update data
                #self.problem_factors_dictionary[keys[index]] = datatype(nextVal)
                #(ast.literal_eval(problem_factor.get()) , keys[index])
                if keys[index] == 'initial_solution' and type(ast.literal_eval(problem_factor.get())) == int:
                    t = (ast.literal_eval(problem_factor.get()),)
                    #(t)
                    self.problem_factors_dictionary[keys[index]] = t
                else:
                    self.problem_factors_dictionary[keys[index]] = ast.literal_eval(problem_factor.get())
                #("datatype of factor -> ", type(datatype(problem_factor.get())))
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
            if index < len(keys):
                #(self.solver_factors_types[index])
                datatype = self.solver_factors_types[index]
                self.solver_factors_dictionary[keys[index]] = datatype(solver_factor.get())
                #("datatype of factor -> ", type(datatype(solver_factor.get())))
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
                    self.exp_num = tk.Label(master=self.tab_one,
                                                    text=str(self.count_experiment_queue),
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.exp_num.grid(row=self.count_experiment_queue, column=0, sticky='nsew', padx=10, pady=3)

                    self.problem_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.problem.name,
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.problem_added.grid(row=self.count_experiment_queue, column=1, sticky='nsew', padx=10, pady=3)

                    self.solver_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.solver.name,
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.solver_added.grid(row=self.count_experiment_queue, column=2, sticky='nsew', padx=10, pady=3)

                    self.macros_added = tk.Label(master=self.tab_one,
                                                    text=self.my_experiment.n_macroreps,
                                                    font = "Calibri 12",
                                                    justify="center")
                    self.macros_added.grid(row=self.count_experiment_queue, column=3, sticky='nsew', padx=10, pady=3)

                    self.run_button_added = ttk.Button(master=self.tab_one,
                                                        text="Run",
                                                        command= partial(self.run_row_function, self.count_experiment_queue))
                    self.run_button_added.grid(row=self.count_experiment_queue, column=4, sticky='nsew', padx=10, pady=3)

                    self.viewEdit_button_added = ttk.Button(master=self.tab_one,
                                                        text="View / Edit" ,
                                                        command= partial(self.viewEdit_function, self.count_experiment_queue))
                    self.viewEdit_button_added.grid(row=self.count_experiment_queue, column=5, sticky='nsew', padx=10, pady=3)

                    self.clear_button_added = ttk.Button(master=self.tab_one,
                                                        text="Remove  " ,
                                                        command= partial(self.clearRow_function, self.count_experiment_queue))
                    self.clear_button_added.grid(row=self.count_experiment_queue, column=6, sticky='nsew', padx=10, pady=3)

                    self.postprocess_button_added = ttk.Button(master=self.tab_one,
                                                        text="Post-Process",
                                                        command= partial(self.post_rep_function, self.count_experiment_queue),
                                                        state = "disabled")
                    self.postprocess_button_added.grid(row=self.count_experiment_queue, column=7, sticky='nsew', padx=10, pady=3)


                    self.widget_row = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.viewEdit_button_added, self.clear_button_added, self.postprocess_button_added, self.exp_num]
                    self.widget_list.insert(place,self.widget_row)

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

                        separator = ttk.Separator(master=self.tab_one, orient='horizontal')

                        separator.place(x=0.1, y=self.prev, relwidth=1)
                        self.prev += 30

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
        self.selected = self.experiment_object_list[row_index]
        self.post_rep_function_row_index = integer
        # calls postprocessing window
        self.postrep_window = tk.Tk()
        self.postrep_window.geometry("500x250")
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
        self.crossdesign_window.geometry("350x350")
        self.crossdesign_window.title("Cross-Design Experiments")
        self.cross_app = Cross_Design_Window(self.crossdesign_window, self)

    def add_meta_exp_to_frame(self):
        row_num = self.count_meta_experiment_queue + 1
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

        self.macros_added = tk.Label(master=self.tab_two,
                                        text="10",
                                        font = "Calibri 12",
                                        justify="center")
        self.macros_added.grid(row=row_num, column=2, sticky='nsew', padx=10, pady=3)

        self.run_button_added = ttk.Button(master=self.tab_two,
                                            text="Run" ,
                                            command = partial(self.run_meta_function,row_num))
        self.run_button_added.grid(row=row_num, column=3, sticky='nsew', padx=10, pady=3)

        self.clear_button_added = ttk.Button(master=self.tab_two,
                                            text="Remove",
                                            command= partial(self.clear_meta_function,row_num))
        self.clear_button_added.grid(row=row_num, column=4, sticky='nsew', padx=10, pady=3)

        self.postprocess_button_added = ttk.Button(master=self.tab_two,
                                            text="Post-Process",
                                            command = partial(self.post_rep_meta_function,row_num),
                                            state = "disabled")
        self.postprocess_button_added.grid(row=row_num, column=5, sticky='nsew', padx=10, pady=3)

        self.plot_button_added = ttk.Button(master=self.tab_two,
                                            text="Plot",
                                            command = partial(self.plot_meta_function,row_num),
                                            state = "disabled")
        self.plot_button_added.grid(row=row_num, column=6, sticky='nsew', padx=10, pady=3)


        # self.select_checkbox = tk.Checkbutton(self.tab_one,text="",state="disabled",command=partial(self.checkbox_function, self.count_experiment_queue - 1))
        # self.select_checkbox.grid(row=self.count_experiment_queue, column=7, sticky='nsew', padx=10, pady=3)

        self.widget_row_meta = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.clear_button_added, self.postprocess_button_added, self.plot_button_added]
        self.widget_meta_list.insert(row_num-1,self.widget_row_meta)
        self.meta_experiment_master_list.insert(row_num-1,self.cross_app.crossdesign_MetaExperiment)
        # self.select_checkbox.deselect()

        self.count_meta_experiment_queue += 1
        self.notebook.select(self.tab_two)

    def plot_meta_function(self,integer):
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
        # self.master.destroy()
        # Plot_Window(self.postrep_window, self.my_experiment.experiments[0], self, True, self.meta_experiment_master_list[row_index])
        Plot_Window(self.postrep_window, exps, self)

    def run_meta_function(self, integer):
        row_index = integer - 1
        self.widget_meta_list[row_index][5]["state"] = "normal"
        self.widget_meta_list[row_index][3]["state"] = "disabled"


        self.my_experiment = self.meta_experiment_master_list[row_index]
        # self.macro_reps = self.selected[2]
        self.macro_reps = 10

        #(self.my_experiment.n_solvers)
        #(self.my_experiment.n_problems)
        #(self.macro_reps)

        self.my_experiment.run(n_macroreps=self.macro_reps)

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

        self.heading_list = ["Problem", "Solvers", "Select", "", "", "", "",""]
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
        # wrapper_base.post_normalize(self.post_norm_exp_list, n_postreps_init_opt, crn_across_init_opt=True, proxy_init_val=None, proxy_opt_val=None, proxy_opt_x=None)

    def post_norm_return_func(self):
        #('IN post_process_disable_button ', self.post_rep_function_row_index)
        print("youve returned")

class Cross_Design_Window():
    def __init__(self, master, main_widow):

        self.master = master
        self.main_window = main_widow

        self.crossdesign_title_label = tk.Label(master=self.master,
                                                text = "Create Cross-Design Experiments",
                                                font = "Calibri 13 bold")
        self.crossdesign_title_label.place(x=10, y=25)

        self.crossdesign_problem_label = tk.Label(master=self.master,
                                                    text = "Select Problems:",
                                                    font = "Calibri 13")
        self.crossdesign_problem_label.place(x=10, y=55)

        self.crossdesign_solver_label = tk.Label(master=self.master,
                                                    text = "Select Solvers:",
                                                    font = "Calibri 13")
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
            self.crossdesign_problem_checkbox.place(x=10, y=85+(25*problem_cnt))

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
            self.crossdesign_macro_label = tk.Label(master=self.master,
                                                    text = "Number of Macroreplications:",
                                                    font = "Calibri 13")
            self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

            self.crossdesign_macro_var = tk.StringVar(self.master)
            self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT)
            self.crossdesign_macro_entry.insert(index=tk.END, string="10")
            self.crossdesign_macro_entry.place(x=15, y=105+(25*solver_cnt))

            self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Add Cross-Design Experiments",
                                            width = 30,
                                            command = self.confirm_cross_design_function)
            self.crossdesign_button.place(x=15, y=135+(25*solver_cnt))

        if problem_cnt > solver_cnt:
            problem_cnt += 1

            self.crossdesign_macro_label = tk.Label(master=self.master,
                                                    text = "Number of Macroreplications:",
                                                    font = "Calibri 13")
            self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

            self.crossdesign_macro_var = tk.StringVar(self.master)
            self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT)
            self.crossdesign_macro_entry.insert(index=tk.END, string="10")

            self.crossdesign_macro_entry.place(x=15, y=105+(25*problem_cnt))

            self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Add Cross-Design Experiments",
                                            width = 30,
                                            command = self.confirm_cross_design_function)
            self.crossdesign_button.place(x=15, y=135+(25*problem_cnt))

        if problem_cnt == solver_cnt:
            problem_cnt += 1

            self.crossdesign_macro_label = tk.Label(master=self.master,
                                                    text = "Number of Macroreplications:",
                                                    font = "Calibri 13")
            self.crossdesign_macro_label.place(x=15, y=80+(25*problem_cnt))

            self.crossdesign_macro_var = tk.StringVar(self.master)
            self.crossdesign_macro_entry = ttk.Entry(master=self.master, textvariable = self.crossdesign_macro_var, justify = tk.LEFT)
            self.crossdesign_macro_entry.insert(index=tk.END, string="10")
            self.crossdesign_macro_entry.place(x=15, y=105+(25*problem_cnt))

            self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Add Cross-Design Experiments",
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
                #(self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)] + " was selected (problem)")
                #problem_list.append(problem_directory[self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)]])
                problem_list.append(problem_names_list[self.crossdesign_checkbox_problem_list.index(checkbox)])

        for checkbox in self.crossdesign_checkbox_solver_list:
            if checkbox.get() == True:
                #(self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)] + " was selected (solver)")
                #solver_list.append(solver_directory[self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)]])
                solver_list.append(solver_names_list[self.crossdesign_checkbox_solver_list.index(checkbox)])

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
        # self.crossdesign_MetaExperiment = MetaExperiment(solver_names=solver_list, problem_names=problem_list, fixed_factors_filename="all_factors")
        self.crossdesign_MetaExperiment = MetaExperiment(solver_names=solver_list, problem_names=problem_list)

        # if self.count_meta_experiment_queue == 0:
        #     self.create_meta_exp_frame()
        self.master.destroy()
        Experiment_Window.add_meta_exp_to_frame( self.main_window)

        return self.crossdesign_MetaExperiment

        #(self.crossdesign_MetaExperiment)

    def test_function(self, *args):
        print("test function connected")

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
    def __init__(self, master, myexperiment, experiment_list, main_window,meta=False):

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
                                text = "Welcome to the Post-Processing \nand Post-Normalizing Page",
                                font = "Calibri 15 bold",justify="center")

        self.n_postreps_label = tk.Label(master = self.master,
                                    text = "Number of Postreplications at each Recommended Solution:",
                                    font = "Calibri 13",
                                    wraplength = "300")

        self.n_postreps_var = tk.StringVar(self.master)
        self.n_postreps_entry = ttk.Entry(master=self.master, textvariable = self.n_postreps_var, justify = tk.LEFT)
        self.n_postreps_entry.insert(index=tk.END, string="100")


        self.crn_across_budget_label = tk.Label(master=self.master,
                                    text = "Use CRN for Postreplications at Solutions Recommended at Different Times?",
                                    font = "Calibri 13",
                                    wraplength = "300")

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
                                        wraplength = "300")

        self.crn_across_macroreps_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_macroreps_var = tk.StringVar(self.master)

        self.crn_across_macroreps_menu = ttk.OptionMenu(self.master, self.crn_across_macroreps_var, "False", *self.crn_across_macroreps_list)

        self.crn_norm_budget_label = tk.Label(master=self.master,
                                    text = "Use CRN for Postreplications at x\u2080 and x\u002A?",
                                    font = "Calibri 13",
                                    wraplength = "300")
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
        self.n_norm_postreps_entry = ttk.Entry(master=self.master, textvariable = self.n_norm_postreps_var, justify = tk.LEFT)
        self.n_norm_postreps_entry.insert(index=tk.END, string="200")

        self.post_processing_run_label = tk.Label(master=self.master, # window label is used for
                        text = "Complete Post-Processing of the Experiment",
                        font = "Calibri 13",
                        wraplength = "300")

        if self.meta:
            self.post_processing_run_label = tk.Label(master=self.master, # window label is used for
                            text = "Complete Post-Processing and Post-Normalization of the Experiment(s)",
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
            self.crn_across_budget_menu.place(x=300, y=105)

            self.crn_across_macroreps_label.place(x=10, y=165)
            self.crn_across_macroreps_menu.place(x=300, y=165)

            self.post_processing_run_label.place(x=10, y=221)
            self.post_processing_run_button.place(x=300, y=215)
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
            # self.n_postreps_init_opt = self.experiment_list[4] # int
            self.crn_across_budget = self.experiment_list[1] # boolean
            self.crn_across_macroreps = self.experiment_list[2] # boolean

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
            top_lab = top_lab + ", " + solv

        self.title = tk.Label(master = self.master,
                                text = top_lab,
                                font = "Calibri 15 bold",
                                background="#fff",justify="center")
        initsol = self.post_norm_exp_list[0].problem.factors['initial_solution']
        if len(initsol) == 1:
            initsol = str(initsol[0])
        else:
            initsol = str(initsol)

        self.n_init_label = tk.Label(master = self.master,
                                text = "The initial solution, x\u2080, is " + initsol +".",
                                font = "Calibri 13",
                                wraplength = "400")

        self.n_opt_label = tk.Label(master = self.master,
                                text = "The optimal solution, x\u002A, is " + opt +  " for this " + minmax + "imization problem. \nIf the proxy optimal value or the proxy optimal solution is unspecified, SimOpt uses the best value found in the experiments as the proxy optimal value.",
                                font = "Calibri 13",
                                wraplength = "600",
                                justify="left")

        self.n_optimal_label = tk.Label(master = self.master,
                                text = "Optimal Solution (optional):",
                                font = "Calibri 13",
                                wraplength = "250")
        self.n_proxy_val_label = tk.Label(master = self.master,
                                text = "Proxy Optimal Value, f(x\u002A):",
                                font = "Calibri 13",
                                wraplength = "250")
        self.n_proxy_sol_label = tk.Label(master = self.master,
                                text = "Proxy Optimal Solution, x\u002A:",
                                font = "Calibri 13",
                                wraplength = "250")


        t = ["x","f(x)"]
        self.n_proxy_sol_entry = ttk.Entry(master=self.master, textvariable = self.proxy_sol, justify = tk.LEFT, width=10)
        self.n_proxy_val_entry = ttk.Entry(master=self.master, textvariable = self.proxy_var, justify = tk.LEFT, width=10)
        self.n_initial_entry = ttk.Entry(master=self.master, textvariable = self.init_var, justify = tk.LEFT)

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
        self.n_postreps_init_opt_entry = ttk.Entry(master=self.master, textvariable = self.n_postreps_init_opt_var, justify = tk.LEFT)
        self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

        self.post_processing_run_label = tk.Label(master=self.master, # window label is used for
                        text = "Complete Post-Normalization of the Experiment(s)",
                        font = "Calibri 13",
                        wraplength = "310")

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
        self.n_proxy_val_entry.place(x=190, y=190)
        self.n_proxy_sol_entry.place(x=500, y=190)

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
            experiment_base.post_normalize(self.post_norm_exp_list, n_postreps_init_opt, crn_across_init_opt=crn, proxy_init_val=None, proxy_opt_val=proxy_val, proxy_opt_x=proxy_sol)
            # self.master.destroy()
            self.post_processed_bool = True

            self.postrep_window = tk.Toplevel()
            self.postrep_window.geometry("1000x800")
            self.postrep_window.title("Plotting Page")
            self.master.destroy()
            Plot_Window(self.postrep_window, self.post_norm_exp_list, self.main_window)

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
        def __init__(self, master, experiment_list, main_window, meta=False, metaList=None):

            self.metaList = metaList
            self.master = master
            self.experiment_list = experiment_list
            self.main_window = main_window
            self.plot_types_inputs = ["cdf_solvability", "quantile_solvability","diff_cdf_solvability","diff_quantile_solvability"]
            self.plot_type_names = ["Mean Progress Curve", "Quatile Progress Curve", "Solve Time CDF", "Scatter Plot", "CDF Solvability","Quantile Solvability","CDF Difference Plot", "Quanitle Difference Plot"]
            self.num_plots = 0
            self.plot_exp_list = []
            self.plot_type_list = []
            self.checkbox_list = []
            self.plot_CI_list = []
            self.plot_param_list = []
            self.all_path_names = []
            self.bad_label = None
            self.plot_var = tk.StringVar(master=self.master)

            self.params = [tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master)]

            self.problem_menu = Listbox(self.master,selectmode = "multiple",exportselection=False, width=10,height=6)
            self.solver_menu = Listbox(self.master,selectmode = "multiple",exportselection=False, width=10,height=6)


            self.all_problems = []
            i = 0
            for problem in self.experiment_list:
                if problem.problem.name not in self.all_problems:
                    self.all_problems.append(problem.problem.name)
                    self.problem_menu.insert(i,problem.problem.name)
                    i += 1



            #("solvers:",self.all_solvers)
            if meta:
                i = 0
                for name in self.metaList.solver_names:
                    self.solver_menu.insert(i,name)
                    i += 1
            else:
                self.all_solvers = []
                i = 0
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
            self.problem_list = problem_directory
            # stays the same, has to change into a special type of variable via tkinter function
            self.problem_var = tk.StringVar(master=self.master)


            # self.problem_menu = tk.Listbox(self.master, self.problem_var, "Problem", *self.all_problems, command=self.experiment_list[0].problem.name)
            self.plot_menu = ttk.OptionMenu(self.master, self.plot_var, "Plot", *self.plot_type_names, command=partial(self.get_parameters, self.plot_var))

            self.solver_label = tk.Label(master=self.master, # window label is used in
                            text = "Select Solver(s):*",
                            font = "Calibri 13")

            # from experiments.inputs.all_factors.py:
            self.solver_list = solver_directory
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


            self.queue_label_frame = ttk.LabelFrame(master=self.master, text="Plots")

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

            self.notebook = ttk.Notebook(master=self.queue_frame)
            self.notebook.pack(fill="both")

            self.tab_one = tk.Frame(master=self.notebook)

            #
            self.notebook.add(self.tab_one, text="Experiments to Plot")

            self.tab_one.grid_rowconfigure(0)

            self.heading_list = ["Problem", "Solver", "Plot Type", "Clear Row", "View Plot", "Parameters", "PNG File Path"]

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

            self.add_button.place(relx=.4, rely=.38)

            separator = ttk.Separator(master=self.master, orient='horizontal')
            separator.place(relx=0.35, rely=.08, relheight=.4)

            self.post_normal_all_button.place(relx=.3,rely=.95)

            self.queue_label_frame.place(x=10, rely=.5, relheight=.4, relwidth=1)

            self.param_label = []
            self.param_entry = []
            self.factor_label_frame_problem = None

            self.CI_label_frame = ttk.LabelFrame(master=self.master, text="Plot Parameters (Optional)")
            self.CI_canvas = tk.Canvas(master=self.CI_label_frame, borderwidth=0)
            self.CI_frame = ttk.Frame(master=self.CI_canvas)

            self.CI_canvas.pack(side="left", fill="both", expand=True)
            self.CI_canvas.create_window((0,0), window=self.CI_frame, anchor="nw",
                                    tags="self.queue_frame")

            self.CI_label_frame.place(relx=.4, rely=.15, relheight=.2, relwidth=.3)

            self.settings_label_frame = ttk.LabelFrame(master=self.master, text="Plot Settings (Optional)")
            self.settings_canvas = tk.Canvas(master=self.settings_label_frame, borderwidth=0)
            self.settings_frame = ttk.Frame(master=self.settings_canvas)

            self.settings_canvas.pack(side="left", fill="both", expand=True)
            self.settings_canvas.create_window((0,0), window=self.settings_frame, anchor="nw",
                                    tags="self.queue_frame")
            self.settings_canvas.grid_rowconfigure(0)

            tf_list = ['True','False']
            self.settings_label_frame.place(relx=.65, rely=.15, relheight=.2, relwidth=.3)

            entry1 = tk.Checkbutton(self.settings_canvas, variable=self.params[0], onvalue="True", offvalue="False")
            entry1.select()
            # entry1 = ttk.OptionMenu(self.settings_canvas, self.params[0], "True", *tf_list)
            label1 = tk.Label(master=self.settings_canvas, text="Confidence Intervals", font="Calibri 14")
            label1.grid(row=0, column=0, padx=10, pady=3)
            entry1.grid(row=0, column=1, padx=10, pady=3)

            entry = tk.Checkbutton(self.settings_canvas, variable=self.params[1], onvalue="True", offvalue="False")
            entry.select()
            label = tk.Label(master=self.settings_canvas, text="Plot Together", font="Calibri 14")
            label.grid(row=1, column=0, padx=10, pady=3)
            entry.grid(row=1, column=1, padx=10, pady=3)

            entry2 = tk.Checkbutton(self.settings_canvas, variable=self.params[2], onvalue="True", offvalue="False")
            entry2.select()
            label2 = tk.Label(master=self.settings_canvas, text="Print Max HW", font="Calibri 14")
            label2.grid(row=2, column=0, padx=10, pady=3)
            entry2.grid(row=2, column=1, padx=10, pady=3)


            # self.frame.pack(fill='both')

        def test_funct(self):
            for i in self.solver_menu.curselection():
                print(self.solver_menu.get(i))

        def add_plot(self):


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
            if len(probs) == 0 or len(self.solvers) == 0 or str(self.plot_var.get()) == "Plot":
                txt = "At least 1 Problem, 1 Solver, and 1 Plot Type must be selected."
                self.bad_label = tk.Label(master=self.master,text=txt,font = "Calibri 12",justify="center")
                self.bad_label.place(relx=.5, rely=.45)
                return
            elif self.bad_label != None:
                self.bad_label.destroy()
                self.bad_label = None

            self.plot_exp_list.append(self.solvers)

            plotType = str(self.plot_var.get())
            self.plot_type_list.append(plotType)

            i = len(self.plot_type_list)-1
            exp = self.plot_exp_list[len(self.plot_exp_list)-1]
            exp2 = [[e] for e in exp]
            #keep as list of list for multiple solvers if using exp2
            #one problem, multiple solvers

            param_value_list = []
            for t in self.params:
                #(t.get())
                if t.get() == "True":
                    param_value_list.append(True)
                elif t.get() == "False":
                    param_value_list.append(False)
                elif t.get() != "":
                    try:
                        param_value_list.append(float(t.get()))
                    except ValueError:
                        param_value_list.append(t.get())

            ci = param_value_list[0]
            hw = param_value_list[2]

            if self.plot_type_list[i] == "Mean Progress Curve":
                path_name = experiment_base.plot_progress_curves(exp,"mean", plot_CIs=ci, all_in_one=param_value_list[1], print_max_hw=hw, normalize=param_value_list[3])
                param_list = {"plot CIs":ci, "print max hw":hw, "normalize":param_value_list[3]}
            elif self.plot_type_list[i] == "Quatile Progress Curve":
                path_name = experiment_base.plot_progress_curves(exp,"quantile", plot_CIs=ci, all_in_one=param_value_list[1], print_max_hw=hw, beta=param_value_list[3],normalize=param_value_list[4])
                param_list = {"plot CIs":ci, "print max hw":hw, "normalize":param_value_list[4], "beta":param_value_list[3]}
            elif self.plot_type_list[i] == "Solve time CDF":
                path_name = experiment_base.plot_solvability_cdfs(exp, plot_CIs=ci, print_max_hw=hw, solve_tol=param_value_list[3])
                param_list = {"plot CIs":ci, "print max hw":hw, "solve tol":param_value_list[3]}
            elif self.plot_type_list[i] == "Scatter Plot":
                path_name = experiment_base.plot_area_scatterplots(exp2, plot_CIs=ci, print_max_hw=hw)
                param_list = {}
            elif self.plot_type_list[i] == "CDF Solvability":
                path_name = experiment_base.plot_solvability_profiles(exp2, "cdf_solvability", plot_CIs=ci, print_max_hw=hw, solve_tol=param_value_list[3],ref_solver=None)
                param_list = {"plot CIs":ci, "print max hw":hw, "solve tol":param_value_list[3]}
            elif self.plot_type_list[i] == "Quantile Solvability":
                param_list = {"plot CIs":ci, "print max hw":hw, "solve tol":param_value_list[3],"beta":param_value_list[4]}
                path_name = experiment_base.plot_solvability_profiles(exp2, "quantile_solvability", plot_CIs=ci, print_max_hw=hw, solve_tol=param_value_list[3],beta=param_value_list[4],ref_solver=None)
            elif self.plot_type_list[i] == "CDF Difference Plot":
                param_list = {"plot CIs":ci, "print max hw":hw, "solve tol":param_value_list[3],"ref solver":param_value_list[4]}
                path_name = experiment_base.plot_solvability_profiles(exp2, "diff_cdf_solvability", plot_CIs=ci,print_max_hw=hw,solve_tol=param_value_list[3],ref_solver=param_value_list[4])
            elif self.plot_type_list[i] == "Quanitle Difference Plot":
                param_list = {"plot CIs":ci, "print max hw":hw, "solve tol":param_value_list[3],"ref solver":param_value_list[5],"beta":param_value_list[4]}
                path_name = experiment_base.plot_solvability_profiles(exp2, "diff_quantile_solvability", plot_CIs=ci,print_max_hw=hw,solve_tol=param_value_list[3],beta=param_value_list[4],ref_solver=param_value_list[5])
            else:
                print(self.plot_type_list[i])
            for i,new_plot in enumerate(path_name):
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


                self.clear_plot = tk.Button(master=self.tab_one,
                                                        text="Clear",
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
                self.get_parameters(0, self.plot_var.get())

        def get_parameters(self,a, plot_choice):
            # ref solver needs to a drop down of solvers that is selected in the problem
            # numbers between 0 and 1
            # checkbox for normalize
            # move CI to parameters
            # checkbox with print_max_hw checkbox
            # remove CI from experiment box

            # beta=0.50, normalize=True
            if plot_choice == "Mean Progress Curve":
                param_list = {'normalize':True}
            elif plot_choice == "Quatile Progress Curve":
                param_list = {'beta':0.50, 'normalize':True}
            elif plot_choice == "Solve Time CDF":
                param_list = {'solve_tol':0.1}
            elif plot_choice == "Scatter Plot":
                param_list = {}
            elif plot_choice == "CDF Solvability":
                param_list = {'solve_tol':0.1}
            elif plot_choice == "Quantile Solvability":
                param_list = {'solve_tol':0.1, 'beta':0.5}
            elif plot_choice == "CDF Difference Plot":
                param_list = {'solve_tol':0.1, 'ref_solver':None}
            elif plot_choice == "Quanitle Difference Plot":
                param_list = {'solve_tol':0.1, 'beta':0.5, 'ref_solver':None}
            else:
                print("invalid plot?")
            self.param_list = param_list


            # self.params = [tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master)]

            self.CI_label_frame.destroy()
            self.CI_label_frame = ttk.LabelFrame(master=self.master, text="Plot Parameters (Optional)")
            self.CI_canvas = tk.Canvas(master=self.CI_label_frame, borderwidth=0)
            self.CI_frame = ttk.Frame(master=self.CI_canvas)

            self.CI_canvas.pack(side="left", fill="both", expand=True)
            self.CI_canvas.create_window((0,0), window=self.CI_frame, anchor="nw",
                                    tags="self.queue_frame")
            self.CI_canvas.grid_rowconfigure(0)

            self.CI_label_frame.place(relx=.4, rely=.15, relheight=.2, relwidth=.25)


            tf_list = ['True','False']

            self.solvers_names = []
            for i in self.solver_menu.curselection():
                self.solvers_names.append(self.solver_menu.get(i))


            i = 1
            for param, param_val in param_list.items():

                if param == 'normalize':
                    entry = ttk.OptionMenu(self.CI_canvas, self.params[i+2], "True", *tf_list)
                    label = tk.Label(master=self.CI_canvas, text="Normalize By Relative Optimality Gap", font="Calibri 14", wraplength="200")
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                elif param == 'ref_solver':
                    label = tk.Label(master=self.CI_canvas, text="Select Solver", font="Calibri 14")
                    if len(self.solvers_names) != 0:
                        label = tk.Label(master=self.CI_canvas, text="Benchmark Solver", font="Calibri 14")
                        entry = ttk.OptionMenu(self.CI_canvas, self.params[i+2], self.solvers_names[0], *self.solvers_names)
                        entry.grid(row=i, column=1, padx=10, pady=3)
                    label.grid(row=i, column=0, padx=10, pady=3)
                elif param == 'solve_tol':
                    label = tk.Label(master=self.CI_canvas, text="Optimality Gap Threshold", font="Calibri 14", wraplength="100")
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry = ttk.Entry(master=self.CI_canvas, textvariable = self.params[i+2], justify = tk.LEFT)
                    if param_val is not None:
                        entry.delete(0, 'end')
                        entry.insert(index=tk.END, string=param_val)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                elif param == 'beta':
                    label = tk.Label(master=self.CI_canvas, text="Quantile Probability", font="Calibri 14", wraplength="100")
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry = ttk.Entry(master=self.CI_canvas, textvariable = self.params[i+2], justify = tk.LEFT)
                    if param_val is not None:
                        entry.delete(0, 'end')
                        entry.insert(index=tk.END, string=param_val)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                else:
                    label = tk.Label(master=self.CI_canvas, text=param, font="Calibri 14")
                    label.grid(row=i, column=0, padx=10, pady=3)
                    entry = ttk.Entry(master=self.CI_canvas, textvariable = self.params[i+2], justify = tk.LEFT)
                    if param_val is not None:
                        entry.delete(0, 'end')
                        entry.insert(index=tk.END, string=param_val)
                    entry.grid(row=i, column=1, padx=10, pady=3)
                i += 1

        def clear_row(self, place):
            # self.plot_CI_list.pop(place)
            # self.plot_exp_list.pop(place)
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
            #(self.plot_exp_list)
            #  self.plot_types_inputs = ["cdf_solvability", "quantile_solvability","diff_cdf_solvability","diff_quantile_solvability"]
            # self.plot_type_names = ["Mean Progress Curve", "Quatile Progress Curve", "Solve time cdf", "Scatter Plot", "cdf Solvability","Quantile Solvability","cdf Difference Plot", "Quanitle Difference Plot"]

            for i,path_name in enumerate(self.all_path_names):

                width = 350
                height = 350
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
                if c == 3:
                    c = 0
                    ro += 1

                # panel.place(x=10,y=0)

            i = 0

        def view_one_pot(self, path_name):
            self.postrep_window = tk.Toplevel()
            self.postrep_window.geometry("400x400")
            self.postrep_window.title("View One Plot")

            ro = 0
            c = 0

            width = 400
            height = 400
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


def main():
    root = tk.Tk()
    root.title("SimOpt Library Graphical User Interface")
    root.geometry("1200x1000")
    root.pack_propagate(False)

    app = Experiment_Window(root)
    root.mainloop()

if __name__ == '__main__':
    main()
