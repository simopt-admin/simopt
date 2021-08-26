import tkinter as tk
from tkinter import ttk, Scrollbar, filedialog
from timeit import timeit
from functools import partial

from directory import problem_directory
from directory import solver_directory
from directory import oracle_directory
from wrapper_base import Experiment, MetaExperiment
import pickle

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

        self.experiment_master_list = []
        self.widget_list = []
        self.experiment_object_list = []
        self.count_experiment_queue = 1
        
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
                        width = 15, # width of button
                        command = self.run_single_function) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click)

        self.add_button = ttk.Button(master=self.master,
                                    text = "Add Experiment",
                                    width = 15,
                                    command=self.add_function)

        self.clear_queue_button = ttk.Button(master=self.master,
                                    text = "Clear All Experiments",
                                    width = 20,
                                    command = self.clear_queue)#(self.experiment_added, self.problem_added, self.solver_added, self.macros_added, self.run_button_added))

        self.crossdesign_button = ttk.Button(master=self.master,
                                            text = "Cross-Design Experiment",
                                            width = 25,
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

        self.tab_one = tk.Frame(master=self.notebook)

        self.notebook.add(self.tab_one, text="Queue of Experiments")

        self.tab_one.grid_rowconfigure(0)

        self.heading_list = ["Problem", "Solver", "Macro Reps", "", "", "", ""]

        for heading in self.heading_list:
            self.tab_one.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(master=self.tab_one, text=heading, font="Calibri 14 bold")
            label.grid(row=0, column=self.heading_list.index(heading), padx=5, pady=3)

        #

        # self.crossdesign_title_label = ttk.Label(master=self.master,
        #                                         text = "Create a Cross-Design Experiment",
        #                                         font = "Calibri 13 bold")
        # self.crossdesign_title_label.place(x=830, y=425)
        
        # self.crossdesign_problem_label = ttk.Label(master=self.master,
        #                                             text = "Select Problems:",
        #                                             font = "Calibri 11 bold")
        # self.crossdesign_problem_label.place(x=830, y=455)

        # self.crossdesign_solver_label = ttk.Label(master=self.master,
        #                                             text = "Select Solvers:",
        #                                             font = "Calibri 11 bold")
        # self.crossdesign_solver_label.place(x=970, y=455)

        # self.crossdesign_checkbox_problem_list = []
        # self.crossdesign_checkbox_problem_names = []
        # self.crossdesign_checkbox_solver_list = [] 
        # self.crossdesign_checkbox_solver_names = []

        # problem_cnt = 0
        # for problem in problem_directory:
        #     self.crossdesign_problem_checkbox_var = tk.BooleanVar()
        #     self.crossdesign_problem_checkbox = tk.Checkbutton(master=self.master,
        #                                         text = problem,
        #                                         variable = self.crossdesign_problem_checkbox_var)
        #     self.crossdesign_problem_checkbox.place(x=830, y=485+(25*problem_cnt))

        #     self.crossdesign_checkbox_problem_list.append(self.crossdesign_problem_checkbox_var)
        #     self.crossdesign_checkbox_problem_names.append(problem)
            
        #     problem_cnt += 1
        
        # solver_cnt = 0
        # for solver in solver_directory:
        #     self.crossdesign_solver_checkbox_var = tk.BooleanVar()
        #     self.crossdesign_solver_checkbox = tk.Checkbutton(master=self.master,
        #                                                     text = solver,
        #                                                     variable = self.crossdesign_solver_checkbox_var)
        #     self.crossdesign_solver_checkbox.place(x=970, y=485+(25*solver_cnt))

        #     self.crossdesign_checkbox_solver_list.append(self.crossdesign_solver_checkbox_var)
        #     self.crossdesign_checkbox_solver_names.append(solver)

        #     solver_cnt += 1

        self.instruction_label.place(x=0, y=0)
        #self.instruction_label.grid(row=0, column=1)

        self.problem_label.place(x=0, y=85)
        #self.problem_label.grid(row=1, column=0, pady=25)
        self.problem_menu.place(x=225, y=85)
        #self.problem_menu.grid(row=1, column=0, sticky='s')

        self.solver_label.place(x=0, y=165)
        #self.solver_label.grid(row=2, column=0, pady=25)
        self.solver_menu.place(x=225, y=165)
        #self.solver_menu.grid(row=2, column=0, sticky='s')

        self.macro_label.place(x=0, y=245)
        #self.macro_label.grid(row=3, column=0, pady=25)
        self.macro_entry.place(x=225, y=245)
        #self.macro_entry.grid(row=3, column=0, sticky='s')

        self.run_button.place(x=5, y=285)
        self.crossdesign_button.place(x=115, y=285)
        self.add_button.place(x=5, y=325)

        self.pickle_file_select_label.place(x=850, y=375)
        self.pickle_file_select_button.place(x=1040, y=375)
        self.pickle_file_load_button.place(x=1150, y=375)
        self.pickle_file_pathname_label.place(x=850, y=400)
        self.pickle_file_pathname_show.place(x=950, y=400)

        #self.add_button.grid(row=4, column=0, pady=25)

        #self.run_button.grid(row=4, column=0, sticky='s')

        self.clear_queue_button.place(x=115, y=325)

        # self.pathname_label.grid(row=3, column=1, sticky='n')
        # self.pathname_entry.grid(row=3, column=1, sticky='s')
        # self.pathname_button.grid(row=3, column=1)

        self.queue_label_frame.place(x=0, y=375, height=400, width=800)

        self.frame.pack(fill='both')

    def show_problem_factors(self, *args):
        # print("Got the problem: ", self.problem_var.get())
        if args and len(args) == 2:
            print("ARGS: ", args[1])
        print("arg length:", len(args))

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

        # self.problem_factor_confirm_button = ttk.Button(master=self.master, # window button is used in
        #             # aesthetic of button and specific formatting options
        #             text = "Confirm Problem Factors",
        #             command = self.confirm_problem_factors)

        # self.problem_factor_confirm_button.place(x=80, y=115)

        self.problem_object = problem_directory[self.problem_var.get()]

        count_factors_problem = 1
        for num, factor_type in enumerate(self.problem_object().specifications, start=0):
            # print("size of dictionary", len(self.problem_object().specifications[factor_type]))
            # print("first", factor_type)
            # print("second", self.problem_object().specifications[factor_type].get("description"))
            # print("third", self.problem_object().specifications[factor_type].get("datatype"))    
            # print("fourth", self.problem_object().specifications[factor_type].get("default"))   

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

                print("yes!")
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
                print(datatype)

                self.problem_factors_list.append(self.boolean_var_problem)
                self.problem_factors_types.append(datatype)

                count_factors_problem += 1

        count_factors_problem += 1

        self.save_label_problem = tk.Label(master=self.factor_tab_one_problem,
                                            text = "Save Problem As",
                                            font = "Calibri 11 bold")

        self.save_var_problem = tk.StringVar(self.factor_tab_one_problem)
        self.save_entry_problem = ttk.Entry(master=self.factor_tab_one_problem, textvariable = self.save_var_problem, justify = tk.LEFT)
        self.save_entry_problem.insert(index=tk.END, string=self.problem_var.get())

        self.save_label_problem.grid(row=count_factors_problem, column=0, sticky='nsew')
        self.save_entry_problem.grid(row=count_factors_problem, column=1, sticky='nsew')

        self.problem_factors_list.append(self.save_var_problem)
        self.problem_factors_types.append(str)

        # print(self.problem_factors_list)
        self.factor_label_frame_problem.place(x=400, y=70, height=150, width=475)

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

        # self.oracle_factor_confirm_button = ttk.Button(master=self.master, # window button is used in
        #             # aesthetic of button and specific formatting options
        #             text = "Confirm Oracle Factors",
        #             command = self.confirm_oracle_factors)

        # self.oracle_factor_confirm_button.place(x=220, y=115)

        count_factors_oracle = 1
        for factor_type in self.oracle_object().specifications:
            # print("size of dictionary", len(self.oracle_object().specifications[factor_type]))
            # print("first", factor_type)
            # print("second", self.oracle_object().specifications[factor_type].get("description"))
            # print("third", self.oracle_object().specifications[factor_type].get("datatype"))    
            # print("fourth", self.oracle_object().specifications[factor_type].get("default"))   

            self.dictionary_size_oracle = len(self.oracle_object().specifications[factor_type])

            if self.oracle_object().specifications[factor_type].get("datatype") != bool:

                # print("yes?")
                self.int_float_description_oracle = tk.Label(master=self.factor_tab_one_oracle,
                                                    text = str(self.oracle_object().specifications[factor_type].get("description")),
                                                    font = "Calibri 11 bold")

                self.int_float_var_oracle = tk.StringVar(self.factor_tab_one_oracle)
                self.int_float_entry_oracle = ttk.Entry(master=self.factor_tab_one_oracle, textvariable = self.int_float_var_oracle, justify = tk.LEFT, width = "50")
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
        self.factor_label_frame_oracle.place(x=900, y=70, height=300, width=600)

    def show_solver_factors(self, *args):
        # print("Got the solver: ", self.solver_var.get())

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

        # self.solver_factor_confirm_button = ttk.Button(master=self.master, # window button is used in
        #             # aesthetic of button and specific formatting options
        #             text = "Confirm Solver Factors",
        #             command = self.confirm_solver_factors)

        # self.solver_factor_confirm_button.place(x=220, y=195)

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

        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(master=self.factor_tab_one_solver, textvariable = self.save_var_solver, justify = tk.LEFT)
        self.save_entry_solver.insert(index=tk.END, string=self.solver_var.get())

        self.save_label_solver.grid(row=count_factors_solver, column=0, sticky='nsew')
        self.save_entry_solver.grid(row=count_factors_solver, column=1, sticky='nsew')

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)

        self.factor_label_frame_solver.place(x=400, y=220, height=150, width=475)

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

            self.factor_label_frame_problem.destroy()
            self.factor_label_frame_oracle.destroy()
            self.factor_label_frame_solver.destroy()

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

                    tk.messagebox.showinfo(title="Status Update", message="Function will now begin running")

                    self.my_experiment.run(n_macroreps=self.macro_reps)

                    # calls postprocessing window
                    self.postrep_window = tk.Tk()
                    self.postrep_window.geometry("1500x1000")
                    self.postrep_window.title("Post Processing Page")
                    self.app = Post_Processing_Window(self.postrep_window, self.my_experiment, self.selected)

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

    def crossdesign_function(self):
        self.crossdesign_window = tk.Tk()
        self.crossdesign_window.geometry("300x350+830+425")
        self.crossdesign_window.title("Cross-Design Experiment")
        self.app = Cross_Design_Window(self.crossdesign_window)

    def clearRow_function(self, integer):
        print("this is the integer passed in by the lambda function", integer)
        
        for widget in self.widget_list[integer-1]:
            widget.grid_remove()

        print(F"Size of self.experiment_master_list BEFORE running is { len(self.experiment_master_list) }")
        print(F"Size of self.experiment_object_list BEFORE running is { len(self.experiment_object_list) }")
        print(F"Size of self.wiedget_list BEFORE running is { len(self.widget_list) }")

        self.experiment_master_list.pop(integer-1)      
        self.experiment_object_list.pop(integer-1)
        self.widget_list.pop(integer-1)

        print(F"Size of self.experiment_master_list AFTER running is { len(self.experiment_master_list) }")
        print(F"Size of self.experiment_object_list AFTER running is { len(self.experiment_object_list) }")
        print(F"Size of self.wiedget_list AFTER running is { len(self.widget_list) }")


        for row_of_widgets in self.widget_list:
            row_index = self.widget_list.index(row_of_widgets)
            print("row_index = ", row_index)

            run_button_added = row_of_widgets[3]
            text_on_run = run_button_added["text"]
            # print(text_on_run)
            # print("BEFORE: ", text_on_run.split(" "))
            split_text = text_on_run.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            # print("AFTER: ", split_text)
            new_text = " ".join(split_text)
            # print("new_text = ", new_text)
            run_button_added["text"] = new_text
            run_button_added["command"] = partial(self.run_row_function, row_index+1)

            row_of_widgets[3] = run_button_added

            viewEdit_button_added = row_of_widgets[4]
            text_on_viewEdit = viewEdit_button_added["text"]
            # print(text_on_viewEdit)
            # print("BEFORE: ", text_on_viewEdit.split(" "))
            split_text = text_on_viewEdit.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            # print("AFTER: ", split_text)
            new_text = " ".join(split_text)
            # print("new_text = ", new_text)
            viewEdit_button_added["text"] = new_text
            viewEdit_button_added["command"] = partial(self.viewEdit_function, row_index+1)

            row_of_widgets[4] = viewEdit_button_added

            clear_button_added = row_of_widgets[5]
            text_on_clear = clear_button_added["text"]
            # print(text_on_clear)
            # print("BEFORE: ", text_on_clear.split(" "))
            split_text = text_on_clear.split(" ")
            split_text[len(split_text)-1] = str(row_index+1)
            # print("AFTER: ", split_text)
            new_text = " ".join(split_text)
            # print("new_text = ", new_text)
            clear_button_added["text"] = new_text
            clear_button_added["command"] = partial(self.clearRow_function, row_index+1)   

            row_of_widgets[5] = clear_button_added

            postprocess_button_added = row_of_widgets[6]
            postprocess_button_added["command"] = partial(self.test_function, row_index+1)   

            row_of_widgets[6] = postprocess_button_added

            row_of_widgets[0].grid(row= (row_index+1), column=0, sticky='nsew', padx=5, pady=3)
            row_of_widgets[1].grid(row= (row_index+1), column=1, sticky='nsew', padx=5, pady=3)
            row_of_widgets[2].grid(row= (row_index+1), column=2, sticky='nsew', padx=5, pady=3)
            row_of_widgets[3].grid(row= (row_index+1), column=3, sticky='nsew', padx=5, pady=3)
            row_of_widgets[4].grid(row= (row_index+1), column=4, sticky='nsew', padx=5, pady=3)
            row_of_widgets[5].grid(row= (row_index+1), column=5, sticky='nsew', padx=5, pady=3)
            row_of_widgets[6].grid(row= (row_index+1), column=6, sticky='nsew', padx=5, pady=3)

        self.count_experiment_queue = len(self.widget_list) + 1

    def viewEdit_function(self, integer):
        row_index = integer
        print(F"This was the row selected {row_index}")

        current_experiment = self.experiment_object_list[row_index-1]
        # print(current_experiment)
        current_experiment_arguments = self.experiment_master_list[row_index-1]
        # print(current_experiment_arguments)

        self.problem_var.set(current_experiment_arguments[0])
        self.solver_var.set(current_experiment_arguments[1])
        self.macro_var.set(current_experiment_arguments[2])
        self.show_problem_factors(True, current_experiment_arguments)
        self.show_solver_factors(True, current_experiment_arguments)

        viewEdit_button_added = self.widget_list[row_index-1][4]
        viewEdit_button_added["text"] = "Save Changes"
        viewEdit_button_added["command"] = partial(self.save_edit_function, row_index)
        viewEdit_button_added.grid(row= (row_index), column=4, sticky='nsew', padx=5, pady=3)

    def clear_queue(self):

        for row in self.widget_list:
            for widget in row:
                widget.grid_remove()

        print(F"Size of self.experiment_master_list BEFORE running is { len(self.experiment_master_list) }")
        print(F"Size of self.experiment_object_list BEFORE running is { len(self.experiment_object_list) }")
        print(F"Size of self.wiedget_list BEFORE running is { len(self.widget_list) }")

        self.experiment_master_list.clear()
        self.experiment_object_list.clear()
        self.widget_list.clear()

    def add_function(self, *args):
        if len(args) == 1:
            place = args[0] - 1
        else:
            place = len(self.experiment_object_list)
        
        print("place ", place)
            
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

            self.factor_label_frame_problem.destroy()
            self.factor_label_frame_oracle.destroy()
            self.factor_label_frame_solver.destroy()

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
                compatibility_result = self.my_experiment.check_compatibility()
                if compatibility_result == "":
                    self.experiment_object_list.append(self.my_experiment)
                    self.experiment_master_list.append(self.selected)
                    #this option list doesnt autoupdate - not sure why but this will force it to update
                    self.experiment_master_list[len( self.experiment_master_list) - 1][5][0]['crn_across_solns'] = self.boolean_var.get()
                    
                    self.rows = 5
                    
                    self.problem_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[0],
                                                    font = "Calibri 10",
                                                    justify="center")
                    self.problem_added.grid(row=self.count_experiment_queue, column=0, sticky='nsew', padx=5, pady=3)

                    self.solver_added = tk.Label(master=self.tab_one,
                                                    text=self.selected[1],
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
                                                        command= self.test_function,
                                                        state = "disabled")
                    self.postprocess_button_added.grid(row=self.count_experiment_queue, column=6, sticky='nsew', padx=5, pady=3)

                    self.widget_row = [self.problem_added, self.solver_added, self.macros_added, self.run_button_added, self.viewEdit_button_added, self.clear_button_added, self.postprocess_button_added]
                    self.widget_list.append(self.widget_row)

                    self.count_experiment_queue += 1

                    # print(self.experiment_master_list)
                    # print(self.widget_list)
                    # print(self.experiment_object_list)

                
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
            #print("it works", self.selected)

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
        print("keys ->", keys)
        print("self.problem_factors_types -> ", self.problem_factors_types)

        for problem_factor in self.problem_factors_list:
            index = self.problem_factors_list.index(problem_factor)
            #print(problem_factor.get())
            if index < len(keys):
                print(self.problem_factors_types[index])
                datatype = self.problem_factors_types[index]
                # if the data type is tuple update data
                self.problem_factors_dictionary[keys[index]] = datatype(problem_factor.get())
                print("datatype of factor -> ", type(datatype(problem_factor.get())))
            if index == len(keys):
                if problem_factor.get() == self.problem_var.get():
                    self.problem_factors_return.append(None)
                else:
                    self.problem_factors_return.append(problem_factor.get())
                    # self.problem_factors_dictionary["rename"] = problem_factor.get()
        
        self.problem_factors_return.insert(0, self.problem_factors_dictionary)
        # print(self.problem_factors_dictionary)
        print("self.problem_factors_return", self.problem_factors_return)
        return self.problem_factors_return

    def confirm_oracle_factors(self):
        self.oracle_factors_return = []
        self.oracle_factors_dictionary = dict()

        keys = list(self.oracle_object().specifications.keys())
        print("keys ->", keys)
        print("self.oracle_factors_types -> ", self.oracle_factors_types)

        keys = list(self.oracle_object().specifications.keys())

        for oracle_factor in self.oracle_factors_list:
            index = self.oracle_factors_list.index(oracle_factor)
            self.oracle_factors_dictionary[keys[index]] = oracle_factor.get()

            print(self.oracle_factors_types[index])
            datatype = self.oracle_factors_types[index]
            self.oracle_factors_dictionary[keys[index]] = datatype(oracle_factor.get())
            print("datatype of factor -> ", type(datatype(oracle_factor.get())))
        
        self.oracle_factors_return.append(self.oracle_factors_dictionary)
        # print(self.oracle_factors_dictionary)
        print("self.oracle_factors_return ", self.oracle_factors_return)
        return self.oracle_factors_return

    def confirm_solver_factors(self):
        self.solver_factors_return = []
        self.solver_factors_dictionary = dict()

        keys = list(self.solver_object().specifications.keys())
        print("keys ->", keys)
        print("self.solver_factors_types -> ", self.solver_factors_types)

        for solver_factor in self.solver_factors_list:
            index = self.solver_factors_list.index(solver_factor)
            #print(solver_factor.get())
            if index < len(keys):
                print(self.solver_factors_types[index])
                datatype = self.solver_factors_types[index]
                self.solver_factors_dictionary[keys[index]] = datatype(solver_factor.get())
                print("datatype of factor -> ", type(datatype(solver_factor.get())))
            if index == len(keys):
                if solver_factor.get() == self.solver_var.get():
                    self.solver_factors_return.append(None)
                else:
                    self.solver_factors_return.append(solver_factor.get())
                    # self.solver_factors_dictionary["rename"] = solver_factor.get()
        
        self.solver_factors_return.insert(0, self.solver_factors_dictionary)
        # print(self.solver_factors_dictionary)
        print("self.solver_factors_return", self.solver_factors_return)
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
        print(self.boolean_var.get())
        
        row_index = integer
        self.experiment_master_list[row_index-1]
        self.experiment_master_list[row_index-1][5][0]['crn_across_solns'] = self.boolean_var.get()
        current_experiment_arguments = self.experiment_master_list[row_index-1][5]
        print(current_experiment_arguments)
        integer = integer
        print(F"test function connected to the number {integer}")
    
    def save_edit_function(self, integer):
        print("save edit")
        
        row_index = integer
        self.experiment_master_list[row_index-1]
        self.experiment_master_list[row_index-1][5][0]['crn_across_solns'] = self.boolean_var.get()

        self.clearRow_function(row_index)
        self.add_function(row_index)
        

    def select_pickle_file_fuction(self, *args):
        filename = filedialog.askopenfilename(parent = self.master,
                                            initialdir = "./",
                                            title = "Select a Pickle File",
                                            filetypes = (("Pickle files", "*.pickle;*.pck;*.pcl;*.pkl;*.db")
                                                         ,("Python files", "*.py")
                                                         ,("All files", "*.*") ))
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
                print(experiment_pathname)
                print(type(experiment_pathname))
                with open(experiment_pathname) as file:
                    experiment = pickle.load(file)
                    print(experiment)
            else:
                message = f"You have loaded a file, but {filetype} files are not acceptable!\nPlease try again."
                tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)
        else:
            message = "You are attempting to load a file, but haven't selected one yet.\nPlease select a file first."
            tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)

    def run_row_function(self, integer):
        # stringtuple[1:-1].split(separator=",")
        print(F"This is the value passed into the function: {integer}")
        row_index = integer - 1
        print(F"This is the index that it would be associated with for self.experiment_object_list: {row_index}")
        row_of_widgets = self.widget_list[row_index]

        post_processing_button = row_of_widgets[6]
        post_processing_button["state"] = "normal"
        print(post_processing_button["text"], post_processing_button["state"])
        post_processing_button.grid(row=integer, column=6, sticky='nsew', padx=5, pady=3)

        run_button = row_of_widgets[3]
        run_button["state"] = "disabled"
        print(run_button["text"], run_button["state"])
        run_button.grid(row=integer, column=3, sticky='nsew', padx=5, pady=3)

        widget_row = [row_of_widgets[0], row_of_widgets[1], row_of_widgets[2], run_button, row_of_widgets[4], row_of_widgets[5], post_processing_button]
        self.widget_list[row_index] = widget_row

        self.my_experiment = self.experiment_object_list[row_index]

        self.selected = self.experiment_master_list[row_index]
        print(self.selected)
        self.macro_reps = self.selected[2]
        print(self.macro_reps)

        tk.messagebox.showinfo(title="Status Update", message="Function will now begin running")

        # self.my_experiment.run(n_macroreps=self.macro_reps)

        # calls postprocessing window
        self.postrep_window = tk.Tk()
        self.postrep_window.geometry("1500x1000")
        self.postrep_window.title("Post Processing Page")
        self.app = Post_Processing_Window(self.postrep_window, self.my_experiment, self.selected)

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
    def __init__(self, master, myexperiment, experiment_list):

        self.master = master
        self.my_experiment = myexperiment
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

        self.n_postreps_init_opt_label = ttk.Label(master = self.master,
                                text = "Number of postreplications to take at initial x\u2070 and optimal x\u002A:",
                                font = "Calibri 11 bold",
                                wraplength = "250")

        self.n_postreps_init_opt_var = tk.StringVar(self.master)
        self.n_postreps_init_opt_entry = ttk.Entry(master=self.master, textvariable = self.n_postreps_init_opt_var, justify = tk.LEFT)
        self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

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
        self.crn_across_budget_menu = ttk.OptionMenu(self.master, self.crn_across_budget_var, "Options", *self.crn_across_budget_list)

        self.crn_across_macroreps_label = ttk.Label(master=self.master,
                                        text = "Use CRN for post-replications at solutions recommended on different macroreplications?",
                                        font = "Calibri 11 bold",
                                        wraplength = "250")

        self.crn_across_macroreps_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_macroreps_var = tk.StringVar(self.master)
        # sets the default OptionMenu selection
        # self.crn_across_macroreps_var.set("False")
        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.crn_across_macroreps_menu = ttk.OptionMenu(self.master, self.crn_across_macroreps_var, "Options", *self.crn_across_macroreps_list)

        self.post_processing_run_label = ttk.Label(master=self.master, # window label is used for
                        text = "Finish Post-Replication of Experiment",
                        font = "Calibri 11 bold",
                        wraplength = "250")

        self.post_processing_run_button = ttk.Button(master=self.master, # window button is used in
                        # aesthetic of button and specific formatting options
                        text = "Run", 
                        width = 15, # width of button
                        command = self.post_processing_run_function) # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click

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
                                                command = self.test_function)

        self.pickle_file_pathname_label = ttk.Label(master=self.master,
                                                    text = "File Selected:",
                                                    font = "Calibri 11 bold")

        self.pickle_file_pathname_show = ttk.Label(master=self.master,
                                                    text = "No file selected",
                                                    font = "Calibri 11 italic",
                                                    foreground = "red")

        self.title.place(x=15, y=15)

        self.n_postreps_label.place(x=0, y=95)
        self.n_postreps_entry.place(x=255, y=95)

        self.n_postreps_init_opt_label.place(x=0, y=180)
        self.n_postreps_init_opt_entry.place(x=255, y=180)

        self.crn_across_budget_label.place(x=0, y=275)
        self.crn_across_budget_menu.place(x=255, y=295)

        self.crn_across_macroreps_label.place(x=0, y=350)
        self.crn_across_macroreps_menu.place(x=255, y=375)

        self.post_processing_run_label.place(x=0, y=435)
        self.post_processing_run_button.place(x=255, y=435)

        self.pickle_file_select_label.place(x=0, y=520)
        self.pickle_file_select_button.place(x=190, y=520)
        self.pickle_file_load_button.place(x=300, y=520)
        self.pickle_file_pathname_label.place(x=0, y=555)
        self.pickle_file_pathname_show.place(x=100, y=555)

        self.frame.pack(side="top", fill="both", expand=True)

    def post_processing_run_function(self):
        self.experiment_list = [self.problem, self.solver, self.macro_reps]

        if self.n_postreps_entry.get().isnumeric() != False and self.n_postreps_init_opt_entry.get().isnumeric() != False and self.crn_across_budget_var.get() in self.crn_across_budget_list and self.crn_across_macroreps_var.get() in self.crn_across_macroreps_list:
            self.experiment_list.append(int(self.n_postreps_entry.get()))
            self.experiment_list.append(int(self.n_postreps_init_opt_entry.get()))

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
            self.n_postreps_init_opt_entry.delete(0, len(self.n_postreps_init_opt_entry.get()))
            self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

            # reset crn_across_budget_bar
            self.crn_across_budget_var.set("True")

            # reset crn_across_macroreps_var 
            self.crn_across_macroreps_var.set("False")

            self.n_postreps = self.experiment_list[3] # int
            self.n_postreps_init_opt = self.experiment_list[4] # int
            self.crn_across_budget = self.experiment_list[5] # boolean
            self.crn_across_macroreps = self.experiment_list[6] # boolean

            self.my_experiment.post_replicate(n_postreps=self.n_postreps, n_postreps_init_opt=self.n_postreps_init_opt, crn_across_budget=self.crn_across_budget, crn_across_macroreps=self.crn_across_macroreps)
            # print("post-replicate ran successfully")

            # print(self.experiment_list)
            return self.experiment_list

        elif self.n_postreps_entry.get().isnumeric() == False:
            message = "Please enter a valid value for the number of post replications to take at each recommended solution."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_entry.delete(0, len(self.n_postreps_entry.get()))
            self.n_postreps_entry.insert(index=tk.END, string="100")

        elif self.n_postreps_init_opt_entry.get().isnumeric() == False:
            message = "Please enter a valid value for the number of post repliactions at the initial x\u2070 and optimal x\u002A."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_init_opt_entry.delete(0, len(self.n_postreps_init_opt_entry.get()))
            self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

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
                                            filetypes = (("Pickle files", "*.pickle;*.pck;*.pcl;*.pkl;*.db")
                                                         ,("Python files", "*.py")
                                                         ,("All files", "*.*") ))
        if filename != "":
            self.pickle_file_pathname_show["text"] = filename
            self.pickle_file_pathname_show["foreground"] = "blue"
            self.pickle_file_pathname_show.place(x=100, y=555)
        else:
            message = "You attempted to select a file but failed, please try again if necessary"
            tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)

    def test_function(self, *args):
        print("connection enabled")

class Cross_Design_Window():
    def __init__(self, master):

        self.master = master

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
            print("problem < solver")
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
            print("problem > solver")

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
        problem_list = []
        solver_list = []

        for checkbox in self.crossdesign_checkbox_problem_list:
            if checkbox.get() == True:
                print(self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)] + " was selected (problem)")
                problem_list.append(problem_directory[self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)]])

        for checkbox in self.crossdesign_checkbox_solver_list:
            if checkbox.get() == True:
                print(self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)] + " was selected (solver)")
                solver_list.append(solver_directory[self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)]])
        
        macro_reps = self.crossdesign_macro_var.get()

        self.crossdesign_MetaExperiment = MetaExperiment(solver_names=solver_list, problem_names=problem_list, fixed_factors_filename="all_factors")

        return self.crossdesign_MetaExperiment

    def test_function(self, *args):
        print("test function connected")

def main(): 
    root = tk.Tk()
    root.title("SimOpt Application")
    root.geometry("1500x1000")
    root.pack_propagate(False)

    app = Experiment_Window(root)
    root.mainloop()

if __name__ == '__main__':
    main()

