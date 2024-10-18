import ast
import os
import pickle
import re
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.font import nametofont
from typing import Callable, Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageTk

from simopt.base import Model, Problem, Solver
from simopt.data_farming_base import DATA_FARMING_DIR, DesignType
from simopt.directory import (
    problem_directory,
    problem_unabbreviated_directory,
    solver_directory,
    solver_unabbreviated_directory,
)
from simopt.experiment_base import (
    ProblemsSolvers,
    create_design,
    plot_area_scatterplots,
    plot_progress_curves,
    plot_solvability_cdfs,
    plot_solvability_profiles,
    plot_terminal_progress,
    plot_terminal_scatterplots,
)
from simopt.gui.df_object import DFFactor, spec_dict_to_df_dict
from simopt.gui.toplevel_custom import Toplevel

# Layout:
# -----------------------------------------------------------------------------
# |     Created Experiments     |           Problem/Solver Notebook           |
# | --------------------------- | | Add Prob | Add Solver | Add Defaults |    |
# | |                         | | ------------------------------------------- |
# | |    Experiment List      | | |                                         | |
# | |                         | | |                                         | |
# | --------------------------- | |           Notebook Contents             | |
# | [     Change Defaults     ] | |                                         | |
# | [  Open Plotting Window   ] | |                                         | |
# | [     Load Experiment     ] | ------------------------------------------- |
# |-----------------------------|---------------------------------------------|
# |     Current Experiment      |                                             |
# | --------------------------- |                                             |
# |   Problems   |   Solvers    |              Generated Design               |
# | ------------ | ------------ |    (merged w/ notebook when not in use)     |
# | |  Problem | | |  Solver  | |                                             |
# | |   List   | | |   List   | |                                             |
# | ------------ | ------------ |                                             |
# | [        Clear List       ] |---------------------------------------------|
# | Exper Name    [___________] | Design Type: [______^]                      |
# | Make Pickle?  [___________] | # of Stacks: [_______]  [ Generate Design ] |
# | [    Create Experiment    ] | Design Name: [_______]                      |
# -----------------------------------------------------------------------------

# Frames in the Window:
# main
# |--Experiments (exps)
# |  |--Experiment List (list)
# |  |--Experiment Fields (fields)
# |--Current Experiment (curr_exp)
# |  |--Problem/Solver Lists (lists)
# |  |  |--Problem List (problems)
# |  |  |--Solver List (solvers)
# |  |--Current Experiment Fields (fields)
# |--Notebook (notebook)
# |  |--Problem/Solver Adding (Notebook) (ps_adding)
# |     |--Add Problem (problem)
# |     |--Add Solver (solver)
# |     |--Quick-Add Problems/Solvers (quick_add)
# |--Generated Design (gen_design)
# |--Design Options (design_opts)


class NewExperimentWindow(Toplevel):
    """New Experiment Window."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize New Experiment Window."""
        super().__init__(
            root, title="SimOpt GUI - New Experiment", exit_on_close=True
        )

        self.center_window(0.8)
        self.minsize(1280, 720)

        # Variables
        # Using dictionaries to store TK variables so they don't clutter
        # the namespace with this.____
        self.labels: dict[str, tk.Label] = {}
        self.buttons: dict[str, tk.Button] = {}
        self.entries: dict[str, tk.Entry] = {}
        self.checkbuttons: dict[str, tk.Checkbutton] = {}
        self.canvases: dict[str, tk.Canvas] = {}
        self.notebooks: dict[str, ttk.Notebook] = {}
        self.frames: dict[str, tk.Frame] = {}

        # Setup the main frame
        self.frames["main"] = ttk.Frame(self)
        self.frames["main"].pack(fill="both", expand=True)
        self.frames["main"].grid_rowconfigure(0, weight=1)
        self.frames["main"].grid_rowconfigure(1, weight=1)
        self.frames["main"].grid_columnconfigure(0, weight=1)
        self.frames["main"].grid_columnconfigure(1, weight=2)

        # Setup the experiments frame
        self.frames["exps"] = ttk.Frame(
            self.frames["main"], borderwidth=1, relief="solid"
        )
        self.frames["exps"].grid(row=0, column=0, sticky="nsew")
        self.frames["exps"].grid_columnconfigure(0, weight=1)
        self.frames["exps"].grid_rowconfigure(1, weight=1)
        self.labels["exps.header"] = ttk.Label(
            self.frames["exps"],
            text="Created Experiments",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.labels["exps.header"].grid(row=0, column=0, sticky="nsew")
        self.frames["exps.list"] = ttk.Frame(
            self.frames["exps"],
        )
        self.frames["exps.list"].grid(row=1, column=0, sticky="nsew")
        self.frames["exps.list"].grid_columnconfigure(0, weight=1)
        self.frames["exps.fields"] = ttk.Frame(
            self.frames["exps"],
        )
        self.frames["exps.fields"].grid(row=2, column=0, sticky="nsew")
        self.frames["exps.fields"].grid_columnconfigure(0, weight=1)
        self.buttons["exps.fields.default_opts"] = ttk.Button(
            self.frames["exps.fields"],
            text="Change Default Experiment Options",
            command=self.change_experiment_defaults,
        )
        self.buttons["exps.fields.default_opts"].grid(
            row=0, column=0, sticky="ew"
        )
        self.buttons["exps.fields.open_plot_win"] = ttk.Button(
            self.frames["exps.fields"],
            text="Open Plotting Window",
            command=self.open_plotting_window,
        )
        self.buttons["exps.fields.open_plot_win"].grid(
            row=1, column=0, sticky="ew"
        )
        self.buttons["exps.fields.load_exp"] = ttk.Button(
            self.frames["exps.fields"],
            text="Load Experiment",
            command=self.load_experiment,
        )
        self.buttons["exps.fields.load_exp"].grid(row=2, column=0, sticky="ew")

        # Setup the current experiment frame
        self.frames["curr_exp"] = ttk.Frame(
            self.frames["main"], borderwidth=1, relief="solid"
        )
        self.frames["curr_exp"].grid(row=1, column=0, sticky="nsew", rowspan=2)
        self.frames["curr_exp"].grid_columnconfigure(0, weight=1)
        self.frames["curr_exp"].grid_rowconfigure(1, weight=1)
        self.labels["curr_exp.header"] = ttk.Label(
            self.frames["curr_exp"],
            text="Current Experiment",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.labels["curr_exp.header"].grid(row=0, column=0, sticky="nsew")
        self.frames["curr_exp.lists"] = ttk.Frame(self.frames["curr_exp"])
        self.frames["curr_exp.lists"].grid(row=1, column=0, sticky="nsew")
        self.frames["curr_exp.lists"].grid_columnconfigure(0, weight=1)
        self.frames["curr_exp.lists"].grid_columnconfigure(1, weight=1)
        self.labels["curr_exp.lists.problem_header"] = ttk.Label(
            self.frames["curr_exp.lists"],
            text="Problems",
            anchor="center",
        )
        self.labels["curr_exp.lists.problem_header"].grid(
            row=0, column=0, sticky="nsew"
        )
        self.labels["curr_exp.lists.solver_header"] = ttk.Label(
            self.frames["curr_exp.lists"], text="Solvers", anchor="center"
        )
        self.labels["curr_exp.lists.solver_header"].grid(
            row=0, column=1, sticky="nsew"
        )
        self.frames["curr_exp.fields"] = ttk.Frame(
            self.frames["curr_exp"],
        )
        self.frames["curr_exp.fields"].grid(row=2, column=0, sticky="nsew")
        self.frames["curr_exp.fields"].grid_columnconfigure(1, weight=1)
        self.buttons["curr_exp.fields.clear_list"] = ttk.Button(
            self.frames["curr_exp.fields"],
            text="Clear Problem/Solver Lists",
            command=self.clear_experiment,
        )
        self.buttons["curr_exp.fields.clear_list"].grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        self.labels["curr_exp.fields.exp_name"] = ttk.Label(
            self.frames["curr_exp.fields"],
            text="Experiment Name ",
            anchor="e",
        )
        self.labels["curr_exp.fields.exp_name"].grid(
            row=1, column=0, sticky="ew"
        )
        self.current_experiment_name = tk.StringVar()
        self.entries["curr_exp.fields.exp_name"] = ttk.Entry(
            self.frames["curr_exp.fields"],
            textvariable=self.current_experiment_name,
        )
        self.entries["curr_exp.fields.exp_name"].grid(
            row=1, column=1, sticky="ew"
        )
        self.labels["curr_exp.fields.make_pickle"] = ttk.Label(
            self.frames["curr_exp.fields"],
            text="Create Pickles for each Problem-Solver Pair? ",
            anchor="e",
        )
        self.labels["curr_exp.fields.make_pickle"].grid(
            row=2, column=0, sticky="ew"
        )
        self.current_experiment_enable_pickle = tk.BooleanVar()
        self.checkbuttons["curr_exp.fields.make_pickle"] = ttk.Checkbutton(
            self.frames["curr_exp.fields"],
            variable=self.current_experiment_enable_pickle,
        )
        self.checkbuttons["curr_exp.fields.make_pickle"].grid(
            row=2, column=1, sticky="w"
        )
        self.buttons["curr_exp.fields.create_exp"] = ttk.Button(
            self.frames["curr_exp.fields"],
            text="Create Experiment",
            command=self.create_experiment,
        )
        self.buttons["curr_exp.fields.create_exp"].grid(
            row=3, column=0, columnspan=2, sticky="ew"
        )

        # Setup the notebook frame
        self.frames["ntbk"] = ttk.Frame(
            self.frames["main"], borderwidth=1, relief="solid"
        )
        self.frames["ntbk"].grid(row=0, column=1, sticky="nsew")
        self.frames["ntbk"].grid_columnconfigure(0, weight=1)
        self.frames["ntbk"].grid_rowconfigure(1, weight=1)
        self.labels["ntbk.header"] = ttk.Label(
            self.frames["ntbk"],
            text="Create Problems/Solvers",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.labels["ntbk.header"].grid(row=0, column=0, sticky="nsew")
        self.notebooks["ntbk.ps_adding"] = ttk.Notebook(self.frames["ntbk"])
        self.notebooks["ntbk.ps_adding"].grid(row=1, column=0, sticky="nsew")
        self.frames["ntbk.ps_adding.problem"] = ttk.Frame(
            self.notebooks["ntbk.ps_adding"]
        )
        self.frames["ntbk.ps_adding.solver"] = ttk.Frame(
            self.notebooks["ntbk.ps_adding"]
        )
        self.frames["ntbk.ps_adding.quick_add"] = ttk.Frame(
            self.notebooks["ntbk.ps_adding"]
        )
        self.notebooks["ntbk.ps_adding"].add(
            self.frames["ntbk.ps_adding.problem"], text="Add Problem"
        )
        self.notebooks["ntbk.ps_adding"].add(
            self.frames["ntbk.ps_adding.solver"], text="Add Solver"
        )
        self.notebooks["ntbk.ps_adding"].add(
            self.frames["ntbk.ps_adding.quick_add"],
            text="Quick-Add Problems/Solvers",
        )

        # Setup the generated design frame
        self.frames["gen_design"] = ttk.Frame(
            self.frames["main"], borderwidth=1, relief="solid"
        )
        self.frames["gen_design"].grid(row=1, column=1, sticky="nsew")
        self.frames["gen_design"].grid_columnconfigure(0, weight=1)
        self.frames["gen_design"].grid_rowconfigure(1, weight=1)
        self.labels["gen_design.header"] = ttk.Label(
            self.frames["gen_design"],
            text="Generated Design",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.labels["gen_design.header"].grid(row=0, column=0, sticky="nsew")
        self.canvases["gen_design.display"] = tk.Canvas(
            self.frames["gen_design"],
            height=10,
            width=10,
        )
        self.canvases["gen_design.display"].grid(row=1, column=0, sticky="nsew")

        # Setup the design options frame
        self.frames["design_opts"] = ttk.Frame(
            self.frames["main"], borderwidth=1, relief="solid"
        )
        self.frames["design_opts"].grid(row=2, column=1, sticky="nsew")
        self.frames["design_opts"].grid_columnconfigure(1, weight=2)
        self.frames["design_opts"].grid_columnconfigure(2, weight=1)
        self.labels["design_opts.header"] = ttk.Label(
            self.frames["design_opts"],
            text="Design Options",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.labels["design_opts.header"].grid(
            row=0, column=0, sticky="nsew", columnspan=3
        )
        self.labels["design_opts.type"] = ttk.Label(
            self.frames["design_opts"], text="Design Type ", anchor="e"
        )
        self.labels["design_opts.type"].grid(row=1, column=0, sticky="ew")
        self.design_type = tk.StringVar()
        self.entries["design_opts.type"] = ttk.Entry(
            self.frames["design_opts"], textvariable=self.design_type
        )
        self.entries["design_opts.type"].grid(row=1, column=1, sticky="ew")
        self.labels["design_opts.num_stacks"] = ttk.Label(
            self.frames["design_opts"], text="# of Stacks ", anchor="e"
        )
        self.labels["design_opts.num_stacks"].grid(row=2, column=0, sticky="ew")
        self.number_of_stacks = tk.IntVar()
        self.entries["design_opts.num_stacks"] = ttk.Entry(
            self.frames["design_opts"], textvariable=self.number_of_stacks
        )
        self.entries["design_opts.num_stacks"].grid(
            row=2, column=1, sticky="ew"
        )
        self.labels["design_opts.name"] = ttk.Label(
            self.frames["design_opts"], text="Design Name ", anchor="e"
        )
        self.labels["design_opts.name"].grid(row=3, column=0, sticky="ew")
        self.design_name = tk.StringVar()
        self.entries["design_opts.name"] = ttk.Entry(
            self.frames["design_opts"], textvariable=self.design_name
        )
        self.entries["design_opts.name"].grid(row=3, column=1, sticky="ew")
        # TODO: make this correclty pick the right function instead of being
        # hardcoded for problem design
        self.buttons["design_opts.generate"] = ttk.Button(
            self.frames["design_opts"],
            text="Generate Design",
            command=self.create_problem_design,
        )
        self.buttons["design_opts.generate"].grid(
            row=1, column=2, sticky="nsew", rowspan=3
        )

        self.frames["gen_design"].grid_forget()
        self.frames["ntbk"].grid(rowspan=2)

        self.frames["ntbk"].grid(rowspan=1)
        self.frames["gen_design"].grid(row=1, column=1, sticky="nsew")

        # master row numbers
        self.notebook_row = 1
        self.load_design_button_row = 2
        self.sol_prob_list_display_row = 3
        self.experiment_button_row = 5
        self.experiment_list_display_row = 6
        # master list variables
        self.root_solver_dict = {}  # for each name of solver or solver design has list that includes: [[solver factors], solver name]
        self.root_problem_dict = {}  # for each name of solver or solver design has list that includes: [[problem factors], [model factors], problem name]
        self.root_experiment_dict = {}  # dictionary of experiment name and related solver/problem lists (solver_factor_list, problem_factor_list, solver_name_list, problem_name_list)
        self.ran_experiments_dict = {}  # dictionary of experiments that have been run orgainized by experiment name
        self.design_types = Literal[
            "nolhs"
        ]  # available design types that can be used during datafarming
        self.design_types_list = self.design_types
        self.macro_reps = {}  # dict that contains user specified macroreps for each experiment
        self.post_reps = {}  # dict that contains user specified postrep numbers for each experiment
        self.init_post_reps = {}  # dict that contains number of postreps to take at initial & optimal solution for normalization for each experiment
        self.crn_budgets = {}  # contains bool val for if crn is used across budget for each experiment
        self.crn_macros = {}  # contains bool val for if crn is used across macroreps for each experiment
        self.crn_inits = {}  # contains bool val for if crn is used across initial and optimal solution for each experiment
        self.solve_tols = {}  # solver tolerance gaps for each experiment (inserted as list)
        self.pickle_checkstates = {}  # contains bool val for if pickles should be created for each individual problem-solver pair

        # widget lists for enable/delete functions
        self.solver_list_labels = {}  # holds widgets for solver name labels in solver list display
        self.problem_list_labels = {}  # holds widgets for problem name labels in problem list display
        self.experiment_list_labels = {}  # holds widgets for experimet name labels in experiment list display
        self.solver_edit_buttons = {}
        self.solver_del_buttons = {}
        self.problem_edit_buttons = {}
        self.problem_del_buttons = {}
        self.run_buttons = {}
        self.macro_entries = {}
        self.experiment_del_buttons = {}
        self.post_process_opt_buttons = {}
        self.post_process_buttons = {}
        self.post_norm_buttons = {}
        self.log_buttons = {}
        self.all_buttons = {}
        self.macro_vars = []  # list used for updated macro entries when default is changed

        # Default experiment options (can be changed in GUI)
        self.macro_default = 10
        self.post_default = 100
        self.init_default = 100
        self.crn_budget_default = True
        self.crn_macro_default = True
        self.crn_init_default = True
        self.solve_tols_default = [0.05, 0.10, 0.20, 0.50]

        """Solver/Problem Notebook & Selection Menus"""

        self.sol_prob_book = ttk.Notebook(master=self.main_frame)
        self.sol_prob_book.grid(row=self.notebook_row, column=0, sticky="nsew")

        self.solver_datafarm_notebook_frame = ttk.Frame(
            master=self.sol_prob_book
        )
        self.problem_datafarm_notebook_frame = ttk.Frame(
            master=self.sol_prob_book
        )
        self.mass_add_notebook_frame = ttk.Frame(master=self.sol_prob_book)

        self.sol_prob_book.add(
            self.solver_datafarm_notebook_frame,
            text="Add Data Farmed Solver",
        )
        self.sol_prob_book.add(
            self.problem_datafarm_notebook_frame,
            text="Add Data Farmed Problem",
        )
        self.sol_prob_book.add(
            self.mass_add_notebook_frame,
            text="Add Problems & Solvers with Default Settings",
        )

        # Solver selection w/ data farming
        self.solver_datafarm_selection_frame = tk.Frame(
            master=self.solver_datafarm_notebook_frame
        )
        self.solver_datafarm_selection_frame.grid(row=0, column=0)

        # Option menu to select solver
        self.solver_select_label = tk.Label(
            master=self.solver_datafarm_selection_frame,
            text="Select Solver:",
            width=20,
        )
        self.solver_select_label.grid(row=0, column=0, sticky=tk.N + tk.W)

        # Variable to store selected solver
        self.selected_solver = tk.StringVar()

        # Directory of solver names
        self.solver_list = solver_unabbreviated_directory

        self.solver_selection_dropdown = ttk.OptionMenu(
            self.solver_datafarm_selection_frame,
            self.selected_solver,
            "Solver",
            *self.solver_list,
            command=self.show_solver_datafarm,
        )
        self.solver_selection_dropdown.grid(row=0, column=1, sticky=tk.N + tk.W)

        # Problem selection w/ data farming
        self.problem_datafarm_selection_frame = tk.Frame(
            master=self.problem_datafarm_notebook_frame
        )
        self.problem_datafarm_selection_frame.grid(
            row=0, column=0, sticky=tk.N + tk.W
        )

        # Option menu to select problem
        self.problem_select_label = tk.Label(
            master=self.problem_datafarm_selection_frame,
            text="Select Problem",
            width=20,
        )
        self.problem_select_label.grid(row=0, column=0, sticky=tk.N + tk.W)

        # Variable to store selected problem
        self.selected_problem = tk.StringVar()

        # Directory of problem names
        self.problem_list = problem_unabbreviated_directory

        self.problem_datafarm_select_menu = ttk.OptionMenu(
            self.problem_datafarm_selection_frame,
            self.selected_problem,
            "Problem",
            *self.problem_list,
            command=self.show_problem_datafarm,
        )
        self.problem_datafarm_select_menu.grid(row=0, column=1)

        # load design button
        self.load_design_button = tk.Button(
            master=self.main_frame,
            text="Load Design from CSV",
            command=self.load_design,
        )
        self.load_design_button.grid(row=self.load_design_button_row, column=0)

        def update_correct_tab(event: tk.Event) -> None:
            # TODO: figure out a less hard-coded way to do this
            # Starts at 0, so the index of the "Add Problems & Solvers with
            # Default Settings" tab is 2
            if self.sol_prob_book.index("current") == 2:
                self.show_cross_design_window()

        # Whenever the tab is changed to "Add Problems & Solvers with Default
        # Settings", populate the tab
        # self.show_cross_design_window
        self.sol_prob_book.bind("<<NotebookTabChanged>>", update_correct_tab)

        """Display solver & problem lists"""

        self.display_sol_prob_list_frame = tk.Frame(master=self.main_frame)
        self.display_sol_prob_list_frame.grid(
            row=self.sol_prob_list_display_row, column=0
        )
        self.display_solver_list_frame = tk.Frame(
            master=self.display_sol_prob_list_frame
        )
        self.display_solver_list_frame.grid(row=0, column=0)
        self.display_problem_list_frame = tk.Frame(
            master=self.display_sol_prob_list_frame
        )
        self.display_problem_list_frame.grid(row=0, column=1)

        self.solver_list_canvas = tk.Canvas(
            master=self.display_solver_list_frame
        )
        self.solver_list_canvas.grid(row=1, column=0)

        self.problem_list_canvas = tk.Canvas(
            master=self.display_problem_list_frame
        )
        self.problem_list_canvas.grid(row=1, column=0)

        self.solver_list_title = tk.Label(
            master=self.display_solver_list_frame,
            text="Created Solvers",
            font=nametofont("TkHeadingFont"),
        )
        self.solver_list_title.grid(row=0, column=0)

        self.problem_list_title = tk.Label(
            master=self.display_problem_list_frame,
            text="Created Problems",
            font=nametofont("TkHeadingFont"),
        )
        self.problem_list_title.grid(row=0, column=1)

        # set experiment name and create experiment
        self.experiment_button_frame = tk.Frame(master=self.main_frame)
        self.experiment_button_frame.grid(
            row=self.experiment_button_row, column=0
        )
        # clear experiment selections
        self.clear_experiment_button = tk.Button(
            master=self.experiment_button_frame,
            text="Clear Current Experiment",
            command=self.clear_experiment,
        )
        self.clear_experiment_button.grid(row=0, column=0, columnspan=2)
        self.experiment_name_label = tk.Label(
            master=self.experiment_button_frame,
            text="Experiment Name",
        )
        self.experiment_name_label.grid(row=1, column=0)
        self.experiment_name_var = tk.StringVar()
        self.experiment_name_var.set("experiment")
        self.experiment_name_entry = tk.Entry(
            master=self.experiment_button_frame,
            textvariable=self.experiment_name_var,
            width=30,
            justify="left",
        )
        self.experiment_name_entry.grid(row=1, column=1)

        # ind pair pickle checkbox
        self.pickle_label = tk.Label(
            master=self.experiment_button_frame,
            text="Create pickles for each problem-solver pair?",
        )
        self.pickle_label.grid(row=2, column=0)
        self.pickle_checkstate = tk.BooleanVar()
        self.pickle_checkbox = tk.Checkbutton(
            master=self.experiment_button_frame,
            variable=self.pickle_checkstate,
            width=5,
        )
        self.pickle_checkbox.grid(row=2, column=1)

        self.run_experiment_button = tk.Button(
            master=self.experiment_button_frame,
            text="Create experiment with listed solvers & problems",
            command=self.create_experiment,
        )
        self.run_experiment_button.grid(row=3, column=0, columnspan=2)

        """ Display experiment list and run options"""
        self.experiment_list_display_frame = tk.Frame(master=self.main_frame)
        self.experiment_list_display_frame.grid(
            row=self.experiment_list_display_row, column=0
        )
        self.experiment_display_canvas = tk.Canvas(
            master=self.experiment_list_display_frame
        )
        self.experiment_display_canvas.grid(row=1, column=0)

        self.experiment_list_title = tk.Label(
            master=self.experiment_list_display_frame,
            text="Created Experiments",
            font=nametofont("TkHeadingFont"),
        )
        self.experiment_list_title.grid(row=0, column=0)
        self.experiment_defaults_button = tk.Button(
            master=self.experiment_list_display_frame,
            text="Change default experiment options",
            command=self.open_defaults_window,
        )
        self.experiment_defaults_button.grid(row=0, column=1, padx=10)

        # plots window button
        self.plot_window_button = tk.Button(
            master=self.experiment_list_display_frame,
            text="Open Plotting Window",
            command=self.open_plotting_window,
        )
        self.plot_window_button.grid(row=0, column=2, padx=10)

        # load experiment button
        self.load_exp_button = tk.Button(
            master=self.experiment_list_display_frame,
            text="Load Experiment",
            command=self.load_experiment,
        )
        self.load_exp_button.grid(row=0, column=3, padx=10)

    def update_main_window_scroll(self, event: tk.Event) -> None:
        self.root_canvas.configure(scrollregion=self.root_canvas.bbox("all"))

    def on_mousewheel(self, event: tk.Event) -> None:
        self.root_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def check_problem_compatibility(self) -> None:
        # create temp objects for current selected solvers and all possilble problems
        temp_solvers = []
        for solver_group in self.root_solver_dict:
            dp_0 = self.root_solver_dict[solver_group][
                0
            ]  # frist design point if design, only design pt if no design
            solver_name = dp_0[1]
            temp_solver = solver_directory[solver_name]()
            temp_solvers.append(temp_solver)
        # check solver selections based on which tab is open
        current_tab = self.sol_prob_book.index("current")
        if current_tab == 0:
            selected_solver = self.solver_var.get()
        if current_tab == 2:
            selected_solver = self.selected_solver.get()
        if selected_solver != "Solver":
            temp_solver = solver_unabbreviated_directory[selected_solver]()
            temp_solvers.append(temp_solver)
        all_problems = problem_unabbreviated_directory
        self.problem_list = {}  # clear current problem selection options
        for problem_name in all_problems:
            temp_problem = [all_problems[problem_name]()]
            temp_exp = ProblemsSolvers(
                solvers=temp_solvers, problems=temp_problem
            )  # temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            if not error:
                self.problem_list[problem_name] = all_problems[problem_name]

        # update problem & problem datafarming selections
        self.problem_select_menu.destroy()
        self.problem_select_menu = ttk.OptionMenu(
            self.problem_selection_frame,
            self.problem_var,
            "Problem",
            *self.problem_list,
            command=self.show_problem_factors,
        )
        self.problem_select_menu.grid(row=0, column=1)
        self.problem_datafarm_select_menu.destroy()
        self.problem_datafarm_select_menu = ttk.OptionMenu(
            self.problem_datafarm_selection_frame,
            self.selected_problem,
            "Problem",
            *self.problem_list,
            command=self.show_problem_datafarm,
        )
        self.problem_datafarm_select_menu.grid(row=0, column=1)

    def check_solver_compatibility(self) -> None:
        # create temp objects for current selected solvers and all possilble problems
        temp_problems = []
        for problem_group in self.root_problem_dict:
            dp_0 = self.root_problem_dict[problem_group][
                0
            ]  # frist design point if design, only design pt if no design
            problem_name = dp_0[1]
            temp_problem = problem_directory[problem_name]()
            temp_problems.append(temp_problem)
        # check problem selections based on which tab is open
        current_tab = self.sol_prob_book.index("current")
        if current_tab == 1:
            selected_problem = self.problem_var.get()
        if current_tab == 3:
            selected_problem = self.selected_problem.get()
        if selected_problem != "Problem":
            temp_problem = problem_unabbreviated_directory[selected_problem]()
            temp_problems.append(temp_problem)
        all_solvers = solver_unabbreviated_directory
        self.solver_list = {}  # clear current problem selection options
        for solver_name in all_solvers:
            temp_solver = [all_solvers[solver_name]()]
            temp_exp = ProblemsSolvers(
                solvers=temp_solver, problems=temp_problems
            )  # temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            if not error:
                self.solver_list[solver_name] = all_solvers[solver_name]

        # update solver & solver datafarming selections
        self.solver_selection_dropdown.destroy()
        self.solver_selection_dropdown = ttk.OptionMenu(
            self.solver_datafarm_selection_frame,
            self.selected_solver,
            "Solver",
            *self.solver_list,
            command=self.show_solver_datafarm,
        )
        self.solver_selection_dropdown.grid(row=0, column=1)

    def show_cross_design_window(self) -> None:
        self.cross_design_window = self.mass_add_notebook_frame

        # Configure the grid layout to expand properly
        self.cross_design_window.grid_rowconfigure(0, weight=1)
        self.cross_design_window.grid_columnconfigure(0, weight=1)
        self.cross_design_window.grid_rowconfigure(1, weight=1)
        self.cross_design_window.grid_columnconfigure(1, weight=1)
        self.cross_design_window.grid_rowconfigure(2, weight=1)
        self.cross_design_window.grid_columnconfigure(2, weight=1)

        self.solvers_canvas = tk.Canvas(master=self.cross_design_window)
        self.solvers_canvas.grid(row=2, column=0, sticky="nsew")
        self.problems_canvas = tk.Canvas(master=self.cross_design_window)
        self.problems_canvas.grid(row=2, column=2, sticky="nsew")

        # Create vertical scrollbar for solvers
        solver_scroll = ttk.Scrollbar(
            self.cross_design_window,
            orient=tk.VERTICAL,
            command=self.solvers_canvas.yview,
        )
        solver_scroll.grid(row=2, column=1, sticky="ns")

        # Create vertical scrollbar for problems
        problem_scroll = ttk.Scrollbar(
            self.cross_design_window,
            orient=tk.VERTICAL,
            command=self.problems_canvas.yview,
        )
        problem_scroll.grid(row=2, column=3, sticky="ns")

        # Configure canvas to use the scrollbars
        self.solvers_canvas.configure(yscrollcommand=solver_scroll.set)
        self.problems_canvas.configure(yscrollcommand=problem_scroll.set)

        # create master frame inside the canvas
        self.solvers_frame = tk.Frame(self.solvers_canvas)
        self.solvers_canvas.create_window(
            (0, 0), window=self.solvers_frame, anchor="nw"
        )
        self.problems_frame = tk.Frame(self.problems_canvas)
        self.problems_canvas.create_window(
            (0, 0), window=self.problems_frame, anchor="nw"
        )

        # Bind the configure event to update the scroll region
        self.solvers_frame.bind(
            "<Configure>", self.update_solvers_canvas_scroll
        )
        self.problems_frame.bind(
            "<Configure>", self.update_problems_canvas_scroll
        )

        self.cross_design_title = tk.Label(
            master=self.cross_design_window,
            text="Select solvers and problems to be included in cross-design. \n Solvers and problems will be run with default factor settings.",
            font=nametofont("TkHeadingFont"),
        )
        self.cross_design_title.grid(row=0, column=0, columnspan=4, sticky="n")
        self.solvers_label = tk.Label(
            master=self.cross_design_window,
            text="Select Solvers:",
        )
        self.solvers_label.grid(row=1, column=0, sticky="nw")
        self.problems_label = tk.Label(
            master=self.cross_design_window,
            text="Select Problems:",
        )
        self.problems_label.grid(row=1, column=2, sticky="nw")
        self.solver_checkboxes = {}  # holds checkbutton widgets, store as dictonary for now
        self.solver_check_vars = {}  # holds check boolvars, store as dictonary for now
        # display all potential solvers
        for solver in solver_unabbreviated_directory:
            row = self.solvers_frame.grid_size()[1]
            checkstate = tk.BooleanVar()
            solver_checkbox = tk.Checkbutton(
                master=self.solvers_frame,
                text=solver,
                variable=checkstate,
                command=self.cross_design_problem_compatibility,
            )
            solver_checkbox.grid(row=row, column=0, sticky="w", padx=10)
            self.solver_checkboxes[solver] = solver_checkbox
            self.solver_check_vars[solver] = checkstate
        self.problem_checkboxes = {}  # holds checkbutton widgets, store as dictonary for now
        self.problem_check_vars = {}  # holds check boolvars, store as dictonary for now
        # display all potential problems
        for problem in problem_unabbreviated_directory:
            row = self.problems_frame.grid_size()[1]
            checkstate = tk.BooleanVar()
            problem_checkbox = tk.Checkbutton(
                master=self.problems_frame,
                text=problem,
                variable=checkstate,
                command=self.cross_design_solver_compatibility,
            )
            problem_checkbox.grid(row=row, column=0, sticky="w", padx=10)
            self.problem_checkboxes[problem] = problem_checkbox
            self.problem_check_vars[problem] = checkstate
        self.create_cross_button = tk.Button(
            master=self.cross_design_window,
            text="Add Cross Design to Experiment",
            command=self.create_cross_design,
        )
        self.create_cross_button.grid(row=3, column=0)
        self.cross_design_problem_compatibility()  # run to check solvers already in experiment
        self.cross_design_solver_compatibility()  # run to check problems already in experiment

    def cross_design_problem_compatibility(self) -> None:
        # create temp objects for current selected solvers and all possilble problems
        temp_solvers = []
        # solvers previously added to experiment
        for solver_group in self.root_solver_dict:
            dp_0 = self.root_solver_dict[solver_group][
                0
            ]  # frist design point if design, only design pt if no design
            solver_name = dp_0[1]
            temp_solver = solver_directory[solver_name]()
            temp_solvers.append(temp_solver)
        # solvers currently added to cross design
        for solver in self.solver_check_vars:
            checkstate = self.solver_check_vars[solver].get()
            if checkstate:
                temp_solver = solver_unabbreviated_directory[solver]()
                temp_solvers.append(temp_solver)
        all_problems = problem_unabbreviated_directory
        for problem_name in all_problems:
            temp_problem = [all_problems[problem_name]()]
            temp_exp = ProblemsSolvers(
                solvers=temp_solvers, problems=temp_problem
            )  # temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            if error:
                self.problem_checkboxes[problem_name].configure(
                    state="disabled"
                )
            else:
                self.problem_checkboxes[problem_name].configure(state="normal")

        # update problem & problem datafarming selections
        self.problem_datafarm_select_menu.destroy()
        self.problem_datafarm_select_menu = ttk.OptionMenu(
            self.problem_datafarm_selection_frame,
            self.selected_problem,
            "Problem",
            *self.problem_list,
            command=self.show_problem_datafarm,
        )
        self.problem_datafarm_select_menu.grid(row=0, column=1)

    def cross_design_solver_compatibility(self) -> None:
        # create temp objects for current selected solvers and all possilble problems
        temp_problems = []
        # solvers previously added to experiment
        for problem_group in self.root_problem_dict:
            dp_0 = self.root_problem_dict[problem_group][
                0
            ]  # frist design point if design, only design pt if no design
            problem_name = dp_0[1]
            temp_problem = problem_directory[problem_name]()
            temp_problems.append(temp_problem)
        # problems currently added to cross design
        for problem in self.problem_check_vars:
            checkstate = self.problem_check_vars[problem].get()
            if checkstate:
                temp_problem = problem_unabbreviated_directory[problem]()
                temp_problems.append(temp_problem)
        all_solvers = solver_unabbreviated_directory
        for solver_name in all_solvers:
            temp_solver = [all_solvers[solver_name]()]
            temp_exp = ProblemsSolvers(
                solvers=temp_solver, problems=temp_problems
            )  # temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            if error:
                self.solver_checkboxes[solver_name].configure(state="disabled")
            else:
                self.solver_checkboxes[solver_name].configure(state="normal")

        # update solver & solver datafarming selections
        self.solver_selection_dropdown.destroy()
        self.solver_selection_dropdown = ttk.OptionMenu(
            self.solver_datafarm_selection_frame,
            self.selected_solver,
            "Solver",
            *self.solver_list,
            command=self.show_solver_datafarm,
        )
        self.solver_selection_dropdown.grid(row=0, column=1)

    def create_cross_design(self) -> None:
        for solver in self.solver_check_vars:
            checkstate = self.solver_check_vars[solver].get()
            if (
                checkstate
            ):  # add solver with default factor settings to master dict
                temp_solver = solver_unabbreviated_directory[solver]()
                factors = {
                    factor: value["default"]
                    for factor, value in temp_solver.specifications.items()
                }
                solver_name = temp_solver.name
                solver_save_name = self.get_unique_name(
                    self.root_solver_dict, solver_name
                )
                self.root_solver_dict[solver_save_name] = [
                    [factors, solver_name]
                ]
                # add solver row to list display
                solver_row = len(self.root_solver_dict) - 1
                self.solver_list_label = tk.Label(
                    master=self.solver_list_canvas,
                    text=solver_save_name,
                )
                self.solver_list_label.grid(row=solver_row, column=1)
                self.solver_list_labels[solver_save_name] = (
                    self.solver_list_label
                )

                # add delete and view/edit buttons
                self.solver_edit_button = tk.Button(
                    master=self.solver_list_canvas,
                    text="View/Edit",
                    command=lambda solver_name=solver_save_name: self.edit_solver(
                        solver_name
                    ),
                )
                self.solver_edit_button.grid(row=solver_row, column=2)
                self.solver_edit_buttons[solver_save_name] = (
                    self.solver_edit_button
                )
                self.solver_del_button = tk.Button(
                    master=self.solver_list_canvas,
                    text="Delete",
                    command=lambda solver_name=solver_save_name: self.delete_solver(
                        solver_name
                    ),
                )
                self.solver_del_button.grid(row=solver_row, column=3)
                self.solver_del_buttons[solver_save_name] = (
                    self.solver_del_button
                )

        for problem in self.problem_check_vars:
            checkstate = self.problem_check_vars[problem].get()
            if checkstate:  # add problem with default factor settings to master dict, ignore disabled boxes
                temp_problem = problem_unabbreviated_directory[problem]()
                factors = {
                    factor: value["default"]
                    for factor, value in temp_problem.specifications.items()
                }
                model_factors = {
                    factor: value["default"]
                    for factor, value in temp_problem.model.specifications.items()
                }
                factors.update(model_factors)
                problem_name = temp_problem.name
                problem_save_name = self.get_unique_name(
                    self.root_problem_dict, problem_name
                )
                self.root_problem_dict[problem_save_name] = [
                    [factors, problem_name]
                ]
                # add problem row to list display
                problem_row = len(self.root_problem_dict) - 1
                self.problem_list_label = tk.Label(
                    master=self.problem_list_canvas,
                    text=problem_save_name,
                )
                self.problem_list_label.grid(row=problem_row, column=1)
                self.problem_list_labels[problem_save_name] = (
                    self.problem_list_label
                )

                # add delete and view/edit buttons
                self.problem_edit_button = tk.Button(
                    master=self.problem_list_canvas,
                    text="View/Edit",
                    command=lambda problem_name=problem_save_name: self.edit_problem(
                        problem_name
                    ),
                )
                self.problem_edit_button.grid(row=problem_row, column=2)
                self.problem_edit_buttons[problem_save_name] = (
                    self.problem_edit_button
                )
                self.problem_del_button = tk.Button(
                    master=self.problem_list_canvas,
                    text="Delete",
                    command=lambda problem_name=problem_save_name: self.delete_problem(
                        problem_name
                    ),
                )
                self.problem_del_button.grid(row=problem_row, column=3)
                self.problem_del_buttons[problem_save_name] = (
                    self.problem_del_button
                )

    def update_solvers_canvas_scroll(self, event: tk.Event) -> None:
        self.solvers_canvas.configure(
            scrollregion=self.solvers_canvas.bbox("all")
        )

    def update_problems_canvas_scroll(self, event: tk.Event) -> None:
        self.problems_canvas.configure(
            scrollregion=self.problems_canvas.bbox("all")
        )

    def load_design(self) -> None:
        # Open file dialog to select design file
        # CSV files only, but all files can be selected (in case someone forgets to change file type)
        design_file = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        # Return if no file is selected
        if design_file == "":
            return
        # Convert whatever is in the file to a dataframe
        self.design_df = pd.read_csv(design_file, index_col=False)
        # Read in the filename and the directory path
        file_name = os.path.basename(design_file)
        self.dir_path = os.path.dirname(design_file)
        # Get design information from table
        name = self.design_df.at[1, "name"]
        if name in solver_directory:  # loaded a solver
            self.obj = solver_directory[
                name
            ]()  # create placeholder objects of the solver to get factor names
            is_problem = False
            factors = self.obj.specifications
            self.initialize_solver_frame()
            self.design_tree_frame = tk.Frame(master=self.solver_notebook_frame)
            self.design_tree_frame.grid(row=2, column=0)
            self.sol_prob_book.select(0)  # change view to solver tab

        elif name in problem_directory:  # loaded a problem
            self.obj = problem_directory[
                name
            ]()  # create placeholder objects of the problem to get factor names
            is_problem = True
            model_factors = self.obj.model.specifications
            problem_factors = self.obj.specifications
            factors = model_factors | problem_factors
            self.initialize_problem_frame()
            self.design_tree_frame = tk.Frame(
                master=self.problem_notebook_frame
            )
            self.design_tree_frame.grid(row=3, column=0)
            self.sol_prob_book.select(1)  # change view to problem tab

        # drop columns that aren't factor names
        drop_col = []
        for col in self.design_df.columns:
            if col not in factors:
                drop_col.append(col)
        self.filtered_design_df = self.design_df.drop(columns=drop_col)

        # determine design factors & default values
        self.design_factors = []
        problem_defaults = {}  # only for factors that do not change in design
        model_defaults = {}
        solver_defaults = {}
        for col in self.filtered_design_df.columns:
            factor_set = set(
                self.filtered_design_df[col]
            )  # determine if all factor values are the same
            if len(factor_set) > 1:
                self.design_factors.append(col)
            else:  # factor is not part of design (only has one value)
                if is_problem:
                    if col in problem_factors:
                        problem_defaults[col] = self.filtered_design_df.at[
                            1, col
                        ]
                    elif col in model_factors:
                        model_defaults[col] = self.filtered_design_df.at[1, col]
                else:
                    solver_defaults[col] = self.filtered_design_df.at[1, col]

        if is_problem:
            # show problem factors and store default widgets and values to this dict
            self.design_defaults, last_row = self.show_factor_defaults(
                self.obj,
                self.problem_factor_display_canvas,
                factor_dict=problem_defaults,
                design_factors=self.design_factors,
            )
            # # show model factors and store default widgets and default values to these
            self.model_defaults, new_last_row = self.show_factor_defaults(
                base_object=self.obj,
                frame=self.problem_factor_display_canvas,
                IsModel=True,
                first_row=last_row + 1,
                factor_dict=model_defaults,
                design_factors=self.design_factors,
            )
            # combine default dictionaries
            self.design_defaults.update(self.model_defaults)
            # entry for problem name and add problem button
            self.problem_name_label = tk.Label(
                master=self.model_frame,
                text="Problem Name",
                width=20,
            )
            self.problem_name_label.grid(row=2, column=0)
            self.design_name = tk.StringVar()
            # get unique problem name
            problem_name = self.get_unique_name(
                self.root_problem_dict, file_name
            )
            self.design_name.set(problem_name)
            self.problem_name_entry = tk.Entry(
                master=self.model_frame,
                textvariable=self.design_name,
                width=20,
            )
            self.problem_name_entry.grid(row=2, column=1)
            self.add_prob_to_exp_button = tk.Button(
                master=self.model_frame,
                text="Add this problem design to experiment",
                command=self.add_loaded_problem_to_experiment,
            )
            self.add_prob_to_exp_button.grid(row=5, column=0)
            # display problem name in menu
            self.problem_var.set(name)
            self.tree_frame = self.model_frame  # use when calling design tree

        else:
            # show problem factors and store default widgets to this dict
            self.design_defaults, last_row = self.show_factor_defaults(
                self.obj,
                self.factor_display_canvas,
                factor_dict=solver_defaults,
                design_factors=self.design_factors,
            )

            # entry for solver name and add solver button
            self.solver_name_label = tk.Label(
                master=self.solver_frame,
                text="Solver Name",
                width=20,
            )
            self.solver_name_label.grid(row=2, column=0)
            self.design_name = tk.StringVar()
            # get unique solver name
            solver_name = self.get_unique_name(self.root_solver_dict, file_name)
            self.design_name.set(solver_name)
            self.solver_name_entry = tk.Entry(
                master=self.solver_frame,
                textvariable=self.design_name,
                width=20,
            )
            self.solver_name_entry.grid(row=2, column=1)
            self.add_sol_to_exp_button = tk.Button(
                master=self.solver_frame,
                text="Add this solver design to experiment",
                command=self.add_loaded_solver_to_experiment,
            )
            self.add_sol_to_exp_button.grid(row=5, column=0)
            # display solver name in menu
            self.solver_var.set(name)
            self.tree_frame = self.solver_frame  # use when calling design tree
        # modify fixed factors button
        self.change_fixed_factors_button = tk.Button(
            master=self.tree_frame,
            text="Modify Fixed Factors",
            command=self.change_fixed_factors,
        )
        self.change_fixed_factors_button.grid(row=3, column=0)
        # display design tre
        self.display_design_tree(
            csv_filename=design_file, frame=self.tree_frame, row=4
        )

    def change_fixed_factors(self) -> None:
        # get new fixed factors
        fixed_factors = {}
        for factor in self.design_defaults:
            if factor not in self.design_factors:
                fixed_val = self.design_defaults[factor].get()
                fixed_factors[factor] = fixed_val

        # update design df
        for factor in fixed_factors:  # update both versions of the data frame
            self.filtered_design_df[factor] = fixed_factors[factor]
            self.design_df[factor] = fixed_factors[factor]

        # create new design csv file that follows original format
        csv_filename = f"{self.dir_path}/{self.design_name.get()}.csv"
        self.design_df.to_csv(csv_filename, mode="w", header=True, index=False)

        # update design tree
        self.design_tree.destroy()
        self.display_design_tree(csv_filename, frame=self.tree_frame, row=4)

    def add_loaded_solver_to_experiment(self) -> None:
        # convert df to list of dictionaries
        self.design_list = self.filtered_design_df.to_dict(orient="records")

        design_name = self.design_name.get()

        solver_holder_list = []  # used so solver list matches datafarming format
        for dp in self.design_list:
            converted_dp = self.convert_proper_datatype(dp, self.obj, var=False)
            solver_list = []  # holds dictionary of dps and solver name
            solver_list.append(converted_dp)
            solver_list.append(self.obj.name)
            solver_holder_list.append(solver_list)

        self.root_solver_dict[design_name] = solver_holder_list

        # add solver name to solver index
        solver_row = len(self.root_solver_dict) - 1
        self.solver_list_label = tk.Label(
            master=self.solver_list_canvas,
            text=design_name,
        )
        self.solver_list_label.grid(row=solver_row, column=1)
        self.solver_list_labels[design_name] = self.solver_list_label

        # add delete and view/edit buttons
        self.solver_edit_button = tk.Button(
            master=self.solver_list_canvas,
            text="View/Edit",
        )
        self.solver_edit_button.grid(row=solver_row, column=2)
        self.solver_edit_buttons[design_name] = self.solver_edit_button
        self.solver_del_button = tk.Button(
            master=self.solver_list_canvas,
            text="Delete",
            command=lambda: self.delete_solver(design_name),
        )
        self.solver_del_button.grid(row=solver_row, column=3)
        self.solver_del_buttons[design_name] = self.solver_del_button

        # refresh solver design name entry box
        self.design_name.set(
            self.get_unique_name(self.root_solver_dict, design_name)
        )

    def add_loaded_problem_to_experiment(self) -> None:
        # convert df to list of dictionaries
        self.design_list = self.filtered_design_df.to_dict(orient="records")

        design_name = self.design_name.get()

        problem_holder_list = []  # holds all problem lists within design name
        for dp in self.design_list:
            dp_list = []  # holds dictionary of factors for current dp
            converted_dp = self.convert_proper_datatype(dp, self.obj, var=False)
            dp_list.append(converted_dp)  # append problem factors
            dp_list.append(self.obj.name)  # append name of problem
            problem_holder_list.append(
                dp_list
            )  # add current dp information to holder list

        self.root_problem_dict[design_name] = problem_holder_list

        # add solver name to solver index
        problem_row = len(self.root_problem_dict) - 1
        self.problem_list_label = tk.Label(
            master=self.problem_list_canvas,
            text=design_name,
        )
        self.problem_list_label.grid(row=problem_row, column=1)
        self.problem_list_labels[design_name] = self.problem_list_label

        # add delete and view/edit buttons
        self.problem_edit_button = tk.Button(
            master=self.problem_list_canvas,
            text="View/Edit",
        )
        self.problem_edit_button.grid(row=problem_row, column=2)
        self.problem_edit_buttons[design_name] = self.problem_edit_button
        self.problem_del_button = tk.Button(
            master=self.problem_list_canvas,
            text="Delete",
            command=lambda: self.delete_problem(design_name),
        )
        self.problem_del_button.grid(row=problem_row, column=3)
        self.problem_del_buttons[design_name] = self.problem_del_button

        # refresh problem design name entry box
        self.problem_design_name_var.set(
            self.get_unique_name(self.root_problem_dict, design_name)
        )

    def load_experiment(self) -> None:
        # ask user for pickle file location
        file_path = filedialog.askopenfilename()
        base = os.path.basename(file_path)
        exp_name = os.path.splitext(base)[0]

        # make sure name is unique
        self.experiment_name = self.get_unique_name(
            self.root_experiment_dict, exp_name
        )

        # load pickle
        tk.messagebox.showinfo(
            "Loading",
            "Loading pickle file. This may take a few minutes. Experiment will appear within created experiments list once loaded.",
        )
        with open(file_path, "rb") as f:
            exp = pickle.load(f)

        self.root_experiment_dict[self.experiment_name] = exp
        self.add_exp_row()

        # determine if exp has been post processed and post normalized and set display
        self.run_buttons[self.experiment_name].configure(state="disabled")
        self.all_buttons[self.experiment_name].configure(state="disabled")
        post_rep = exp.check_postreplicate()
        post_norm = exp.check_postnormalize()
        if not post_rep:
            self.post_process_buttons[self.experiment_name].configure(
                state="normal"
            )
        if post_rep and not post_norm:
            self.post_norm_buttons[self.experiment_name].configure(
                state="normal"
            )

    def clear_frame(self, frame: tk.Frame) -> None:
        """Clear frame of all widgets."""
        for widget in frame.winfo_children():
            widget.destroy()

    def insert_factor_headers(
        self,
        frame: tk.Frame,
        first_row: int = 0,
    ) -> int:
        """Insert the headers for the factors into the frame.

        Parameters
        ----------
        frame : tk.Frame
            Frame to display factors.
        factor_heading_list : list[str]
            List of factor headings.
        first_row : int, optional
            First row to display factors.

        Returns
        -------
        int
            Index of the last row inserted.

        """
        header_columns = [
            "Factor Name",
            "Factor Description",
            "Factor Type",
            "Default Value",
            "Include in Design?",
            "Min Value",
            "Max Value",
            "# Decimals",
        ]
        for heading in header_columns:
            frame.grid_columnconfigure(header_columns.index(heading))
            label = tk.Label(
                master=frame,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=first_row,
                column=header_columns.index(heading),
            )
        # Insert horizontal separator
        ttk.Separator(frame, orient="horizontal").grid(
            row=first_row + 1, columnspan=len(header_columns), sticky="ew"
        )
        return first_row + 1

    def insert_factors(
        self,
        frame: tk.Frame,
        factor_dict: dict[str, DFFactor],
        first_row: int = 2,
    ) -> int:
        """Insert the factors into the frame.

        Parameters
        ----------
        frame : tk.Frame
            Frame to display factors.
        factors : dict[str, DFFactor]
            Dictionary of factors.
        first_row : int, optional
            First row to display factors.

        Returns
        -------
        int
            Index of the last row displayed.
        """

        row_index = first_row
        # Loop through and add everything to the frame
        for factor_index, factor_name in enumerate(factor_dict):
            # Skip every other row to allow for the separator
            row_index = factor_index * 2 + first_row

            # Get the factor object
            factor_obj = factor_dict[factor_name]
            # Make a list of functions that will return the widgets for each
            # column in the frame
            column_functions: list[Callable[[tk.Frame], tk.Widget | None]] = [
                factor_obj.get_name_label,
                factor_obj.get_description_label,
                factor_obj.get_type_label,
                factor_obj.get_default_entry,
                factor_obj.get_include_checkbutton,
                factor_obj.get_minimum_entry,
                factor_obj.get_maximum_entry,
                factor_obj.get_num_decimals_entry,
            ]

            # If it's not the last row, add a separator
            if factor_index != len(factor_dict) - 1:
                ttk.Separator(frame, orient="horizontal").grid(
                    row=row_index + 1,
                    column=0,
                    columnspan=len(column_functions),
                    sticky=tk.E + tk.W,
                )

            # Loop through and insert the factor data into the frame
            for column_index, function in enumerate(column_functions):
                # Configure the column
                frame.grid_columnconfigure(column_index)
                # Call the function to get the widget
                widget = function(frame)
                # Display the widget if it exists
                if widget is not None:
                    widget.grid(
                        row=row_index, column=column_index, padx=10, pady=3
                    )
        return row_index

    def show_factor_defaults(
        self,
        base_object: Solver | Problem,
        frame: tk.Frame,
        factor_dict: dict | None = None,
        is_model: bool = False,
        first_row: int = 1,
        empty_rows_between: int = 0,
    ) -> tuple[dict, int]:
        """Show default factors for a solver or problem.

        Parameters
        ----------
        base_object : Solver | Problem
            Solver or Problem object.
        frame : tk.Frame
            Frame to display factors.
        factor_dict : dict, optional
            Dictionary of factors and their default values.
        is_model : bool, optional
            If True, base_object is a model.
        first_row : int, optional
            First row to display factors.
        empty_rows_between : int, optional
            Number of empty rows between factors.

        Returns
        -------
        dict
            Dictionary of factors and their default values.
        int
            Last row to display factors.

        """
        # Widget lists
        defaults = {}
        # Initial variable values
        # self.factor_que_length = 0
        entry_width = 10
        # append_list
        if is_model:
            base_object = base_object.model

        # Store outside of loop so we can return at end
        row_index = first_row

        factor_spec = base_object.specifications
        for index, factor in enumerate(factor_spec):
            # Update the row index
            # Skip every other row to allow for the separator
            row_index = first_row + index * (empty_rows_between + 1)

            # Get the factor's datatype, description, and default value
            f_type = factor_spec[factor].get("datatype")
            f_type_str = f_type.__name__
            f_description = factor_spec[factor].get("description")
            if factor_dict is not None:
                f_default = factor_dict[factor]
            else:
                f_default = factor_spec[factor].get("default")

            # Loop through and insert the factor data into the frame
            column_data = [factor, f_description, f_type_str]
            for column_index, column in enumerate(column_data):
                frame.grid_columnconfigure(column_index)
                label = tk.Label(
                    master=frame,
                    text=column,
                    wraplength=250,
                    justify=tk.LEFT,
                )
                label.grid(
                    row=row_index,
                    column=column_index,
                    padx=10,
                    pady=3,
                    sticky=tk.W,
                )

            default_value = tk.StringVar()
            if f_type is bool:
                # Add option menu for true/false
                default_value.set(str(True))  # Set default bool option
                bool_menu = ttk.OptionMenu(
                    frame, default_value, str(True), str(True), str(False)
                )
                bool_menu.grid(row=row_index, column=3, sticky=tk.W + tk.E)

            elif f_type in (list, tuple):
                # Add entry box for default value
                default_len = len(str(f_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                default_value = tk.StringVar()
                default_entry = tk.Entry(
                    master=frame,
                    width=entry_width,
                    textvariable=default_value,
                    justify=tk.LEFT,
                )
                default_entry.grid(
                    row=row_index, column=3, sticky=tk.W + tk.E, columnspan=5
                )
                # Display original default value
                default_entry.insert(0, str(f_default))

            # Add entry box for default value
            elif f_type in (int, float):
                default_entry = tk.Entry(
                    master=frame,
                    width=entry_width,
                    textvariable=default_value,
                    justify=tk.RIGHT,
                )
                default_entry.grid(row=row_index, column=3, sticky=tk.W + tk.E)
                # Display original default value
                default_entry.insert(0, f_default)

            # add varibles to default list
            defaults[factor] = default_value

        return defaults, row_index

    def show_data_farming_options(
        self,
        base_object: Solver | Problem | Model,
        frame: tk.Frame,
        first_row: int = 1,
        empty_rows_between: int = 0,
    ) -> tuple[dict, dict, dict, dict, dict, int]:
        """Show data farming options for a solver or problem.

        Parameters
        ----------
        base_object : Solver | Problem | Model
            Solver, Problem, or Model object.
        frame : tk.Frame
            Frame to display factors.
        first_row : int, optional
            First row to display factors.
        empty_rows_between : int, optional
            Number of empty rows between factors.

        Returns
        -------
        dict
            Dictionary of factors and their check states.
        dict
            Dictionary of factors and their minimum values.
        dict
            Dictionary of factors and their maximum values.
        dict
            Dictionary of float factors and their number of decimals.
        dict
            Dictionary of factors and their widgets.
        int
            Last row to display factors.

        """
        checkstates = {}  # holds variable for each factor's check state
        min_vals = {}  # holds variable for each factor's min value
        max_vals = {}  # holds variable for each factor's max values
        dec_vals = {}  # holds variable for each float factor's # decimals
        widgets = {}  # holds a list of each widget for min, max, and dec entry for each factor

        # TODO: Investigate if this is a valid approach for models
        # The specification path looks like it's querying a problem object
        if isinstance(base_object, Model):
            specifications = base_object.model.specifications
        else:
            specifications = base_object.specifications

        for index, factor in enumerate(specifications):
            factor_datatype = specifications[factor].get("datatype")
            class_type = type(base_object).__base__
            widget_list = []
            row_index = first_row + index * (empty_rows_between + 1)

            # If the factor is an int, float, or bool, we can farm it
            if factor_datatype in (int, float, bool):
                # Add check box to include in design
                checkstate = tk.BooleanVar()
                checkbox = tk.Checkbutton(
                    master=frame,
                    variable=checkstate,
                    width=5,
                    command=lambda class_type=class_type: self.enable_datafarm_entry(
                        class_type
                    ),
                )
                checkbox.grid(
                    row=row_index, column=4, sticky=tk.W + tk.E, padx=5
                )
                checkstates[factor] = checkstate

                if factor_datatype in (int, float):
                    # Add entry box for min val
                    min_val = tk.StringVar()
                    min_entry = tk.Entry(
                        master=frame,
                        width=10,
                        textvariable=min_val,
                        justify="right",
                    )
                    min_entry.grid(
                        row=row_index, column=5, sticky=tk.W + tk.E, padx=5
                    )
                    min_entry.configure(state="disabled")
                    min_vals[factor] = min_val
                    widget_list.append(min_entry)

                    # Add entry box for max val
                    max_val = tk.StringVar()
                    max_entry = tk.Entry(
                        master=frame,
                        width=10,
                        textvariable=max_val,
                        justify="right",
                    )
                    max_entry.grid(
                        row=row_index, column=6, sticky=tk.W + tk.E, padx=5
                    )
                    max_entry.configure(state="disabled")
                    max_vals[factor] = max_val
                    widget_list.append(max_entry)

                    # Add entry box for dec val for float factors
                    if factor_datatype is float:
                        dec_val = tk.StringVar()
                        dec_entry = tk.Entry(
                            master=frame,
                            width=10,
                            textvariable=dec_val,
                            justify="right",
                        )
                        dec_entry.grid(
                            row=row_index, column=7, sticky=tk.W + tk.E, padx=5
                        )
                        dec_entry.configure(state="disabled")
                        dec_vals[factor] = dec_val
                        widget_list.append(dec_entry)

                widgets[factor] = widget_list
        return checkstates, min_vals, max_vals, dec_vals, widgets, row_index

    def show_problem_factors(self, event: tk.Event) -> None:
        # clear previous selections
        self.clear_frame(self.problem_frame)
        self.clear_frame(self.model_frame)
        # check solver compatibility
        self.check_solver_compatibility()

        """ Initialize frames, headers, and data farming buttons"""

        # self.prob_mod_frame = tk.Frame(master = self)
        # self.prob_mod_frame.grid(row = self.factors_display_row, column = 0)

        self.problem_frame = tk.Frame(master=self.problem_notebook_frame)
        self.problem_frame.grid(row=1, column=0, sticky=tk.N + tk.W)
        self.model_frame = tk.Frame(master=self.problem_notebook_frame)
        self.model_frame.grid(row=2, column=0, sticky=tk.N + tk.W)

        # self.IsSolver = False #used when adding problem to experiment list

        self.datafarm_prob_button = tk.Button(
            master=self.problem_selection_frame, text="Data Farm this Problem"
        )
        self.datafarm_prob_button.grid(row=1, column=0)

        self.datafarm_mod_button = tk.Button(
            master=self.problem_selection_frame, text="Data Farm this Model"
        )
        self.datafarm_mod_button.grid(row=1, column=2)

        # Create column for solver factor names
        self.problem_headername_label = tk.Label(
            master=self.problem_frame,
            text="Problem Factors",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.problem_headername_label.grid(row=0, column=0, sticky=tk.N + tk.W)

        # self.model_headername_label = tk.Label(master = self.model_frame, text = 'Model Factors', font = f"{TEXT_FAMILY} 13 bold", width = 20, anchor = 'w')
        # self.model_headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W)

        # Create column for factor type
        self.header_lbl_type = tk.Label(
            master=self.problem_frame,
            text="Factor Type",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor=tk.N + tk.W,
        )
        self.header_lbl_type.grid(row=0, column=1, sticky=tk.N + tk.W)

        # Create column for factor default values
        self.header_lbl_include = tk.Label(
            master=self.problem_frame,
            text="Default Value",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor=tk.N + tk.W,
        )
        self.header_lbl_include.grid(row=0, column=2, sticky=tk.N + tk.W)

        """ Get problem information from directory and display"""
        # Get problem info from directory
        self.selected_problem = self.problem_var.get()
        self.problem_object = self.problem_list[self.selected_problem]()

        # show problem factors and store default widgets and values to this dict
        self.problem_defaults, last_row = self.show_factor_defaults(
            self.problem_object, self.problem_frame
        )

        # Create column for solver factor names
        self.problem_headername_label = tk.Label(
            master=self.problem_frame,
            text="Model Factors",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor=tk.N + tk.W,
        )
        self.problem_headername_label.grid(
            row=last_row + 1, column=0, sticky=tk.N + tk.W
        )

        """ Get model information from dicrectory and display"""
        # self.model_problem_dict = model_problem_class_directory # directory that relates problem name to model class
        # self.model_object = self.model_problem_dict[self.selected_problem]()
        # # show model factors and store default widgets and default values to these
        self.model_defaults, new_last_row = self.show_factor_defaults(
            base_object=self.problem_object,
            frame=self.problem_frame,
            is_model=True,
            first_row=last_row + 2,
        )
        # combine default dictionaries
        self.problem_defaults.update(self.model_defaults)
        # entry for problem name and add problem button
        self.problem_name_label = tk.Label(
            master=self.model_frame,
            text="problem Name",
            width=20,
        )
        self.problem_name_label.grid(row=new_last_row + 1, column=0)
        self.problem_name_var = tk.StringVar()
        # get unique problem name
        problem_name = self.get_unique_name(
            self.root_problem_dict, self.problem_object.name
        )
        self.problem_name_var.set(problem_name)
        self.problem_name_entry = tk.Entry(
            master=self.model_frame,
            textvariable=self.problem_name_var,
            width=20,
        )
        self.problem_name_entry.grid(row=new_last_row + 1, column=1)
        self.add_prob_to_exp_button = tk.Button(
            master=self.model_frame,
            text="Add this problem to experiment",
            command=self.add_problem_to_experiment,
        )
        self.add_prob_to_exp_button.grid(row=new_last_row + 2, column=0)

    def show_solver_factors(self, event: tk.Event) -> None:
        # clear previous selections
        self.clear_frame(self.solver_frame)
        # update problem selections
        self.check_problem_compatibility()

        """ Initialize frames and headers"""

        # TODO:
        self.solver_frame = tk.Frame(
            master=self.solver_notebook_frame, bg="green"
        )
        self.solver_frame.grid(row=1, column=0)

        # Create column for solver factor names
        self.header_lbl_name = tk.Label(
            master=self.solver_frame,
            text="Solver Factors",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.header_lbl_name.grid(row=0, column=0, sticky=tk.N + tk.W)

        # Create column for factor type
        self.header_lbl_type = tk.Label(
            master=self.solver_frame,
            text="Factor Type",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.header_lbl_type.grid(row=0, column=1, sticky=tk.N + tk.W)

        # Create column for factor default values
        self.header_lbl_include = tk.Label(
            master=self.solver_frame,
            text="Default Value",
            font=nametofont("TkHeadingFont"),
            width=20,
        )
        self.header_lbl_include.grid(row=0, column=2, sticky=tk.N + tk.W)

        """ Get solver information from dicrectory and display"""
        # Get solver info from dictionary
        self.selected_solver = self.solver_var.get()
        self.solver_object = self.solver_list[self.selected_solver]()
        # show problem factors and store default widgets to this dict
        self.solver_defaults, last_row = self.show_factor_defaults(
            self.solver_object, frame=self.solver_frame, first_row=1
        )

        # entry for solver name and add solver button
        self.solver_name_label = tk.Label(
            master=self.solver_frame,
            text="Solver Name",
            width=20,
        )
        self.solver_name_label.grid(row=last_row + 1, column=0)
        self.solver_name_var = tk.StringVar()
        # get unique solver name
        solver_name = self.get_unique_name(
            self.root_solver_dict, self.solver_object.name
        )
        self.solver_name_var.set(solver_name)
        self.solver_name_entry = tk.Entry(
            master=self.solver_frame,
            textvariable=self.solver_name_var,
            width=20,
        )
        self.solver_name_entry.grid(row=last_row + 1, column=1)
        self.add_sol_to_exp_button = tk.Button(
            master=self.solver_frame,
            text="Add this solver to experiment",
            command=self.add_solver_to_experiment,
        )
        self.add_sol_to_exp_button.grid(row=last_row + 2, column=0)

    def get_unique_name(self, dict_lookup: dict, base_name: str) -> str:
        """Determine unique name from dictionary.

        Parameters
        ----------
        dict_lookup : dict
            dictionary where you want to determine unique name from.
        base_name : str
            base name that you want appended to become unique.

        Returns
        -------
        new_name : str
            new unique name.

        """
        if base_name in dict_lookup:
            # remove suffix from base_name if applicable
            if re.match(r".*_[0-9]+$", base_name):
                base_name = base_name.rsplit("_", 1)[0]
            count = 0
            test_name = base_name
            while test_name in dict_lookup:
                test_name = base_name
                count += 1
                test_name = f"{base_name}_{count!s}"
            new_name = test_name
        else:
            new_name = base_name

        # if base_name != new_name:
        #     print(
        #         f"Name {base_name} already exists. New name: {new_name}"
        #     )

        return new_name

    def __show_data_farming_core(
        self, base_object: Literal["Problem", "Solver"]
    ) -> None:
        """Show data farming options for a solver or problem.

        Parameters
        ----------
        base_object : Literal["Problem", "Solver"]
            Solver or Problem object.

        """
        # Check if the base object is a Problem or Solver
        if base_object not in ("Problem", "Solver"):
            raise TypeError("base_object must be 'Problem' or 'Solver'")

        # Run compatability checks
        # TODO: make these not dependent on self attributes
        # if base_object == "Problem":
        #     self.check_problem_compatibility()
        # else:
        #     self.check_solver_compatibility()

        # Initialize the frame for the data farming factor options
        if hasattr(self, "factor_frame"):
            self.factor_frame.destroy()
        if base_object == "Problem":
            self.factor_frame = tk.Frame(
                master=self.problem_datafarm_notebook_frame
            )
        else:
            self.factor_frame = tk.Frame(master=self.frames["add solver"])
        self.factor_frame.grid(row=1, column=0, sticky=tk.N + tk.W)

        # Get the name of the problem or solver from the GUI
        # and use it to get the object
        if base_object == "Problem":
            selected_name = self.selected_problem.get()
            datafarm_object = self.problem_list[selected_name]()
            # TODO: revamp problem so this isn't needed
            self.problem_save_for_later = datafarm_object
        else:
            selected_name = self.selected_solver.get()
            datafarm_object = self.solver_list[selected_name]()
            # TODO: revamp solver so this isn't needed
            self.solver_save_for_later = datafarm_object

        # Convert the specifications to factors so they can be displayed
        specifications = datafarm_object.specifications
        self.factor_dict = spec_dict_to_df_dict(specifications)
        # If the object is a Problem, we need to display the model factors
        if base_object == "Problem":
            model_specifications = datafarm_object.model.specifications
            model_factor_dict = spec_dict_to_df_dict(model_specifications)
            for factor in model_factor_dict:
                self.factor_dict[factor] = model_factor_dict[factor]

        # Add all the column headers
        header_end_row = self.insert_factor_headers(frame=self.factor_frame)
        # Add all the factors
        factor_end_row = self.insert_factors(
            frame=self.factor_frame,
            factor_dict=self.factor_dict,
            first_row=header_end_row + 1,
        )

        # Create the design options
        self.design_frame = tk.Frame(master=self.factor_frame)
        self.design_frame.grid(row=factor_end_row + 1, column=0, columnspan=8)

        # Design type for problem
        self.design_type_label = tk.Label(
            master=self.design_frame,
            text="Design Type",
            width=20,
        )
        self.design_type_label.grid(row=0, column=0)

        self.design_type = tk.StringVar()
        self.design_type.set("nolhs")
        self.design_type_menu = ttk.OptionMenu(
            self.design_frame,
            self.design_type,
            "nolhs",
            *DesignType._member_names_,
        )
        self.design_type_menu.grid(row=0, column=1, padx=30)

        # Stack selection menu
        self.stack_label = tk.Label(
            self.design_frame,
            text="Number of Stacks",
            width=20,
        )
        self.stack_label.grid(row=1, column=0)
        self.stack_count = tk.StringVar()
        self.stack_count.set("1")
        self.stack_menu = ttk.Entry(
            master=self.design_frame,
            width=10,
            textvariable=self.stack_count,
            justify="right",
        )
        self.stack_menu.grid(row=1, column=1)

        # Design name entry
        self.name_label = tk.Label(
            master=self.design_frame,
            text="Name of Design",
            width=20,
        )
        self.name_label.grid(row=2, column=0)
        self.design_name = tk.StringVar()
        # Get unique design name
        if base_object == "Problem":
            lookup_dict = self.root_problem_dict
        else:
            lookup_dict = self.root_solver_dict
        unique_name = self.get_unique_name(
            lookup_dict, f"{datafarm_object.name}_design"
        )
        # Set the design name
        self.design_name.set(unique_name)
        self.design_name_entry = tk.Entry(
            master=self.design_frame,
            textvariable=self.design_name,
            width=20,
        )
        self.design_name_entry.grid(row=2, column=1)

        # Delete the create design button if it exists
        if hasattr(self, "create_design_button"):
            self.create_design_button.destroy()
        # Create design button
        if base_object == "Problem":
            create_function = self.create_problem_design
        else:
            create_function = self.create_solver_design
        self.create_design_button = tk.Button(
            master=self.design_frame,
            text="Create Design",
            command=create_function,
        )
        self.create_design_button.grid(row=3, column=0, columnspan=2)

    def show_problem_datafarm(self, option: tk.StringVar | None = None) -> None:
        self.__show_data_farming_core(base_object="Problem")

    def show_solver_datafarm(self, option: tk.StringVar | None = None) -> None:
        self.__show_data_farming_core(base_object="Solver")

    def create_solver_design(self) -> None:
        if self.design_name.get() in self.root_solver_dict:
            tk.messagebox.showerror(
                "Error",
                "A design with this name already exists. Please choose a different name.",
            )
            return

        self.solver_design_name = self.design_name.get()
        # # Get unique solver design name
        # self.solver_design_name = self.get_unique_name(
        #     self.root_solver_dict, self.design_name.get()
        # )

        # get n stacks and design type from user input
        n_stacks = int(self.stack_count.get())
        design_type = self.design_type.get()

        """ Determine factors included in design """
        # List of names of factors included in the design
        self.design_factors: list[str] = []
        # Dict of cross design factors w/ lists of possible values
        # TODO: figure out if this will ever be anything other than bools
        self.cross_design_factors: dict[str, list[str]] = {}
        # Dict of factors not included in the design
        # Key is the factor name, value is the default value
        self.fixed_factors: dict[str, bool | float | int] = {}
        for factor in self.factor_dict:
            # If the factor is not included in the design, it's a fixed factor
            if (
                self.factor_dict[factor].include is None
                or not self.factor_dict[factor].include.get()  # type: ignore
            ):
                fixed_val = self.factor_dict[factor].default_eval
                self.fixed_factors[factor] = fixed_val
            # If the factor is included in the design, add it to the list of factors
            else:
                if self.factor_dict[factor].type.get() in ("int", "float"):
                    self.design_factors.append(factor)
                elif self.factor_dict[factor].type.get() == "bool":
                    self.cross_design_factors[factor] = ["True", "False"]

        """ Check if there are any factors included in the design """
        if not self.design_factors and not self.cross_design_factors:
            # Create a non-datafarmed solver design
            solver_list = []
            solver_list.append(self.fixed_factors)

            # Nested loop to search for the right solver without knowing its name
            design_factors = set(self.factor_dict)
            for solver in solver_directory:
                # Set of factors for the solver
                solver_object = solver_directory[solver]()
                solver_factors = set(solver_object.specifications.keys())
                # If the sets are equal, we have found the right solver
                if solver_factors == design_factors:
                    solver_list.append(str(solver))
                    break

            self.root_solver_dict[self.solver_design_name] = [solver_list]

            # add solver name to solver index
            solver_row = len(self.root_solver_dict) - 1
            self.solver_list_label = tk.Label(
                master=self.solver_list_canvas,
                text=self.solver_design_name,
            )
            self.solver_list_label.grid(row=solver_row, column=1)
            self.solver_list_labels[self.solver_design_name] = (
                self.solver_list_label
            )

            # add delete and view/edit buttons
            self.solver_edit_button = tk.Button(
                master=self.solver_list_canvas,
                text="View/Edit",
                command=lambda solver_design_name=self.solver_design_name: self.edit_solver(
                    solver_design_name
                ),
            )
            self.solver_edit_button.grid(row=solver_row, column=2)
            self.solver_edit_buttons[self.solver_design_name] = (
                self.solver_edit_button
            )
            self.solver_del_button = tk.Button(
                master=self.solver_list_canvas,
                text="Delete",
                command=lambda solver_design_name=self.solver_design_name: self.delete_solver(
                    solver_design_name
                ),
            )
            self.solver_del_button.grid(row=solver_row, column=3)
            self.solver_del_buttons[self.solver_design_name] = (
                self.solver_del_button
            )

            # refresh solver name entry box
            self.design_name.set(
                self.get_unique_name(
                    self.root_solver_dict, self.solver_design_name
                )
            )

        else:
            """ Create factor settings txt file"""
            # Check if folder exists, if not create it
            if not os.path.exists(DATA_FARMING_DIR):
                os.makedirs(DATA_FARMING_DIR)
            # If file already exists, clear it and make a new, empty file of the same name
            filepath = os.path.join(
                DATA_FARMING_DIR, f"{self.solver_design_name}.txt"
            )
            if os.path.exists(filepath):
                os.remove(filepath)

            # Write the factor settings to the file
            with open(filepath, "x") as settings_file:
                # For each factor, write the min, max, and decimal values to the file
                for factor_name in self.design_factors:
                    # Lookup the factor in the dictionary
                    factor = self.factor_dict[factor_name]
                    # Make sure the factor has a minimum and maximum value
                    assert factor.minimum is not None
                    assert factor.maximum is not None
                    min_val = factor.minimum.get()
                    max_val = factor.maximum.get()
                    if factor.type.get() == "float":
                        assert factor.num_decimals is not None
                        dec_val = factor.num_decimals.get()
                    else:
                        dec_val = "0"

                    # Write the values to the file
                    data_insert = f"{min_val} {max_val} {dec_val}\n"
                    settings_file.write(data_insert)

            try:
                self.solver_design_list = create_design(
                    name=self.solver_save_for_later.name,
                    factor_headers=self.design_factors,
                    factor_settings_filename=self.solver_design_name,
                    fixed_factors=self.fixed_factors,
                    cross_design_factors=self.cross_design_factors,
                    n_stacks=n_stacks,
                    design_type=design_type,
                    class_type="solver",
                )
            except Exception as e:
                # Give error message if design creation fails
                tk.messagebox.showerror("Error Creating Design", str(e))
                return
            # display design tree
            self.display_design_tree(
                os.path.join(
                    DATA_FARMING_DIR, f"{self.solver_design_name}_design.csv"
                ),
                self.factor_frame,
                row=999,
                column=0,
                columnspan=8,
            )
            # button to add solver design to experiment
            self.add_solver_design_button = tk.Button(
                master=self.factor_frame,
                text="Add this solver to experiment",
                command=self.add_solver_design_to_experiment,
            )
            self.add_solver_design_button.grid(row=1000, column=0, columnspan=8)
            # disable design name entry
            self.design_name_entry.configure(state="disabled")

    def create_problem_design(self) -> None:
        # Get unique solver design name
        self.problem_design_name = self.get_unique_name(
            self.root_problem_dict, self.design_name.get()
        )

        # get n stacks and design type from user input
        n_stacks = int(self.stack_count.get())
        design_type = self.design_type.get()

        """ Determine factors included in design """
        # List of names of factors included in the design
        self.design_factors: list[str] = []
        # Dict of cross design factors w/ lists of possible values
        # TODO: figure out if this will ever be anything other than bools
        self.cross_design_factors: dict[str, list[str]] = {}
        # Dict of factors not included in the design
        # Key is the factor name, value is the default value
        self.fixed_factors: dict[str, bool | float | int] = {}
        for factor in self.factor_dict:
            # If the factor is not included in the design, it's a fixed factor
            if (
                self.factor_dict[factor].include is None
                or not self.factor_dict[factor].include.get()  # type: ignore
            ):
                fixed_val = self.factor_dict[factor].default_eval
                self.fixed_factors[factor] = fixed_val
            # If the factor is included in the design, add it to the list of factors
            else:
                if self.factor_dict[factor].type.get() in ("int", "float"):
                    self.design_factors.append(factor)
                elif self.factor_dict[factor].type.get() == "bool":
                    self.cross_design_factors[factor] = ["True", "False"]

        """ Check if there are any factors included in the design """
        if not self.design_factors and not self.cross_design_factors:
            # Create a non-datafarmed problem design
            problem_list = []
            problem_list.append(self.fixed_factors)

            # Nested loop to search for the right solver without knowing its name
            design_factors = set(self.factor_dict)
            for problem in problem_directory:
                # Set of factors for the solver
                problem_object = problem_directory[problem]()
                problem_factors = set(
                    problem_object.specifications.keys()
                ).union(set(problem_object.model.specifications.keys()))
                # If the sets are equal, we have found the right solver
                if problem_factors == design_factors:
                    problem_list.append(str(problem))
                    break
            assert (
                len(problem_list) == 2
            ), "Unable to find problem in problem directory"

            self.root_problem_dict[self.problem_design_name] = [problem_list]

            # add solver name to solver index
            problem_row = len(self.root_problem_dict) - 1
            self.problem_list_label = tk.Label(
                master=self.problem_list_canvas,
                text=self.problem_design_name,
            )
            self.problem_list_label.grid(row=problem_row, column=1)
            self.problem_list_labels[self.problem_design_name] = (
                self.problem_list_label
            )

            # add delete and view/edit buttons
            self.problem_edit_button = tk.Button(
                master=self.problem_list_canvas,
                text="View/Edit",
                command=lambda problem_design_name=self.problem_design_name: self.edit_problem(
                    problem_design_name
                ),
            )
            self.problem_edit_button.grid(row=problem_row, column=2)
            self.problem_edit_buttons[self.problem_design_name] = (
                self.problem_edit_button
            )
            self.problem_del_button = tk.Button(
                master=self.problem_list_canvas,
                text="Delete",
                command=lambda problem_design_name=self.problem_design_name: self.delete_problem(
                    problem_design_name
                ),
            )
            self.problem_del_button.grid(row=problem_row, column=3)
            self.problem_del_buttons[self.problem_design_name] = (
                self.problem_del_button
            )

            # refresh solver name entry box
            self.design_name.set(
                self.get_unique_name(
                    self.root_problem_dict, self.problem_design_name
                )
            )

        else:
            """ Create factor settings txt file"""
            # Check if folder exists, if not create it
            if not os.path.exists(DATA_FARMING_DIR):
                os.makedirs(DATA_FARMING_DIR)
            # If file already exists, clear it and make a new, empty file of the same name
            filepath = os.path.join(
                DATA_FARMING_DIR, f"{self.problem_design_name}.txt"
            )
            if os.path.exists(filepath):
                os.remove(filepath)

            # Write the factor settings to the file
            with open(filepath, "x") as settings_file:
                # For each factor, write the min, max, and decimal values to the file
                for factor_name in self.design_factors:
                    # Lookup the factor in the dictionary
                    factor = self.factor_dict[factor_name]
                    # Make sure the factor has a minimum and maximum value
                    assert factor.minimum is not None
                    assert factor.maximum is not None
                    min_val = factor.minimum.get()
                    max_val = factor.maximum.get()
                    if factor.type.get() == "float":
                        assert factor.num_decimals is not None
                        dec_val = factor.num_decimals.get()
                    else:
                        dec_val = "0"

                    # Write the values to the file
                    data_insert = f"{min_val} {max_val} {dec_val}\n"
                    settings_file.write(data_insert)

            try:
                self.problem_design_list = create_design(
                    name=self.problem_save_for_later.name,
                    factor_headers=self.design_factors,
                    factor_settings_filename=self.problem_design_name,
                    fixed_factors=self.fixed_factors,
                    cross_design_factors=self.cross_design_factors,
                    n_stacks=n_stacks,
                    design_type=design_type,
                    class_type="problem",
                )
            except Exception as e:
                # Give error message if design creation fails
                tk.messagebox.showerror("Error Creating Design", str(e))
                return
            # display design tree
            self.display_design_tree(
                os.path.join(
                    DATA_FARMING_DIR, f"{self.problem_design_name}_design.csv"
                ),
                self.factor_frame,
                row=999,
                column=0,
                columnspan=8,
            )
            # button to add solver design to experiment
            self.add_problem_design_button = tk.Button(
                master=self.factor_frame,
                text="Add this problem to experiment",
                command=self.add_problem_design_to_experiment,
            )
            self.add_problem_design_button.grid(
                row=1000, column=0, columnspan=8
            )
            # disable design name entry
            self.design_name_entry.configure(state="disabled")

    def display_design_tree(
        self,
        csv_filename: str,
        frame: tk.Frame,
        row: int = 0,
        column: int = 0,
        columnspan: int = 1,
    ) -> None:
        # Initialize design tree
        self.create_design_frame = tk.Frame(master=frame)
        self.create_design_frame.grid(
            row=row, column=column, columnspan=columnspan
        )

        self.design_tree = ttk.Treeview(master=self.create_design_frame)
        self.design_tree.grid(row=1, column=0, sticky="nsew", padx=10)
        self.style = ttk.Style()
        self.style.configure(
            "Treeview.Heading",
            font=nametofont("TkHeadingFont"),
        )
        self.style.configure(
            "Treeview",
            foreground="black",
            font=nametofont("TkTextFont"),
        )
        self.design_tree.heading("#0", text="Design #")

        # Get design point values from csv
        design_table = pd.read_csv(csv_filename, index_col="design_num")
        num_dp = len(design_table)  # used for label
        self.create_design_label = tk.Label(
            master=self.create_design_frame,
            text=f"Total Design Points: {num_dp}",
            font=nametofont("TkHeadingFont"),
            width=50,
        )
        self.create_design_label.grid(row=0, column=0, sticky=tk.W)

        # Enter design values into treeview
        self.design_tree["columns"] = tuple(design_table.columns)[:-3]

        for column in design_table.columns[:-3]:
            self.design_tree.heading(column, text=column)
            self.design_tree.column(column, width=100)

        for index, row in design_table.iterrows():
            self.design_tree.insert(
                "", index, text=index, values=tuple(row)[:-3]
            )

        # Create a horizontal scrollbar
        xscrollbar = ttk.Scrollbar(
            master=self.create_design_frame,
            orient="horizontal",
            command=self.design_tree.xview,
        )
        xscrollbar.grid(row=2, column=0, sticky="nsew")

        # Configure the Treeview to use the horizontal scrollbar
        self.design_tree.configure(xscrollcommand=xscrollbar.set)

    def convert_proper_datatype(
        self,
        fixed_factors: dict,
        base_object: Problem | Solver | Model,
        var: bool = False,
    ) -> dict:
        # TODO: figure out if VAR is supposed to be true or false
        converted_fixed_factors = {}

        for factor in fixed_factors:
            if (
                var
            ):  # determine if factors are still variable objects or strings
                fixed_val = fixed_factors[factor].get()
            else:
                fixed_val = fixed_factors[factor]
            if factor in base_object.specifications:
                datatype = base_object.specifications[factor].get("datatype")
            else:
                datatype = base_object.model.specifications[factor].get(
                    "datatype"
                )

            if datatype in (int, float):
                converted_fixed_factors[factor] = datatype(fixed_val)
            if datatype is list:
                converted_fixed_factors[factor] = ast.literal_eval(fixed_val)
            if datatype is tuple:
                last_val = fixed_val[-2]
                tuple_str = fixed_val[1:-1].split(",")
                # determine if last tuple value is empty
                if last_val != ",":
                    converted_fixed_factors[factor] = tuple(
                        float(s) for s in tuple_str
                    )
                else:
                    tuple_exclude_last = tuple_str[:-1]
                    float_tuple = [float(s) for s in tuple_exclude_last]
                    converted_fixed_factors[factor] = tuple(float_tuple)
            if datatype is bool:
                if fixed_val == "True":
                    converted_fixed_factors[factor] = True
                else:
                    converted_fixed_factors[factor] = False

        return converted_fixed_factors

    def add_solver_to_experiment(self) -> None:
        # get solver name entered by user & ensure it is unique
        solver_name = self.get_unique_name(
            self.root_solver_dict, self.solver_name_var.get()
        )
        # convert factor values to proper data type
        fixed_factors = self.convert_proper_datatype(
            self.solver_defaults, self.solver_object
        )

        solver_list = []  # holds dictionary of dps and solver name
        solver_list.append(fixed_factors)
        solver_list.append(self.solver_object.name)

        solver_holder_list = []  # used so solver list matches datafarming format
        solver_holder_list.append(solver_list)

        self.root_solver_dict[solver_name] = solver_holder_list
        # add solver name to solver index
        solver_row = len(self.root_solver_dict) - 1
        self.solver_list_label = tk.Label(
            master=self.solver_list_canvas,
            text=solver_name,
        )
        self.solver_list_label.grid(row=solver_row, column=1)
        self.solver_list_labels[solver_name] = self.solver_list_label

        # add delete and view/edit buttons
        self.solver_edit_button = tk.Button(
            master=self.solver_list_canvas,
            text="View/Edit",
            command=lambda: self.edit_solver(solver_name),
        )
        self.solver_edit_button.grid(row=solver_row, column=2)
        self.solver_edit_buttons[solver_name] = self.solver_edit_button
        self.solver_del_button = tk.Button(
            master=self.solver_list_canvas,
            text="Delete",
            command=lambda: self.delete_solver(solver_name),
        )
        self.solver_del_button.grid(row=solver_row, column=3)
        self.solver_del_buttons[solver_name] = self.solver_del_button

        # refresh solver name entry box
        self.solver_name_var.set(
            self.get_unique_name(self.root_solver_dict, solver_name)
        )

    def add_problem_to_experiment(self) -> None:
        # Convect problem and model factor values to proper data type
        prob_fixed_factors = self.convert_proper_datatype(
            self.problem_defaults, self.problem_object
        )
        # mod_fixed_factors = self.convert_proper_datatype(self.model_defaults, self.model_object.specifications)

        # get problem name and ensure it is unique
        problem_name = self.get_unique_name(
            self.root_problem_dict, self.problem_name_var.get()
        )

        problem_list = []  # holds dictionary of dps and solver name
        problem_list.append(prob_fixed_factors)
        problem_list.append(self.problem_object.name)

        problem_holder_list = []  # used so solver list matches datafarming format
        problem_holder_list.append(problem_list)

        self.root_problem_dict[problem_name] = problem_holder_list

        # add problem name to problem index
        problem_row = len(self.root_problem_dict) - 1
        self.problem_list_label = tk.Label(
            master=self.problem_list_canvas,
            text=problem_name,
        )
        self.problem_list_label.grid(row=problem_row, column=1)
        self.problem_list_labels[problem_name] = self.problem_list_label

        # add delete and view/edit buttons
        self.problem_edit_button = tk.Button(
            master=self.problem_list_canvas,
            text="View/Edit",
        )
        self.problem_edit_button.grid(row=problem_row, column=2)
        self.problem_edit_buttons[problem_name] = self.problem_edit_button
        self.problem_del_button = tk.Button(
            master=self.problem_list_canvas,
            text="Delete",
            command=lambda: self.delete_problem(problem_name),
        )
        self.problem_del_button.grid(row=problem_row, column=3)
        self.problem_del_buttons[problem_name] = self.problem_del_button

        # refresh problem name entry box
        self.problem_name_var.set(
            self.get_unique_name(self.root_problem_dict, problem_name)
        )

    def add_problem_design_to_experiment(self) -> None:
        problem_design_name = self.problem_design_name

        problem_holder_list = []  # holds all problem lists within design name
        for _, dp in enumerate(self.problem_design_list):
            dp_list = []  # holds dictionary of factors for current dp
            dp_list.append(dp)  # append problem factors
            dp_list.append(self.design_name.get())  # append name of problem
            problem_holder_list.append(
                dp_list
            )  # add current dp information to holder list

        self.root_problem_dict[problem_design_name] = problem_holder_list
        self.add_problem_design_to_list()

    def add_problem_design_to_list(self) -> None:
        problem_design_name = self.problem_design_name

        # add solver name to solver index
        problem_row = len(self.root_problem_dict) - 1
        self.problem_list_label = tk.Label(
            master=self.problem_list_canvas,
            text=problem_design_name,
        )
        self.problem_list_label.grid(row=problem_row, column=1)
        self.problem_list_labels[problem_design_name] = self.problem_list_label

        # add delete and view/edit buttons
        self.problem_edit_button = tk.Button(
            master=self.problem_list_canvas,
            text="View/Edit",
        )
        self.problem_edit_button.grid(row=problem_row, column=2)
        self.problem_edit_buttons[problem_design_name] = (
            self.problem_edit_button
        )
        self.problem_del_button = tk.Button(
            master=self.problem_list_canvas,
            text="Delete",
            command=lambda: self.delete_problem(problem_design_name),
        )
        self.problem_del_button.grid(row=problem_row, column=3)
        self.problem_del_buttons[problem_design_name] = self.problem_del_button

        # refresh problem design name entry box

    def add_solver_design_to_experiment(self) -> None:
        solver_design_name = self.solver_design_name

        solver_holder_list = []  # used so solver list matches datafarming format
        for dp in self.solver_design_list:
            solver_list = []  # holds dictionary of dps and solver name
            solver_list.append(dp)
            solver_list.append(solver_design_name)
            solver_holder_list.append(solver_list)

        self.root_solver_dict[solver_design_name] = solver_holder_list
        # add solver name to solver index
        solver_row = len(self.root_solver_dict) - 1
        self.solver_list_label = tk.Label(
            master=self.solver_list_canvas,
            text=solver_design_name,
        )
        self.solver_list_label.grid(row=solver_row, column=1)
        self.solver_list_labels[solver_design_name] = self.solver_list_label

        # add delete and view/edit buttons
        self.solver_edit_button = tk.Button(
            master=self.solver_list_canvas,
            text="View/Edit",
        )
        self.solver_edit_button.grid(row=solver_row, column=2)
        self.solver_edit_buttons[solver_design_name] = self.solver_edit_button
        self.solver_del_button = tk.Button(
            master=self.solver_list_canvas,
            text="Delete",
            command=lambda: self.delete_solver(solver_design_name),
        )
        self.solver_del_button.grid(row=solver_row, column=3)
        self.solver_del_buttons[solver_design_name] = self.solver_del_button

        # refresh solver design name entry box
        self.design_name.set(
            self.get_unique_name(self.root_solver_dict, solver_design_name)
        )

    def edit_solver(self, solver_save_name: str) -> None:
        # clear previous selections
        self.clear_frame(self.solver_frame)

        """ Initialize frames and headers"""

        self.solver_frame = tk.Frame(master=self.solver_notebook_frame)
        self.solver_frame.grid(row=1, column=0)
        self.factor_display_canvas = tk.Canvas(master=self.solver_frame)
        self.factor_display_canvas.grid(row=1, column=0)

        # Create column for solver factor names
        self.header_lbl_name = tk.Label(
            master=self.solver_frame,
            text="Solver Factors",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.header_lbl_name.grid(row=0, column=0, sticky=tk.N + tk.W)

        # Create column for factor type
        self.header_lbl_type = tk.Label(
            master=self.solver_frame,
            text="Factor Type",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.header_lbl_type.grid(row=0, column=1, sticky=tk.N + tk.W)

        # Create column for factor default values
        self.headerdefault_label = tk.Label(
            master=self.solver_frame,
            text="Default Value",
            font=nametofont("TkHeadingFont"),
            width=20,
        )
        self.headerdefault_label.grid(row=0, column=2, sticky=tk.N + tk.W)

        """ Get solver information from master solver dict and display"""
        solver = self.root_solver_dict[solver_save_name][0][1]

        self.solver_object = solver_directory[solver]()

        # show problem factors and store default widgets to this dict
        self.solver_defaults = self.show_factor_defaults(
            self.solver_object,
            self.factor_display_canvas,
            factor_dict=self.root_solver_dict[solver_save_name][0][0],
        )

        # save previous name if user edits name
        self.solver_prev_name = solver_save_name

        # entry for solver name and add solver button
        self.solver_name_label = tk.Label(
            master=self.solver_frame,
            text="Solver Name",
            width=20,
        )
        self.solver_name_label.grid(row=2, column=0)
        self.solver_name_var = tk.StringVar()
        self.solver_name_var.set(solver_save_name)
        self.solver_name_entry = tk.Entry(
            master=self.solver_frame,
            textvariable=self.solver_name_var,
            width=20,
        )
        self.solver_name_entry.grid(row=2, column=1)
        self.add_sol_to_exp_button = tk.Button(
            master=self.solver_frame,
            text="Save Edits",
            command=self.save_solver_edits,
        )
        self.add_sol_to_exp_button.grid(row=3, column=0)

    def save_solver_edits(self) -> None:
        # convert factor values to proper data type
        fixed_factors = self.convert_proper_datatype(
            self.solver_defaults, self.solver_object
        )

        # Update fixed factors in solver master dict
        self.root_solver_dict[self.solver_prev_name][0][0] = fixed_factors

        # Change solver save name if applicable
        new_solver_name = self.solver_name_var.get()
        if new_solver_name != self.solver_prev_name:
            self.root_solver_dict[new_solver_name] = self.root_solver_dict[
                self.solver_prev_name
            ]
            del self.root_solver_dict[self.solver_prev_name]
            # change solver name in labels & buttons
            self.solver_list_labels[self.solver_prev_name].configure(
                text=new_solver_name
            )
            self.solver_list_labels[new_solver_name] = self.solver_list_labels[
                self.solver_prev_name
            ]
            self.solver_edit_buttons[new_solver_name] = (
                self.solver_edit_buttons[self.solver_prev_name]
            )
            self.solver_del_buttons[new_solver_name] = self.solver_del_buttons[
                self.solver_prev_name
            ]
            del self.solver_list_labels[self.solver_prev_name]
            del self.solver_edit_buttons[self.solver_prev_name]
            del self.solver_del_buttons[self.solver_prev_name]

    def delete_solver(self, solver_name: str) -> None:
        # delete from master list
        del self.root_solver_dict[solver_name]

        # delete label & edit/delete buttons
        self.solver_list_labels[solver_name].destroy()
        self.solver_edit_buttons[solver_name].destroy()
        self.solver_del_buttons[solver_name].destroy()
        del self.solver_list_labels[solver_name]
        del self.solver_edit_buttons[solver_name]
        del self.solver_del_buttons[solver_name]

        # re-display solver labels & buttons
        for row, solver_group in enumerate(self.solver_list_labels):
            self.solver_list_labels[solver_group].grid(row=row, column=1)
            self.solver_edit_buttons[solver_group].grid(row=row, column=2)
            self.solver_del_buttons[solver_group].grid(row=row, column=3)

    def delete_problem(self, problem_name: str) -> None:
        # delete from master list
        del self.root_problem_dict[problem_name]

        # delete label & edit/delete buttons
        self.problem_list_labels[problem_name].destroy()
        self.problem_edit_buttons[problem_name].destroy()
        self.problem_del_buttons[problem_name].destroy()
        del self.problem_list_labels[problem_name]
        del self.problem_edit_buttons[problem_name]
        del self.problem_del_buttons[problem_name]

        # re-display problem labels & buttons
        for row, problem_group in enumerate(self.problem_list_labels):
            self.problem_list_labels[problem_group].grid(row=row, column=1)
            self.problem_edit_buttons[problem_group].grid(row=row, column=2)
            self.problem_del_buttons[problem_group].grid(row=row, column=3)

    def create_experiment(self) -> None:
        # get unique experiment name
        self.experiment_name = self.get_unique_name(
            self.root_experiment_dict, self.experiment_name_var.get()
        )

        # get pickle checkstate
        pickle_checkstate = self.pickle_checkstate.get()

        # Extract solver and problem information from master dictionaries
        master_solver_factor_list = []  # holds dict of factors for each dp
        master_solver_name_list = []  # holds name of each solver for each dp
        master_problem_factor_list = []  # holds dict of factors for each dp
        master_problem_name_list = []  # holds name of each solver for each dp
        solver_renames = []  # holds rename for each solver
        problem_renames = []  # holds rename for each problem

        for solver_group_name in self.root_solver_dict:
            solver_group = self.root_solver_dict[solver_group_name]
            for index, dp in enumerate(solver_group):
                factors = dp[0]
                solver_name = dp[1]
                if len(solver_group) > 1:
                    solver_rename = f"{solver_group_name}_dp_{index}"
                else:
                    solver_rename = f"{solver_group_name}"

                master_solver_factor_list.append(factors)
                master_solver_name_list.append(solver_name)
                solver_renames.append(solver_rename)

        for problem_group_name in self.root_problem_dict:
            problem_group = self.root_problem_dict[problem_group_name]
            for index, dp in enumerate(problem_group):
                factors = dp[0]
                problem_name = dp[1]
                if len(problem_group) > 1:
                    problem_rename = f"{problem_group_name}_dp_{index}"
                else:
                    problem_rename = f"{problem_group_name}"
                master_problem_factor_list.append(factors)
                master_problem_name_list.append(problem_name)
                problem_renames.append(problem_rename)

        # use ProblemsSolvers to initialize exp
        self.experiment = ProblemsSolvers(
            solver_factors=master_solver_factor_list,
            problem_factors=master_problem_factor_list,
            solver_names=master_solver_name_list,
            problem_names=master_problem_name_list,
            solver_renames=solver_renames,
            problem_renames=problem_renames,
            experiment_name=self.experiment_name,
            create_pair_pickles=pickle_checkstate,
        )

        # run check on solver/problem compatibility
        self.experiment.check_compatibility()

        # add to master experiment list
        self.root_experiment_dict[self.experiment_name] = self.experiment

        # reset default experiment name for next experiment
        self.experiment_name_var.set(
            self.get_unique_name(
                self.root_experiment_dict, self.experiment_name
            )
        )

        # add exp to row
        self.add_exp_row()

    def clear_experiment(self) -> None:
        # clear solver and problem lists
        self.root_solver_factor_list = []
        self.root_solver_name_list = []
        self.root_problem_factor_list = []
        self.root_problem_name_list = []
        self.root_solver_dict = {}
        self.root_problem_dict = {}
        self.clear_frame(self.solver_list_canvas)
        self.clear_frame(self.problem_list_canvas)

    def add_exp_row(self) -> None:
        """Display experiment in list."""
        experiment_row = self.experiment_display_canvas.grid_size()[1]
        self.current_experiment_frame = tk.Frame(
            master=self.experiment_display_canvas
        )
        self.current_experiment_frame.grid(row=experiment_row, column=0)
        self.experiment_list_label = tk.Label(
            master=self.current_experiment_frame,
            text=self.experiment_name,
        )
        self.experiment_list_label.grid(row=0, column=0)

        # run button
        self.run_experiment_button = tk.Button(
            master=self.current_experiment_frame,
            text="Run",
            command=lambda name=self.experiment_name: self.run_experiment(
                experiment_name=name
            ),
        )
        self.run_experiment_button.grid(row=0, column=2, pady=10)
        self.run_buttons[self.experiment_name] = (
            self.run_experiment_button
        )  # add run button to widget dict
        # experiment options button
        self.post_process_opt_button = tk.Button(
            master=self.current_experiment_frame,
            text="Experiment Options",
            command=lambda name=self.experiment_name: self.open_post_processing_window(
                name
            ),
        )
        self.post_process_opt_button.grid(row=0, column=1)
        self.post_process_opt_buttons[self.experiment_name] = (
            self.post_process_opt_button
        )  # add option button to widget dict
        # post replication button
        self.post_process_button = tk.Button(
            master=self.current_experiment_frame,
            text="Post-Replicate",
            command=lambda name=self.experiment_name: self.post_process(name),
        )
        self.post_process_button.grid(row=0, column=3)
        self.post_process_button.configure(state="disabled")
        self.post_process_buttons[self.experiment_name] = (
            self.post_process_button
        )  # add post process button to widget dict
        # post normalize button
        self.post_norm_button = tk.Button(
            master=self.current_experiment_frame,
            text="Post-Normalize",
            command=lambda name=self.experiment_name: self.post_normalize(name),
        )
        self.post_norm_button.grid(row=0, column=4)
        self.post_norm_button.configure(state="disabled")
        self.post_norm_buttons[self.experiment_name] = (
            self.post_norm_button
        )  # add post process button to widget dict
        # log results button
        self.log_button = tk.Button(
            master=self.current_experiment_frame,
            text="Log Results",
            command=lambda name=self.experiment_name: self.log_results(name),
        )
        self.log_button.grid(row=0, column=5)
        self.log_button.configure(state="disabled")
        self.log_buttons[self.experiment_name] = (
            self.log_button
        )  # add post process button to widget dict
        # all in one
        self.all_button = tk.Button(
            master=self.current_experiment_frame,
            text="All",
            command=lambda name=self.experiment_name: self.do_all_steps(name),
        )
        self.all_button.grid(row=0, column=6)
        self.all_buttons[self.experiment_name] = (
            self.all_button
        )  # add post process button to widget dict
        # delete experiment
        self.delete_exp_button = tk.Button(
            master=self.current_experiment_frame,
            text="Delete Experiment",
            command=lambda name=self.experiment_name,
            f=self.current_experiment_frame: self.delete_experiment(name, f),
        )
        self.delete_exp_button.grid(row=0, column=7)

    def delete_experiment(
        self, experiment_name: str, experiment_frame: tk.Frame
    ) -> None:
        del self.root_experiment_dict[experiment_name]
        self.clear_frame(experiment_frame)
        # move up other frames below deleted one
        row = experiment_frame.grid_info()["row"]
        experiment_frames = self.experiment_display_canvas.winfo_children()
        for frame in experiment_frames:
            current_row = frame.grid_info()["row"]
            if current_row > row:
                frame.grid(row=current_row - 1, column=0)

    def run_experiment(self, experiment_name: str) -> None:
        # get experiment object from master dict
        experiment = self.root_experiment_dict[experiment_name]

        # get specified number of macro reps
        if experiment_name in self.macro_reps:
            n_macroreps = int(self.macro_reps[experiment_name].get())
        else:
            n_macroreps = self.macro_default
        # use ProblemsSolvers run
        experiment.run(n_macroreps=n_macroreps)
        # disable run buttons
        # self.macro_entries[experiment_name].configure(state = 'disabled')
        self.run_buttons[experiment_name].configure(state="disabled")

        # enable post-processing buttons
        self.post_process_buttons[experiment_name].configure(state="normal")

        # disable all button
        self.all_buttons[experiment_name].configure(state="disabled")

    def open_defaults_window(self) -> None:
        # create new winow
        self.experiment_defaults_window = Toplevel(self)
        self.experiment_defaults_window.title(
            "Simopt Graphical User Interface - Experiment Options Defaults"
        )
        self.center_window(0.8)
        self.set_theme()

        self.main_frame = tk.Frame(master=self.experiment_defaults_window)
        self.main_frame.grid(row=0, column=0)

        # Title label
        self.title_label = tk.Label(
            master=self.main_frame,
            text=" Default experiment options for all experiments. Any changes made will affect all future and current un-run or processed experiments.",
            font=nametofont("TkHeadingFont"),
        )
        self.title_label.grid(row=0, column=0, sticky="nsew")

        # Macro replication number input
        self.macro_rep_label = tk.Label(
            master=self.main_frame,
            text="Number of macro-replications of the solver run on the problem",
        )
        self.macro_rep_label.grid(row=1, column=0)
        self.macro_rep_var = tk.IntVar()
        self.macro_rep_var.set(self.macro_default)
        self.macro_rep_entry = tk.Entry(
            master=self.main_frame,
            textvariable=self.macro_rep_var,
            width=10,
            justify="right",
        )
        self.macro_rep_entry.grid(row=1, column=1)

        # Post replication number input
        self.post_rep_label = tk.Label(
            master=self.main_frame,
            text="Number of post-replications",
        )
        self.post_rep_label.grid(row=2, column=0)
        self.post_rep_var = tk.IntVar()
        self.post_rep_var.set(self.post_default)
        self.post_rep_entry = tk.Entry(
            master=self.main_frame,
            textvariable=self.post_rep_var,
            width=10,
            justify="right",
        )
        self.post_rep_entry.grid(row=2, column=1)

        # CRN across budget
        self.crn_budget_label = tk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for solutions recommended at different times?",
        )
        self.crn_budget_label.grid(row=3, column=0)
        self.crn_budget_var = tk.StringVar()
        self.crn_budget_opt = ttk.OptionMenu(
            self.main_frame,
            self.crn_budget_var,
            self.crn_budget_default,
            "yes",
            "no",
        )
        self.crn_budget_opt.grid(row=3, column=1)

        # CRN across macroreps
        self.crn_macro_label = tk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for solutions recommended on different macro-replications?",
        )
        self.crn_macro_label.grid(row=4, column=0)
        self.crn_macro_var = tk.StringVar()
        self.crn_macro_opt = ttk.OptionMenu(
            self.main_frame,
            self.crn_macro_var,
            self.crn_macro_default,
            "yes",
            "no",
        )
        self.crn_macro_opt.grid(row=4, column=1)

        # Post reps at inital & optimal solution input
        self.init_post_rep_label = tk.Label(
            master=self.main_frame,
            text="Number of post-replications at initial and optimal solutions",
        )
        self.init_post_rep_label.grid(row=5, column=0)
        self.init_post_rep_var = tk.IntVar()
        self.init_post_rep_var.set(self.init_default)
        self.init_post_rep_entry = tk.Entry(
            master=self.main_frame,
            textvariable=self.init_post_rep_var,
            width=10,
            justify="right",
        )
        self.init_post_rep_entry.grid(row=5, column=1)

        # CRN across init solutions
        self.crn_init_label = tk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for initial and optimal solution?",
        )
        self.crn_init_label.grid(row=6, column=0)
        self.crn_init_var = tk.StringVar()
        self.crn_init_opt = ttk.OptionMenu(
            self.main_frame,
            self.crn_init_var,
            self.crn_init_default,
            "yes",
            "no",
        )
        self.crn_init_opt.grid(row=6, column=1)

        # solve tols
        self.solve_tols_label = tk.Label(
            master=self.main_frame,
            text="Relative optimality gap(s) definining when a problem is solved; must be between 0 & 1, list in increasing order.",
        )
        self.solve_tols_label.grid(row=7, column=1)
        self.solve_tols_frame = tk.Frame(master=self.main_frame)
        self.solve_tols_frame.grid(row=8, column=0, columnspan=2)
        self.solve_tol_1_var = tk.StringVar()
        self.solve_tol_2_var = tk.StringVar()
        self.solve_tol_3_var = tk.StringVar()
        self.solve_tol_4_var = tk.StringVar()
        self.solve_tol_1_var.set(self.solve_tols_default[0])
        self.solve_tol_2_var.set(self.solve_tols_default[1])
        self.solve_tol_3_var.set(self.solve_tols_default[2])
        self.solve_tol_4_var.set(self.solve_tols_default[3])
        self.solve_tol_1_entry = tk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_1_var,
            width=5,
            justify="right",
        )
        self.solve_tol_2_entry = tk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_2_var,
            width=5,
            justify="right",
        )
        self.solve_tol_3_entry = tk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_3_var,
            width=5,
            justify="right",
        )
        self.solve_tol_4_entry = tk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_4_var,
            width=5,
            justify="right",
        )
        self.solve_tol_1_entry.grid(row=0, column=0, padx=5)
        self.solve_tol_2_entry.grid(row=0, column=1, padx=5)
        self.solve_tol_3_entry.grid(row=0, column=2, padx=5)
        self.solve_tol_4_entry.grid(row=0, column=3, padx=5)

        # set options as default for future experiments
        self.set_as_default_button = tk.Button(
            master=self.main_frame,
            text="Set options as default for all experiments",
            command=self.change_experiment_defaults,
        )
        self.set_as_default_button.grid(row=9, column=0)

    def change_experiment_defaults(self) -> None:
        # Change default values to user input
        self.macro_default = self.macro_rep_var.get()
        self.post_default = self.post_rep_var.get()
        self.init_default = self.init_post_rep_var.get()

        crn_budget_str = self.crn_budget_var.get()
        if crn_budget_str == "yes":
            self.crn_budget_default = True
        else:
            self.crn_budget_default = False
        crn_macro_str = self.crn_macro_var.get()
        if crn_macro_str == "yes":
            self.crn_macro_default = True
        else:
            self.crn_macro_default = False
        crn_init_str = self.crn_init_var.get()
        if crn_init_str == "yes":
            self.crn_init_default = True
        else:
            self.crn_init_default = False

        solve_tol_1 = float(self.solve_tol_1_var.get())
        solve_tol_2 = float(self.solve_tol_2_var.get())
        solve_tol_3 = float(self.solve_tol_3_var.get())
        solve_tol_4 = float(self.solve_tol_4_var.get())
        self.solve_tols_default = [
            solve_tol_1,
            solve_tol_2,
            solve_tol_3,
            solve_tol_4,
        ]

        # update macro entry widgets
        for var in self.macro_vars:
            var.set(self.macro_default)

    def find_option_setting(
        self, exp_name: str, search_dict: dict[str, any], default_val: any
    ) -> any:
        if exp_name in search_dict:
            value = search_dict[exp_name].get()
        else:
            value = default_val
        return value

    def open_post_processing_window(self, experiment_name: str) -> None:
        # check if options have already been set
        n_macroreps = self.find_option_setting(
            experiment_name, self.macro_reps, self.macro_default
        )
        n_postreps = self.find_option_setting(
            experiment_name, self.post_reps, self.post_default
        )
        crn_budget = self.find_option_setting(
            experiment_name, self.crn_budgets, self.crn_budget_default
        )
        crn_macro = self.find_option_setting(
            experiment_name, self.crn_macros, self.crn_macro_default
        )
        n_initreps = self.find_option_setting(
            experiment_name, self.init_post_reps, self.init_default
        )
        crn_init = self.find_option_setting(
            experiment_name, self.crn_inits, self.crn_init_default
        )
        if experiment_name in self.solve_tols:
            solve_tols = []
            for tol in self.solve_tols[experiment_name]:
                solve_tols.append(tol.get())
        else:
            solve_tols = self.solve_tols_default

        # create new winow
        self.post_processing_window = Toplevel(self)
        self.post_processing_window.title(
            "Simopt Graphical User Interface - Experiment Options"
        )
        self.center_window(0.8)
        self.set_theme()

        self.main_frame = tk.Frame(master=self.post_processing_window)
        self.main_frame.grid(row=0, column=0)

        # Title label
        self.title_label = tk.Label(
            master=self.main_frame,
            text=f"Options for {experiment_name}.",
            font=nametofont("TkHeadingFont"),
        )
        self.title_label.grid(row=0, column=0, sticky="nsew")

        # Macro replication number input
        self.macro_rep_label = tk.Label(
            master=self.main_frame,
            text="Number of macro-replications of the solver run on the problem",
        )
        self.macro_rep_label.grid(row=1, column=0)
        self.macro_rep_var = tk.IntVar()
        self.macro_rep_var.set(n_macroreps)
        self.macro_rep_entry = tk.Entry(
            master=self.main_frame,
            textvariable=self.macro_rep_var,
            width=10,
            justify="right",
        )
        self.macro_rep_entry.grid(row=1, column=1)

        # Post replication number input
        self.post_rep_label = tk.Label(
            master=self.main_frame,
            text="Number of post-replications",
        )
        self.post_rep_label.grid(row=2, column=0)
        self.post_rep_var = tk.IntVar()
        self.post_rep_var.set(n_postreps)
        self.post_rep_entry = tk.Entry(
            master=self.main_frame,
            textvariable=self.post_rep_var,
            width=10,
            justify="right",
        )
        self.post_rep_entry.grid(row=2, column=1)

        # CRN across budget
        self.crn_budget_label = tk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for solutions recommended at different times?",
        )
        self.crn_budget_label.grid(row=3, column=0)
        self.crn_budget_var = tk.StringVar()
        self.crn_budget_opt = ttk.OptionMenu(
            self.main_frame, self.crn_budget_var, crn_budget, "yes", "no"
        )
        self.crn_budget_opt.grid(row=3, column=1)

        # CRN across macroreps
        self.crn_macro_label = tk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for solutions recommended on different macro-replications?",
        )
        self.crn_macro_label.grid(row=4, column=0)
        self.crn_macro_var = tk.StringVar()
        self.crn_macro_opt = ttk.OptionMenu(
            self.main_frame, self.crn_macro_var, crn_macro, "yes", "no"
        )
        self.crn_macro_opt.grid(row=4, column=1)

        # Post reps at inital & optimal solution input
        self.init_post_rep_label = tk.Label(
            master=self.main_frame,
            text="Number of post-replications at initial and optimal solutions",
        )
        self.init_post_rep_label.grid(row=5, column=0)
        self.init_post_rep_var = tk.IntVar()
        self.init_post_rep_var.set(n_initreps)
        self.init_post_rep_entry = tk.Entry(
            master=self.main_frame,
            textvariable=self.init_post_rep_var,
            width=10,
            justify="right",
        )
        self.init_post_rep_entry.grid(row=5, column=1)

        # CRN across init solutions
        self.crn_init_label = tk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for initial and optimal solution?",
        )
        self.crn_init_label.grid(row=6, column=0)
        self.crn_init_var = tk.StringVar()
        self.crn_init_opt = ttk.OptionMenu(
            self.main_frame, self.crn_init_var, crn_init, "yes", "no"
        )
        self.crn_init_opt.grid(row=6, column=1)

        # solve tols
        self.solve_tols_label = tk.Label(
            master=self.main_frame,
            text="Relative optimality gap(s) definining when a problem is solved; must be between 0 & 1, list in increasing order.",
        )
        self.solve_tols_label.grid(row=7, column=1)
        self.solve_tols_frame = tk.Frame(master=self.main_frame)
        self.solve_tols_frame.grid(row=8, column=0, columnspan=2)
        self.solve_tol_1_var = tk.StringVar()
        self.solve_tol_2_var = tk.StringVar()
        self.solve_tol_3_var = tk.StringVar()
        self.solve_tol_4_var = tk.StringVar()
        self.solve_tol_1_var.set(solve_tols[0])
        self.solve_tol_2_var.set(solve_tols[1])
        self.solve_tol_3_var.set(solve_tols[2])
        self.solve_tol_4_var.set(solve_tols[3])
        self.solve_tol_1_entry = tk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_1_var,
            width=5,
            justify="right",
        )
        self.solve_tol_2_entry = tk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_2_var,
            width=5,
            justify="right",
        )
        self.solve_tol_3_entry = tk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_3_var,
            width=5,
            justify="right",
        )
        self.solve_tol_4_entry = tk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_4_var,
            width=5,
            justify="right",
        )
        self.solve_tol_1_entry.grid(row=0, column=0, padx=5)
        self.solve_tol_2_entry.grid(row=0, column=1, padx=5)
        self.solve_tol_3_entry.grid(row=0, column=2, padx=5)
        self.solve_tol_4_entry.grid(row=0, column=3, padx=5)

        # save options for current experiment
        self.set_as_default_button = tk.Button(
            master=self.main_frame,
            text="Save options for current experiment",
            command=lambda: self.save_experiment_options(experiment_name),
        )
        self.set_as_default_button.grid(row=7, column=0)

    def save_experiment_options(self, experiment_name: str) -> None:
        # get user specified values and save to dictionaries
        self.post_reps[experiment_name] = self.post_rep_var
        self.init_post_reps[experiment_name] = self.init_post_rep_var

        self.crn_budgets[experiment_name] = self.crn_budget_var

        self.macro_reps[experiment_name] = self.macro_rep_var

        self.crn_macros[experiment_name] = self.crn_macro_var

        self.crn_inits[experiment_name] = self.crn_init_var

        self.solve_tols[experiment_name] = [
            self.solve_tol_1_var,
            self.solve_tol_2_var,
            self.solve_tol_3_var,
            self.solve_tol_4_var,
        ]

    def post_process(self, experiment_name: str) -> None:
        # get experiment object from master dict
        experiment = self.root_experiment_dict[experiment_name]

        # get user specified options
        if experiment_name in self.post_reps:
            post_reps = self.post_reps[experiment_name].get()
        else:
            post_reps = self.post_default
        if experiment_name in self.crn_budgets:
            crn_budget_str = self.crn_budgets[experiment_name].get()
            if crn_budget_str == "yes":
                crn_budget = True
            else:
                crn_budget = False
        else:
            crn_budget_str = self.crn_budget_default
        if crn_budget_str == "yes":
            crn_budget = True
        else:
            crn_budget = False

        if experiment_name in self.crn_macros:
            crn_macro_str = self.crn_macros[experiment_name].get()
            if crn_macro_str == "yes":
                crn_macro = True
            else:
                crn_macro = False
        else:
            crn_macro = self.crn_macro_default

        # run post processing with parameters from dictionaries
        experiment.post_replicate(
            n_postreps=post_reps,
            crn_across_budget=crn_budget,
            crn_across_macroreps=crn_macro,
        )

        # disable post processing button & enable normalize buttons
        self.post_process_buttons[experiment_name].configure(state="disabled")
        self.post_norm_buttons[experiment_name].configure(state="normal")

    def post_normalize(self, experiment_name: str) -> None:
        # get experiment object from master dict
        experiment = self.root_experiment_dict[experiment_name]

        # get user specified options
        if experiment_name in self.init_post_reps:
            reps = self.init_post_reps[experiment_name].get()
        else:
            reps = self.init_default
        if experiment_name in self.crn_inits:
            crn_str = self.crn_inits[experiment_name].get()
            if crn_str == "yes":
                crn = True
            else:
                crn = False
        else:
            crn = self.crn_init_default

        # run post normalization
        experiment.post_normalize(
            n_postreps_init_opt=reps, crn_across_init_opt=crn
        )

        # disable post normalization button
        self.post_norm_buttons[experiment_name].configure(state="disabled")
        self.log_buttons[experiment_name].configure(state="normal")

    def log_results(self, experiment_name: str) -> None:
        # get experiment object from master dict
        experiment = self.root_experiment_dict[experiment_name]

        # get user specified options
        if experiment_name in self.solve_tols:
            tol_1 = self.solve_tols[experiment_name][0].get()
            tol_2 = self.solve_tols[experiment_name][1].get()
            tol_3 = self.solve_tols[experiment_name][2].get()
            tol_4 = self.solve_tols[experiment_name][3].get()
            solve_tols = [tol_1, tol_2, tol_3, tol_4]
        else:
            solve_tols = self.solve_tols_default

        # log results
        experiment.log_group_experiment_results()
        experiment.report_group_statistics(solve_tols=solve_tols)

        # disable log button
        self.log_buttons[experiment_name].configure(state="disabled")

    def do_all_steps(self, experiment_name: str) -> None:
        # run experiment
        self.run_experiment(experiment_name)
        # post replicate experiment
        self.post_process(experiment_name)
        # post normalize experiment
        self.post_normalize(experiment_name)
        # log experiment results
        self.log_results(experiment_name)

    def open_plotting_window(self) -> None:
        # create new window
        self.plotting_window = Toplevel(self)
        self.plotting_window.title(
            "Simopt Graphical User Interface - Experiment Plots"
        )
        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        self.center_window(0.8)
        self.set_theme()

        # Configure the grid layout to expand properly
        self.plotting_window.grid_rowconfigure(0, weight=1)
        self.plotting_window.grid_columnconfigure(0, weight=1)

        # create master canvas
        self.plotting_canvas = tk.Canvas(self.plotting_window)
        self.plotting_canvas.grid(row=0, column=0, sticky="nsew")

        # Create vertical scrollbar
        vert_scroll = ttk.Scrollbar(
            self.plotting_window,
            orient=tk.VERTICAL,
            command=self.plotting_canvas.yview,
        )
        vert_scroll.grid(row=0, column=1, sticky="ns")

        # Create horizontal scrollbar
        horiz_scroll = ttk.Scrollbar(
            self.plotting_window,
            orient=tk.HORIZONTAL,
            command=self.plotting_canvas.xview,
        )
        horiz_scroll.grid(row=1, column=0, sticky="ew")

        # Configure canvas to use the scrollbars
        self.plotting_canvas.configure(
            yscrollcommand=vert_scroll.set, xscrollcommand=horiz_scroll.set
        )

        # create master frame inside the canvas
        self.plot_main_frame = tk.Frame(self.plotting_canvas)
        self.plotting_canvas.create_window(
            (0, 0), window=self.plot_main_frame, anchor="nw"
        )

        # Bind the configure event to update the scroll region
        self.plot_main_frame.bind("<Configure>", self.update_plot_window_scroll)

        # create frames so clear frames function can execute
        self.plot_main_frame.grid_columnconfigure(0, weight=1)
        self.plot_main_frame.grid_columnconfigure(1, weight=1)
        # self.plot_main_frame.grid(row=0, column =0)
        self.plot_options_frame = tk.Frame(master=self.plot_main_frame)
        self.more_options_frame = tk.Frame(self.plot_options_frame)
        self.plotting_workspace_frame = tk.Frame(self.plot_main_frame)

        # dictonaries/variables to store plotting information
        self.experiment_tabs = {}  # holds names of experiments that already have a plotting tab created
        self.selected_solvers = []  # holds solvers that have been selected
        self.selected_problems = []  # holds problems that have been selected
        self.ref_menu_created = (
            False  # tracks if there if a ref solver menu currently existing
        )
        self.plot_check_vars = {}  # holds selection variables of plots orgainized by filepath
        self.ext_options = [
            ".png",
            ".jpeg",
            ".pdf",
        ]  # acceptable extension types for plot images

        # page title
        self.title_frame = tk.Frame(master=self.plot_main_frame)
        self.title_frame.grid(row=0, column=0)
        self.title_label = tk.Label(
            master=self.title_frame,
            text="Welcome to the Plotting Page of SimOpt.",
            font=nametofont("TkHeadingFont"),
        )
        self.title_label.grid(row=0, column=0)
        subtitle = "Select Solvers and Problems to Plot from Experiments that have been Post-Normalized. \n Solver/Problem factors will only be displayed if all solvers/problems within the experiment are the same."
        self.subtitle_label = tk.Label(
            master=self.title_frame,
            text=subtitle,
        )
        self.subtitle_label.grid(row=1, column=0, columnspan=2)

        # load plot button
        self.load_plot_button = tk.Button(
            master=self.title_frame,
            text="Load Plot from Pickle",
            command=self.load_plot,
        )
        self.load_plot_button.grid(row=2, column=0)

        # refresh experiment button
        self.refresh_button = tk.Button(
            master=self.title_frame,
            text="Refresh Experiments",
            command=self.refresh_experiments,
        )
        self.refresh_button.grid(row=2, column=1)

        # experiment selection
        self.plot_selection_frame = tk.Frame(
            master=self.plot_main_frame, width=10
        )
        self.plot_selection_frame.grid_columnconfigure(0, weight=0)
        self.plot_selection_frame.grid_columnconfigure(1, weight=0)
        self.plot_selection_frame.grid(row=1, column=0)
        self.experiment_selection_label = tk.Label(
            master=self.plot_selection_frame,
            text="Select Experiment",
        )
        self.experiment_selection_label.grid(row=0, column=0)
        # find experiments that have been postnormalized
        postnorm_experiments = []  # list to hold names of all experiments that have been postnormalized
        for exp_name in self.root_experiment_dict:
            experiment = self.root_experiment_dict[exp_name]
            status = experiment.check_postnormalize()
            if status:
                postnorm_experiments.append(exp_name)
        self.experiment_var = tk.StringVar()
        self.experiment_menu = ttk.OptionMenu(
            self.plot_selection_frame,
            self.experiment_var,
            "Experiment",
            *postnorm_experiments,
            command=self.update_plot_menu,
        )
        self.experiment_menu.grid(row=0, column=1)

        # solver selection (treeview)
        self.solver_tree_frame = tk.Frame(
            master=self.plot_selection_frame, width=500, height=250
        )  # frame just to hold solver tree
        self.solver_tree_frame.grid(
            row=2, column=0, columnspan=2, padx=10, pady=10
        )
        self.solver_tree_frame.grid_rowconfigure(0, weight=1)
        self.solver_tree_frame.grid_columnconfigure(0, weight=1)
        self.solver_tree_frame.grid_propagate(False)

        self.select_plot_solvers_label = tk.Label(
            master=self.plot_selection_frame,
            text="Select Solver(s)",
        )
        self.select_plot_solvers_label.grid(row=1, column=0)
        self.solver_tree = ttk.Treeview(master=self.solver_tree_frame)
        self.solver_tree.grid(row=0, column=0, sticky="nsew")
        self.style = ttk.Style()
        self.style.configure(
            "Treeview.Heading", font=nametofont("TkHeadingFont")
        )
        self.style.configure(
            "Treeview", foreground="black", font=nametofont("TkDefaultFont")
        )

        self.solver_tree.bind("<<TreeviewSelect>>", self.get_selected_solvers)

        # Create a horizontal scrollbar
        solver_xscrollbar = ttk.Scrollbar(
            master=self.solver_tree_frame,
            orient="horizontal",
            command=self.solver_tree.xview,
        )
        self.solver_tree.configure(xscrollcommand=solver_xscrollbar.set)
        solver_xscrollbar.grid(row=1, column=0, sticky="ew")

        # plot all solvers checkbox
        self.all_solvers_var = tk.BooleanVar()
        self.all_solvers_check = tk.Checkbutton(
            master=self.plot_selection_frame,
            variable=self.all_solvers_var,
            text="Plot all solvers from this experiment",
            command=self.get_selected_solvers,
        )
        self.all_solvers_check.grid(row=4, column=0, columnspan=2)

        # problem selection (treeview)
        self.problem_tree_frame = tk.Frame(
            master=self.plot_selection_frame, width=500, height=250
        )
        self.problem_tree_frame.grid(
            row=6, column=0, columnspan=2, padx=10, pady=10
        )
        self.problem_tree_frame.grid_rowconfigure(0, weight=1)
        self.problem_tree_frame.grid_columnconfigure(0, weight=1)
        self.problem_tree_frame.grid_propagate(False)
        self.select_plot_problems_label = tk.Label(
            master=self.plot_selection_frame,
            text="Select Problem(s)",
        )
        self.select_plot_problems_label.grid(row=5, column=0)
        self.problem_tree = ttk.Treeview(master=self.problem_tree_frame)
        self.problem_tree.grid(row=0, column=0, sticky="nsew")
        self.style = ttk.Style()
        self.style.configure(
            "Treeview.Heading", font=nametofont("TkHeadingFont")
        )
        self.style.configure(
            "Treeview", foreground="black", font=nametofont("TkDefaultFont")
        )
        self.problem_tree.bind("<<TreeviewSelect>>", self.get_selected_problems)

        # Create a horizontal scrollbar
        problem_xscrollbar = ttk.Scrollbar(
            master=self.problem_tree_frame,
            orient="horizontal",
            command=self.problem_tree.xview,
        )
        self.problem_tree.configure(xscrollcommand=problem_xscrollbar.set)
        problem_xscrollbar.grid(row=1, column=0, sticky="ew")

        # plot all problems checkbox
        self.all_problems_var = tk.BooleanVar()
        self.all_problems_check = tk.Checkbutton(
            master=self.plot_selection_frame,
            variable=self.all_problems_var,
            text="Plot all problems from this experiment",
            command=self.get_selected_problems,
        )
        self.all_problems_check.grid(row=8, column=0, columnspan=2)

        self.plot_types_inputs = [
            "cdf_solvability",
            "quantile_solvability",
            "diff_cdf_solvability",
            "diff_quantile_solvability",
        ]
        plot_types = [
            "All Progress Curves",
            "Mean Progress Curve",
            "Quantile Progress Curve",
            "Solve time CDF",
            "Area Scatter Plot",
            "CDF Solvability",
            "Quantile Solvability",
            "CDF Difference Plot",
            "Quantile Difference Plot",
            "Terminal Progress Plot",
            "Terminal Scatter Plot",
        ]

        # plot options
        self.plot_options_frame.grid(row=1, column=1)
        self.plot_type_label = tk.Label(
            master=self.plot_options_frame,
            text="Select Plot Type",
        )
        self.plot_type_label.grid(row=0, column=0)
        self.plot_type_var = tk.StringVar()
        plot_types = [
            "Solvability CDF",
            "Solvability Profile",
            "Area Scatter Plot",
            "Progress Curve",
            "Terminal Progress",
            "Terminal Scatter Plot",
        ]
        self.plot_type_menu = ttk.OptionMenu(
            self.plot_options_frame,
            self.plot_type_var,
            "Plot Type",
            *plot_types,
            command=self.show_plot_options,
        )
        self.plot_type_menu.grid(row=0, column=1)

        # blank plotting workspace
        self.plotting_workspace_frame.grid(row=2, column=0, columnspan=2)
        self.workspace_label = tk.Label(
            master=self.plotting_workspace_frame,
            text="Created Plots by Experiment",
            font=nametofont("TkHeadingFont"),
        )
        self.workspace_label.grid(row=0, column=0)
        # view selected plots button
        self.view_selected_plots_button = tk.Button(
            master=self.plotting_workspace_frame,
            text="View Selected Plots",
            command=self.view_selected_plots,
        )
        self.view_selected_plots_button.grid(row=0, column=1, padx=20)
        # view all plots button
        self.view_all_plots_button = tk.Button(
            master=self.plotting_workspace_frame,
            text="View All Created Plots",
            command=self.view_all_plots,
        )
        self.view_all_plots_button.grid(row=0, column=2, padx=20)
        # empty notebook to hold plots
        self.plot_notebook = ttk.Notebook(self.plotting_workspace_frame)
        self.plot_notebook.grid(row=1, column=0, columnspan=3)

        # loaded plots tab
        self.loaded_plots_frame = tk.Frame(self.plot_notebook)
        self.plot_notebook.add(
            self.loaded_plots_frame, text="Loaded Plots & Copies"
        )

        self.select_header = tk.Label(
            master=self.loaded_plots_frame,
            text="Select Plot(s)",
            font=nametofont("TkHeadingFont"),
        )
        self.select_header.grid(row=0, column=0)
        self.plot_name_header = tk.Label(
            master=self.loaded_plots_frame,
            text="Plot Name",
            font=nametofont("TkHeadingFont"),
        )
        self.plot_name_header.grid(row=0, column=1)
        self.view_header = tk.Label(
            master=self.loaded_plots_frame,
            text="View/Edit",
            font=nametofont("TkHeadingFont"),
        )
        self.view_header.grid(row=0, column=2, pady=10)
        self.file_path_header = tk.Label(
            master=self.loaded_plots_frame,
            text="File Location",
            font=nametofont("TkHeadingFont"),
        )
        self.file_path_header.grid(row=0, column=3)
        self.del_header = tk.Label(
            master=self.loaded_plots_frame,
            text="Delete Plot",
            font=nametofont("TkHeadingFont"),
        )
        self.del_header.grid(row=0, column=4)

    def refresh_experiments(self) -> None:
        self.experiment_menu.destroy()

        # find experiments that have been postnormalized
        postnorm_experiments = []  # list to hold names of all experiments that have been postnormalized
        for exp_name in self.root_experiment_dict:
            experiment = self.root_experiment_dict[exp_name]
            status = experiment.check_postnormalize()
            if status:
                postnorm_experiments.append(exp_name)
        self.experiment_menu = ttk.OptionMenu(
            self.plot_selection_frame,
            self.experiment_var,
            "Experiment",
            *postnorm_experiments,
            command=self.update_plot_menu,
        )
        self.experiment_menu.grid(row=0, column=1)

    def update_plot_window_scroll(self, event: tk.Event) -> None:
        self.plotting_canvas.configure(
            scrollregion=self.plotting_canvas.bbox("all")
        )

    def update_plot_menu(self, experiment_name: str) -> None:
        self.plot_solver_options = [
            "All"
        ]  # holds names of potential solvers to plot
        self.plot_problem_options = [
            "All"
        ]  # holds names of potential problems to plot
        self.plot_experiment = self.root_experiment_dict[experiment_name]
        solver_factor_set = set()  # holds names of solver factors
        problem_factor_set = set()  # holds names of problem factors
        for solver in self.plot_experiment.solvers:
            self.plot_solver_options.append(solver.name)
            for factor in solver.factors.keys():
                solver_factor_set.add(factor)  # append factor names to list

        for problem in self.plot_experiment.problems:
            self.plot_problem_options.append(problem.name)
            for factor in problem.factors.keys():
                problem_factor_set.add(factor)
            for factor in problem.model.factors.keys():
                problem_factor_set.add(factor)

        # determine if all solvers in experiment have the same factor options
        if len(solver_factor_set) == len(
            self.plot_experiment.solvers[0].factors
        ):  # if set length is the same as the fist solver in the experiment
            self.all_same_solver = True
        else:
            self.all_same_solver = False

        # determine if all problems in experiment have the same factor options
        n_prob_factors = len(self.plot_experiment.problems[0].factors) + len(
            self.plot_experiment.problems[0].model.factors
        )
        if (
            len(problem_factor_set) == n_prob_factors
        ):  # if set length is the same as the fist problem in the experiment
            self.all_same_problem = True
        else:
            self.all_same_problem = False

        # clear previous values in the solver tree
        for row in self.solver_tree.get_children():
            self.solver_tree.delete(row)

        # create first column of solver tree view
        self.solver_tree.column("#0", width=75)
        if self.all_same_solver:
            columns = [
                "Solver Name",
                *list(self.plot_experiment.solvers[0].factors.keys()),
            ]
            self.solver_tree["columns"] = (
                columns  # set column names to factor names
            )
            self.solver_tree.heading(
                "Solver Name", text="Solver Name"
            )  # set heading for name column
            for factor in self.plot_experiment.solvers[0].factors.keys():
                self.solver_tree.heading(
                    factor, text=factor
                )  # set column header text to factor names
            for index, solver in enumerate(self.plot_experiment.solvers):
                row = [solver.name]  # list to hold data for this row
                for factor in solver.factors:
                    row.append(solver.factors[factor])
                self.solver_tree.insert("", index, text=index, values=row)
        else:
            self.solver_tree["columns"] = [
                "Solver Name"
            ]  # set columns just to solver name
            self.solver_tree.heading("Solver Name", text="Solver Name")
            for index, solver in enumerate(self.plot_experiment.solvers):
                self.solver_tree.insert(
                    "", index, text=index, values=[solver.name]
                )

        # clear previous values in the problem tree
        for row in self.problem_tree.get_children():
            self.problem_tree.delete(row)

        # create first column of problem tree view
        self.problem_tree.heading("#0", text="Problem #")
        self.problem_tree.column("#0", width=75)
        if self.all_same_problem:
            factors = list(
                self.plot_experiment.problems[0].factors.keys()
            ) + list(self.plot_experiment.problems[0].model.factors.keys())
            columns = ["Problem Name", *factors]
            self.problem_tree["columns"] = (
                columns  # set column names to factor names
            )
            self.problem_tree.heading(
                "Problem Name", text="Problem Name"
            )  # set heading for name column
            for factor in factors:
                self.problem_tree.heading(
                    factor, text=factor
                )  # set column header text to factor names
            for index, problem in enumerate(self.plot_experiment.problems):
                row = [problem.name]  # list to hold data for this row
                for factor in problem.factors:
                    row.append(problem.factors[factor])
                for factor in problem.model.factors:
                    row.append(problem.model.factors[factor])
                self.problem_tree.insert("", index, text=index, values=row)
        else:
            self.problem_tree["columns"] = ["Problem Name"]
            self.problem_tree.heading(
                "Problem Name", text="Problem Name"
            )  # set heading for name column
            for index, problem in enumerate(self.plot_experiment.problems):
                self.problem_tree.insert(
                    "", index, text=index, values=[problem.name]
                )

    def show_plot_options(self, plot_type: str) -> None:
        self.clear_frame(self.more_options_frame)
        self.more_options_frame.grid(row=1, column=0, columnspan=2)

        self.plot_type = plot_type

        # all in one entry (option is present for all plot types)
        self.all_label = tk.Label(
            master=self.more_options_frame,
            text="Plot all solvers together?",
        )
        self.all_label.grid(row=1, column=0)
        self.all_var = tk.StringVar()
        self.all_var.set("Yes")
        self.all_menu = ttk.OptionMenu(
            self.more_options_frame,
            self.all_var,
            "Yes",
            "Yes",
            "No",
            command=self.disable_legend,
        )
        self.all_menu.grid(row=1, column=1)

        self.ref_menu_created = False  # reset to false

        if plot_type == "Progress Curve":
            # plot description
            description = "Plot individual or aggregate progress curves for one or more solvers on a single problem."
            self.plot_description = tk.Label(
                master=self.more_options_frame,
                text=description,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2)

            # select subplot type
            self.subplot_type_label = tk.Label(
                master=self.more_options_frame,
                text="Type",
            )
            self.subplot_type_label.grid(row=2, column=0)
            subplot_type_options = ["all", "mean", "quantile"]
            self.subplot_type_var = tk.StringVar()
            self.subplot_type_var.set("all")
            self.subplot_type_menu = ttk.OptionMenu(
                self.more_options_frame,
                self.subplot_type_var,
                "all",
                *subplot_type_options,
                command=self.enable_ref_solver,
            )
            self.subplot_type_menu.grid(row=2, column=1)

            # beta entry
            self.beta_label = tk.Label(
                master=self.more_options_frame,
                text="Quantile Probability (0.0-1.0)",
            )
            self.beta_label.grid(row=4, column=0)
            self.beta_var = tk.StringVar()
            self.beta_var.set("0.5")  # default value
            self.beta_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.beta_var
            )
            self.beta_entry.grid(row=4, column=1)
            self.beta_entry.configure(state="disabled")

            # normalize entry
            self.normalize_label = tk.Label(
                master=self.more_options_frame,
                text="Normalize Optimality Gaps?",
            )
            self.normalize_label.grid(row=3, column=0)
            self.normalize_var = tk.StringVar()
            self.normalize_var.set("Yes")
            self.normalize_menu = ttk.OptionMenu(
                self.more_options_frame, self.normalize_var, "Yes", "Yes", "No"
            )
            self.normalize_menu.grid(row=3, column=1)

            # num bootstraps entry
            self.boot_label = tk.Label(
                master=self.more_options_frame,
                text="Number Bootstrap Samples",
            )
            self.boot_label.grid(row=5, column=0)
            self.boot_var = tk.StringVar()
            self.boot_var.set("100")  # default value
            self.boot_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.boot_var
            )
            self.boot_entry.grid(row=5, column=1)
            self.boot_entry.configure(state="disabled")

            # confidence level entry
            self.con_level_label = tk.Label(
                master=self.more_options_frame,
                text="Confidence Level (0.0-1.0)",
            )
            self.con_level_label.grid(row=6, column=0)
            self.con_level_var = tk.StringVar()
            self.con_level_var.set("0.95")  # default value
            self.con_level_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.con_level_var
            )
            self.con_level_entry.grid(row=6, column=1)
            self.con_level_entry.configure(state="disabled")

            # plot CIs entry
            self.plot_CI_label = tk.Label(
                master=self.more_options_frame,
                text="Plot Confidence Intervals?",
            )
            self.plot_CI_label.grid(row=7, column=0)
            self.plot_CI_var = tk.StringVar()
            self.plot_CI_var.set("Yes")
            self.plot_CI_menu = ttk.OptionMenu(
                self.more_options_frame, self.plot_CI_var, "Yes", "Yes", "No"
            )
            self.plot_CI_menu.grid(row=7, column=1)
            self.plot_CI_menu.configure(state="disabled")

            # plot max HW entry
            self.plot_hw_label = tk.Label(
                master=self.more_options_frame,
                text="Plot Max Halfwidth?",
            )
            self.plot_hw_label.grid(row=8, column=0)
            self.plot_hw_var = tk.StringVar()
            self.plot_hw_var.set("Yes")
            self.plot_hw_menu = ttk.OptionMenu(
                self.more_options_frame, self.plot_hw_var, "Yes", "Yes", "No"
            )
            self.plot_hw_menu.grid(row=8, column=1)
            self.plot_hw_menu.configure(state="disabled")

            # legend location
            self.legend_label = tk.Label(
                master=self.more_options_frame,
                text="Legend Location",
            )
            self.legend_label.grid(row=9, column=0)
            loc_opt = [
                "best",
                "upper right",
                "upper left",
                "lower left",
                "lower right",
                "right",
                "center left",
                "center right",
                "lower center",
                "upper center",
                "center",
            ]
            self.legend_var = tk.StringVar()
            self.legend_var.set("best")
            self.legend_menu = ttk.OptionMenu(
                self.more_options_frame, self.legend_var, "best", *loc_opt
            )
            self.legend_menu.grid(row=9, column=1)

        if plot_type == "Solvability CDF":
            # plot description
            description = "Plot the solvability cdf for one or more solvers on a single problem."
            self.plot_description = tk.Label(
                master=self.more_options_frame,
                text=description,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2)

            # solve tol entry
            self.solve_tol_label = tk.Label(
                master=self.more_options_frame,
                text="Solve Tolerance",
            )
            self.solve_tol_label.grid(row=2, column=0)
            self.solve_tol_var = tk.StringVar()
            self.solve_tol_var.set(0.1)
            self.solve_tol_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.solve_tol_var
            )
            self.solve_tol_entry.grid(row=2, column=1)

            # num bootstraps entry
            self.boot_label = tk.Label(
                master=self.more_options_frame,
                text="Number Bootstrap Samples",
            )
            self.boot_label.grid(row=3, column=0)
            self.boot_var = tk.StringVar()
            self.boot_var.set("100")  # default value
            self.boot_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.boot_var
            )
            self.boot_entry.grid(row=3, column=1)

            # confidence level entry
            self.con_level_label = tk.Label(
                master=self.more_options_frame,
                text="Confidence Level (0.0-1.0)",
            )
            self.con_level_label.grid(row=4, column=0)
            self.con_level_var = tk.StringVar()
            self.con_level_var.set("0.95")  # default value
            self.con_level_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.con_level_var
            )
            self.con_level_entry.grid(row=4, column=1)

            # plot CIs entry
            self.plot_CI_label = tk.Label(
                master=self.more_options_frame,
                text="Plot Confidence Intervals?",
            )
            self.plot_CI_label.grid(row=5, column=0)
            self.plot_CI_var = tk.StringVar()
            self.plot_CI_var.set("Yes")
            self.plot_CI_menu = ttk.OptionMenu(
                self.more_options_frame, self.plot_CI_var, "Yes", "Yes", "No"
            )
            self.plot_CI_menu.grid(row=5, column=1)

            # plot max HW entry
            self.plot_hw_label = tk.Label(
                master=self.more_options_frame,
                text="Plot Max Halfwidth?",
            )
            self.plot_hw_label.grid(row=6, column=0)
            self.plot_hw_var = tk.StringVar()
            self.plot_hw_var.set("Yes")
            self.plot_hw_menu = ttk.OptionMenu(
                self.more_options_frame, self.plot_hw_var, "Yes", "Yes", "No"
            )
            self.plot_hw_menu.grid(row=6, column=1)

            # legend location
            self.legend_label = tk.Label(
                master=self.more_options_frame,
                text="Legend Location",
            )
            self.legend_label.grid(row=7, column=0)
            loc_opt = [
                "best",
                "upper right",
                "upper left",
                "lower left",
                "lower right",
                "right",
                "center left",
                "center right",
                "lower center",
                "upper center",
                "center",
            ]
            self.legend_var = tk.StringVar()
            self.legend_var.set("best")
            self.legend_menu = ttk.OptionMenu(
                self.more_options_frame, self.legend_var, "best", *loc_opt
            )
            self.legend_menu.grid(row=7, column=1)

        if plot_type == "Area Scatter Plot":
            # plot description
            description = "Plot a scatter plot of mean and standard deviation of area under progress curves."
            self.plot_description = tk.Label(
                master=self.more_options_frame,
                text=description,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2)

            # num bootstraps entry
            self.boot_label = tk.Label(
                master=self.more_options_frame,
                text="Number Bootstrap Samples",
            )
            self.boot_label.grid(row=2, column=0)
            self.boot_var = tk.StringVar()
            self.boot_var.set("100")  # default value
            self.boot_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.boot_var
            )
            self.boot_entry.grid(row=2, column=1)

            # confidence level entry
            self.con_level_label = tk.Label(
                master=self.more_options_frame,
                text="Confidence Level (0.0-1.0)",
            )
            self.con_level_label.grid(row=3, column=0)
            self.con_level_var = tk.StringVar()
            self.con_level_var.set("0.95")  # default value
            self.con_level_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.con_level_var
            )
            self.con_level_entry.grid(row=3, column=1)

            # plot CIs entry
            self.plot_CI_label = tk.Label(
                master=self.more_options_frame,
                text="Plot Confidence Intervals?",
            )
            self.plot_CI_label.grid(row=4, column=0)
            self.plot_CI_var = tk.StringVar()
            self.plot_CI_var.set("Yes")
            self.plot_CI_menu = ttk.OptionMenu(
                self.more_options_frame, self.plot_CI_var, "Yes", "Yes", "No"
            )
            self.plot_CI_menu.grid(row=4, column=1)

            # plot max HW entry
            self.plot_hw_label = tk.Label(
                master=self.more_options_frame,
                text="Plot Max Halfwidth?",
            )
            self.plot_hw_label.grid(row=5, column=0)
            self.plot_hw_var = tk.StringVar()
            self.plot_hw_var.set("Yes")
            self.plot_hw_menu = ttk.OptionMenu(
                self.more_options_frame, self.plot_hw_var, "Yes", "Yes", "No"
            )
            self.plot_hw_menu.grid(row=5, column=1)

            # legend location
            self.legend_label = tk.Label(
                master=self.more_options_frame,
                text="Legend Location",
            )
            self.legend_label.grid(row=6, column=0)
            loc_opt = [
                "best",
                "upper right",
                "upper left",
                "lower left",
                "lower right",
                "right",
                "center left",
                "center right",
                "lower center",
                "upper center",
                "center",
            ]
            self.legend_var = tk.StringVar()
            self.legend_var.set("best")
            self.legend_menu = ttk.OptionMenu(
                self.more_options_frame, self.legend_var, "best", *loc_opt
            )
            self.legend_menu.grid(row=6, column=1)

        if plot_type == "Solvability Profile":
            self.ref_menu_created = True  # track that menu exists
            # plot description
            description = "Plot the (difference of) solvability profiles for each solver on a set of problems."
            self.plot_description = tk.Label(
                master=self.more_options_frame,
                text=description,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2)

            # select subplot type
            self.subplot_type_label = tk.Label(
                master=self.more_options_frame,
                text="Type",
            )
            self.subplot_type_label.grid(row=2, column=0)
            subplot_type_options = [
                "CDF Solvability",
                "Quantile Solvability",
                "Difference of CDF Solvablility",
                "Difference of Quantile Solvability",
            ]
            self.subplot_type_var = tk.StringVar()
            self.subplot_type_var.set("CDF Solvability")
            self.subplot_type_menu = ttk.OptionMenu(
                self.more_options_frame,
                self.subplot_type_var,
                "CDF Solvability",
                *subplot_type_options,
                command=self.enable_ref_solver,
            )
            self.subplot_type_menu.grid(row=2, column=1)

            # num bootstraps entry
            self.boot_label = tk.Label(
                master=self.more_options_frame,
                text="Number Bootstrap Samples",
            )
            self.boot_label.grid(row=3, column=0)
            self.boot_var = tk.StringVar()
            self.boot_var.set("100")  # default value
            self.boot_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.boot_var
            )
            self.boot_entry.grid(row=3, column=1)

            # confidence level entry
            self.con_level_label = tk.Label(
                master=self.more_options_frame,
                text="Confidence Level (0.0-1.0)",
            )
            self.con_level_label.grid(row=4, column=0)
            self.con_level_var = tk.StringVar()
            self.con_level_var.set("0.95")  # default value
            self.con_level_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.con_level_var
            )
            self.con_level_entry.grid(row=4, column=1)

            # plot CIs entry
            self.plot_CI_label = tk.Label(
                master=self.more_options_frame,
                text="Plot Confidence Intervals?",
            )
            self.plot_CI_label.grid(row=5, column=0)
            self.plot_CI_var = tk.StringVar()
            self.plot_CI_var.set("Yes")
            self.plot_CI_menu = ttk.OptionMenu(
                self.more_options_frame, self.plot_CI_var, "Yes", "Yes", "No"
            )
            self.plot_CI_menu.grid(row=5, column=1)

            # plot max HW entry
            self.plot_hw_label = tk.Label(
                master=self.more_options_frame,
                text="Plot Max Halfwidth?",
            )
            self.plot_hw_label.grid(row=6, column=0)
            self.plot_hw_var = tk.StringVar()
            self.plot_hw_var.set("Yes")
            self.plot_hw_menu = ttk.OptionMenu(
                self.more_options_frame, self.plot_hw_var, "Yes", "Yes", "No"
            )
            self.plot_hw_menu.grid(row=6, column=1)

            # solve tol entry
            self.solve_tol_label = tk.Label(
                master=self.more_options_frame,
                text="Solve Tolerance",
            )
            self.solve_tol_label.grid(row=7, column=0)
            self.solve_tol_var = tk.StringVar()
            self.solve_tol_var.set(0.1)
            self.solve_tol_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.solve_tol_var
            )
            self.solve_tol_entry.grid(row=7, column=1)

            # beta entry (quantile size)
            self.beta_label = tk.Label(
                master=self.more_options_frame,
                text="Quantile Probability (0.0-1.0)",
            )
            self.beta_label.grid(row=8, column=0)
            self.beta_var = tk.StringVar()
            self.beta_var.set("0.5")  # default value
            self.beta_entry = tk.Entry(
                master=self.more_options_frame, textvariable=self.beta_var
            )
            self.beta_entry.grid(row=8, column=1)
            self.beta_entry.configure(state="disabled")

            # reference solver
            self.ref_solver_label = tk.Label(
                master=self.more_options_frame,
                text="Solver to use for difference benchmark",
            )
            self.ref_solver_label.grid(row=9, column=0)

            # set none if no solvers selected yet
            self.ref_solver_var = tk.StringVar()
            solver_options = []
            if len(self.selected_solvers) == 0:
                solver_display = "No solvers selected"
            else:
                for solver in self.selected_solvers:
                    solver_options.append(solver.name)
                    solver_display = solver_options[0]
                    self.ref_solver_var.set(solver_display)
            self.ref_solver_menu = ttk.OptionMenu(
                self.more_options_frame,
                self.ref_solver_var,
                solver_display,
                *solver_options,
            )
            self.ref_solver_menu.grid(row=9, column=1)
            self.ref_solver_menu.configure(state="disabled")

            # legend location
            self.legend_label = tk.Label(
                master=self.more_options_frame,
                text="Legend Location",
            )
            self.legend_label.grid(row=10, column=0)
            loc_opt = [
                "best",
                "upper right",
                "upper left",
                "lower left",
                "lower right",
                "right",
                "center left",
                "center right",
                "lower center",
                "upper center",
                "center",
            ]
            self.legend_var = tk.StringVar()
            self.legend_var.set("best")
            self.legend_menu = ttk.OptionMenu(
                self.more_options_frame, self.legend_var, "best", *loc_opt
            )
            self.legend_menu.grid(row=10, column=1)

        if plot_type == "Terminal Progress":
            # plot description
            description = "Plot individual or aggregate terminal progress for one or more solvers on a single problem. Each unique selected problem will produce its own plot. "
            self.plot_description = tk.Label(
                master=self.more_options_frame,
                text=description,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2)

            # select subplot type
            self.subplot_type_label = tk.Label(
                master=self.more_options_frame,
                text="Type",
            )
            self.subplot_type_label.grid(row=2, column=0)
            subplot_type_options = ["box", "violin"]
            self.subplot_type_var = tk.StringVar()
            self.subplot_type_var.set("violin")
            self.subplot_type_menu = ttk.OptionMenu(
                self.more_options_frame,
                self.subplot_type_var,
                "violin",
                *subplot_type_options,
            )
            self.subplot_type_menu.grid(row=2, column=1)

            # normalize entry
            self.normalize_label = tk.Label(
                master=self.more_options_frame,
                text="Normalize Optimality Gaps?",
            )
            self.normalize_label.grid(row=3, column=0)
            self.normalize_var = tk.StringVar()
            self.normalize_var.set("Yes")
            self.normalize_menu = ttk.OptionMenu(
                self.more_options_frame, self.normalize_var, "Yes", "Yes", "No"
            )
            self.normalize_menu.grid(row=3, column=1)

        if plot_type == "Terminal Scatter Plot":
            # plot description
            description = "Plot a scatter plot of mean and standard deviation of terminal progress."
            self.plot_description = tk.Label(
                master=self.more_options_frame,
                text=description,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2)

            # legend location
            self.legend_label = tk.Label(
                master=self.more_options_frame,
                text="Legend Location",
            )
            self.legend_label.grid(row=2, column=0)
            loc_opt = [
                "best",
                "upper right",
                "upper left",
                "lower left",
                "lower right",
                "right",
                "center left",
                "center right",
                "lower center",
                "upper center",
                "center",
            ]
            self.legend_var = tk.StringVar()
            self.legend_var.set("best")
            self.legend_menu = ttk.OptionMenu(
                self.more_options_frame, self.legend_var, "best", *loc_opt
            )
            self.legend_menu.grid(row=2, column=1)

        # get last used row for more options frame
        new_row = self.more_options_frame.grid_size()[1]

        # solver and problem set names
        self.solver_set_label = tk.Label(
            master=self.more_options_frame,
            text="Solver Group Name to be Used in Title",
        )
        self.solver_set_label.grid(row=new_row, column=0)
        self.solver_set_var = tk.StringVar()
        self.solver_set_var.set("SOLVER_SET")
        self.solver_set_entry = tk.Entry(
            master=self.more_options_frame, textvariable=self.solver_set_var
        )
        self.solver_set_entry.grid(row=new_row, column=1)
        self.solver_set_entry.configure(
            state="normal"
        )  # set disabled unless all in is true

        if plot_type in [
            "Terminal Scatter Plot",
            "Solvability Profile",
            "Area Scatter Plot",
        ]:
            self.problem_set_label = tk.Label(
                master=self.more_options_frame,
                text="Problem Group Name to be Used in Title",
            )
            self.problem_set_label.grid(row=new_row + 1, column=0)
            self.problem_set_var = tk.StringVar()
            self.problem_set_var.set("PROBLEM_SET")
            self.problem_set_entry = tk.Entry(
                master=self.more_options_frame,
                textvariable=self.problem_set_var,
            )
            self.problem_set_entry.grid(row=new_row + 1, column=1)
            self.problem_set_entry.configure(
                state="normal"
            )  # set disabled unlass all in is true

        # file extension
        self.ext_label = tk.Label(
            master=self.more_options_frame,
            text="Save image as:",
        )
        self.ext_label.grid(row=new_row + 2, column=0)
        ext_options = [".png", ".jpeg", ".pdf", ".eps"]
        self.ext_var = tk.StringVar()
        self.ext_var.set(".png")
        self.ext_menu = ttk.OptionMenu(
            self.more_options_frame, self.ext_var, ".png", *ext_options
        )
        self.ext_menu.grid(row=new_row + 2, column=1)

        # plot button
        self.plot_button = ttk.Button(
            master=self.plot_options_frame, text="Plot", command=self.plot
        )
        self.plot_button.grid(row=2, column=0)

    def disable_legend(
        self, event: tk.Event
    ) -> None:  # also enables/disables solver & problem group names
        if (
            self.all_var.get() == "Yes"
            and self.plot_type != "Terminal Progress"
        ):
            self.legend_menu.configure(state="normal")
            self.solver_set_entry.configure(state="normal")
            if self.plot_type in [
                "Terminal Scatter Plot",
                "Solvability Profile",
                "Area Scatter Plot",
            ]:
                self.problem_set_entry.configure(state="normal")
        else:
            self.legend_menu.configure(state="disabled")
            self.solver_set_entry.configure(state="disabled")

    def enable_ref_solver(self, plot_type: str) -> None:
        # enable reference solver option
        if plot_type in ["CDF Solvability", "Quantile Solvability"]:
            self.ref_solver_menu.configure(state="disabled")
        elif plot_type in [
            "Difference of CDF Solvablility",
            "Difference of Quantile Solvability",
        ]:
            self.ref_solver_menu.configure(state="normal")

        # enable beta (also works for progress curves )
        if plot_type in [
            "Quantile Solvability",
            "Difference of Quantile Solvability",
            "quantile",
        ]:
            self.beta_entry.configure(state="normal")
        else:
            self.beta_entry.configure(state="disabled")

        # enable confidence level settings for progress curves
        if plot_type in ["mean", "quantile"]:
            self.boot_entry.configure(state="normal")
            self.con_level_entry.configure(state="normal")
            self.plot_CI_menu.configure(state="normal")
            self.plot_hw_menu.configure(state="normal")
        elif plot_type == "all":
            self.boot_entry.configure(state="disabled")
            self.con_level_entry.configure(sate="disabled")
            self.plot_CI_menu.configure(state="disabled")
            self.plot_hw_menu.configure(state="disabled")

    def get_selected_solvers(
        self, event: tk.Event
    ) -> None:  # upddates solver list and options menu for reference solver when relevant
        all_solvers = self.all_solvers_var.get()
        if all_solvers:
            self.selected_solvers = self.plot_experiment.solvers
        else:  # get selected solvers from treeview
            selected_items = self.solver_tree.selection()
            self.selected_solvers = []
            for item in selected_items:
                solver_index = self.solver_tree.item(item, "text")
                solver = self.plot_experiment.solvers[
                    solver_index
                ]  # get corresponding solver from experiment
                self.selected_solvers.append(solver)

        if self.ref_menu_created:  # if reference solver menu exists update menu
            self.update_ref_solver()

    def get_selected_problems(self, event: tk.Event) -> None:
        all_problems = self.all_problems_var.get()
        if all_problems:
            self.selected_problems = self.plot_experiment.problems
        else:  # get selected problems from treeview
            selected_items = self.problem_tree.selection()
            self.selected_problems = []
            for item in selected_items:
                problem_index = self.problem_tree.item(item, "text")
                problem = self.plot_experiment.problems[
                    problem_index
                ]  # get corresponding problem from experiment
                self.selected_problems.append(problem)

    def update_ref_solver(self) -> None:
        saved_solver = (
            self.ref_solver_var.get()
        )  # save previously selected reference solver
        if len(self.selected_solvers) != 0:
            solver_options = []
            for (
                solver
            ) in self.selected_solvers:  # append solver names to options list
                solver_options.append(solver.name)
            if (
                saved_solver not in solver_options
            ):  # set new default if previous option was deselected
                saved_solver = solver_options[0]
        else:
            solver_options = ["No solvers selected"]
            saved_solver = ["No solvers selected"]
        self.ref_solver_var.set(saved_solver)
        # destroy old menu and create new one
        self.ref_solver_menu.destroy()
        self.ref_solver_menu = ttk.OptionMenu(
            self.more_options_frame,
            self.ref_solver_var,
            saved_solver,
            *solver_options,
        )
        self.ref_solver_menu.grid(row=9, column=1)
        if self.subplot_type_var.get() in [
            "CDF Solvability",
            "Quantile Solvability",
        ]:  # disable if not correct plot type
            self.ref_solver_menu.configure(state="disabled")

    def plot(self) -> None:
        if (
            len(self.selected_problems) == 0 or len(self.selected_solvers) == 0
        ):  # show message that no solvers or problems have been selected
            if (
                len(self.selected_problems) == 0
                and len(self.selected_solvers) == 0
            ):
                text = "Please select solvers and problems to plot."
            elif len(self.selected_solvers) == 0:
                text = "Please select solvers to plot."
            elif len(self.selected_problems) == 0:
                text = "Please select problems to plot."

            # show popup message
            tk.messagebox.showerror("Error", text)
        else:  # create plots
            # get selected solvers & problems
            exp_sublist = []  # sublist of experiments to be plotted (each index represents a group of problems over a single solver)
            for solver in self.selected_solvers:
                solver_list = []
                for problem in self.selected_problems:
                    for solver_group in self.plot_experiment.experiments:
                        for dp in solver_group:
                            if id(dp.solver) == id(solver) and id(
                                dp.problem
                            ) == id(problem):
                                solver_list.append(dp)
                exp_sublist.append(solver_list)
            n_problems = len(exp_sublist[0])

            # get user input common across all plot types
            all_str = self.all_var.get()
            if all_str == "Yes":
                all_in = True

            else:
                all_in = False

            if (
                all_in and self.plot_type != "Terminal Progress"
            ):  # only get legend location if all in one is selected
                legend = self.legend_var.get()
            else:
                legend = None
            ext = self.ext_var.get()  # file extension type

            # get solver set name (pass through even if all in is false, will just be ignored by plotting function)
            solver_set_name = self.solver_set_var.get()
            if self.plot_type in [
                "Terminal Scatter Plot",
                "Solvability Profile",
                "Area Scatter Plot",
            ]:
                problem_set_name = self.problem_set_var.get()

            if self.plot_type == "Progress Curve":
                # get user input
                subplot_type = self.subplot_type_var.get()
                beta = float(self.beta_var.get())
                normalize_str = self.normalize_var.get()
                if normalize_str == "Yes":
                    norm = True
                else:
                    norm = False

                n_boot = int(self.boot_var.get())
                con_level = float(self.con_level_var.get())
                plot_ci_str = self.plot_CI_var.get()
                if plot_ci_str == "Yes":
                    plot_ci = True
                else:
                    plot_ci = False
                plot_hw_str = self.plot_hw_var.get()
                if plot_hw_str == "Yes":
                    plot_hw = True
                else:
                    plot_hw = False
                parameters = {}  # holds relevant parameter info for display
                parameters["Plot Type"] = subplot_type
                parameters["Normalize Optimality Gaps"] = normalize_str
                if subplot_type == "quantile":
                    parameters["Quantile Probability"] = beta
                parameters["Number Bootstrap Samples"] = n_boot
                parameters["Confidence Level"] = con_level
                # create new plot for each problem
                for i in range(n_problems):
                    prob_list = []
                    for solver_group in exp_sublist:
                        prob_list.append(solver_group[i])
                    returned_path = plot_progress_curves(
                        experiments=prob_list,
                        plot_type=subplot_type,
                        beta=beta,
                        normalize=norm,
                        all_in_one=all_in,
                        n_bootstraps=n_boot,
                        conf_level=con_level,
                        plot_conf_ints=plot_ci,
                        print_max_hw=plot_hw,
                        legend_loc=legend,
                        save_as_pickle=True,
                        ext=ext,
                        solver_set_name=solver_set_name,
                    )
                    # get plot info and call add plot
                    file_path = [
                        item for item in returned_path if item is not None
                    ]  # remove None items from list
                    n_plots = len(file_path)
                    if all_in:
                        solver_names = [solver_set_name]
                    else:
                        solver_names = []
                        for i in range(n_plots):
                            solver_names.append(prob_list[i].solver.name)
                    problem_names = []
                    for i in range(n_plots):
                        problem_names.append(
                            prob_list[i].problem.name
                        )  # should all be the same

                    self.add_plot(
                        file_paths=file_path,
                        solver_names=solver_names,
                        problem_names=problem_names,
                        parameters=parameters,
                    )

            if self.plot_type == "Solvability CDF":
                solve_tol = float(self.solve_tol_var.get())
                n_boot = int(self.boot_var.get())
                con_level = float(self.con_level_var.get())
                plot_ci_str = self.plot_CI_var.get()
                if plot_ci_str == "Yes":
                    plot_ci = True
                else:
                    plot_ci = False
                plot_hw_str = self.plot_hw_var.get()
                if plot_hw_str == "Yes":
                    plot_hw = True
                else:
                    plot_hw = False

                parameters = {}  # holds relevant parameter info for display
                parameters["Solve Tolerance"] = solve_tol
                parameters["Number Bootstrap Samples"] = n_boot
                parameters["Confidence Level"] = con_level
                # create a new plot for each problem
                for i in range(n_problems):
                    prob_list = []
                    for solver_group in exp_sublist:
                        prob_list.append(solver_group[i])
                    returned_path = plot_solvability_cdfs(
                        experiments=prob_list,
                        solve_tol=solve_tol,
                        all_in_one=all_in,
                        n_bootstraps=n_boot,
                        conf_level=con_level,
                        plot_conf_ints=plot_ci,
                        print_max_hw=plot_hw,
                        legend_loc=legend,
                        save_as_pickle=True,
                        ext=ext,
                        solver_set_name=solver_set_name,
                    )
                    # get plot info and call add plot
                    file_path = [
                        item for item in returned_path if item is not None
                    ]  # remove None items from list
                    n_plots = len(file_path)
                    if all_in:
                        solver_names = [solver_set_name]
                    else:
                        solver_names = []
                        for i in range(n_plots):
                            solver_names.append(prob_list[i].solver.name)
                    problem_names = []
                    for i in range(n_plots):
                        problem_names.append(
                            prob_list[i].problem.name
                        )  # should all be the same

                    self.add_plot(
                        file_paths=file_path,
                        solver_names=solver_names,
                        problem_names=problem_names,
                        parameters=parameters,
                    )

            if self.plot_type == "Area Scatter Plot":
                if (
                    len(self.selected_solvers) > 7 and all_in
                ):  # check if too many solvers selected
                    tk.messagebox.showerror(
                        "Exceeds Solver Limit",
                        "Area scatter plot can plot at most 7 solvers at one time. Please select fewer solvers and plot again.",
                    )
                else:
                    # get user input
                    n_boot = int(self.boot_var.get())
                    con_level = float(self.con_level_var.get())
                    plot_ci_str = self.plot_CI_var.get()
                    if plot_ci_str == "Yes":
                        plot_ci = True
                    else:
                        plot_ci = False
                    plot_hw_str = self.plot_hw_var.get()
                    if plot_hw_str == "Yes":
                        plot_hw = True
                    else:
                        plot_hw = False
                    parameters = {}  # holds relevant parameter info for display
                    parameters["Number Bootstrap Samples"] = n_boot
                    parameters["Confidence Level"] = con_level
                    # create plots
                    returned_path = plot_area_scatterplots(
                        experiments=exp_sublist,
                        all_in_one=all_in,
                        n_bootstraps=n_boot,
                        conf_level=con_level,
                        plot_conf_ints=plot_ci,
                        print_max_hw=plot_hw,
                        save_as_pickle=True,
                        ext=ext,
                        solver_set_name=solver_set_name,
                        problem_set_name=problem_set_name,
                    )
                    # get plot info and call add plot
                    file_path = [
                        item for item in returned_path if item is not None
                    ]  # remove None items from list
                    n_plots = len(file_path)
                    if all_in:
                        solver_names = [solver_set_name]
                        problem_names = [problem_set_name]
                    else:
                        solver_names = []
                        problem_names = []
                        for i in range(n_plots):
                            solver_names.append(
                                exp_sublist[i][0].solver.name
                            )  # get name of first solver since should all be the same
                            problem_names.append(problem_set_name)

                    self.add_plot(
                        file_paths=file_path,
                        solver_names=solver_names,
                        problem_names=problem_names,
                        parameters=parameters,
                    )

            if self.plot_type == "Terminal Progress":
                # get user input
                subplot_type = self.subplot_type_var.get()
                normalize_str = self.normalize_var.get()
                if normalize_str == "Yes":
                    norm = True
                else:
                    norm = False
                parameters = {}  # holds relevant parameter info for display
                parameters["Plot Type"] = subplot_type
                parameters["Normalize Optimality Gaps"] = normalize_str
                # create a new plot for each problem
                for i in range(n_problems):
                    prob_list = []
                    for solver_group in exp_sublist:
                        prob_list.append(solver_group[i])
                    returned_path = plot_terminal_progress(
                        experiments=prob_list,
                        plot_type=subplot_type,
                        all_in_one=all_in,
                        normalize=norm,
                        save_as_pickle=True,
                        ext=ext,
                        solver_set_name=solver_set_name,
                    )
                    # get plot info and call add plot
                    file_path = [
                        item for item in returned_path if item is not None
                    ]  # remove None items from list
                    n_plots = len(file_path)
                    if all_in:
                        solver_names = [solver_set_name]
                    else:
                        solver_names = []
                        for i in range(n_plots):
                            solver_names.append(prob_list[i].solver.name)
                    problem_names = []
                    for i in range(n_plots):
                        problem_names.append(
                            prob_list[i].problem.name
                        )  # should all be the same

                    self.add_plot(
                        file_paths=file_path,
                        solver_names=solver_names,
                        problem_names=problem_names,
                        parameters=parameters,
                    )

            if self.plot_type == "Terminal Scatter Plot":
                returned_path = plot_terminal_scatterplots(
                    experiments=exp_sublist,
                    all_in_one=all_in,
                    legend_loc=legend,
                    save_as_pickle=True,
                    ext=ext,
                    solver_set_name=solver_set_name,
                    problem_set_name=problem_set_name,
                )
                # get plot info and call add plot
                file_path = [
                    item for item in returned_path if item is not None
                ]  # remove None items from list
                n_plots = len(file_path)
                if all_in:
                    solver_names = [solver_set_name]
                    problem_names = [problem_set_name]
                else:
                    solver_names = []
                    problem_names = []
                    for i in range(n_plots):
                        solver_names.append(
                            exp_sublist[i][0].solver.name
                        )  # get name of first solver since should all be the same
                        problem_names.append(problem_set_name)

                self.add_plot(
                    file_paths=file_path,
                    solver_names=solver_names,
                    problem_names=problem_names,
                    parameters=parameters,
                )

            if self.plot_type == "Solvability Profile":
                # get user input
                subplot_type = self.subplot_type_var.get()
                if subplot_type == "CDF Solvability":
                    plot_input = "cdf_solvability"
                elif subplot_type == "Quantile Solvability":
                    plot_input = "quantile_solvability"
                elif subplot_type == "Difference of CDF Solvablility":
                    plot_input = "diff_cdf_solvability"
                elif subplot_type == "Difference of Quantile Solvability":
                    plot_input = "diff_quantile_solvability"

                beta = float(self.beta_var.get())
                n_boot = int(self.boot_var.get())
                con_level = float(self.con_level_var.get())
                plot_ci_str = self.plot_CI_var.get()
                if plot_ci_str == "Yes":
                    plot_ci = True
                else:
                    plot_ci = False
                plot_hw_str = self.plot_hw_var.get()
                if plot_hw_str == "Yes":
                    plot_hw = True
                else:
                    plot_hw = False
                solve_tol = float(self.solve_tol_var.get())
                parameters = {}  # holds relevant parameter info for display
                parameters["Plot Type"] = subplot_type
                parameters["Solve Tolerance"] = solve_tol
                parameters["Number Bootstrap Samples"] = n_boot
                parameters["Confidence Level"] = con_level
                parameters["Solve Tolerance"] = solve_tol
                if subplot_type in [
                    "Quantile Solvability",
                    "Difference of Quantile Solvability",
                ]:
                    parameters["Quantile Probability"] = beta

                if subplot_type in ["CDF Solvability", "Quantile Solvability"]:
                    returned_path = plot_solvability_profiles(
                        experiments=exp_sublist,
                        plot_type=plot_input,
                        all_in_one=all_in,
                        n_bootstraps=n_boot,
                        conf_level=con_level,
                        plot_conf_ints=plot_ci,
                        print_max_hw=plot_hw,
                        legend_loc=legend,
                        beta=beta,
                        save_as_pickle=True,
                        ext=ext,
                        solver_set_name=solver_set_name,
                        problem_set_name=problem_set_name,
                    )

                else:  # performing a difference solvability profile
                    ref_solver = self.ref_solver_var.get()
                    parameters["Reference Solver"] = ref_solver
                    returned_path = plot_solvability_profiles(
                        experiments=exp_sublist,
                        plot_type=plot_input,
                        all_in_one=all_in,
                        n_bootstraps=n_boot,
                        conf_level=con_level,
                        plot_conf_ints=plot_ci,
                        print_max_hw=plot_hw,
                        legend_loc=legend,
                        beta=beta,
                        ref_solver=ref_solver,
                        save_as_pickle=True,
                        ext=ext,
                        solver_set_name=solver_set_name,
                        problem_set_name=problem_set_name,
                    )
                # get plot info and call add plot
                file_path = [
                    item for item in returned_path if item is not None
                ]  # remove None items from list
                n_plots = len(file_path)

                if all_in:
                    solver_names = [solver_set_name]
                    problem_names = [problem_set_name]
                else:
                    solver_names = []
                    problem_names = []
                    for i in range(n_plots):
                        solver_names.append(
                            exp_sublist[i][0].solver.name
                        )  # get name of first solver since should all be the same
                        problem_names.append(problem_set_name)

                self.add_plot(
                    file_paths=file_path,
                    solver_names=solver_names,
                    problem_names=problem_names,
                    parameters=parameters,
                )

    def add_plot(
        self,
        file_paths: list[str],
        solver_names: str,
        problem_names: str,
        parameters: dict | None = None,
    ) -> None:
        # add new tab for exp if applicable
        exp_name = self.experiment_var.get()
        if exp_name not in self.experiment_tabs:
            tab_frame = tk.Frame(self.plot_notebook)
            self.plot_notebook.add(tab_frame, text=exp_name)
            self.experiment_tabs[exp_name] = (
                tab_frame  # save tab frame to dictionary
            )

            # set up tab first time it is created
            select_header = tk.Label(
                master=tab_frame,
                text="Select Plot(s)",
                font=nametofont("TkDefaultFont"),
            )
            select_header.grid(row=0, column=0)
            solver_header = tk.Label(
                master=tab_frame,
                text="Solver(s)",
                font=nametofont("TkDefaultFont"),
            )
            solver_header.grid(row=0, column=1)
            problem_header = tk.Label(
                master=tab_frame,
                text="Problem(s)",
                font=nametofont("TkDefaultFont"),
            )
            problem_header.grid(row=0, column=2)
            type_header = tk.Label(
                master=tab_frame,
                text="Plot Type",
                font=nametofont("TkDefaultFont"),
            )
            type_header.grid(row=0, column=3)
            view_header = tk.Label(
                master=tab_frame,
                text="View/Edit",
                font=nametofont("TkDefaultFont"),
            )
            view_header.grid(row=0, column=4, pady=10)
            file_path_header = tk.Label(
                master=tab_frame,
                text="File Location",
                font=nametofont("TkDefaultFont"),
            )
            file_path_header.grid(row=0, column=5)
            parameters_header = tk.Label(
                master=tab_frame,
                text="Plot Parameters",
                font=nametofont("TkDefaultFont"),
            )
            parameters_header.grid(row=0, column=6)
            del_header = tk.Label(
                master=tab_frame,
                text="Delete Plot",
                font=nametofont("TkDefaultFont"),
            )
            del_header.grid(row=0, column=7)

        else:  # add more plots if tab has already been created
            tab_frame = self.experiment_tabs[
                exp_name
            ]  # access previously created experiment tab
            row = tab_frame.grid_size()[1]

        if parameters is not None:
            display_str = []
            for parameter in parameters:
                text = f"{parameter} = {parameters[parameter]}"
                display_str.append(text)
        para_display = " , ".join(display_str)

        # add plots to display
        for index, file_path in enumerate(file_paths):
            row = tab_frame.grid_size()[1]
            self.plot_check_var = tk.BooleanVar()
            check = tk.Checkbutton(
                master=tab_frame, variable=self.plot_check_var
            )
            check.grid(row=row, column=0, padx=5)
            self.plot_check_vars[file_path] = self.plot_check_var
            solver_label = tk.Label(
                master=tab_frame,
                text=solver_names[index],
            )
            solver_label.grid(row=row, column=1, padx=5)
            problem_label = tk.Label(
                master=tab_frame,
                text=problem_names[index],
            )
            problem_label.grid(row=row, column=2, padx=5)
            type_label = tk.Label(
                master=tab_frame,
                text=self.plot_type,
            )
            type_label.grid(row=row, column=3, padx=5)
            view_button = tk.Button(
                master=tab_frame,
                text="View/Edit",
                command=lambda fp=file_path: self.view_plot(fp),
            )
            view_button.grid(row=row, column=4, padx=5)
            path_label = tk.Label(
                master=tab_frame,
                text=file_path,
            )
            path_label.grid(row=row, column=5, padx=5)
            para_label = tk.Label(
                master=tab_frame,
                text=para_display,
            )
            para_label.grid(row=row, column=6, padx=5)
            del_button = tk.Button(
                master=tab_frame,
                text="Delete",
                command=lambda r=row,
                frame=tab_frame,
                fp=file_path: self.delete_plot(r, frame, fp),
            )
            del_button.grid(row=row, column=7, pady=10)

        # open correct notebook tab
        for index in range(self.plot_notebook.index("end")):
            if self.plot_notebook.tab(index, "text") == exp_name:
                self.plot_notebook.select(index)

    def delete_plot(
        self, row: int, frame: tk.Frame, file_path: os.PathLike | str
    ) -> None:
        for widget in frame.winfo_children():  # remove plot from list display
            info = widget.grid_info()
            if info["row"] == row:
                widget.destroy()
        del self.plot_check_vars[
            file_path
        ]  # remove check variable from dictionary

    def load_plot(self) -> None:
        # ask user for pickle file location
        file_path = filedialog.askopenfilename()

        # load plot pickle
        with open(file_path, "rb") as f:
            fig = pickle.load(f)
        ax = fig.axes[0]
        # get current plot information
        title = ax.get_title()

        # display image with same file name, if no image, create one
        lead_path = os.path.splitext(file_path)[0]
        image = False
        for ext in self.ext_options:
            if not image:  # image has not yet been found (display first file type found if more than one exist)
                image_path = lead_path + ext
                image = os.path.exists(image_path)

        if not image:  # create image to display if path not found
            image_path = lead_path + ".png"
            plt.savefig(image_path, bbox_inches="tight")

        # add plot info to display list
        row = self.loaded_plots_frame.grid_size()[1]
        self.plot_check_var = tk.BooleanVar()
        check = tk.Checkbutton(
            master=self.loaded_plots_frame, variable=self.plot_check_var
        )
        check.grid(row=row, column=0, padx=5)
        self.plot_check_vars[image_path] = self.plot_check_var
        plot_name_label = tk.Label(
            master=self.loaded_plots_frame,
            text=title,
        )
        plot_name_label.grid(row=row, column=1)
        view_button = tk.Button(
            master=self.loaded_plots_frame,
            text="View/Edit",
            command=lambda fp=image_path: self.view_plot(fp),
        )
        view_button.grid(row=row, column=2, padx=5)
        path_label = tk.Label(
            master=self.loaded_plots_frame,
            text=image_path,
        )
        path_label.grid(row=row, column=3, padx=5)
        del_button = tk.Button(
            master=self.loaded_plots_frame,
            text="Delete",
            command=lambda r=row,
            frame=self.loaded_plots_frame,
            fp=image_path: self.delete_plot(r, frame, fp),
        )
        del_button.grid(row=row, column=4, pady=10)

        # display loaded plots tab
        self.plot_notebook.select(0)

    def view_plot(
        self, file_path: os.PathLike | str
    ) -> None:  # this window also allows for the editing of individual plots by accessing the created pickle file
        # create new window
        self.view_single_window = tk.Toplevel(self)
        self.view_single_window.title(
            "Simopt Graphical User Interface - View Plot"
        )
        self.view_single_window.geometry("800x500")

        # self.view_single_frame = tk.Frame(self.view_single_window)
        # self.view_single_frame.grid(row=0,column=0)

        # Configure the grid layout to expand properly
        self.view_single_window.grid_rowconfigure(0, weight=1)
        self.view_single_window.grid_columnconfigure(0, weight=1)

        # create master canvas
        self.view_single_canvas = tk.Canvas(self.view_single_window)
        self.view_single_canvas.grid(row=0, column=0, sticky="nsew")

        # Create vertical scrollbar
        vert_scroll = ttk.Scrollbar(
            self.view_single_window,
            orient=tk.VERTICAL,
            command=self.view_single_canvas.yview,
        )
        vert_scroll.grid(row=0, column=1, sticky="ns")

        # Create horizontal scrollbar
        horiz_scroll = ttk.Scrollbar(
            self.view_single_window,
            orient=tk.HORIZONTAL,
            command=self.view_single_canvas.xview,
        )
        horiz_scroll.grid(row=1, column=0, sticky="ew")

        # Configure canvas to use the scrollbars
        self.view_single_canvas.configure(
            yscrollcommand=vert_scroll.set, xscrollcommand=horiz_scroll.set
        )

        # create master frame inside the canvas
        self.view_single_frame = tk.Frame(self.view_single_canvas)
        self.view_single_canvas.create_window(
            (0, 0), window=self.view_single_frame, anchor="nw"
        )

        # Bind the configure event to update the scroll region
        self.view_single_frame.bind(
            "<Configure>", self.update_view_single_window_scroll
        )

        # open image of plot
        self.image_frame = tk.Frame(self.view_single_frame)
        self.image_frame.grid(row=0, column=0)
        plot_image = Image.open(file_path)
        plot_photo = ImageTk.PhotoImage(plot_image)
        plot_display = tk.Label(master=self.image_frame, image=plot_photo)
        plot_display.image = plot_photo
        plot_display.grid(row=0, column=0, padx=10, pady=10)

        # menu options supported by matplotlib
        self.font_weight_options = [
            "ultralight",
            "light",
            "normal",
            "medium",
            "semibold",
            "bold",
            "heavy",
            "extra bold",
            "black",
        ]
        self.font_style_options = ["normal", "italic", "oblique"]
        self.font_options = [
            "Arial",
            "Verdana",
            "Geneva",
            "Calibri",
            "Trebuchet MS",
            "Times New Roman",
            "Times",
            "Palatino",
            "Georgia",
            "Courier New",
            "Courier",
            "Lucida Console",
            "Monaco",
            "DejaVu Sans",
        ]
        self.color_options = [
            "blue",
            "green",
            "red",
            "cyan",
            "magenta",
            "yellow",
            "black",
            "white",
            "lightgray",
            "darkgray",
        ]

        self.scale_options = [
            "linear",
            "logarithmic",
            "symmetrical logarithmic",
            "logit",
        ]

        self.font_family_options = [
            "serif",
            "sans-serif",
            "monospace",
            "fantasy",
            "cursive",
        ]

        # plot editing options
        self.edit_frame = tk.Frame(self.view_single_frame)
        self.edit_frame.grid(row=0, column=1)
        self.edit_title_button = tk.Button(
            master=self.edit_frame,
            text="Edit Plot Title",
            command=lambda frame=self.image_frame,
            fp=file_path: self.edit_plot_title(fp, frame),
        )
        self.edit_title_button.grid(row=0, column=0, padx=10, pady=10)
        self.edit_axes_button = tk.Button(
            master=self.edit_frame,
            text="Edit Plot Axes",
            command=lambda frame=self.image_frame,
            fp=file_path: self.edit_plot_x_axis(fp, frame),
        )
        self.edit_axes_button.grid(row=1, column=0, padx=10, pady=10)
        self.edit_text_button = tk.Button(
            master=self.edit_frame,
            text="Edit Plot Caption",
            command=lambda frame=self.image_frame,
            fp=file_path: self.edit_plot_text(fp, frame),
        )
        self.edit_text_button.grid(row=2, column=0)
        self.edit_image_button = tk.Button(
            master=self.edit_frame,
            text="Edit Image File",
            command=lambda frame=self.image_frame,
            fp=file_path: self.edit_plot_image(fp, frame),
        )
        self.edit_image_button.grid(row=3, column=0, pady=10)

    def save_plot_changes(
        self,
        fig: plt.Figure,
        pickle_path: os.PathLike | str,
        file_path: os.PathLike | str,
        image_frame: tk.Frame,
        copy: bool = False,
    ) -> None:
        if not copy:
            # overwrite pickle with new plot
            with open(pickle_path, "wb") as f:
                pickle.dump(fig, f)

            # overwrite image with new plot
            plt.savefig(file_path, bbox_inches="tight")
            # display new image in view window
            for photo in image_frame.winfo_children():
                photo.destroy()
            plot_image = Image.open(file_path)
            plot_photo = ImageTk.PhotoImage(plot_image)
            plot_display = tk.Label(master=image_frame, image=plot_photo)
            plot_display.image = plot_photo
            plot_display.grid(row=0, column=0, padx=10, pady=10)
        else:
            path_name, ext = os.path.splitext(file_path)
            # Check to make sure file does not override previous images
            counter = 1
            extended_path_name = file_path
            while os.path.exists(extended_path_name):
                extended_path_name = f"{path_name} ({counter}){ext}"
                new_path_name = f"{path_name} ({counter})"  # use for pickle
                counter += 1
            plt.savefig(extended_path_name, bbox_inches="tight")  # save image
            # save pickle with new name
            pickle_file = new_path_name + ".pkl"
            with open(pickle_file, "wb") as f:
                pickle.dump(fig, f)
            # add new row to loaded plots tab
            title = fig.axes[0].get_title()  # title for display
            row = self.loaded_plots_frame.grid_size()[1]
            self.plot_check_var = tk.BooleanVar()
            check = tk.Checkbutton(
                master=self.loaded_plots_frame, variable=self.plot_check_var
            )
            check.grid(row=row, column=0, padx=5)
            self.plot_check_vars[extended_path_name] = self.plot_check_var
            plot_name_label = tk.Label(
                master=self.loaded_plots_frame,
                text=title,
            )
            plot_name_label.grid(row=row, column=1)
            view_button = tk.Button(
                master=self.loaded_plots_frame,
                text="View/Edit",
                command=lambda fp=extended_path_name: self.view_plot(fp),
            )
            view_button.grid(row=row, column=2, padx=5)
            path_label = tk.Label(
                master=self.loaded_plots_frame,
                text=extended_path_name,
            )
            path_label.grid(row=row, column=3, padx=5)
            del_button = tk.Button(
                master=self.loaded_plots_frame,
                text="Delete",
                command=lambda r=row,
                frame=self.loaded_plots_frame,
                fp=extended_path_name: self.delete_plot(r, frame, fp),
            )
            del_button.grid(row=row, column=4, pady=10)

            # display loaded plots tab
            self.plot_notebook.select(0)

    def edit_plot_title(
        self, file_path: os.PathLike | str, image_frame: tk.Frame
    ) -> None:
        # create new window
        self.edit_title_window = tk.Toplevel(self)
        self.edit_title_window.title(
            "Simopt Graphical User Interface - Edit Plot Title"
        )
        self.edit_title_window.geometry("800x500")

        self.edit_title_frame = tk.Frame(self.edit_title_window)
        self.edit_title_frame.grid(row=0, column=0)
        # load plot pickle
        root, ext = os.path.splitext(file_path)
        pickle_path = f"{root}.pkl"
        with open(pickle_path, "rb") as f:
            fig = pickle.load(f)
        ax = fig.axes[0]

        # get current plot information
        title = ax.get_title()
        font_properties = ax.title.get_fontproperties()
        font_name = font_properties.get_name()
        font_size = font_properties.get_size_in_points()
        font_style = font_properties.get_style()
        font_weight = font_properties.get_weight()
        color = ax.title.get_color()
        title_position = ax.title.get_position()  # will be a tuple
        alignment = ax.title.get_ha()

        # display current information in entry widgets
        self.title_label = tk.Label(
            master=self.edit_title_window,
            text="Plot Title",
        )
        self.title_label.grid(row=0, column=0)
        self.title_var = tk.StringVar()
        self.title_var.set(str(title))
        self.title_entry = tk.Entry(
            master=self.edit_title_window, textvariable=self.title_var, width=50
        )
        self.title_entry.grid(row=0, column=1, padx=10)
        description = r"Use \n to represent a new line in the title"
        self.title_description_label = tk.Label(
            master=self.edit_title_window,
            text=description,
        )
        self.title_description_label.grid(row=0, column=2, padx=10)

        self.font_label = tk.Label(
            master=self.edit_title_window,
            text="Title Font",
        )
        self.font_label.grid(row=1, column=0)
        self.font_var = tk.StringVar()
        self.font_var.set(font_name)
        self.font_menu = ttk.OptionMenu(
            self.edit_title_window, self.font_var, font_name, *self.font_options
        )
        self.font_menu.grid(row=1, column=1, padx=10)

        self.font_size_label = tk.Label(
            master=self.edit_title_window,
            text="Font Size",
        )
        self.font_size_label.grid(row=2, column=0)
        self.font_size_var = tk.StringVar()
        self.font_size_var.set(font_size)
        self.font_size_entry = tk.Entry(
            master=self.edit_title_window, textvariable=self.font_size_var
        )
        self.font_size_entry.grid(row=2, column=1, padx=10)

        self.font_style_label = tk.Label(
            master=self.edit_title_window,
            text="Font Style",
        )
        self.font_style_label.grid(row=3, column=0)
        self.font_style_var = tk.StringVar()
        self.font_style_var.set(font_style)
        self.font_style_menu = ttk.OptionMenu(
            self.edit_title_window,
            self.font_style_var,
            font_style,
            *self.font_style_options,
        )
        self.font_style_menu.grid(row=3, column=1, padx=10)

        self.font_weight_label = tk.Label(
            master=self.edit_title_window,
            text="Font Weight",
        )
        self.font_weight_label.grid(row=4, column=0)
        self.font_weight_var = tk.StringVar()
        self.font_weight_var.set(font_weight)
        self.font_weight_menu = ttk.OptionMenu(
            self.edit_title_window,
            self.font_weight_var,
            font_weight,
            *self.font_weight_options,
        )
        self.font_weight_menu.grid(row=4, column=1, padx=10)

        self.font_color_label = tk.Label(
            master=self.edit_title_window,
            text="Font Color",
        )
        self.font_color_label.grid(row=5, column=0)
        self.font_color_var = tk.StringVar()
        self.font_color_var.set(color)
        self.font_color_menu = ttk.OptionMenu(
            self.edit_title_window,
            self.font_color_var,
            color,
            *self.color_options,
        )
        self.font_color_menu.grid(row=5, column=1, padx=10)

        self.position_x_label = tk.Label(
            master=self.edit_title_window,
            text="X Position \n (determines centerpoint of title)",
        )
        self.position_x_label.grid(row=7, column=0)
        self.position_x_var = tk.StringVar()
        self.position_x_var.set(title_position[0])
        self.position_x_entry = tk.Entry(
            master=self.edit_title_window, textvariable=self.position_x_var
        )
        self.position_x_entry.grid(row=7, column=1, padx=10)

        self.align_label = tk.Label(
            master=self.edit_title_window,
            text="Alignment",
        )
        self.align_label.grid(row=6, column=0)
        self.align_var = tk.StringVar()
        self.align_var.set(alignment)
        self.align_menu = ttk.OptionMenu(
            self.edit_title_window,
            self.align_var,
            alignment,
            *["left", "center", "right"],
        )
        self.align_menu.grid(row=6, column=1, padx=10)

        self.save_title_button = tk.Button(
            master=self.edit_title_window,
            text="Save Changes",
            command=lambda: self.save_title_changes(
                fig, pickle_path, file_path, image_frame
            ),
        )
        self.save_title_button.grid(row=8, column=0, pady=10)

        self.save_title_to_copy_button = tk.Button(
            master=self.edit_title_window,
            text="Save Changes to Copy",
            command=lambda: self.save_title_changes(
                fig, pickle_path, file_path, image_frame, copy=True
            ),
        )
        self.save_title_to_copy_button.grid(row=8, column=1, padx=20)

    def save_title_changes(
        self,
        fig: plt.figure,
        pickle_path: os.PathLike | str,
        file_path: os.PathLike | str,
        image_frame: tk.Frame,
        copy: bool = False,
    ) -> None:
        # get user input variables
        title_text = self.title_var.get().replace("\\n", "\n")
        font = self.font_var.get()
        size = self.font_size_var.get()
        style = self.font_style_var.get()
        weight = self.font_weight_var.get()
        color = self.font_color_var.get()
        pos_x = self.position_x_var.get()
        alignment = self.align_var.get()

        # change plot to user specifications
        ax = fig.axes[0]
        # get current y pos
        pos_y = ax.title.get_position()[1]

        font_specs = {
            "fontsize": size,
            "fontweight": weight,
            "fontname": font,
            "color": color,
            "fontstyle": style,
        }
        title_pos = (float(pos_x), float(pos_y))

        for align in [
            "left",
            "center",
            "right",
        ]:  # remove old title from all alignments
            ax.set_title("", loc=align)

        ax.set_title(
            f"{title_text}", **font_specs, position=title_pos, loc=alignment
        )

        self.save_plot_changes(
            fig, pickle_path, file_path, image_frame, copy
        )  # save changes and display new image
        self.edit_title_window.destroy()  # close editing window

    def edit_plot_x_axis(
        self, file_path: os.PathLike | str, image_frame: tk.Frame
    ) -> None:  # actualy edits both axes
        # create new window
        self.edit_x_axis_window = tk.Toplevel(self)
        self.edit_x_axis_window.title(
            "Simopt Graphical User Interface - Edit Plot Axes"
        )
        self.edit_x_axis_window.geometry("800x500")

        # select axis
        self.select_axis_label = tk.Label(
            master=self.edit_x_axis_window,
            text="Select Axis",
        )
        self.select_axis_label.grid(row=0, column=0)
        self.axis_var = tk.StringVar()
        self.select_axis_menu = ttk.OptionMenu(
            self.edit_x_axis_window,
            self.axis_var,
            "Select Axis",
            *["X-Axis", "Y-Axis"],
            command=lambda axis: self.show_axis_options(
                axis, file_path, image_frame
            ),
        )
        self.select_axis_menu.grid(row=0, column=1)
        self.edit_x_axis_frame = tk.Frame(
            self.edit_x_axis_window
        )  # create editing frame

    def show_axis_options(
        self,
        axis: Literal["X-Axis", "Y-Axis"],
        file_path: os.PathLike | str,
        image_frame: tk.Frame,
    ) -> None:
        self.clear_frame(self.edit_x_axis_frame)
        self.edit_x_axis_frame.grid(row=1, column=0)

        # load plot pickle
        root, ext = os.path.splitext(file_path)
        pickle_path = f"{root}.pkl"
        with open(pickle_path, "rb") as f:
            fig = pickle.load(f)
        ax = fig.axes[0]

        if axis == "X-Axis":
            # get current plot information
            label = ax.get_xlabel()
            limits = ax.get_xlim()
            # scale = ax.get_xscale()
            font_properties = ax.xaxis.label.get_fontproperties()
            font_name = font_properties.get_name()
            font_size = font_properties.get_size_in_points()
            font_weight = font_properties.get_weight()
            font_style = font_properties.get_style()
            font_color = ax.xaxis.label.get_color()
            # label_pos = ax.xaxis.label.get_position()
            tick_pos = ax.get_xticks()
            alignment = ax.xaxis.label.get_ha()
            axis_display = "X"
            align_options = ["left", "center", "right"]
        else:
            # get current plot information
            label = ax.get_ylabel()
            limits = ax.get_ylim()
            scale = ax.get_yscale()
            font_properties = ax.yaxis.label.get_fontproperties()
            font_name = font_properties.get_name()
            font_size = font_properties.get_size_in_points()
            font_weight = font_properties.get_weight()
            font_style = font_properties.get_style()
            font_color = ax.yaxis.label.get_color()
            # label_pos = ax.yaxis.label.get_position()
            tick_pos = ax.get_yticks()
            alignment = ax.yaxis.label.get_ha()
            axis_display = "Y"
            align_options = ["top", "center", "bottom"]
        # get spacing between ticks
        if len(tick_pos) > 1:
            space = tick_pos[1] - tick_pos[0]
        else:
            space = "none"

        # display current information in entry widgets
        self.x_title_label = tk.Label(
            master=self.edit_x_axis_frame,
            text=f"{axis_display}-Axis Title",
        )
        self.x_title_label.grid(row=0, column=0)
        self.x_title_var = tk.StringVar()
        self.x_title_var.set(label)
        self.x_title_entry = tk.Entry(
            master=self.edit_x_axis_frame,
            textvariable=self.x_title_var,
            width=50,
        )
        self.x_title_entry.grid(row=0, column=1)
        description = r"Use \n to represent a new line in the title"
        self.x_title_description_label = tk.Label(
            master=self.edit_x_axis_frame,
            text=description,
        )
        self.x_title_description_label.grid(row=0, column=2, padx=10)

        self.x_font_label = tk.Label(
            master=self.edit_x_axis_frame,
            text="Title Font",
        )
        self.x_font_label.grid(row=1, column=0)
        self.x_font_var = tk.StringVar()
        self.x_font_var.set(font_name)
        self.x_font_menu = ttk.OptionMenu(
            self.edit_x_axis_frame,
            self.x_font_var,
            font_name,
            *self.font_options,
        )
        self.x_font_menu.grid(row=1, column=1, padx=10)

        self.x_font_size_label = tk.Label(
            master=self.edit_x_axis_frame,
            text="Font Size",
        )
        self.x_font_size_label.grid(row=2, column=0)
        self.x_font_size_var = tk.StringVar()
        self.x_font_size_var.set(font_size)
        self.x_font_size_entry = tk.Entry(
            master=self.edit_x_axis_frame, textvariable=self.x_font_size_var
        )
        self.x_font_size_entry.grid(row=2, column=1, padx=10)

        self.x_font_style_label = tk.Label(
            master=self.edit_x_axis_frame,
            text="Font Style",
        )
        self.x_font_style_label.grid(row=3, column=0)
        self.x_font_style_var = tk.StringVar()
        self.x_font_style_var.set(font_style)
        self.x_font_style_menu = ttk.OptionMenu(
            self.edit_x_axis_frame,
            self.x_font_style_var,
            font_style,
            *self.font_style_options,
        )
        self.x_font_style_menu.grid(row=3, column=1, padx=10)

        self.x_font_weight_label = tk.Label(
            master=self.edit_x_axis_frame,
            text="Font Weight",
        )
        self.x_font_weight_label.grid(row=4, column=0)
        self.x_font_weight_var = tk.StringVar()
        self.x_font_weight_var.set(font_weight)
        self.x_font_weight_menu = ttk.OptionMenu(
            self.edit_x_axis_frame,
            self.x_font_weight_var,
            font_weight,
            *self.font_weight_options,
        )
        self.x_font_weight_menu.grid(row=4, column=1, padx=10)

        self.x_font_color_label = tk.Label(
            master=self.edit_x_axis_frame,
            text="Font Color",
        )
        self.x_font_color_label.grid(row=5, column=0)
        self.x_font_color_var = tk.StringVar()
        self.x_font_color_var.set(font_color)
        self.x_font_color_menu = ttk.OptionMenu(
            self.edit_x_axis_frame,
            self.x_font_color_var,
            font_color,
            *self.color_options,
        )
        self.x_font_color_menu.grid(row=5, column=1, padx=10)

        self.align_label = tk.Label(
            master=self.edit_x_axis_frame,
            text="Title Alignment",
        )
        self.align_label.grid(row=6, column=0)
        self.align_var = tk.StringVar()
        self.align_var.set(alignment)
        self.align_menu = ttk.OptionMenu(
            self.edit_x_axis_frame, self.align_var, alignment, *align_options
        )
        self.align_menu.grid(row=6, column=1, padx=10)

        if axis == "Y-Axis":
            self.x_scale_label = tk.Label(
                master=self.edit_x_axis_frame,
                text=f"{axis_display}-Axis Scale",
            )
            self.x_scale_label.grid(row=7, column=0)
            self.x_scale_var = tk.StringVar()
            self.x_scale_var.set(scale)
            self.x_scale_menu = ttk.OptionMenu(
                self.edit_x_axis_frame,
                self.x_scale_var,
                scale,
                *self.scale_options,
            )
            self.x_scale_menu.grid(row=7, column=1, padx=10)
        else:
            self.x_scale_var = None

        self.min_x_label = tk.Label(
            master=self.edit_x_axis_frame,
            text=f"Min {axis_display} Value",
        )
        self.min_x_label.grid(row=8, column=0)
        self.min_x_var = tk.StringVar()
        self.min_x_var.set(limits[0])
        self.min_x_entry = tk.Entry(
            master=self.edit_x_axis_frame, textvariable=self.min_x_var
        )
        self.min_x_entry.grid(row=8, column=1, padx=10)

        self.max_x_label = tk.Label(
            master=self.edit_x_axis_frame,
            text=f"Max {axis_display} Value",
        )
        self.max_x_label.grid(row=9, column=0)
        self.max_x_var = tk.StringVar()
        self.max_x_var.set(limits[1])
        self.max_x_entry = tk.Entry(
            master=self.edit_x_axis_frame, textvariable=self.max_x_var
        )
        self.max_x_entry.grid(row=9, column=1, padx=10)

        self.x_space_label = tk.Label(
            master=self.edit_x_axis_frame,
            text=f"Space Between {axis_display} Ticks",
        )
        self.x_space_label.grid(row=10, column=0)
        self.x_space_var = tk.StringVar()
        self.x_space_var.set(space)
        self.x_space_entry = tk.Entry(
            master=self.edit_x_axis_frame, textvariable=self.x_space_var
        )
        self.x_space_entry.grid(row=10, column=1, padx=10)

        self.save_axes_button = tk.Button(
            master=self.edit_x_axis_window,
            text="Save Changes",
            command=lambda: self.save_x_axis_changes(
                fig, pickle_path, file_path, image_frame, axis
            ),
        )
        self.save_axes_button.grid(row=2, column=0)

        self.save_axes_to_copy_button = tk.Button(
            master=self.edit_x_axis_window,
            text="Save Changes to Copy",
            command=lambda: self.save_x_axis_changes(
                fig, pickle_path, file_path, image_frame, axis, copy=True
            ),
        )
        self.save_axes_to_copy_button.grid(row=2, column=1, padx=20)

    def save_x_axis_changes(
        self,
        fig: plt.figure,
        pickle_path: os.PathLike | str,
        file_path: os.PathLike | str,
        image_frame: tk.Frame,
        axis: Literal["X-Axis", "Y-Axis"],
        copy: bool = False,
    ) -> None:  # actually saves changes for both axes
        # get user input variables
        title = self.x_title_var.get().replace("\\n", "\n")
        font = self.x_font_var.get()
        size = self.x_font_size_var.get()
        style = self.x_font_style_var.get()
        weight = self.x_font_weight_var.get()
        color = self.x_font_color_var.get()
        scale = self.x_scale_var  # only one left as variable
        min_x = self.min_x_var.get()
        max_x = self.max_x_var.get()
        space = self.x_space_var.get()
        alignment = self.align_var.get()

        # change plot to user specifications
        ax = fig.axes[0]
        font_specs = {
            "fontsize": size,
            "fontweight": weight,
            "fontname": font,
            "color": color,
            "fontstyle": style,
        }
        # title_pos = (float(pos_x), float(pos_y))

        if axis == "X-Axis":
            for align in ["left", "center", "right"]:
                ax.set_xlabel("", loc=align)  # remove old title
            # set new title
            ax.set_xlabel(f"{title}", **font_specs, loc=alignment)
            ax.set_xlim(float(min_x), float(max_x))
            # update scale to match input format
            if scale is not None:
                scale = scale.get()
                if scale == "symmetrical logarithmic":
                    scale = "symlog"
                elif scale == "logarithmic":
                    scale = "log"
                ax.set_xscale(scale)

            # set axis spacing
            if space != "none":
                ax.xaxis.set_major_locator(MultipleLocator(float(space)))
        else:
            for align in ["top", "center", "bottom"]:
                ax.set_ylabel("", loc=align)  # remove old title
            # set new title
            ax.set_ylabel(f"{title}", **font_specs, loc=alignment)
            ax.set_ylim(float(min_x), float(max_x))
            # update scale to match input format
            if scale is not None:
                scale = scale.get()
                if scale == "symmetrical logarithmic":
                    scale = "symlog"
                elif scale == "logarithmic":
                    scale = "log"
                ax.set_yscale(scale)

            # set axis spacing
            if space != "none":
                ax.yaxis.set_major_locator(MultipleLocator(float(space)))

        self.save_plot_changes(
            fig, pickle_path, file_path, image_frame, copy
        )  # save changes and display new image
        self.edit_x_axis_window.destroy()

    def edit_plot_text(
        self, file_path: os.PathLike | str, image_frame: tk.Frame
    ) -> None:
        # create new window
        self.edit_text_window = tk.Toplevel(self)
        self.edit_text_window.title(
            "Simopt Graphical User Interface - Edit Plot Caption"
        )
        self.edit_text_window.geometry("800x500")

        self.edit_text_frame = tk.Frame(self.edit_text_window)
        self.edit_text_frame.grid(row=0, column=0)
        # load plot pickle
        root, ext = os.path.splitext(file_path)
        pickle_path = f"{root}.pkl"
        with open(pickle_path, "rb") as f:
            fig = pickle.load(f)
        ax = fig.axes[0]
        # test to make sure not editing title or axes

        # get current text info
        text_objects = [i for i in ax.get_children() if isinstance(i, plt.Text)]
        filtered_text = [
            text
            for text in text_objects
            if text not in (ax.title, ax.xaxis.label, ax.yaxis.label)
        ]  # remove plot and axis title from list
        non_blank = [
            text for text in filtered_text if text.get_text().strip()
        ]  # filter out blank text objects
        if len(non_blank) != 0:  # there is already a text caption present
            # text properties
            text = non_blank[0]
            description = text.get_text()
            font_props = text.get_fontproperties()
            font_family = font_props.get_family()[0]
            font_size = text.get_fontsize()
            font_style = font_props.get_style()
            font_weight = font_props.get_weight()
            color = text.get_color()
            position = text.get_position()
            h_alignment = text.get_ha()
            v_alignment = text.get_va()
            # bbox properties
            bbox = text.get_bbox_patch()
            if bbox is not None:
                face_color = bbox.get_facecolor()
                edge_color = bbox.get_edgecolor()
                line_width = bbox.get_linewidth()
                alpha = bbox.get_alpha()  # transparency
            else:
                face_color = "none"
                edge_color = "none"
                line_width = ""
                alpha = ""

        else:  # the plot currently does not have a caption
            text = ax.text(-0.5, -0.45, "")  # create new blank text object
            position = text.get_position()
            description = ""
            font_family = "sans-serif"
            font_size = "11"
            font_weight = "normal"
            font_style = "normal"
            color = "black"
            face_color = "none"
            edge_color = "none"
            line_width = ""
            h_alignment = "center"
            v_alignment = "center"
            alpha = ""

        self.text_label = tk.Label(
            master=self.edit_text_frame,
            text="Plot Caption",
        )
        self.text_label.grid(row=0, column=0)
        self.text_entry = tk.Text(self.edit_text_frame, height=5, width=75)
        self.text_entry.grid(row=0, column=1, padx=10)
        self.text_entry.insert(tk.END, description)

        self.text_font_label = tk.Label(
            master=self.edit_text_frame,
            text="Font Family",
        )
        self.text_font_label.grid(row=1, column=0)
        self.text_font_var = tk.StringVar()
        self.text_font_var.set(font_family)
        self.text_font_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_font_var,
            font_family,
            *self.font_family_options,
        )
        self.text_font_menu.grid(row=1, column=1, padx=10)

        self.text_font_size_label = tk.Label(
            master=self.edit_text_frame,
            text="Font Size",
        )
        self.text_font_size_label.grid(row=2, column=0)
        self.text_font_size_var = tk.StringVar()
        self.text_font_size_var.set(font_size)
        self.text_font_size_entry = tk.Entry(
            master=self.edit_text_frame, textvariable=self.text_font_size_var
        )
        self.text_font_size_entry.grid(row=2, column=1, padx=10)

        self.text_font_style_label = tk.Label(
            master=self.edit_text_frame,
            text="Font Style",
        )
        self.text_font_style_label.grid(row=3, column=0)
        self.text_font_style_var = tk.StringVar()
        self.text_font_style_var.set(font_style)
        self.text_font_style_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_font_style_var,
            font_style,
            *self.font_style_options,
        )
        self.text_font_style_menu.grid(row=3, column=1, padx=10)

        self.text_font_weight_label = tk.Label(
            master=self.edit_text_frame,
            text="Font Weight",
        )
        self.text_font_weight_label.grid(row=4, column=0)
        self.text_font_weight_var = tk.StringVar()
        self.text_font_weight_var.set(font_weight)
        self.text_font_weight_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_font_weight_var,
            font_weight,
            *self.font_weight_options,
        )
        self.text_font_weight_menu.grid(row=4, column=1, padx=10)

        self.text_font_color_label = tk.Label(
            master=self.edit_text_frame,
            text="Font Color",
        )
        self.text_font_color_label.grid(row=5, column=0)
        self.text_font_color_var = tk.StringVar()
        self.text_font_color_var.set(color)
        self.text_font_color_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_font_color_var,
            color,
            *self.color_options,
        )
        self.text_font_color_menu.grid(row=5, column=1, padx=10)

        self.text_align_label = tk.Label(
            master=self.edit_text_frame,
            text="Horizontal Alignment",
        )
        self.text_align_label.grid(row=6, column=0)
        self.text_align_var = tk.StringVar()
        self.text_align_var.set(color)
        self.text_align_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_align_var,
            h_alignment,
            *["left", "right", "center"],
        )
        self.text_align_menu.grid(row=6, column=1, padx=10)

        self.text_valign_label = tk.Label(
            master=self.edit_text_frame,
            text="Vertical Alignment",
        )
        self.text_valign_label.grid(row=7, column=0)
        self.text_valign_var = tk.StringVar()
        self.text_valign_var.set(color)
        self.text_valign_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_valign_var,
            v_alignment,
            *["top", "bottom", "center", "baseline"],
        )
        self.text_valign_menu.grid(row=7, column=1, padx=10)

        self.text_position_x_label = tk.Label(
            master=self.edit_text_frame,
            text="Description X Position \n (can be + or -)",
        )
        self.text_position_x_label.grid(row=8, column=0)
        self.text_position_x_var = tk.StringVar()
        self.text_position_x_var.set(position[0])
        self.text_position_x_entry = tk.Entry(
            master=self.edit_text_frame, textvariable=self.text_position_x_var
        )
        self.text_position_x_entry.grid(row=8, column=1, padx=10)

        self.text_position_y_label = tk.Label(
            master=self.edit_text_frame,
            text="Description Y Position \n (can be + or -)",
        )
        self.text_position_y_label.grid(row=9, column=0)
        self.text_position_y_var = tk.StringVar()
        self.text_position_y_var.set(position[1])
        self.text_position_y_entry = tk.Entry(
            master=self.edit_text_frame, textvariable=self.text_position_y_var
        )
        self.text_position_y_entry.grid(row=9, column=1, padx=10)

        self.background_color_label = tk.Label(
            master=self.edit_text_frame,
            text="Background Color",
        )
        self.background_color_label.grid(row=10, column=0)
        self.background_color_var = tk.StringVar()
        self.background_color_var.set(face_color)
        self.background_color_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.background_color_var,
            face_color,
            *([*self.color_options, "none"]),
        )
        self.background_color_menu.grid(row=10, column=1, padx=10)

        self.border_color_label = tk.Label(
            master=self.edit_text_frame,
            text="Border Color",
        )
        self.border_color_label.grid(row=11, column=0)
        self.border_color_var = tk.StringVar()
        self.border_color_var.set(face_color)
        self.border_color_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.border_color_var,
            edge_color,
            *([*self.color_options, "none"]),
        )
        self.border_color_menu.grid(row=11, column=1, padx=10)

        self.border_weight_label = tk.Label(
            master=self.edit_text_frame,
            text="Border Weight",
        )
        self.border_weight_label.grid(row=12, column=0)
        self.border_weight_var = tk.StringVar()
        self.border_weight_var.set(line_width)
        self.border_weight_menu = tk.Entry(
            master=self.edit_text_frame, textvariable=self.border_weight_var
        )
        self.border_weight_menu.grid(row=12, column=1, padx=10)

        self.alpha_label = tk.Label(
            master=self.edit_text_frame,
            text="Transparency",
        )
        self.alpha_label.grid(row=13, column=0)
        self.alpha_var = tk.StringVar()
        self.alpha_var.set(alpha)
        self.alpha_menu = tk.Entry(
            master=self.edit_text_frame, textvariable=self.alpha_var
        )
        self.alpha_menu.grid(row=13, column=1, padx=10)

        self.save_text_button = tk.Button(
            master=self.edit_text_window,
            text="Save Changes",
            command=lambda: self.save_text_changes(
                fig, pickle_path, file_path, image_frame, text
            ),
        )
        self.save_text_button.grid(row=1, column=0, pady=10)
        self.save_text_to_copy_button = tk.Button(
            master=self.edit_text_window,
            text="Save Changes to Copy",
            command=lambda: self.save_text_changes(
                fig, pickle_path, file_path, image_frame, text, copy=True
            ),
        )
        self.save_text_to_copy_button.grid(row=1, column=1, pady=10, padx=20)

    def save_text_changes(
        self,
        fig: plt.figure,
        pickle_path: os.PathLike | str,
        file_path: os.PathLike | str,
        image_frame: tk.Frame,
        text: plt.Text,
        copy: bool = False,
    ) -> None:
        # get text properties from user inputs
        text_entry = self.text_entry
        font = self.text_font_var.get()
        font_size = self.text_font_size_var.get()
        font_weight = self.text_font_weight_var.get()
        font_color = self.text_font_color_var.get()
        pos_x = self.text_position_x_var.get()
        pos_y = self.text_position_y_var.get()
        ha = self.text_align_var.get()
        va = self.text_valign_var.get()
        background = self.background_color_var.get()
        border_color = self.border_color_var.get()
        border_weight = self.border_weight_var.get()
        alpha = self.alpha_var.get()

        description = text_entry.get("1.0", tk.END).replace("\\n", "\n")
        position = (float(pos_x), float(pos_y))

        # modify text obj to new properties
        text.set_text(description)
        text.set_fontsize(font_size)
        text.set_color(font_color)
        text.set_family(font)
        text.set_weight(font_weight)
        text.set_position(position)
        text.set_ha(ha)
        text.set_va(va)

        bbox = {}  # holds bbox properties
        if background != "none":
            bbox["facecolor"] = background
        if border_color != "none":
            bbox["edgecolor"] = border_color
        if border_weight != "":
            bbox["linewidth"] = float(border_weight)
        if alpha != "":
            bbox["alpha"] = float(alpha)
        if len(bbox) != 0:
            text.set_bbox(bbox)

        self.save_plot_changes(
            fig, pickle_path, file_path, image_frame, copy
        )  # save changes and display new image
        self.edit_text_window.destroy()

    def edit_plot_image(
        self, file_path: os.PathLike | str, image_frame: tk.Frame
    ) -> None:
        # create new window
        self.edit_image_window = tk.Toplevel(self)
        self.edit_image_window.title(
            "Simopt Graphical User Interface - Edit Plot Image File"
        )
        self.edit_image_window.geometry("800x500")

        self.edit_image_frame = tk.Frame(self.edit_image_window)
        self.edit_image_frame.grid(row=0, column=0)
        # load plot pickle
        root, ext = os.path.splitext(file_path)
        pickle_path = f"{root}.pkl"
        with open(pickle_path, "rb") as f:
            fig = pickle.load(f)
        dpi = fig.get_dpi()  # get current dpi

        self.dpi_label = tk.Label(
            master=self.edit_image_frame,
            text="DPI (dots per square inch)",
        )
        self.dpi_label.grid(row=0, column=0)
        self.dpi_var = tk.StringVar()
        self.dpi_var.set(dpi)
        self.dpi_entry = tk.Entry(
            master=self.edit_image_frame, textvariable=self.dpi_var
        )
        self.dpi_entry.grid(row=0, column=1, padx=10)

        self.ext_label = tk.Label(
            master=self.edit_image_frame,
            text="Font Family",
        )
        self.ext_label.grid(row=1, column=0)
        self.ext_var = tk.StringVar()
        self.ext_var.set(ext)
        self.ext_menu = ttk.OptionMenu(
            self.edit_image_frame, self.ext_var, ext, *self.ext_options
        )
        self.ext_menu.grid(row=1, column=1, padx=10)

        self.save_image_button = tk.Button(
            master=self.edit_image_window,
            text="Save Changes",
            command=lambda: self.save_image_changes(
                fig, pickle_path, file_path, image_frame
            ),
        )
        self.save_image_button.grid(row=1, column=0, pady=10)
        self.save_image_to_copy_button = tk.Button(
            master=self.edit_image_window,
            text="Save Changes to Copy",
            command=lambda: self.save_image_changes(
                fig, pickle_path, file_path, image_frame, copy=True
            ),
        )
        self.save_image_to_copy_button.grid(row=1, column=1, pady=10, padx=20)

    def save_image_changes(
        self,
        fig: plt.figure,
        pickle_path: os.PathLike | str,
        file_path: os.PathLike | str,
        image_frame: tk.Frame,
        copy: bool = False,
    ) -> None:
        dpi = float(self.dpi_var.get())
        ext = self.ext_var.get()
        path_name = os.path.splitext(file_path)[0]
        save_path = path_name + ext
        if not copy:
            # overwrite pickle with new plot
            with open(pickle_path, "wb") as f:
                pickle.dump(fig, f)

            # overwrite image with new plot
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
            # display new image in view window
            for photo in image_frame.winfo_children():
                photo.destroy()
            plot_image = Image.open(save_path)
            plot_photo = ImageTk.PhotoImage(plot_image)
            plot_display = tk.Label(master=image_frame, image=plot_photo)
            plot_display.image = plot_photo
            plot_display.grid(row=0, column=0, padx=10, pady=10)

        else:
            # Check to make sure file does not override previous images
            counter = 1
            extended_path_name = save_path
            new_path_name = save_path  # in case while loop doesn't need to run
            while os.path.exists(extended_path_name):
                extended_path_name = f"{path_name} ({counter}){ext}"
                new_path_name = f"{path_name} ({counter})"  # use for pickle
                counter += 1
            plt.savefig(
                extended_path_name, bbox_inches="tight", dpi=dpi
            )  # save image
            # save pickle with new name
            pickle_file = new_path_name + ".pkl"
            with open(pickle_file, "wb") as f:
                pickle.dump(fig, f)
            # add new row to loaded plots tab
            title = fig.axes[0].get_title()  # title for display
            row = self.loaded_plots_frame.grid_size()[1]
            self.plot_check_var = tk.BooleanVar()
            check = tk.Checkbutton(
                master=self.loaded_plots_frame, variable=self.plot_check_var
            )
            check.grid(row=row, column=0, padx=5)
            self.plot_check_vars[extended_path_name] = self.plot_check_var
            plot_name_label = tk.Label(
                master=self.loaded_plots_frame,
                text=title,
            )
            plot_name_label.grid(row=row, column=1)
            view_button = tk.Button(
                master=self.loaded_plots_frame,
                text="View/Edit",
                command=lambda fp=extended_path_name: self.view_plot(fp),
            )
            view_button.grid(row=row, column=2, padx=5)
            path_label = tk.Label(
                master=self.loaded_plots_frame,
                text=extended_path_name,
            )
            path_label.grid(row=row, column=3, padx=5)
            del_button = tk.Button(
                master=self.loaded_plots_frame,
                text="Delete",
                command=lambda r=row,
                frame=self.loaded_plots_frame,
                fp=extended_path_name: self.delete_plot(r, frame, fp),
            )
            del_button.grid(row=row, column=4, pady=10)

        self.edit_image_window.destroy()

    def view_all_plots(self) -> None:
        # create new window
        self.view_all_window = tk.Toplevel(self)
        self.view_all_window.title(
            "Simopt Graphical User Interface - View Selected Plots"
        )
        self.view_all_window.geometry("800x500")

        # self.view_all_frame = tk.Frame(self.view_all_window)
        # self.view_all_frame.grid(row=0,column=0)

        # Configure the grid layout to expand properly
        self.view_all_window.grid_rowconfigure(0, weight=1)
        self.view_all_window.grid_columnconfigure(0, weight=1)

        # create master canvas
        self.view_all_canvas = tk.Canvas(self.view_all_window)
        self.view_all_canvas.grid(row=0, column=0, sticky="nsew")

        # Create vertical scrollbar
        vert_scroll = ttk.Scrollbar(
            self.view_all_window,
            orient=tk.VERTICAL,
            command=self.view_all_canvas.yview,
        )
        vert_scroll.grid(row=0, column=1, sticky="ns")

        # Create horizontal scrollbar
        horiz_scroll = ttk.Scrollbar(
            self.view_all_window,
            orient=tk.HORIZONTAL,
            command=self.view_all_canvas.xview,
        )
        horiz_scroll.grid(row=1, column=0, sticky="ew")

        # Configure canvas to use the scrollbars
        self.view_all_canvas.configure(
            yscrollcommand=vert_scroll.set, xscrollcommand=horiz_scroll.set
        )

        # create master frame inside the canvas
        self.view_all_frame = tk.Frame(self.view_all_canvas)
        self.view_all_canvas.create_window(
            (0, 0), window=self.view_all_frame, anchor="nw"
        )

        # Bind the configure event to update the scroll region
        self.view_all_frame.bind(
            "<Configure>", self.update_view_all_window_scroll
        )

        # open plot images
        row = 0
        col = 0
        for (
            image_path
        ) in self.plot_check_vars:  # get file path of all created plots
            plot_image = Image.open(image_path)
            plot_photo = ImageTk.PhotoImage(plot_image)
            plot_display = tk.Label(
                master=self.view_all_frame, image=plot_photo
            )
            plot_display.image = plot_photo
            plot_display.grid(row=row, column=col, padx=10, pady=10)
            col += 1
            if col == 3:  # reset col val and move down one row
                row += 1
                col = 0

    def view_selected_plots(self) -> None:
        # get selected plots
        selected_plots = []
        for file_path in self.plot_check_vars:
            select = self.plot_check_vars[file_path].get()
            if select:
                selected_plots.append(file_path)

        if len(selected_plots) == 0:
            tk.messagebox.showerror(
                "No Plots Selected",
                " No plots were selected. Please check boxes next to plots you wish to display.",
            )
        else:  # create viewing window
            # create new window
            self.view_window = tk.Toplevel(self)
            self.view_window.title(
                "Simopt Graphical User Interface - View Selected Plots"
            )
            self.view_window.geometry("800x500")

            # self.view_frame = tk.Frame(self.view_window)
            # self.view_frame.grid(row=0,column=0)

            # Configure the grid layout to expand properly
            self.view_window.grid_rowconfigure(0, weight=1)
            self.view_window.grid_columnconfigure(0, weight=1)

            # create master canvas
            self.view_canvas = tk.Canvas(self.view_window)
            self.view_canvas.grid(row=0, column=0, sticky="nsew")

            # Create vertical scrollbar
            vert_scroll = ttk.Scrollbar(
                self.view_window,
                orient=tk.VERTICAL,
                command=self.view_canvas.yview,
            )
            vert_scroll.grid(row=0, column=1, sticky="ns")

            # Create horizontal scrollbar
            horiz_scroll = ttk.Scrollbar(
                self.view_window,
                orient=tk.HORIZONTAL,
                command=self.view_canvas.xview,
            )
            horiz_scroll.grid(row=1, column=0, sticky="ew")

            # Configure canvas to use the scrollbars
            self.view_canvas.configure(
                yscrollcommand=vert_scroll.set, xscrollcommand=horiz_scroll.set
            )

            # create master frame inside the canvas
            self.view_frame = tk.Frame(self.view_canvas)
            self.view_canvas.create_window(
                (0, 0), window=self.view_frame, anchor="nw"
            )

            # Bind the configure event to update the scroll region
            self.view_frame.bind("<Configure>", self.update_view_window_scroll)

            # open plot images
            row = 0
            col = 0
            for image_path in selected_plots:
                plot_image = Image.open(image_path)
                plot_photo = ImageTk.PhotoImage(plot_image)
                plot_display = tk.Label(
                    master=self.view_frame, image=plot_photo
                )
                plot_display.image = plot_photo
                plot_display.grid(row=row, column=col, padx=10, pady=10)
                col += 1
                if col == 3:  # reset col val and move down one row
                    row += 1
                    col = 0

    def update_view_window_scroll(self, event: tk.Event) -> None:
        self.view_canvas.configure(scrollregion=self.view_canvas.bbox("all"))

    def update_view_all_window_scroll(self, event: tk.Event) -> None:
        self.view_all_canvas.configure(
            scrollregion=self.view_all_canvas.bbox("all")
        )

    def update_view_single_window_scroll(self, event: tk.Event) -> None:
        self.view_single_canvas.configure(
            scrollregion=self.view_single_canvas.bbox("all")
        )
