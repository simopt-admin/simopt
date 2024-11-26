import ast
import os
import pickle
import re
import tkinter as tk
from abc import ABCMeta
from tkinter import filedialog, messagebox, ttk
from tkinter.font import nametofont
from typing import Callable, Final, Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageTk

from simopt.base import Model, Problem, Solver
from simopt.data_farming_base import DATA_FARMING_DIR
from simopt.directory import (
    problem_directory,
    problem_unabbreviated_directory,
    solver_directory,
    solver_unabbreviated_directory,
)
from simopt.experiment_base import (
    ProblemSolver,
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


class NewExperimentWindow(Toplevel):
    """New Experiment Window."""

    # Constants
    DEFAULT_NUM_STACKS: Final[int] = 1
    DEFAULT_EXP_NAME: Final[str] = "Experiment"
    DEFAULT_EXP_CHECK: Final[bool] = False

    def __init__(self, root: tk.Tk) -> None:
        """Initialize New Experiment Window."""
        super().__init__(
            root, title="SimOpt GUI - New Experiment", exit_on_close=True
        )

        self.center_window(0.8)
        self.minsize(1280, 720)

        # TODO: integrate this into a class somewhere so it's not based in
        # the GUI
        self.design_types: Final[list[str]] = ["nolhs"]

        # master list variables
        self.root_solver_dict = {}  # for each name of solver or solver design has list that includes: [[solver factors], solver name]
        self.root_problem_dict = {}  # for each name of solver or solver design has list that includes: [[problem factors], [model factors], problem name]
        self.root_experiment_dict = {}  # dictionary of experiment name and related solver/problem lists (solver_factor_list, problem_factor_list, solver_name_list, problem_name_list)
        self.macro_reps = {}  # dict that contains user specified macroreps for each experiment
        self.post_reps = {}  # dict that contains user specified postrep numbers for each experiment
        self.init_post_reps = {}  # dict that contains number of postreps to take at initial & optimal solution for normalization for each experiment
        self.crn_budgets = {}  # contains bool val for if crn is used across budget for each experiment
        self.crn_macros = {}  # contains bool val for if crn is used across macroreps for each experiment
        self.crn_inits = {}  # contains bool val for if crn is used across initial and optimal solution for each experiment
        self.solve_tols = {}  # solver tolerance gaps for each experiment (inserted as list)

        # widget lists for enable/delete functions
        self.macro_vars = []  # list used for updated macro entries when default is changed

        # Default experiment options (can be changed in GUI)
        self.macro_default = 10
        self.post_default = 100
        self.init_default = 100
        self.crn_budget_default = True
        self.crn_macro_default = True
        self.crn_init_default = True
        self.solve_tols_default = [0.05, 0.10, 0.20, 0.50]

        # Variables used by the GUI
        # Add the name of the problem/solver to the displayed description
        # TODO: update the problem/solver implementations so that "name" is an
        # attribute of the class, as currently it is only initialized in the
        # __init__ method of the class. This will eliminate the need to
        # instantiate the class to get the name.
        self.problem_full_name_to_class = {
            f"{problem().name} - {key}": problem
            for key, problem in problem_unabbreviated_directory.items()
        }
        self.solver_full_name_to_class = {
            f"{solver().name} - {key}": solver
            for key, solver in solver_unabbreviated_directory.items()
        }
        # TODO: replace root lists/dicts with more descriptive names
        # All exp variables
        # self.all_exp_problems = []
        # Current exp variables
        # self.current_exp_problems = []
        # self.current_exp_solvers = []
        self.curr_exp_name = tk.StringVar()
        self.curr_exp_name.set(self.DEFAULT_EXP_NAME)
        self.curr_exp_is_pickled = tk.BooleanVar()
        self.curr_exp_is_pickled.set(self.DEFAULT_EXP_CHECK)
        # Notebook variables
        self.selected_problem_name = tk.StringVar()
        self.selected_solver_name = tk.StringVar()
        self.factor_dict: dict[str, DFFactor] = {}
        # Design option variables
        self.design_type = tk.StringVar()
        self.design_type.set(self.design_types[0])
        self.design_num_stacks = tk.IntVar()
        self.design_num_stacks.set(self.DEFAULT_NUM_STACKS)
        self.design_name = tk.StringVar()

        # Using dictionaries to store TK variables so they don't clutter
        # the namespace and can easily be accessed by name
        self.tk_buttons: dict[str, ttk.Button] = {}
        self.tk_canvases: dict[str, tk.Canvas] = {}
        self.tk_checkbuttons: dict[str, ttk.Checkbutton] = {}
        self.tk_comboboxes: dict[str, ttk.Combobox] = {}
        self.tk_entries: dict[str, ttk.Entry] = {}
        self.tk_frames: dict[str, ttk.Frame] = {}
        self.tk_labels: dict[str, ttk.Label] = {}
        self.tk_notebooks: dict[str, ttk.Notebook] = {}
        self.tk_var_bools: dict[str, tk.BooleanVar] = {}
        # self.tk_scrollbars: dict[str, ttk.Scrollbar] = {}

        # Setup the main frame
        self._initialize_main_frame()
        # Setup each subframe
        self._initialize_experiment_frame()
        self._initialize_current_experiment_frame()
        self._initialize_notebook_frame()
        self._initialize_generated_design_frame()
        self._initialize_design_options()
        # Grid each subframe
        self.tk_frames["exps"].grid(row=0, sticky="nsew", pady=5)
        self.tk_frames["curr_exp"].grid(row=1, sticky="nsew", pady=5)
        self.tk_frames["ntbk"].grid(row=0, sticky="nsew", pady=5)
        self.tk_frames["design_opts"].grid(row=1, sticky="nsew", pady=5)
        self.tk_frames["gen_design"].grid(row=2, sticky="nsew", pady=5)
        self.tk_frames["gen_design"].grid_remove()

    def _initialize_main_frame(self) -> None:
        if "main" in self.tk_frames:
            self.tk_frames["main"].destroy()
        # Setup the main frame
        self.tk_frames["main"] = ttk.Frame(self)
        self.tk_frames["main"].pack(fill="both", expand=True)
        self.tk_frames["main"].grid_columnconfigure(0, weight=1)
        self.tk_frames["main"].grid_columnconfigure(1, weight=2)
        self.tk_frames["main"].grid_rowconfigure(0, weight=1)
        # Configure the left side of the window
        self.tk_frames["left"] = ttk.Frame(self.tk_frames["main"])
        self.tk_frames["left"].grid(row=0, column=0, sticky="nsew", padx=5)
        self.tk_frames["left"].grid_columnconfigure(0, weight=1)
        self.tk_frames["left"].grid_rowconfigure(0, weight=1)
        self.tk_frames["left"].grid_rowconfigure(1, weight=1)
        self.tk_frames["left"].grid_propagate(False)
        # Configure the right side of the window
        self.tk_frames["right"] = ttk.Frame(self.tk_frames["main"])
        self.tk_frames["right"].grid(row=0, column=1, sticky="nsew", padx=5)
        self.tk_frames["right"].grid_columnconfigure(0, weight=1)
        self.tk_frames["right"].grid_rowconfigure(0, weight=1)
        self.tk_frames["right"].grid_rowconfigure(1, weight=0)
        self.tk_frames["right"].grid_rowconfigure(2, weight=0)
        self.tk_frames["right"].grid_propagate(False)
        # Apply a custom theme to each frame to achieve a grid-like appearance
        border_style = ttk.Style()
        border_style.configure("Main.TFrame", background="white")
        self.tk_frames["main"].configure(style="Main.TFrame")
        self.tk_frames["left"].configure(style="Main.TFrame")
        self.tk_frames["right"].configure(style="Main.TFrame")

    def _initialize_experiment_frame(self) -> None:
        if "exps" in self.tk_frames:
            self.tk_frames["exps"].destroy()
        self.tk_frames["exps"] = ttk.Frame(
            self.tk_frames["left"], borderwidth=1, relief="solid"
        )
        self.tk_frames["exps"].grid_columnconfigure(0, weight=1)
        self.tk_frames["exps"].grid_rowconfigure(1, weight=1)
        self.tk_labels["exps.header"] = ttk.Label(
            self.tk_frames["exps"],
            text="Created Experiments",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["exps.header"].grid(row=0, column=0, sticky="nsew")
        self.tk_frames["exps.list"] = ttk.Frame(
            self.tk_frames["exps"],
        )
        self.tk_frames["exps.list"].grid(row=1, column=0, sticky="nsew")
        self.tk_frames["exps.list"].grid_columnconfigure(0, weight=1)
        self.tk_canvases["exps.list"] = tk.Canvas(
            self.tk_frames["exps.list"],
        )
        self.tk_canvases["exps.list"].grid(row=0, column=0, sticky="nsew")
        self.tk_frames["exps.fields"] = ttk.Frame(
            self.tk_frames["exps"],
        )
        self.tk_frames["exps.fields"].grid(row=2, column=0, sticky="nsew")
        self.tk_frames["exps.fields"].grid_columnconfigure(0, weight=1)
        self.tk_buttons["exps.fields.default_opts"] = ttk.Button(
            self.tk_frames["exps.fields"],
            text="Change Default Experiment Options",
            command=self.open_defaults_window,
        )
        self.tk_buttons["exps.fields.default_opts"].grid(
            row=0, column=0, sticky="ew"
        )
        self.tk_buttons["exps.fields.open_plot_win"] = ttk.Button(
            self.tk_frames["exps.fields"],
            text="Open Plotting Window",
            command=self.open_plotting_window,
        )
        self.tk_buttons["exps.fields.open_plot_win"].grid(
            row=1, column=0, sticky="ew"
        )
        self.tk_buttons["exps.fields.load_exp"] = ttk.Button(
            self.tk_frames["exps.fields"],
            text="Load Experiment",
            command=self.load_experiment,
        )
        self.tk_buttons["exps.fields.load_exp"].grid(
            row=2, column=0, sticky="ew"
        )

    def _initialize_current_experiment_frame(self) -> None:
        if "curr_exp" in self.tk_frames:
            self.tk_frames["curr_exp"].destroy()
        self.tk_frames["curr_exp"] = ttk.Frame(
            self.tk_frames["left"], borderwidth=1, relief="solid"
        )
        self.tk_frames["curr_exp"].grid_columnconfigure(0, weight=1)
        self.tk_frames["curr_exp"].grid_rowconfigure(1, weight=1)
        self.tk_labels["curr_exp.header"] = ttk.Label(
            self.tk_frames["curr_exp"],
            text="Current Experiment Workspace",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["curr_exp.header"].grid(row=0, column=0, sticky="nsew")
        self.tk_frames["curr_exp.lists"] = ttk.Frame(self.tk_frames["curr_exp"])
        self.tk_frames["curr_exp.lists"].grid(row=1, column=0, sticky="nsew")
        self.tk_frames["curr_exp.lists"].grid_propagate(False)
        self.tk_frames["curr_exp.lists"].grid_columnconfigure(0, weight=1)
        self.tk_frames["curr_exp.lists"].grid_columnconfigure(1, weight=1)
        self.tk_labels["curr_exp.lists.problem_header"] = ttk.Label(
            self.tk_frames["curr_exp.lists"],
            text="Problems",
            anchor="center",
        )
        self.tk_labels["curr_exp.lists.problem_header"].grid(
            row=0, column=0, sticky="nsew"
        )
        self.tk_labels["curr_exp.lists.solver_header"] = ttk.Label(
            self.tk_frames["curr_exp.lists"], text="Solvers", anchor="center"
        )
        self.tk_labels["curr_exp.lists.solver_header"].grid(
            row=0, column=1, sticky="nsew"
        )
        self.tk_canvases["curr_exp.lists.problems"] = tk.Canvas(
            self.tk_frames["curr_exp.lists"],
        )
        self.tk_canvases["curr_exp.lists.problems"].grid(
            row=1, column=0, sticky="nsew"
        )
        self.tk_canvases["curr_exp.lists.solvers"] = tk.Canvas(
            self.tk_frames["curr_exp.lists"],
        )
        self.tk_canvases["curr_exp.lists.solvers"].grid(
            row=1, column=1, sticky="nsew"
        )
        self.tk_frames["curr_exp.fields"] = ttk.Frame(
            self.tk_frames["curr_exp"],
        )
        self.tk_frames["curr_exp.fields"].grid(row=2, column=0, sticky="nsew")
        self.tk_frames["curr_exp.fields"].grid_columnconfigure(1, weight=1)
        self.tk_buttons["curr_exp.fields.load_design"] = ttk.Button(
            self.tk_frames["curr_exp.fields"],
            text="Load Design from CSV",
            command=self.load_design,
        )
        self.tk_buttons["curr_exp.fields.load_design"].grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        self.tk_buttons["curr_exp.fields.clear_list"] = ttk.Button(
            self.tk_frames["curr_exp.fields"],
            text="Clear Problem/Solver Lists",
            command=self.clear_experiment,
        )
        self.tk_buttons["curr_exp.fields.clear_list"].grid(
            row=1, column=0, columnspan=2, sticky="ew"
        )
        self.tk_labels["curr_exp.fields.exp_name"] = ttk.Label(
            self.tk_frames["curr_exp.fields"],
            text="Experiment Name ",
            anchor="e",
        )
        self.tk_labels["curr_exp.fields.exp_name"].grid(
            row=2, column=0, sticky="ew"
        )
        self.tk_entries["curr_exp.fields.exp_name"] = ttk.Entry(
            self.tk_frames["curr_exp.fields"],
            textvariable=self.curr_exp_name,
        )
        self.tk_entries["curr_exp.fields.exp_name"].grid(
            row=2, column=1, sticky="ew"
        )
        self.tk_labels["curr_exp.fields.make_pickle"] = ttk.Label(
            self.tk_frames["curr_exp.fields"],
            text="Create Pickles for each pair? ",
            anchor="e",
        )
        self.tk_labels["curr_exp.fields.make_pickle"].grid(
            row=3, column=0, sticky="ew"
        )
        self.tk_checkbuttons["curr_exp.fields.make_pickle"] = ttk.Checkbutton(
            self.tk_frames["curr_exp.fields"],
            variable=self.curr_exp_is_pickled,
        )
        self.tk_checkbuttons["curr_exp.fields.make_pickle"].grid(
            row=3, column=1, sticky="w"
        )
        self.tk_buttons["curr_exp.fields.create_exp"] = ttk.Button(
            self.tk_frames["curr_exp.fields"],
            text="Create Experiment",
            command=self.create_experiment,
        )
        self.tk_buttons["curr_exp.fields.create_exp"].grid(
            row=4, column=0, columnspan=2, sticky="ew"
        )

    def _initialize_notebook_frame(self) -> None:
        if "ntbk" in self.tk_frames:
            self.tk_frames["ntbk"].destroy()
        self.tk_frames["ntbk"] = ttk.Frame(
            self.tk_frames["right"], borderwidth=1, relief="solid"
        )
        self.tk_frames["ntbk"].grid_propagate(False)
        self.tk_frames["ntbk"].grid_columnconfigure(0, weight=1)
        self.tk_frames["ntbk"].grid_rowconfigure(1, weight=1)
        self.tk_labels["ntbk.header"] = ttk.Label(
            self.tk_frames["ntbk"],
            text="Create Problems/Solvers",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["ntbk.header"].grid(row=0, column=0, sticky="nsew")
        self.tk_notebooks["ntbk.ps_adding"] = ttk.Notebook(
            self.tk_frames["ntbk"]
        )
        self.tk_notebooks["ntbk.ps_adding"].grid(row=1, column=0, sticky="nsew")
        self.tk_frames["ntbk.ps_adding.problem"] = ttk.Frame(
            self.tk_notebooks["ntbk.ps_adding"]
        )
        self.tk_notebooks["ntbk.ps_adding"].add(
            self.tk_frames["ntbk.ps_adding.problem"], text="Add Problem"
        )
        self.tk_frames["ntbk.ps_adding.problem"].grid_columnconfigure(
            1, weight=1
        )
        self.tk_frames["ntbk.ps_adding.problem"].grid_rowconfigure(1, weight=1)
        self.tk_labels["ntbk.ps_adding.problem.select"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.problem"], text="Selected Problem"
        )
        self.tk_labels["ntbk.ps_adding.problem.select"].grid(
            row=0, column=0, padx=5
        )
        # Setting this to readonly prevents the user from typing in the combobox
        self.tk_comboboxes["ntbk.ps_adding.problem.select"] = ttk.Combobox(
            self.tk_frames["ntbk.ps_adding.problem"],
            textvariable=self.selected_problem_name,
            values=sorted(list(self.problem_full_name_to_class.keys())),
            state="readonly",
        )
        self.tk_comboboxes["ntbk.ps_adding.problem.select"].grid(
            row=0, column=1, sticky="ew", padx=5
        )
        self.tk_comboboxes["ntbk.ps_adding.problem.select"].bind(
            "<<ComboboxSelected>>", self._on_problem_combobox_change
        )
        self.tk_canvases["ntbk.ps_adding.problem.factors"] = tk.Canvas(
            self.tk_frames["ntbk.ps_adding.problem"]
        )
        self.tk_canvases["ntbk.ps_adding.problem.factors"].grid(
            row=1, column=0, sticky="nsew", columnspan=2
        )

        self.tk_frames["ntbk.ps_adding.solver"] = ttk.Frame(
            self.tk_notebooks["ntbk.ps_adding"]
        )
        self.tk_notebooks["ntbk.ps_adding"].add(
            self.tk_frames["ntbk.ps_adding.solver"], text="Add Solver"
        )
        self.tk_frames["ntbk.ps_adding.solver"].grid_columnconfigure(
            1, weight=1
        )
        self.tk_frames["ntbk.ps_adding.solver"].grid_rowconfigure(1, weight=1)
        self.tk_labels["ntbk.ps_adding.solver.select"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.solver"], text="Selected Solver"
        )
        self.tk_labels["ntbk.ps_adding.solver.select"].grid(
            row=0, column=0, padx=5
        )
        # Setting this to readonly prevents the user from typing in the combobox
        self.tk_comboboxes["ntbk.ps_adding.solver.select"] = ttk.Combobox(
            self.tk_frames["ntbk.ps_adding.solver"],
            textvariable=self.selected_solver_name,
            values=sorted(list(self.solver_full_name_to_class.keys())),
            state="readonly",
        )
        self.tk_comboboxes["ntbk.ps_adding.solver.select"].grid(
            row=0, column=1, sticky="ew", padx=5
        )
        self.tk_comboboxes["ntbk.ps_adding.solver.select"].bind(
            "<<ComboboxSelected>>", self._on_solver_combobox_change
        )
        self.tk_canvases["ntbk.ps_adding.solver.factors"] = tk.Canvas(
            self.tk_frames["ntbk.ps_adding.solver"]
        )
        self.tk_canvases["ntbk.ps_adding.solver.factors"].grid(
            row=1, column=0, sticky="nsew", columnspan=2
        )

        self.tk_frames["ntbk.ps_adding.quick_add"] = ttk.Frame(
            self.tk_notebooks["ntbk.ps_adding"]
        )
        self.tk_notebooks["ntbk.ps_adding"].add(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            text="Quick-Add Problems/Solvers",
        )
        self.tk_notebooks["ntbk.ps_adding"].bind(
            "<<NotebookTabChanged>>", self._on_notebook_tab_change
        )
        # Initialize the quick-add tab
        # If this doesn't get initialized, the compatability checks will fail
        self._add_with_default_options()

    def _initialize_generated_design_frame(self) -> None:
        if "gen_design" in self.tk_frames:
            self.tk_frames["gen_design"].destroy()
        self.tk_frames["gen_design"] = ttk.Frame(
            self.tk_frames["right"], borderwidth=1, relief="solid"
        )
        self.tk_frames["gen_design"].grid_columnconfigure(0, weight=1)
        self.tk_frames["gen_design"].grid_rowconfigure(1, weight=1)
        self.tk_labels["gen_design.header"] = ttk.Label(
            self.tk_frames["gen_design"],
            text="Generated Design",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["gen_design.header"].grid(row=0, column=0, sticky="nsew")
        self.tk_frames["gen_design.display"] = ttk.Frame(
            self.tk_frames["gen_design"]
        )
        self.tk_frames["gen_design.display"].grid(
            row=1, column=0, sticky="nsew"
        )
        self.tk_frames["gen_design.display"].grid_columnconfigure(0, weight=1)
        self.tk_frames["gen_design.display"].grid_columnconfigure(1, weight=0)
        self.tk_frames["gen_design.display"].grid_rowconfigure(0, weight=1)

    def _initialize_design_options(self) -> None:
        if "design_opts" in self.tk_frames:
            self.tk_frames["design_opts"].destroy()
        self.tk_frames["design_opts"] = ttk.Frame(
            self.tk_frames["right"], borderwidth=1, relief="solid"
        )
        self.tk_frames["design_opts"].grid_columnconfigure(1, weight=1)
        self.tk_labels["design_opts.header"] = ttk.Label(
            self.tk_frames["design_opts"],
            text="Design Options",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["design_opts.header"].grid(
            row=0, column=0, sticky="nsew", columnspan=3
        )
        self.tk_labels["design_opts.type"] = ttk.Label(
            self.tk_frames["design_opts"], text="Design Type ", anchor="e"
        )
        self.tk_labels["design_opts.type"].grid(row=1, column=0, sticky="ew")
        # Setting this to readonly prevents the user from typing in the combobox
        self.tk_comboboxes["design_opts.type"] = ttk.Combobox(
            self.tk_frames["design_opts"],
            textvariable=self.design_type,
            values=sorted(self.design_types),
            state="readonly",
        )
        self.tk_comboboxes["design_opts.type"].grid(
            row=1, column=1, sticky="ew"
        )
        self.tk_labels["design_opts.num_stacks"] = ttk.Label(
            self.tk_frames["design_opts"], text="# of Stacks ", anchor="e"
        )
        self.tk_labels["design_opts.num_stacks"].grid(
            row=2, column=0, sticky="ew"
        )
        self.tk_entries["design_opts.num_stacks"] = ttk.Entry(
            self.tk_frames["design_opts"], textvariable=self.design_num_stacks
        )
        self.tk_entries["design_opts.num_stacks"].grid(
            row=2, column=1, sticky="ew"
        )
        self.tk_labels["design_opts.name"] = ttk.Label(
            self.tk_frames["design_opts"], text="Design Name ", anchor="e"
        )
        self.tk_labels["design_opts.name"].grid(row=3, column=0, sticky="ew")
        self.tk_entries["design_opts.name"] = ttk.Entry(
            self.tk_frames["design_opts"], textvariable=self.design_name
        )
        self.tk_entries["design_opts.name"].grid(row=3, column=1, sticky="ew")
        self.tk_buttons["design_opts.generate"] = ttk.Button(
            self.tk_frames["design_opts"],
            text="Generate Design",
            width=40,
        )
        self.tk_buttons["design_opts.generate"].grid(
            row=1, column=2, sticky="nsew", rowspan=3
        )

    def _hide_gen_design(self) -> None:
        self.tk_frames["gen_design"].grid_remove()

    def _show_gen_design(self) -> None:
        # We don't need to pass any settings to grid since grid_remove
        # remembers the previous settings
        self.tk_frames["gen_design"].grid()

    def _disable_design_opts(self) -> None:
        self.tk_comboboxes["design_opts.type"].configure(state="disabled")
        self.tk_entries["design_opts.num_stacks"].configure(state="disabled")
        self.tk_entries["design_opts.name"].configure(state="disabled")

    def _enable_design_opts(self) -> None:
        self.tk_comboboxes["design_opts.type"].configure(state="readonly")
        self.tk_entries["design_opts.num_stacks"].configure(state="normal")
        self.tk_entries["design_opts.name"].configure(state="normal")

    # Event handler for when the user changes the notebook tab
    def _on_notebook_tab_change(self, event: tk.Event) -> None:
        # Hide the generated design frame
        self._hide_gen_design()

        # Reset the design options to the default values
        self.design_type.set(self.design_types[0])
        self.design_num_stacks.set(self.DEFAULT_NUM_STACKS)
        self.design_name.set("")  # Blank out the design name

        # Figure out what tab is being switched to
        tab_name = event.widget.tab(event.widget.select(), "text")
        # Switch on the tab name
        if tab_name == "Add Problem":
            self.selected_problem_name.set("")
            self._enable_design_opts()
            self._destroy_widget_children(
                self.tk_canvases["ntbk.ps_adding.problem.factors"]
            )
            self.tk_buttons["design_opts.generate"].configure(
                text="Generate Problem Design",
                command=self.create_problem_design,
            )

        elif tab_name == "Add Solver":
            self.selected_solver_name.set("")
            self._enable_design_opts()
            self._destroy_widget_children(
                self.tk_canvases["ntbk.ps_adding.solver.factors"]
            )
            self.tk_buttons["design_opts.generate"].configure(
                text="Generate Solver Design", command=self.create_solver_design
            )

        elif tab_name == "Quick-Add Problems/Solvers":
            self._disable_design_opts()
            self._add_with_default_options()
            self.tk_buttons["design_opts.generate"].configure(
                text="Add Cross Design to Experiment",
                command=self.create_cross_design,
            )

        else:
            error_msg = f"Unknown tab name: {tab_name}"
            raise ValueError(error_msg)

    # Event handler for when the user changes the problem combobox
    def _on_problem_combobox_change(self, _: tk.Event) -> None:
        problem_name = self.selected_problem_name.get()
        # Initialize problem for later. This is needed since the problem has
        # many of its attributes set in the __init__ method, and if it is not
        # initialized here, the problem will not have the correct attributes
        problem_class: ABCMeta = self.problem_full_name_to_class[problem_name]
        problem: Problem = problem_class()
        self._create_problem_factors_canvas(problem)
        self._hide_gen_design()

    def _on_solver_combobox_change(self, _: tk.Event) -> None:
        solver_name = self.selected_solver_name.get()
        # Initialize solver for later. This is needed since the solver has many
        # of its attributes set in the __init__ method, and if it is not
        # initialized here, the solver will not have the correct attributes
        solver_class: ABCMeta = self.solver_full_name_to_class[solver_name]
        solver: Solver = solver_class()
        self._create_solver_factors_canvas(solver)
        self._hide_gen_design()

    def __update_problem_dropdown(self) -> None:
        possible_problems = sorted(list(self.problem_full_name_to_class.keys()))
        # For each solver in the current experiment, check all the possible
        # problems and remove the ones that are not compatible
        # Grab the name (index 1) out of the first element (index 0) of the
        # dictionary (looked up by key) to get the solver class name
        solver_class_list = [self.root_solver_dict[key][0][1] for key in self.root_solver_dict]
        for solver_name in solver_class_list:
            solver_class: ABCMeta = solver_directory[solver_name]
            solver: Solver = solver_class()
            problem_list = possible_problems.copy()
            for problem_name in problem_list:
                short_problem_name = problem_name.split(" - ")[0]
                problem_class: ABCMeta = problem_directory[short_problem_name]
                problem: Problem = problem_class()
                # Create a new ProblemSolver object to check compatibility
                problem_solver = ProblemSolver(
                    problem=problem, solver=solver
                )
                # If there was an error, remove it from the options
                if len(problem_solver.check_compatibility()) > 0:
                    possible_problems.remove(problem_name)
        self.tk_comboboxes["ntbk.ps_adding.problem.select"].configure(
            values=possible_problems
        )

    def __update_solver_dropdown(self) -> None:
        possible_options = sorted(list(self.solver_full_name_to_class.keys()))
        # For each problem in the current experiment, check all the possible
        # solvers and remove the ones that are not compatible
        # Grab the name (index 1) out of the first element (index 0) of the
        # dictionary (looked up by key) to get the problem class name
        problem_class_list = [self.root_problem_dict[key][0][1] for key in self.root_problem_dict]
        for problem_name in problem_class_list:
            problem_class: ABCMeta = problem_directory[problem_name]
            problem: Problem = problem_class()
            solver_list = possible_options.copy()
            for solver_name in solver_list:
                short_solver_name = solver_name.split(" - ")[0]
                solver_class: ABCMeta = solver_directory[short_solver_name]
                solver: Solver = solver_class()
                # Create a new ProblemSolver object to check compatibility
                problem_solver = ProblemSolver(
                    problem=problem, solver=solver
                )
                # If there was an error, remove it from the options
                if len(problem_solver.check_compatibility()) > 0:
                    possible_options.remove(solver_name)
        self.tk_comboboxes["ntbk.ps_adding.solver.select"].configure(
            values=possible_options
        )

    def add_problem_to_curr_exp(
        self, unique_name: str, problem_list: list
    ) -> None:
        self.root_problem_dict[unique_name] = problem_list
        self.add_problem_to_curr_exp_list(unique_name)
        self.__update_solver_dropdown()

    def add_solver_to_curr_exp(
        self, unique_name: str, solver_list: list
    ) -> None:
        self.root_solver_dict[unique_name] = solver_list
        self.add_solver_to_curr_exp_list(unique_name)
        self.__update_problem_dropdown()

    def add_problem_to_curr_exp_list(self, unique_name: str) -> None:
        # Make sure the unique name is in the root problem dict
        if unique_name not in self.root_problem_dict:
            error_msg = f"Problem {unique_name} not found in root problem dict"
            raise ValueError(error_msg)
        # Add the problem to the GUI
        tk_base_name = f"curr_exp.lists.problems.{unique_name}"
        list_entries = self.tk_canvases[
            "curr_exp.lists.problems"
        ].winfo_children()
        problem_row = len(list_entries) // 3
        # Add name label
        name_label_name = f"{tk_base_name}.name"
        self.tk_labels[name_label_name] = ttk.Label(
            master=self.tk_canvases["curr_exp.lists.problems"],
            text=unique_name,
        )
        self.tk_labels[name_label_name].grid(row=problem_row, column=1)
        # Add edit button
        edit_button_name = f"{tk_base_name}.edit"
        self.tk_buttons[edit_button_name] = ttk.Button(
            master=self.tk_canvases["curr_exp.lists.problems"],
            text="View/Edit",
            command=lambda problem_name=unique_name: self.edit_problem(
                problem_name
            ),
        )
        self.tk_buttons[edit_button_name].grid(row=problem_row, column=2)
        # Add delete button
        del_button_name = f"{tk_base_name}.del"
        self.tk_buttons[del_button_name] = ttk.Button(
            master=self.tk_canvases["curr_exp.lists.problems"],
            text="Delete",
            command=lambda problem_name=unique_name: self.delete_problem(
                problem_name
            ),
        )
        self.tk_buttons[del_button_name].grid(row=problem_row, column=3)

    def add_solver_to_curr_exp_list(self, unique_name: str) -> None:
        # Make sure the unique name is in the root solver dict
        if unique_name not in self.root_solver_dict:
            error_msg = f"Solver {unique_name} not found in root solver dict"
            raise ValueError(error_msg)
        # Add the solver to the GUI
        tk_base_name = f"curr_exp.lists.solvers.{unique_name}"
        list_entries = self.tk_canvases[
            "curr_exp.lists.solvers"
        ].winfo_children()
        solver_row = len(list_entries) // 3
        # Add name label
        name_label_name = f"{tk_base_name}.name"
        self.tk_labels[name_label_name] = ttk.Label(
            master=self.tk_canvases["curr_exp.lists.solvers"],
            text=unique_name,
        )
        self.tk_labels[name_label_name].grid(row=solver_row, column=1)
        # Add edit button
        edit_button_name = f"{tk_base_name}.edit"
        self.tk_buttons[edit_button_name] = ttk.Button(
            master=self.tk_canvases["curr_exp.lists.solvers"],
            text="View/Edit",
            command=lambda solver_name=unique_name: self.edit_solver(
                solver_name
            ),
        )
        self.tk_buttons[edit_button_name].grid(row=solver_row, column=2)
        # Add delete button
        del_button_name = f"{tk_base_name}.del"
        self.tk_buttons[del_button_name] = ttk.Button(
            master=self.tk_canvases["curr_exp.lists.solvers"],
            text="Delete",
            command=lambda solver_name=unique_name: self.delete_solver(
                solver_name
            ),
        )
        self.tk_buttons[del_button_name].grid(row=solver_row, column=3)

    def _add_with_default_options(self) -> None:
        # Delete all existing children of the frame
        for child in self.tk_frames[
            "ntbk.ps_adding.quick_add"
        ].winfo_children():
            child.destroy()
        # Configure the grid layout to expand properly
        problem_frame_weight = 2
        solver_frame_weight = 1
        self.tk_frames["ntbk.ps_adding.quick_add"].grid_rowconfigure(
            2, weight=1
        )
        self.tk_frames["ntbk.ps_adding.quick_add"].grid_columnconfigure(
            0, weight=problem_frame_weight
        )
        self.tk_frames["ntbk.ps_adding.quick_add"].grid_columnconfigure(
            1, weight=solver_frame_weight
        )

        # Create labels for the title and the column headers
        title_text = "Select problems/solvers to be included in cross-design."
        title_text += "\nThese will be added with default factor settings."
        self.tk_labels["ntbk.ps_adding.quick_add.title"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            text=title_text,
            anchor="center",
            justify="center",
        )
        self.tk_labels["ntbk.ps_adding.quick_add.title"].grid(
            row=0, column=0, columnspan=2
        )
        self.tk_labels["ntbk.ps_adding.quick_add.problems"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            text="Problems",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["ntbk.ps_adding.quick_add.problems"].grid(
            row=1, column=0, sticky="ew"
        )
        self.tk_labels["ntbk.ps_adding.quick_add.solvers"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            text="Solvers",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["ntbk.ps_adding.quick_add.solvers"].grid(
            row=1, column=1, sticky="ew"
        )

        # Create canvases for the problems and solvers
        self.tk_canvases["ntbk.ps_adding.quick_add.problems"] = tk.Canvas(
            self.tk_frames["ntbk.ps_adding.quick_add"]
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.problems"].grid(
            row=2, column=0, sticky="nsew"
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.solvers"] = tk.Canvas(
            self.tk_frames["ntbk.ps_adding.quick_add"]
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.solvers"].grid(
            row=2, column=1, sticky="nsew"
        )

        # create master frame inside the canvas
        self.tk_frames["ntbk.ps_adding.quick_add.problems_frame"] = ttk.Frame(
            self.tk_canvases["ntbk.ps_adding.quick_add.problems"],
            width=0,
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.problems"].create_window(
            (0, 0),
            window=self.tk_frames["ntbk.ps_adding.quick_add.problems_frame"],
            anchor="nw",
        )
        self.tk_frames["ntbk.ps_adding.quick_add.solvers_frame"] = ttk.Frame(
            self.tk_canvases["ntbk.ps_adding.quick_add.solvers"],
            width=0,
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.solvers"].create_window(
            (0, 0),
            window=self.tk_frames["ntbk.ps_adding.quick_add.solvers_frame"],
            anchor="nw",
        )

        # Calculate how much space for text to wrap
        # TODO: make this check happen whenever the window is resized
        # checkbox_size = 50
        # total_scale = problem_frame_weight + solver_frame_weight
        # problem_scale = problem_frame_weight / total_scale
        # solver_scale = solver_frame_weight / total_scale
        # problem_frame_wrap = (
        #     self.tk_frames["ntbk.ps_adding.quick_add"].winfo_width()
        #     * problem_scale
        #     - checkbox_size
        # )
        # solver_frame_wrap = (
        #     self.tk_frames["ntbk.ps_adding.quick_add"].winfo_width()
        #     * solver_scale
        #     - checkbox_size
        # )

        # display all potential problems
        problem_list = [
            f"{problem().name} - {key}"
            for key, problem in problem_unabbreviated_directory.items()
        ]
        sorted_problems = sorted(problem_list)
        for problem_name in sorted_problems:
            row = self.tk_frames[
                "ntbk.ps_adding.quick_add.problems_frame"
            ].grid_size()[1]
            shortened_name = problem_name.split(" - ")[0]
            tk_name = (
                f"ntbk.ps_adding.quick_add.problems_frame.{shortened_name}"
            )
            self.tk_var_bools[tk_name] = tk.BooleanVar()
            self.tk_checkbuttons[tk_name] = ttk.Checkbutton(
                master=self.tk_frames[
                    "ntbk.ps_adding.quick_add.problems_frame"
                ],
                text=problem_name,
                variable=self.tk_var_bools[tk_name],
                command=self.cross_design_solver_compatibility,
            )
            self.tk_checkbuttons[tk_name].grid(
                row=row, column=0, sticky="w", padx=10
            )
        # display all potential solvers
        solver_list = [
            f"{solver().name} - {key}"
            for key, solver in solver_unabbreviated_directory.items()
        ]
        sorted_solvers = sorted(solver_list)
        for solver_name in sorted_solvers:
            row = self.tk_frames[
                "ntbk.ps_adding.quick_add.solvers_frame"
            ].grid_size()[1]
            shortened_name = solver_name.split(" - ")[0]
            tk_name = (
                f"ntbk.ps_adding.quick_add.problems_frame.{shortened_name}"
            )
            self.tk_var_bools[tk_name] = tk.BooleanVar()
            self.tk_checkbuttons[tk_name] = ttk.Checkbutton(
                master=self.tk_frames["ntbk.ps_adding.quick_add.solvers_frame"],
                text=solver_name,
                variable=self.tk_var_bools[tk_name],
                command=self.cross_design_problem_compatibility,
            )
            self.tk_checkbuttons[tk_name].grid(
                row=row, column=0, sticky="w", padx=10
            )
        # Run the compatibility checks
        self.cross_design_problem_compatibility()
        self.cross_design_solver_compatibility()

    def cross_design_problem_compatibility(self) -> None:
        # If we don't have the tab open, return
        # if self.tk_notebooks["ntbk.ps_adding"].select() != "Quick-Add Problems/Solvers":
        #     return
        # create temp objects for current selected solvers and all possilble problems
        temp_solvers = []
        # solvers previously added to experiment
        for solver_group in self.root_solver_dict:
            dp_0 = self.root_solver_dict[solver_group][
                0
            ]  # first design point if design, only design pt if no design
            solver_name = dp_0[1]
            temp_solver = solver_directory[solver_name]()
            temp_solvers.append(temp_solver)
        # Add all selected solvers to the temp list
        for solver_name in solver_directory:
            dict_name = f"ntbk.ps_adding.quick_add.problems_frame.{solver_name}"
            checkstate = self.tk_var_bools[dict_name].get()
            if checkstate:
                temp_solver = solver_directory[solver_name]()
                temp_solvers.append(temp_solver)
        # Attempt to create a temp experiment with the current solvers and all problems
        for problem_name in problem_directory:
            temp_problem = [problem_directory[problem_name]()]
            temp_exp = ProblemsSolvers(
                solvers=temp_solvers, problems=temp_problem
            )  # temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            dict_name = (
                f"ntbk.ps_adding.quick_add.problems_frame.{problem_name}"
            )
            if error:
                self.tk_checkbuttons[dict_name].configure(state="disabled")
            else:
                self.tk_checkbuttons[dict_name].configure(state="normal")

    def cross_design_solver_compatibility(self) -> None:
        # If we don't have the tab open, return
        # if self.tk_notebooks["ntbk.ps_adding"].select() != "Quick-Add Problems/Solvers":
        #     return
        # create temp objects for current selected solvers and all possilble problems
        temp_problems = []
        # solvers previously added to experiment
        for problem_group in self.root_problem_dict:
            dp_0 = self.root_problem_dict[problem_group][
                0
            ]  # first design point if design, only design pt if no design
            problem_name = dp_0[1]
            temp_problem = problem_directory[problem_name]()
            temp_problems.append(temp_problem)
        # problems currently added to cross design
        for problem in problem_directory:
            dict_name = f"ntbk.ps_adding.quick_add.problems_frame.{problem}"
            checkstate = self.tk_var_bools[dict_name].get()
            if checkstate:
                temp_problem = problem_directory[problem]()
                temp_problems.append(temp_problem)
        # Attempt to create a temp experiment with the current solvers and all problems
        for solver_name in solver_directory:
            temp_solver = [solver_directory[solver_name]()]
            temp_exp = ProblemsSolvers(
                solvers=temp_solver, problems=temp_problems
            )  # temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            dict_name = f"ntbk.ps_adding.quick_add.problems_frame.{solver_name}"
            if error:
                self.tk_checkbuttons[dict_name].configure(state="disabled")
            else:
                self.tk_checkbuttons[dict_name].configure(state="normal")

    def create_cross_design(self) -> None:
        any_added = False
        for solver_name in solver_directory:
            dict_name = f"ntbk.ps_adding.quick_add.problems_frame.{solver_name}"
            checkstate = self.tk_var_bools[dict_name].get()
            # Move on if the solver is not selected
            if not checkstate:
                continue
            any_added = True
            # Otherwise, add the solver with default factor settings to the master dict
            solver = solver_directory[solver_name]()
            factors = {
                factor: value["default"]
                for factor, value in solver.specifications.items()
            }
            solver_save_name = self.get_unique_name(
                self.root_solver_dict, solver.name
            )
            # Add the solver to the experiment
            factor_list = [[factors, solver.name]]
            self.add_solver_to_curr_exp(solver_save_name, factor_list)

        for problem in problem_directory:
            dict_name = f"ntbk.ps_adding.quick_add.problems_frame.{problem}"
            checkstate = self.tk_var_bools[dict_name].get()
            # Move on if the problem is not selected
            if not checkstate:
                continue
            any_added = True
            # Otherwise, add the problem with default factor settings to the master dict
            temp_problem = problem_directory[problem]()
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
            # Add the problem to the experiment
            factor_list = [[factors, problem_name]]
            self.add_problem_to_curr_exp(problem_save_name, factor_list)

        if not any_added:
            messagebox.showerror(
                "No Problems/Solvers Added",
                "No problems or solvers were selected to be added to the experiment.",
            )
            return

        # Reset the quick-add frame
        self._add_with_default_options()
        # Reset all the booleans
        for key in self.tk_var_bools:
            if "ntbk.ps_adding.quick_add.problems_frame" in key:
                self.tk_var_bools[key].set(False)
            if "ntbk.ps_adding.quick_add.solvers_frame" in key:
                self.tk_var_bools[key].set(False)

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
        else:
            raise ValueError(
                "Loaded file does not match any solver or problem in the directory"
            )

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
            self.design_defaults, last_row = self._show_factor_defaults(
                self.obj,
                self.problem_factor_display_canvas,
                factor_dict=problem_defaults,
                design_factors=self.design_factors,
            )
            # # show model factors and store default widgets and default values to these
            self.model_defaults, new_last_row = self._show_factor_defaults(
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
            self.design_defaults, last_row = self._show_factor_defaults(
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
            csv_filename=design_file, master_frame=self.tree_frame
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
        self.display_design_tree(
            csv_filename,
            master_frame=self.tree_frame,
        )

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
        messagebox.showinfo(
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
                assert isinstance(base_object, (Solver, Model))
                datatype = base_object.specifications[factor].get("datatype")
            else:
                assert isinstance(base_object, Problem)
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

    def _destroy_widget_children(self, widget: tk.Widget) -> None:
        """_Destroy all children of a widget._

        Args:
            widget (tk.Widget): _The widget whose children will be destroyed._
        """
        children = widget.winfo_children()
        for child in children:
            child.destroy()

    def __insert_factor_headers(
        self,
        frame: ttk.Frame,
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
            "Description",
            "Type",
            "Default Value",
            "Include in Design?",
            "Minimum",
            "Maximum",
            "# Decimals",
        ]
        for heading in header_columns:
            frame.grid_columnconfigure(header_columns.index(heading), weight=1)
            label = tk.Label(
                master=frame,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=first_row,
                column=header_columns.index(heading),
                padx=10,
            )
        # Insert horizontal separator
        ttk.Separator(frame, orient="horizontal").grid(
            row=first_row + 1, columnspan=len(header_columns), sticky="ew"
        )
        return first_row + 1

    def __insert_factors(
        self,
        frame: ttk.Frame,
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
            column_functions: list[Callable[[ttk.Frame], tk.Widget | None]] = [
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
                widget = function(frame)
                # Display the widget if it exists
                if widget is not None:
                    widget.grid(
                        row=row_index,
                        column=column_index,
                        padx=10,
                        pady=3,
                        sticky="ew",
                    )
                else:
                    break
        return row_index

    def _create_problem_factors_canvas(self, problem: Problem) -> None:
        # Clear the canvas
        self._destroy_widget_children(
            self.tk_canvases["ntbk.ps_adding.problem.factors"]
        )

        # Initialize the frames and headers
        self.tk_frames["ntbk.ps_adding.problem.factors.problems"] = ttk.Frame(
            master=self.tk_canvases["ntbk.ps_adding.problem.factors"],
        )
        self.tk_frames["ntbk.ps_adding.problem.factors.problems"].grid(
            row=0, column=0, sticky="nsew"
        )

        # show problem factors and store default widgets to this dict
        self.__show_data_farming_core(
            problem,
            frame=self.tk_frames["ntbk.ps_adding.problem.factors.problems"],
            row=1,
        )

        # Update the design name to be unique
        unique_name = self.get_unique_name(self.root_problem_dict, problem.name)
        self.design_name.set(unique_name)
        self.tk_entries["design_opts.name"].delete(0, tk.END)
        self.tk_entries["design_opts.name"].insert(0, unique_name)

    def _create_solver_factors_canvas(self, solver: Solver) -> None:
        # Clear the canvas
        self._destroy_widget_children(
            self.tk_canvases["ntbk.ps_adding.solver.factors"]
        )

        # Initialize the frames and headers
        self.tk_frames["ntbk.ps_adding.solver.factors.solvers"] = ttk.Frame(
            master=self.tk_canvases["ntbk.ps_adding.solver.factors"],
        )
        self.tk_frames["ntbk.ps_adding.solver.factors.solvers"].grid(
            row=0, column=0, sticky="nsew"
        )

        # show problem factors and store default widgets to this dict
        self.__show_data_farming_core(
            solver,
            frame=self.tk_frames["ntbk.ps_adding.solver.factors.solvers"],
            row=1,
        )

        # Update the design name to be unique
        unique_name = self.get_unique_name(self.root_solver_dict, solver.name)
        self.design_name.set(unique_name)
        self.tk_entries["design_opts.name"].delete(0, tk.END)
        self.tk_entries["design_opts.name"].insert(0, unique_name)

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
        self, base_object: Solver | Problem, frame: ttk.Frame, row: int = 1
    ) -> None:
        """Show data farming options for a solver or problem.

        Parameters
        ----------
        base_object : Solver or Problem
            Solver or Problem object.

        """
        # Check if the base object is a Problem or Solver
        if not isinstance(base_object, (Problem, Solver)):
            error_msg = "base_object must be a Problem or Solver object."
            error_msg += f" Received {type(base_object)}."
            raise TypeError(error_msg)

        # Grab the specifications from the base object
        specifications = base_object.specifications
        # If we're dealing with a Problem, we also need to grab the
        # specifications for the model
        if isinstance(base_object, Problem):
            model_specifications = base_object.model.specifications
            for factor in model_specifications.keys():
                specifications[factor] = model_specifications[factor]
        # Convert the specifications to a dictionary of DFFactor objects
        self.factor_dict = spec_dict_to_df_dict(specifications)

        # Add all the column headers
        self.__insert_factor_headers(frame=frame)
        # Add all the factors
        self.__insert_factors(
            frame=frame,
            factor_dict=self.factor_dict,
        )

        # Set all the columns to automatically expand if there's room
        for i in range(len(self.factor_dict) + 1):
            frame.grid_rowconfigure(i, weight=1)

    def __create_design_core(self, base_object: str) -> None:
        # Check if the base object is a Problem or Solver
        if base_object not in ("Problem", "Solver"):
            error_msg = "base_object must be 'Problem' or 'Solver'."
            error_msg += f" Received {base_object}."
            raise TypeError(error_msg)

        base_dropdown = (
            self.selected_problem_name.get()
            if base_object == "Problem"
            else self.selected_solver_name.get()
        )
        root_dict = (
            self.root_problem_dict
            if base_object == "Problem"
            else self.root_solver_dict
        )

        # Check to see if the user has selected a problem or solver
        if base_dropdown == "":
            messagebox.showerror(
                "Error",
                f"Please select a {base_object} from the dropdown list.",
            )
            return
        # Check to see if the design name already exists
        if self.design_name.get() in root_dict:
            # Generate a new name
            new_name = self.get_unique_name(root_dict, self.design_name.get())
            # Ask the user if they want to use the new name
            prompt_text = (
                f"A {base_object} with the name {self.design_name.get()}"
            )
            prompt_text += " already exists. Would you like to use the name "
            prompt_text += f"{new_name} instead?\nNote: If you choose 'No',"
            prompt_text += " you will need to choose a different name."
            use_new_name = messagebox.askyesno(
                "Name Exists",
                prompt_text,
            )
            if use_new_name:
                self.design_name.set(new_name)
            else:
                return

        # Get the name of the design
        design_name = self.design_name.get()
        # Get the number of stacks and the type of design
        num_stacks = self.design_num_stacks.get()
        design_type = self.design_type.get()
        # Extract the name of the problem or solver from the dropdown box
        base_name = base_dropdown.split(" - ")[0]

        """ Determine factors included in design """
        # List of names of factors included in the design
        self.design_factors: list[str] = []
        # Dict of cross design factors w/ lists of possible values
        self.cross_design_factors: dict[str, list[str]] = {}
        # Dict of factors not included in the design
        # Key is the factor name, value is the default value
        self.fixed_factors: dict[str, object] = {}
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
            # Create a non-datafarmed design
            design_list = []
            design_list.append(self.fixed_factors)
            design_list.append(base_name)

            # Add the design to the list display
            if base_object == "Problem":
                self.add_problem_to_curr_exp(design_name, [design_list])
            else:
                self.add_solver_to_curr_exp(design_name, [design_list])

            # Refresh the design name entry box
            self.design_name.set(self.get_unique_name(root_dict, design_name))
        else:
            # Create the factor settings txt file
            # Check if the folder exists, if not create it
            if not os.path.exists(DATA_FARMING_DIR):
                os.makedirs(DATA_FARMING_DIR)
            # If the file already exists, clear it and make a new, empty file of the same name
            filepath = os.path.join(DATA_FARMING_DIR, f"{design_name}.txt")
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
                # Create the design
                if base_object == "Problem":
                    self.problem_design_list = create_design(
                        name=base_name,
                        factor_headers=self.design_factors,
                        factor_settings_filename=design_name,
                        fixed_factors=self.fixed_factors,
                        cross_design_factors=self.cross_design_factors,
                        n_stacks=num_stacks,
                        design_type=design_type,  # type: ignore
                        class_type="problem",
                    )
                else:
                    self.solver_design_list = create_design(
                        name=base_name,
                        factor_headers=self.design_factors,
                        factor_settings_filename=design_name,
                        fixed_factors=self.fixed_factors,
                        cross_design_factors=self.cross_design_factors,
                        n_stacks=num_stacks,
                        design_type=design_type,  # type: ignore
                        class_type="solver",
                    )
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"An error occurred while creating the design: {e}",
                )
                return

            # Display the design tree
            filename = os.path.join(
                DATA_FARMING_DIR, f"{design_name}_design.csv"
            )
            self.display_design_tree(
                csv_filename=filename,
            )
            # Button to add the design to the experiment
            command = (
                self.add_problem_design_to_experiment
                if base_object == "Problem"
                else self.add_solver_design_to_experiment
            )
            self.tk_buttons["gen_design.add"] = ttk.Button(
                master=self.tk_frames["gen_design.display"],
                text=f"Add this {base_object} design to experiment",
                command=command,
            )
            self.tk_buttons["gen_design.add"].grid(
                row=1, column=0, sticky="nsew"
            )

    def create_solver_design(self) -> None:
        self.__create_design_core("Solver")

    def create_problem_design(self) -> None:
        self.__create_design_core("Problem")

    def display_design_tree(
        self,
        csv_filename: str,
        master_frame: ttk.Frame | None = None,
    ) -> None:
        if master_frame is None:
            master_frame = self.tk_frames["gen_design.display"]
        # Read the design table from the csv file
        design_table = pd.read_csv(csv_filename, index_col="design_num")
        # Drop the last 3 columns from the design table since they're not needed
        design_table.drop(
            columns=["name", "design_type", "num_stacks"], inplace=True
        )

        # Unhide the generated design frame
        self._show_gen_design()

        # Modify the header to show the # of design points and # of duplicates
        unique_design_points = design_table.drop_duplicates().shape[0]
        num_duplicates = design_table.shape[0] - unique_design_points
        plural_s = "" if num_duplicates == 1 else "s"
        self.tk_labels["gen_design.header"].configure(
            text=f"Generated Design - {len(design_table)} Design Points ({num_duplicates} Duplicate{plural_s})"
        )

        self.design_tree = ttk.Treeview(master=master_frame)
        self.design_tree.grid(row=0, column=0, sticky="nsew")
        self.style = ttk.Style()
        self.style.configure(
            "Treeview.Heading",
            font=nametofont("TkHeadingFont"),
        )
        self.design_tree.heading("#0", text="Design #")
        # Change row height to 30
        std_font_size = nametofont("TkDefaultFont").cget("size")
        # TODO: see if this is the right scaling
        height = 30 * std_font_size / 12
        self.style.configure("Treeview", rowheight=int(height))

        # Enter design values into treeview
        self.design_tree["columns"] = tuple(design_table.columns)

        for column in design_table.columns:
            self.design_tree.heading(column, text=column)
            self.design_tree.column(column, width=100)

        for index, row in design_table.iterrows():
            # TODO: figure out a better (non-warning raising) way to do this
            self.design_tree.insert("", index, text=index, values=tuple(row))  # type: ignore

        # Set the size of each column to the width of the longest entry
        for column in design_table.columns:
            max_width = max(
                design_table[column].astype(str).map(len).max(), len(column)
            )
            header_font_size = nametofont("TkHeadingFont").cget("size")
            width = max_width * header_font_size * 0.8 + 10
            self.design_tree.column(column, width=int(width))

    def add_problem_design_to_experiment(self) -> None:
        design_name = self.design_name.get()
        selected_name = self.selected_problem_name.get()
        selected_name_short = selected_name.split(" - ")[0]

        problem_holder_list = []  # holds all problem lists within design name
        for dp in self.problem_design_list:
            problem_list = []  # holds dictionary of dps and solver name
            problem_list.append(dp)
            problem_list.append(selected_name_short)
            problem_holder_list.append(problem_list)

        # Add the problem to the current experiment
        self.add_problem_to_curr_exp(design_name, problem_holder_list)

        # refresh problem design name entry box
        self.design_name.set(
            self.get_unique_name(self.root_problem_dict, design_name)
        )

        # Hide the design tree
        self._hide_gen_design()

    def add_solver_design_to_experiment(self) -> None:
        design_name = self.design_name.get()
        selected_name = self.selected_solver_name.get()
        selected_name_short = selected_name.split(" - ")[0]

        solver_holder_list = []  # used so solver list matches datafarming format
        for dp in self.solver_design_list:
            solver_list = []  # holds dictionary of dps and solver name
            solver_list.append(dp)
            solver_list.append(selected_name_short)
            solver_holder_list.append(solver_list)

        # Add solver row to list display
        self.add_solver_to_curr_exp(design_name, solver_holder_list)

        # refresh solver design name entry box
        self.design_name.set(
            self.get_unique_name(self.root_solver_dict, design_name)
        )

        # Hide the design tree
        self._hide_gen_design()

    def edit_problem(self, problem_save_name: str) -> None:
        error_msg = "Edit problem function not yet implemented."
        messagebox.showerror("Error", error_msg)

    def edit_solver(self, solver_save_name: str) -> None:
        error_msg = "Edit solver function not yet implemented."
        messagebox.showerror("Error", error_msg)

    def delete_problem(self, problem_name: str) -> None:
        # Delete from master list
        del self.root_problem_dict[problem_name]
        # Delete label & edit/delete buttons
        tk_base_name = f"curr_exp.lists.problems.{problem_name}"
        tk_lbl_name = f"{tk_base_name}.name"
        tk_edit_name = f"{tk_base_name}.edit"
        tk_del_name = f"{tk_base_name}.del"
        self.tk_labels[tk_lbl_name].destroy()
        self.tk_buttons[tk_edit_name].destroy()
        self.tk_buttons[tk_del_name].destroy()
        del self.tk_labels[tk_lbl_name]
        del self.tk_buttons[tk_edit_name]
        del self.tk_buttons[tk_del_name]
        # Clear the canvas
        self._destroy_widget_children(
            self.tk_canvases["curr_exp.lists.problems"]
        )
        # Add all the problems back to the canvas
        for _, problem_group_name in enumerate(self.root_problem_dict):
            self.add_problem_to_curr_exp_list(problem_group_name)
        # Rerun compatibility check
        self.cross_design_solver_compatibility()

    def delete_solver(self, unique_solver_name: str) -> None:
        # Delete from master list
        del self.root_solver_dict[unique_solver_name]
        # Delete label & edit/delete buttons
        tk_base_name = f"curr_exp.lists.solvers.{unique_solver_name}"
        tk_lbl_name = f"{tk_base_name}.name"
        tk_edit_name = f"{tk_base_name}.edit"
        tk_del_name = f"{tk_base_name}.del"
        self.tk_labels[tk_lbl_name].destroy()
        self.tk_buttons[tk_edit_name].destroy()
        self.tk_buttons[tk_del_name].destroy()
        del self.tk_labels[tk_lbl_name]
        del self.tk_buttons[tk_edit_name]
        del self.tk_buttons[tk_del_name]
        # Clear the canvas
        self._destroy_widget_children(
            self.tk_canvases["curr_exp.lists.solvers"]
        )
        # Add all the solvers back to the canvas
        for _, solver_group_name in enumerate(self.root_solver_dict):
            self.add_solver_to_curr_exp_list(solver_group_name)
        # Rerun compatibility check
        self.cross_design_problem_compatibility()

    def create_experiment(self) -> None:
        # Check to make sure theres at least one problem and solver
        if len(self.root_solver_dict) == 0 or len(self.root_problem_dict) == 0:
            messagebox.showerror(
                "Error",
                "Please add at least one solver and one problem to the experiment.",
            )
            return

        # get unique experiment name
        old_name = self.curr_exp_name.get()
        self.experiment_name = self.get_unique_name(
            self.root_experiment_dict, old_name
        )
        self.curr_exp_name.set(self.experiment_name)

        # get pickle checkstate
        pickle_checkstate = self.curr_exp_is_pickled.get()

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

        # add exp to row
        self.add_exp_row()

        # Clear the current experiment
        self.clear_experiment()

    def clear_experiment(self) -> None:
        # Clear dictionaries
        self.root_problem_dict = {}
        self.root_solver_dict = {}
        # Clear the GUI lists
        self._destroy_widget_children(
            self.tk_canvases["curr_exp.lists.problems"]
        )
        self._destroy_widget_children(
            self.tk_canvases["curr_exp.lists.solvers"]
        )
        # Reset the experiment name
        new_name = self.get_unique_name(
            self.root_experiment_dict, self.DEFAULT_EXP_NAME
        )
        self.curr_exp_name.set(new_name)
        # Set default pickle checkstate
        self.curr_exp_is_pickled.set(self.DEFAULT_EXP_CHECK)
        # Run all update functions for ensuring compatible options now that the
        # experiment is empty
        self.cross_design_problem_compatibility()
        self.cross_design_solver_compatibility()
        self.__update_problem_dropdown()
        self.__update_solver_dropdown()

    def add_exp_row(self) -> None:
        """Display experiment in list."""
        experiment_row = self.tk_canvases["exps.list"].grid_size()[1]
        self.current_experiment_frame = tk.Frame(
            master=self.tk_canvases["exps.list"]
        )
        self.current_experiment_frame.grid(
            row=experiment_row, column=0, sticky="nsew"
        )
        self.current_experiment_frame.grid_columnconfigure(0, weight=0)
        self.current_experiment_frame.grid_columnconfigure(1, weight=1)
        self.current_experiment_frame.grid_columnconfigure(2, weight=1)
        self.current_experiment_frame.grid_columnconfigure(3, weight=1)
        self.current_experiment_frame.grid_columnconfigure(4, weight=1)

        name_text_step_0 = self.experiment_name + "\n(Initialized)"
        name_text_step_1 = self.experiment_name + "\n(Ran)"
        name_text_step_2 = self.experiment_name + "\n(Post-Replicated)"
        name_text_step_3 = self.experiment_name + "\n(Post-Normalized)"
        name_text_step_4 = self.experiment_name + "\n(Logged)"

        name_base = "exp." + self.experiment_name
        lbl_name = name_base + ".name"
        action_bttn_name = name_base + ".action"
        all_bttn_name = name_base + ".all"
        opt_bttn_name = name_base + ".options"
        del_bttn_name = name_base + ".delete"

        self.tk_labels[lbl_name] = ttk.Label(
            master=self.current_experiment_frame,
            text=name_text_step_0,
            justify="center",
            anchor="center",
        )
        self.tk_labels[lbl_name].grid(
            row=0, column=0, padx=5, pady=5, sticky="nsew"
        )

        bttn_text_run_all = "Run All\nRemaining Steps"
        bttn_text_run = "Run\nExperiment"
        bttn_text_running = "Running..."
        bttn_text_post_process = "Post-Replicate"
        bttn_text_post_processing = "Post-Replicating..."
        bttn_text_post_norm = "Post-Normalize"
        bttn_text_post_normalizing = "Post-Normalizing..."
        bttn_text_log = "Log\nResults"
        bttn_text_logging = "Logging..."
        bttn_text_done = "Done"

        def exp_run(name: str) -> bool:
            action_button = self.tk_buttons[action_bttn_name]
            action_button.configure(text=bttn_text_running, state="disabled")
            self.update()
            try:
                self.run_experiment(experiment_name=name)
                action_button.configure(
                    text=bttn_text_post_process,
                    state="normal",
                    command=lambda name=name: exp_post_process(name),
                )
                self.tk_labels[lbl_name].configure(text=name_text_step_1)
                return True
            except Exception as e:
                messagebox.showerror("Error", str(e))
                action_button.configure(text=bttn_text_run, state="normal")
                return False

        def exp_post_process(name: str) -> bool:
            action_button = self.tk_buttons[action_bttn_name]
            action_button.configure(
                text=bttn_text_post_processing, state="disabled"
            )
            self.update()
            try:
                self.post_process(experiment_name=name)
                action_button.configure(
                    text=bttn_text_post_norm,
                    state="normal",
                    command=lambda name=name: exp_post_norm(name),
                )
                self.tk_labels[lbl_name].configure(text=name_text_step_2)
                return True
            except Exception as e:
                messagebox.showerror("Error", str(e))
                action_button.configure(
                    text=bttn_text_post_process, state="normal"
                )
                return False

        def exp_post_norm(name: str) -> bool:
            action_button = self.tk_buttons[action_bttn_name]
            action_button.configure(
                text=bttn_text_post_normalizing, state="disabled"
            )
            self.update()
            try:
                self.post_normalize(experiment_name=name)
                action_button.configure(
                    text=bttn_text_log,
                    state="normal",
                    command=lambda name=name: exp_log(name),
                )
                self.tk_labels[lbl_name].configure(text=name_text_step_3)
                return True
            except Exception as e:
                messagebox.showerror("Error", str(e))
                action_button.configure(
                    text=bttn_text_post_norm, state="normal"
                )
                return False

        def exp_log(name: str) -> bool:
            action_button = self.tk_buttons[action_bttn_name]
            action_button.configure(text=bttn_text_logging, state="disabled")
            self.update()
            try:
                self.log_results(experiment_name=name)
                action_button.configure(text=bttn_text_done, state="disabled")
                self.tk_buttons[all_bttn_name].configure(state="disabled")
                self.tk_labels[lbl_name].configure(text=name_text_step_4)
                return True
            except Exception as e:
                messagebox.showerror("Error", str(e))
                action_button.configure(text=bttn_text_log, state="normal")
                return False

        def exp_all(name: str) -> None:
            self.tk_buttons[all_bttn_name].configure(state="disabled")
            if (
                not exp_run(name)
                or not exp_post_process(name)
                or not exp_post_norm(name)
                or not exp_log(name)
            ):
                # We already printed the error message in the individual steps
                self.tk_buttons[all_bttn_name].configure(state="normal")

        def delete_experiment(
            experiment_name: str, experiment_frame: tk.Frame
        ) -> None:
            del self.root_experiment_dict[experiment_name]
            experiment_frame.destroy()

        # Setup initial action button state
        self.tk_buttons[action_bttn_name] = ttk.Button(
            master=self.current_experiment_frame,
            text=bttn_text_run,
            command=lambda name=self.experiment_name: exp_run(name),
        )
        self.tk_buttons[action_bttn_name].grid(
            row=0, column=1, padx=5, pady=5, sticky="nsew"
        )

        # all in one
        self.tk_buttons[all_bttn_name] = ttk.Button(
            master=self.current_experiment_frame,
            text=bttn_text_run_all,
            command=lambda name=self.experiment_name: exp_all(name),
        )
        self.tk_buttons[all_bttn_name].grid(
            row=0, column=2, padx=5, pady=5, sticky="nsew"
        )
        # experiment options button
        self.tk_buttons[opt_bttn_name] = ttk.Button(
            master=self.current_experiment_frame,
            text="Options",
            command=lambda name=self.experiment_name: self.open_post_processing_window(
                name
            ),
        )
        self.tk_buttons[opt_bttn_name].grid(
            row=0, column=3, padx=5, pady=5, sticky="nsew"
        )
        # delete experiment
        self.tk_buttons[del_bttn_name] = ttk.Button(
            master=self.current_experiment_frame,
            text="Delete",
            command=lambda name=self.experiment_name,
            frame=self.current_experiment_frame: delete_experiment(name, frame),
        )
        self.tk_buttons[del_bttn_name].grid(
            row=0, column=4, padx=5, pady=5, sticky="nsew"
        )

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

    def open_defaults_window(self) -> None:
        # create new winow
        self.experiment_defaults_window = Toplevel(self)
        self.experiment_defaults_window.title(
            "Simopt Graphical User Interface - Experiment Options Defaults"
        )
        self.center_window(0.8)
        self.set_style()

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
        self.solve_tol_1_var.set(str(self.solve_tols_default[0]))
        self.solve_tol_2_var.set(str(self.solve_tols_default[1]))
        self.solve_tol_3_var.set(str(self.solve_tols_default[2]))
        self.solve_tol_4_var.set(str(self.solve_tols_default[3]))
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

    # Functionally the same as the below function, but for boolean values
    def _find_option_setting_bool(
        self,
        exp_name: str,
        search_dict: dict[str, tk.BooleanVar],
        default_val: bool,
    ) -> bool:
        if exp_name in search_dict:
            return search_dict[exp_name].get()
        return default_val

    # Functionally the same as the above function, but for integers
    def _find_option_setting_int(
        self,
        exp_name: str,
        search_dict: dict[str, tk.IntVar],
        default_val: int,
    ) -> int:
        if exp_name in search_dict:
            return search_dict[exp_name].get()
        return default_val

    def open_post_processing_window(self, experiment_name: str) -> None:
        # check if options have already been set
        n_macroreps: int = self._find_option_setting_int(
            experiment_name, self.macro_reps, self.macro_default
        )
        n_postreps: int = self._find_option_setting_int(
            experiment_name, self.post_reps, self.post_default
        )
        crn_budget = self._find_option_setting_bool(
            experiment_name, self.crn_budgets, self.crn_budget_default
        )
        crn_macro = self._find_option_setting_bool(
            experiment_name, self.crn_macros, self.crn_macro_default
        )
        n_initreps = self._find_option_setting_int(
            experiment_name, self.init_post_reps, self.init_default
        )
        crn_init = self._find_option_setting_bool(
            experiment_name, self.crn_inits, self.crn_init_default
        )
        if experiment_name in self.solve_tols:
            solve_tols = []
            for tol in self.solve_tols[experiment_name]:
                solve_tols.append(tol.get())
        else:
            solve_tols = self.solve_tols_default

        # create new winow
        self.post_processing_window = Toplevel(self.root)
        self.post_processing_window.title(
            "Simopt Graphical User Interface - Experiment Options"
        )
        self.center_window(0.8)
        self.set_style()

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
        solve_tol_1_str = str(solve_tols[0])
        solve_tol_2_str = str(solve_tols[1])
        solve_tol_3_str = str(solve_tols[2])
        solve_tol_4_str = str(solve_tols[3])
        self.solve_tol_1_var.set(solve_tol_1_str)
        self.solve_tol_2_var.set(solve_tol_2_str)
        self.solve_tol_3_var.set(solve_tol_3_str)
        self.solve_tol_4_var.set(solve_tol_4_str)
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

    def open_plotting_window(self) -> None:
        # create new window
        self.plotting_window = Toplevel(self)
        self.plotting_window.title(
            "Simopt Graphical User Interface - Experiment Plots"
        )
        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        self.center_window(0.8)
        self.set_style()

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
                self.solver_tree.insert("", index, text=str(index), values=row)
        else:
            self.solver_tree["columns"] = [
                "Solver Name"
            ]  # set columns just to solver name
            self.solver_tree.heading("Solver Name", text="Solver Name")
            for index, solver in enumerate(self.plot_experiment.solvers):
                self.solver_tree.insert(
                    "", index, text=str(index), values=[solver.name]
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
                self.problem_tree.insert("", index, text=str(index), values=row)
        else:
            self.problem_tree["columns"] = ["Problem Name"]
            self.problem_tree.heading(
                "Problem Name", text="Problem Name"
            )  # set heading for name column
            for index, problem in enumerate(self.plot_experiment.problems):
                self.problem_tree.insert(
                    "", index, text=str(index), values=[problem.name]
                )

    def show_plot_options(self, plot_type: str) -> None:
        self._destroy_widget_children(self.more_options_frame)
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
            self.solve_tol_var.set("0.1")  # default value
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
            self.con_level_entry.configure(state="disabled")
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
                text = "problems and solvers"
            elif len(self.selected_solvers) == 0:
                text = "solvers"
            else:
                text = "problems"

            # show popup message
            messagebox.showerror("Error", f"Please select {text} to plot.")
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
                    messagebox.showerror(
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
        self._destroy_widget_children(self.edit_x_axis_frame)
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
            messagebox.showerror(
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
