from __future__ import annotations

import os
from pathlib import Path
import pickle
import re
import threading
import tkinter as tk
from abc import ABCMeta
from tkinter import filedialog, messagebox, ttk
from tkinter.font import nametofont
from typing import Callable, Final, Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageTk

import simopt.directory as directory
from simopt.base import Problem, Solver
from simopt.data_farming_base import DATA_FARMING_DIR
from simopt.experiment_base import (
    PlotType,
    ProblemSolver,
    ProblemsSolvers,
    create_design,
    create_design_list_from_table,
    plot_area_scatterplots,
    plot_progress_curves,
    plot_solvability_cdfs,
    plot_solvability_profiles,
    plot_terminal_progress,
    plot_terminal_scatterplots,
)
from simopt.gui.df_object import DFFactor, spec_dict_to_df_dict
from simopt.gui.toplevel_custom import Toplevel

# Workaround for AutoAPI
problem_directory = directory.problem_directory
solver_directory = directory.solver_directory
problem_unabbreviated_directory = directory.problem_unabbreviated_directory
solver_unabbreviated_directory = directory.solver_unabbreviated_directory


class NewExperimentWindow(Toplevel):
    """New Experiment Window."""

    # Constants
    DEFAULT_NUM_STACKS: Final[int] = 1
    DEFAULT_EXP_NAME: Final[str] = "Experiment"
    DEFAULT_EXP_CHECK: Final[bool] = False

    def __init__(self, root: tk.Tk) -> None:
        """Initialize New Experiment Window."""
        super().__init__(
            root,
            title="Simulation Optimization Experiments",
            exit_on_close=True,
        )

        self.center_window(0.8)

        # TODO: integrate this into a class somewhere so it's not based in
        # the GUI
        self.design_types: Final[list[str]] = ["nolhs"]

        # Dictionary to store all the experiments in the GUI
        self.root_experiment_dict: dict[str, ProblemsSolvers] = {}
        # Dictionary to store all the problems and solvers for the current experiment
        # for each name of solver or solver design has list that
        # includes: [[solver factors], solver name]
        self.root_solver_dict: dict[str, list[list]] = {}
        # for each name of solver or solver design has list that
        # includes: [[problem factors], [model factors], problem name]
        self.root_problem_dict: dict[str, list[list]] = {}

        # Dictionaries to keep track of custom settings for each experiment
        # If a custom setting is not found (not in the dictionary), the default
        # setting is used
        # dict that contains user specified macroreps for each experiment
        self.custom_macro_reps: dict[str, tk.IntVar] = {}
        # dict that contains user specified postrep numbers for each experiment
        self.custom_post_reps: dict[str, tk.IntVar] = {}
        # dict that contains number of postreps to take at initial & optimal solution for normalization for each experiment
        self.custom_init_post_reps: dict[str, tk.IntVar] = {}
        # contains bool val for if crn is used across budget for each experiment
        self.custom_crn_budgets: dict[str, tk.StringVar] = {}
        # contains bool val for if crn is used across macroreps for each experiment
        self.custom_crn_macros: dict[str, tk.StringVar] = {}
        # contains bool val for if crn is used across initial and optimal solution for each experiment
        self.custom_crn_inits: dict[str, tk.StringVar] = {}
        # solver tolerance gaps for each experiment (inserted as list)
        # TODO: add checks to ensure that solve_tol lists are always 4 long
        self.custom_solve_tols: dict[str, list[tk.StringVar]] = {}

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
        self.problem_full_name_to_class = {
            f"{problem.class_name_abbr}  --  {key}": problem
            for key, problem in problem_unabbreviated_directory.items()
        }
        self.solver_full_name_to_class = {
            f"{solver.class_name_abbr}  --  {key}": solver
            for key, solver in solver_unabbreviated_directory.items()
        }
        # Current exp variables
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
        self.tk_scrollbars: dict[str, ttk.Scrollbar] = {}
        self.tk_separators: dict[str, ttk.Separator] = {}
        self.tk_var_bools: dict[str, tk.BooleanVar] = {}

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
        self.tk_frames["main"].grid_columnconfigure(0, weight=3)
        self.tk_frames["main"].grid_columnconfigure(1, weight=5)
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
        self.tk_frames["exps.list_canvas"] = ttk.Frame(
            self.tk_frames["exps"],
        )
        self.tk_frames["exps.list_canvas"].grid(row=1, column=0, sticky="nsew")
        self.tk_frames["exps.list_canvas"].grid_columnconfigure(0, weight=1)
        self.tk_frames["exps.list_canvas"].grid_rowconfigure(0, weight=1)
        self.tk_canvases["exps.list_canvas"] = tk.Canvas(
            self.tk_frames["exps.list_canvas"],
        )
        self.tk_canvases["exps.list_canvas"].grid(row=0, column=0, sticky="nsew")
        self.__update_exp_list_scroll_region()
        self.tk_frames["exps.list_canvas.list"] = ttk.Frame(
            self.tk_canvases["exps.list_canvas"],
        )
        self.tk_canvases["exps.list_canvas"].create_window(
            (0, 0),
            window=self.tk_frames["exps.list_canvas.list"],
            anchor="nw",
        )
        self.tk_scrollbars["exps.list_canvas_vert"] = ttk.Scrollbar(
            self.tk_frames["exps.list_canvas"],
            orient="vertical",
            command=self.tk_canvases["exps.list_canvas"].yview,
        )
        self.tk_canvases["exps.list_canvas"].config(
            yscrollcommand=self.tk_scrollbars["exps.list_canvas_vert"].set
        )
        self.tk_scrollbars["exps.list_canvas_vert"].grid(row=0, column=1, sticky="ns")
        self.tk_scrollbars["exps.list_canvas_horiz"] = ttk.Scrollbar(
            self.tk_frames["exps.list_canvas"],
            orient="horizontal",
            command=self.tk_canvases["exps.list_canvas"].xview,
        )
        self.tk_canvases["exps.list_canvas"].config(
            xscrollcommand=self.tk_scrollbars["exps.list_canvas_horiz"].set
        )
        self.tk_scrollbars["exps.list_canvas_horiz"].grid(row=1, column=0, sticky="ew")
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
        self.tk_buttons["exps.fields.default_opts"].grid(row=0, column=0, sticky="ew")
        self.tk_buttons["exps.fields.open_plot_win"] = ttk.Button(
            self.tk_frames["exps.fields"],
            text="Open Plotting Window",
            command=self.open_plotting_window,
        )
        self.tk_buttons["exps.fields.open_plot_win"].grid(row=2, column=0, sticky="ew")
        self.tk_buttons["exps.fields.load_exp"] = ttk.Button(
            self.tk_frames["exps.fields"],
            text="Load Experiment",
            command=self.load_experiment,
        )
        self.tk_buttons["exps.fields.load_exp"].grid(row=3, column=0, sticky="ew")

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
        # Only let the columns with lists expand
        curr_exp_list_uniformity = "curr_exp_list_col_size_uniformity"
        self.tk_frames["curr_exp.lists"].grid_columnconfigure(
            0, weight=1, uniform=curr_exp_list_uniformity
        )
        self.tk_frames["curr_exp.lists"].grid_columnconfigure(
            3, weight=1, uniform=curr_exp_list_uniformity
        )
        self.tk_frames["curr_exp.lists"].grid_rowconfigure(1, weight=1)

        self.tk_labels["curr_exp.lists.problem_header"] = ttk.Label(
            self.tk_frames["curr_exp.lists"],
            text="Problems",
            anchor="center",
        )
        self.tk_labels["curr_exp.lists.problem_header"].grid(
            row=0, column=0, columnspan=2, sticky="nsew"
        )
        self.tk_separators["curr_exp.lists"] = ttk.Separator(
            self.tk_frames["curr_exp.lists"], orient="vertical"
        )
        self.tk_separators["curr_exp.lists"].grid(
            row=0, column=2, rowspan=2, sticky="ns", padx=10
        )
        self.tk_labels["curr_exp.lists.solver_header"] = ttk.Label(
            self.tk_frames["curr_exp.lists"], text="Solvers", anchor="center"
        )
        self.tk_labels["curr_exp.lists.solver_header"].grid(
            row=0, column=3, columnspan=2, sticky="nsew"
        )

        self.tk_canvases["curr_exp.lists.problems"] = tk.Canvas(
            self.tk_frames["curr_exp.lists"],
        )
        self.tk_canvases["curr_exp.lists.problems"].grid(row=1, column=0, sticky="nsew")
        self.tk_frames["curr_exp.lists.problems"] = ttk.Frame(
            self.tk_canvases["curr_exp.lists.problems"],
        )
        self.tk_canvases["curr_exp.lists.problems"].create_window(
            (0, 0),
            window=self.tk_frames["curr_exp.lists.problems"],
            anchor="nw",
        )
        self.tk_scrollbars["curr_exp.lists.problems_vert"] = ttk.Scrollbar(
            self.tk_frames["curr_exp.lists"],
            orient="vertical",
            command=self.tk_canvases["curr_exp.lists.problems"].yview,
        )
        self.tk_canvases["curr_exp.lists.problems"].config(
            yscrollcommand=self.tk_scrollbars["curr_exp.lists.problems_vert"].set
        )
        self.tk_scrollbars["curr_exp.lists.problems_vert"].grid(
            row=1, column=1, sticky="ns"
        )
        self.tk_scrollbars["curr_exp.lists.problems_horiz"] = ttk.Scrollbar(
            self.tk_frames["curr_exp.lists"],
            orient="horizontal",
            command=self.tk_canvases["curr_exp.lists.problems"].xview,
        )
        self.tk_canvases["curr_exp.lists.problems"].config(
            xscrollcommand=self.tk_scrollbars["curr_exp.lists.problems_horiz"].set
        )
        self.tk_scrollbars["curr_exp.lists.problems_horiz"].grid(
            row=2, column=0, sticky="ew"
        )
        self.__update_problem_list_scroll_region()

        self.tk_canvases["curr_exp.lists.solvers"] = tk.Canvas(
            self.tk_frames["curr_exp.lists"],
        )
        self.tk_canvases["curr_exp.lists.solvers"].grid(row=1, column=3, sticky="nsew")
        self.tk_frames["curr_exp.lists.solvers"] = ttk.Frame(
            self.tk_canvases["curr_exp.lists.solvers"],
        )
        self.tk_canvases["curr_exp.lists.solvers"].create_window(
            (0, 0),
            window=self.tk_frames["curr_exp.lists.solvers"],
            anchor="nw",
        )
        self.tk_scrollbars["curr_exp.lists.solvers_vert"] = ttk.Scrollbar(
            self.tk_frames["curr_exp.lists"],
            orient="vertical",
            command=self.tk_canvases["curr_exp.lists.solvers"].yview,
        )
        self.tk_canvases["curr_exp.lists.solvers"].config(
            yscrollcommand=self.tk_scrollbars["curr_exp.lists.solvers_vert"].set
        )
        self.tk_scrollbars["curr_exp.lists.solvers_vert"].grid(
            row=1, column=4, sticky="ns"
        )
        self.tk_scrollbars["curr_exp.lists.solvers_horiz"] = ttk.Scrollbar(
            self.tk_frames["curr_exp.lists"],
            orient="horizontal",
            command=self.tk_canvases["curr_exp.lists.solvers"].xview,
        )
        self.tk_canvases["curr_exp.lists.solvers"].config(
            xscrollcommand=self.tk_scrollbars["curr_exp.lists.solvers_horiz"].set
        )
        self.tk_scrollbars["curr_exp.lists.solvers_horiz"].grid(
            row=2, column=3, sticky="ew"
        )
        self.__update_solver_list_scroll_region()

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
        self.tk_labels["curr_exp.fields.exp_name"].grid(row=2, column=0, sticky="ew")
        self.tk_entries["curr_exp.fields.exp_name"] = ttk.Entry(
            self.tk_frames["curr_exp.fields"],
            textvariable=self.curr_exp_name,
        )
        self.tk_entries["curr_exp.fields.exp_name"].grid(row=2, column=1, sticky="ew")
        self.tk_labels["curr_exp.fields.make_pickle"] = ttk.Label(
            self.tk_frames["curr_exp.fields"],
            text="Create Pickles for each pair? ",
            anchor="e",
        )
        self.tk_labels["curr_exp.fields.make_pickle"].grid(row=3, column=0, sticky="ew")
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
        self.tk_frames["ntbk.header"] = ttk.Frame(self.tk_frames["ntbk"])
        self.tk_frames["ntbk.header"].grid(row=0, column=0, sticky="nsew")
        self.tk_frames["ntbk.header"].grid_columnconfigure(0, weight=1)
        self.tk_labels["ntbk.header.title"] = ttk.Label(
            self.tk_frames["ntbk.header"],
            text="Create Problems/Solvers",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["ntbk.header.title"].grid(
            row=0, column=0, rowspan=2, sticky="nsew"
        )
        self.tk_separators["ntbk.header.sep"] = ttk.Separator(
            self.tk_frames["ntbk.header"], orient="vertical"
        )
        self.tk_separators["ntbk.header.sep"].grid(
            row=0, column=1, rowspan=2, sticky="ns", padx=10
        )
        attr_desc_lines = [
            "Objective: Single [S] | Multiple [M]",
            "Constraint: Unconstrained [U] | Box [B] | Deterministic [D] | Stochastic [S]",
            "Variable: Discrete [D] | Continuous [C] | Mixed [M]",
            "Gradient Available: True [G] | False [N]",
        ]
        attribute_desc = "\n".join(attr_desc_lines)
        self.tk_labels["ntbk.header.attr_desc"] = ttk.Label(
            self.tk_frames["ntbk.header"],
            text=attribute_desc,
            anchor="nw",
        )
        self.tk_labels["ntbk.header.attr_desc"].grid(row=0, column=2, sticky="nsew")
        self.tk_labels["ntbk.header.incomp_desc"] = ttk.Label(
            self.tk_frames["ntbk.header"],
            text="incompatible problems/solvers will be unselectable",
            anchor="center",
        )
        self.tk_labels["ntbk.header.incomp_desc"].grid(row=1, column=2, sticky="nsew")
        self.tk_notebooks["ntbk.ps_adding"] = ttk.Notebook(self.tk_frames["ntbk"])
        self.tk_notebooks["ntbk.ps_adding"].grid(row=1, column=0, sticky="nsew")
        self.tk_frames["ntbk.ps_adding.problem"] = ttk.Frame(
            self.tk_notebooks["ntbk.ps_adding"]
        )
        self.tk_notebooks["ntbk.ps_adding"].add(
            self.tk_frames["ntbk.ps_adding.problem"], text="Add Problem"
        )
        self.tk_frames["ntbk.ps_adding.problem"].grid_columnconfigure(1, weight=1)
        self.tk_frames["ntbk.ps_adding.problem"].grid_rowconfigure(1, weight=1)
        self.tk_labels["ntbk.ps_adding.problem.select"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.problem"], text="Selected Problem"
        )
        self.tk_labels["ntbk.ps_adding.problem.select"].grid(row=0, column=0, padx=5)
        # Setting this to readonly prevents the user from typing in the combobox
        self.tk_comboboxes["ntbk.ps_adding.problem.select"] = ttk.Combobox(
            self.tk_frames["ntbk.ps_adding.problem"],
            textvariable=self.selected_problem_name,
            values=sorted(self.problem_full_name_to_class.keys()),
            state="readonly",
        )
        self.tk_comboboxes["ntbk.ps_adding.problem.select"].grid(
            row=0, column=1, sticky="ew", columnspan=2
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
        self.tk_scrollbars["ntbk.ps_adding.problem.factors_vert"] = ttk.Scrollbar(
            self.tk_frames["ntbk.ps_adding.problem"],
            orient="vertical",
            command=self.tk_canvases["ntbk.ps_adding.problem.factors"].yview,
        )
        self.tk_canvases["ntbk.ps_adding.problem.factors"].config(
            yscrollcommand=self.tk_scrollbars["ntbk.ps_adding.problem.factors_vert"].set
        )
        self.tk_scrollbars["ntbk.ps_adding.problem.factors_vert"].grid(
            row=1, column=2, sticky="ns"
        )
        self.tk_scrollbars["ntbk.ps_adding.problem.factors_horiz"] = ttk.Scrollbar(
            self.tk_frames["ntbk.ps_adding.problem"],
            orient="horizontal",
            command=self.tk_canvases["ntbk.ps_adding.problem.factors"].xview,
        )
        self.tk_canvases["ntbk.ps_adding.problem.factors"].config(
            xscrollcommand=self.tk_scrollbars[
                "ntbk.ps_adding.problem.factors_horiz"
            ].set
        )
        self.tk_scrollbars["ntbk.ps_adding.problem.factors_horiz"].grid(
            row=2,
            column=0,
            sticky="ew",
            columnspan=2,
        )
        self.__update_problem_factor_scroll_region()

        self.tk_frames["ntbk.ps_adding.solver"] = ttk.Frame(
            self.tk_notebooks["ntbk.ps_adding"]
        )
        self.tk_notebooks["ntbk.ps_adding"].add(
            self.tk_frames["ntbk.ps_adding.solver"], text="Add Solver"
        )
        self.tk_frames["ntbk.ps_adding.solver"].grid_columnconfigure(1, weight=1)
        self.tk_frames["ntbk.ps_adding.solver"].grid_rowconfigure(1, weight=1)
        self.tk_labels["ntbk.ps_adding.solver.select"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.solver"], text="Selected Solver"
        )
        self.tk_labels["ntbk.ps_adding.solver.select"].grid(row=0, column=0, padx=5)
        # Setting this to readonly prevents the user from typing in the combobox
        self.tk_comboboxes["ntbk.ps_adding.solver.select"] = ttk.Combobox(
            self.tk_frames["ntbk.ps_adding.solver"],
            textvariable=self.selected_solver_name,
            values=sorted(self.solver_full_name_to_class.keys()),
            state="readonly",
        )
        self.tk_comboboxes["ntbk.ps_adding.solver.select"].grid(
            row=0, column=1, sticky="ew", columnspan=2
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
        self.tk_scrollbars["ntbk.ps_adding.solver.factors_vert"] = ttk.Scrollbar(
            self.tk_frames["ntbk.ps_adding.solver"],
            orient="vertical",
            command=self.tk_canvases["ntbk.ps_adding.solver.factors"].yview,
        )
        self.tk_canvases["ntbk.ps_adding.solver.factors"].config(
            yscrollcommand=self.tk_scrollbars["ntbk.ps_adding.solver.factors_vert"].set
        )
        self.tk_scrollbars["ntbk.ps_adding.solver.factors_vert"].grid(
            row=1, column=2, sticky="ns"
        )
        self.tk_scrollbars["ntbk.ps_adding.solver.factors_horiz"] = ttk.Scrollbar(
            self.tk_frames["ntbk.ps_adding.solver"],
            orient="horizontal",
            command=self.tk_canvases["ntbk.ps_adding.solver.factors"].xview,
        )
        self.tk_canvases["ntbk.ps_adding.solver.factors"].config(
            xscrollcommand=self.tk_scrollbars["ntbk.ps_adding.solver.factors_horiz"].set
        )
        self.tk_scrollbars["ntbk.ps_adding.solver.factors_horiz"].grid(
            row=2, column=0, sticky="ew", columnspan=2
        )
        self.__update_solver_factor_scroll_region()

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
        self.__initialize_quick_add()

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
        self.tk_frames["gen_design.display"] = ttk.Frame(self.tk_frames["gen_design"])
        self.tk_frames["gen_design.display"].grid(row=1, column=0, sticky="nsew")
        self.tk_frames["gen_design.display"].grid_columnconfigure(0, weight=1)
        self.tk_frames["gen_design.display"].grid_columnconfigure(1, weight=0)
        self.tk_frames["gen_design.display"].grid_rowconfigure(0, weight=1)

    def __update_canvas_scroll_region(self, canvas_name: str) -> None:
        # Find the canvas
        if canvas_name not in self.tk_canvases:
            error_msg = f"Canvas {canvas_name} not found in GUI"
            raise ValueError(error_msg)
        canvas = self.tk_canvases[canvas_name]
        # Make sure it's up to date before updating the scroll region
        canvas.update()
        canvas.config(scrollregion=canvas.bbox("all"))

    def __update_exp_list_scroll_region(self) -> None:
        self.__update_canvas_scroll_region("exps.list_canvas")

    def __update_problem_list_scroll_region(self) -> None:
        self.__update_canvas_scroll_region("curr_exp.lists.problems")

    def __update_solver_list_scroll_region(self) -> None:
        self.__update_canvas_scroll_region("curr_exp.lists.solvers")

    def __update_problem_factor_scroll_region(self) -> None:
        self.__update_canvas_scroll_region("ntbk.ps_adding.problem.factors")

    def __update_solver_factor_scroll_region(self) -> None:
        self.__update_canvas_scroll_region("ntbk.ps_adding.solver.factors")

    def __update_quick_add_problems_scroll_region(self) -> None:
        self.__update_canvas_scroll_region("ntbk.ps_adding.quick_add.problems")

    def __update_quick_add_solvers_scroll_region(self) -> None:
        self.__update_canvas_scroll_region("ntbk.ps_adding.quick_add.solvers")

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
        self.tk_comboboxes["design_opts.type"].grid(row=1, column=1, sticky="ew")
        self.tk_labels["design_opts.num_stacks"] = ttk.Label(
            self.tk_frames["design_opts"], text="# of Stacks ", anchor="e"
        )
        self.tk_labels["design_opts.num_stacks"].grid(row=2, column=0, sticky="ew")
        self.tk_entries["design_opts.num_stacks"] = ttk.Entry(
            self.tk_frames["design_opts"], textvariable=self.design_num_stacks
        )
        self.tk_entries["design_opts.num_stacks"].grid(row=2, column=1, sticky="ew")
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
        # We start on the first tab (Add Problem) so we need to initialize the
        # problem factors canvas
        self.__refresh_problem_tab()

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

    def __refresh_problem_tab(self) -> None:
        self.selected_problem_name.set("")
        self._enable_design_opts()
        self._destroy_widget_children(
            self.tk_canvases["ntbk.ps_adding.problem.factors"]
        )
        self.tk_buttons["design_opts.generate"].configure(
            text="Generate Problem Design",
            command=self.create_problem_design,
        )
        self.__update_problem_factor_scroll_region()

    def __refresh_solver_tab(self) -> None:
        self.selected_solver_name.set("")
        self._enable_design_opts()
        self._destroy_widget_children(self.tk_canvases["ntbk.ps_adding.solver.factors"])
        self.tk_buttons["design_opts.generate"].configure(
            text="Generate Solver Design", command=self.create_solver_design
        )
        self.__update_solver_factor_scroll_region()

    def __refresh_quick_add_tab(self) -> None:
        self._disable_design_opts()
        self.__initialize_quick_add()
        self.tk_buttons["design_opts.generate"].configure(
            text="Add Cross Design to Experiment",
            command=self.create_cross_design,
        )
        self.__update_quick_add_problems_scroll_region()
        self.__update_quick_add_solvers_scroll_region()

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
            self.__refresh_problem_tab()
        elif tab_name == "Add Solver":
            self.__refresh_solver_tab()
        elif tab_name == "Quick-Add Problems/Solvers":
            self.__refresh_quick_add_tab()
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
        possible_problems = sorted(self.problem_full_name_to_class.keys())
        # For each solver in the current experiment, check all the possible
        # problems and remove the ones that are not compatible
        # Grab the name (index 1) out of the first element (index 0) of the
        # dictionary (looked up by key) to get the solver class name
        solver_class_list = [
            self.root_solver_dict[key][0][1] for key in self.root_solver_dict
        ]
        for solver_name in solver_class_list:
            solver_class: ABCMeta = solver_directory[solver_name]
            solver: Solver = solver_class()
            problem_list = possible_problems.copy()
            for problem_name in problem_list:
                short_problem_name = problem_name.split(" ")[0]
                problem_class: ABCMeta = problem_directory[short_problem_name]
                problem: Problem = problem_class()
                # Create a new ProblemSolver object to check compatibility
                problem_solver = ProblemSolver(problem=problem, solver=solver)
                # If there was an error, remove it from the options
                if len(problem_solver.check_compatibility()) > 0:
                    possible_problems.remove(problem_name)
        self.tk_comboboxes["ntbk.ps_adding.problem.select"].configure(
            values=possible_problems
        )

    def __update_solver_dropdown(self) -> None:
        possible_options = sorted(self.solver_full_name_to_class.keys())
        # For each problem in the current experiment, check all the possible
        # solvers and remove the ones that are not compatible
        # Grab the name (index 1) out of the first element (index 0) of the
        # dictionary (looked up by key) to get the problem class name
        problem_class_list = [
            self.root_problem_dict[key][0][1] for key in self.root_problem_dict
        ]
        for problem_name in problem_class_list:
            problem_class: ABCMeta = problem_directory[problem_name]
            problem: Problem = problem_class()
            solver_list = possible_options.copy()
            for solver_name in solver_list:
                short_solver_name = solver_name.split(" ")[0]
                solver_class: ABCMeta = solver_directory[short_solver_name]
                solver: Solver = solver_class()
                # Create a new ProblemSolver object to check compatibility
                problem_solver = ProblemSolver(problem=problem, solver=solver)
                # If there was an error, remove it from the options
                if len(problem_solver.check_compatibility()) > 0:
                    possible_options.remove(solver_name)
        self.tk_comboboxes["ntbk.ps_adding.solver.select"].configure(
            values=possible_options
        )

    def add_problem_to_curr_exp(
        self, unique_name: str, problem_list: list[list]
    ) -> None:
        self.root_problem_dict[unique_name] = problem_list
        self.add_problem_to_curr_exp_list(unique_name)
        self.__update_solver_dropdown()

    def add_solver_to_curr_exp(self, unique_name: str, solver_list: list[list]) -> None:
        self.root_solver_dict[unique_name] = solver_list
        self.add_solver_to_curr_exp_list(unique_name)
        self.__update_problem_dropdown()

    def __add_item_to_curr_exp_list(
        self,
        unique_name: str,
        list_name: str,
        view_func: Callable,
        del_func: Callable,
    ) -> None:
        # Get all the information needed to add the item to the GUI
        list_name = f"curr_exp.lists.{list_name}"
        base_name = f"{list_name}.{unique_name}"
        parent_frame = self.tk_frames[list_name]
        insert_row = parent_frame.grid_size()[1]
        # Add the name label
        name_label_name = f"{base_name}.name"
        self.tk_labels[name_label_name] = ttk.Label(
            master=parent_frame,
            text=unique_name,
        )
        self.tk_labels[name_label_name].grid(row=insert_row, column=1)
        # Add the view button
        view_button_name = f"{base_name}.view"
        self.tk_buttons[view_button_name] = ttk.Button(
            master=parent_frame,
            text="View",
            command=lambda unique_name=unique_name: view_func(unique_name),
        )
        self.tk_buttons[view_button_name].grid(row=insert_row, column=2)
        # Add the delete button
        del_button_name = f"{base_name}.del"
        self.tk_buttons[del_button_name] = ttk.Button(
            master=parent_frame,
            text="Delete",
            command=lambda unique_name=unique_name: del_func(unique_name),
        )
        self.tk_buttons[del_button_name].grid(row=insert_row, column=3)

    def add_problem_to_curr_exp_list(self, unique_name: str) -> None:
        # Make sure the unique name is in the root problem dict
        if unique_name not in self.root_problem_dict:
            error_msg = f"Problem {unique_name} not found in root problem dict"
            raise ValueError(error_msg)
        # Add the problem to the GUI
        self.__add_item_to_curr_exp_list(
            unique_name,
            "problems",
            self.view_problem_design,
            self.delete_problem,
        )
        self.__update_problem_list_scroll_region()

    def add_solver_to_curr_exp_list(self, unique_name: str) -> None:
        # Make sure the unique name is in the root solver dict
        if unique_name not in self.root_solver_dict:
            error_msg = f"Solver {unique_name} not found in root solver dict"
            raise ValueError(error_msg)
        # Add the solver to the GUI
        self.__add_item_to_curr_exp_list(
            unique_name,
            "solvers",
            self.view_solver_design,
            self.delete_solver,
        )
        self.__update_solver_list_scroll_region()

    def __initialize_quick_add(self) -> None:
        # Delete all existing children of the frame
        for child in self.tk_frames["ntbk.ps_adding.quick_add"].winfo_children():
            child.destroy()
        # Configure the grid layout to expand properly
        self.tk_frames["ntbk.ps_adding.quick_add"].grid_rowconfigure(2, weight=1)
        self.tk_frames["ntbk.ps_adding.quick_add"].grid_columnconfigure(0, weight=2)
        self.tk_frames["ntbk.ps_adding.quick_add"].grid_columnconfigure(3, weight=1)

        # Create labels for the title and the column headers
        title_text = "Select problems/solvers to be included in cross-design."
        title_text += " These will be added with default factor settings."
        self.tk_labels["ntbk.ps_adding.quick_add.title"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            text=title_text,
            anchor="center",
            justify="center",
        )
        self.tk_labels["ntbk.ps_adding.quick_add.title"].grid(
            row=0, column=0, columnspan=5, sticky="ew"
        )
        self.tk_labels["ntbk.ps_adding.quick_add.problems"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            text="Problems",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["ntbk.ps_adding.quick_add.problems"].grid(
            row=1, column=0, sticky="ew", columnspan=2
        )
        self.tk_separators["ntbk.ps_adding.quick_add"] = ttk.Separator(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            orient="vertical",
        )
        self.tk_separators["ntbk.ps_adding.quick_add"].grid(
            row=1, column=2, sticky="ns", rowspan=2, padx=10
        )
        self.tk_labels["ntbk.ps_adding.quick_add.solvers"] = ttk.Label(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            text="Solvers",
            anchor="center",
            font=nametofont("TkHeadingFont"),
        )
        self.tk_labels["ntbk.ps_adding.quick_add.solvers"].grid(
            row=1, column=3, sticky="ew", columnspan=2
        )

        # Create canvases for the problems and solvers
        self.tk_canvases["ntbk.ps_adding.quick_add.problems"] = tk.Canvas(
            self.tk_frames["ntbk.ps_adding.quick_add"]
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.problems"].grid(
            row=2, column=0, sticky="nsew"
        )
        self.tk_scrollbars["ntbk.ps_adding.quick_add.problems_vert"] = ttk.Scrollbar(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            orient="vertical",
            command=self.tk_canvases["ntbk.ps_adding.quick_add.problems"].yview,
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.problems"].config(
            yscrollcommand=self.tk_scrollbars[
                "ntbk.ps_adding.quick_add.problems_vert"
            ].set
        )
        self.tk_scrollbars["ntbk.ps_adding.quick_add.problems_vert"].grid(
            row=2, column=1, sticky="ns"
        )
        self.tk_scrollbars["ntbk.ps_adding.quick_add.problems_horiz"] = ttk.Scrollbar(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            orient="horizontal",
            command=self.tk_canvases["ntbk.ps_adding.quick_add.problems"].xview,
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.problems"].config(
            xscrollcommand=self.tk_scrollbars[
                "ntbk.ps_adding.quick_add.problems_horiz"
            ].set
        )
        self.tk_scrollbars["ntbk.ps_adding.quick_add.problems_horiz"].grid(
            row=3, column=0, sticky="ew"
        )

        self.tk_canvases["ntbk.ps_adding.quick_add.solvers"] = tk.Canvas(
            self.tk_frames["ntbk.ps_adding.quick_add"]
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.solvers"].grid(
            row=2, column=3, sticky="nsew"
        )
        self.tk_scrollbars["ntbk.ps_adding.quick_add.solvers_vert"] = ttk.Scrollbar(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            orient="vertical",
            command=self.tk_canvases["ntbk.ps_adding.quick_add.solvers"].yview,
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.solvers"].config(
            yscrollcommand=self.tk_scrollbars[
                "ntbk.ps_adding.quick_add.solvers_vert"
            ].set
        )
        self.tk_scrollbars["ntbk.ps_adding.quick_add.solvers_vert"].grid(
            row=2, column=4, sticky="ns"
        )
        self.tk_scrollbars["ntbk.ps_adding.quick_add.solvers_horiz"] = ttk.Scrollbar(
            self.tk_frames["ntbk.ps_adding.quick_add"],
            orient="horizontal",
            command=self.tk_canvases["ntbk.ps_adding.quick_add.solvers"].xview,
        )
        self.tk_canvases["ntbk.ps_adding.quick_add.solvers"].config(
            xscrollcommand=self.tk_scrollbars[
                "ntbk.ps_adding.quick_add.solvers_horiz"
            ].set
        )
        self.tk_scrollbars["ntbk.ps_adding.quick_add.solvers_horiz"].grid(
            row=3, column=3, sticky="ew"
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
        sorted_problems = sorted(self.problem_full_name_to_class)
        for problem_name in sorted_problems:
            row = self.tk_frames["ntbk.ps_adding.quick_add.problems_frame"].grid_size()[
                1
            ]
            shortened_name = problem_name.split(" ")[0]
            tk_name = f"ntbk.ps_adding.quick_add.problems_frame.{shortened_name}"
            self.tk_var_bools[tk_name] = tk.BooleanVar()
            self.tk_checkbuttons[tk_name] = ttk.Checkbutton(
                master=self.tk_frames["ntbk.ps_adding.quick_add.problems_frame"],
                text=problem_name,
                variable=self.tk_var_bools[tk_name],
                command=self.cross_design_solver_compatibility,
            )
            self.tk_checkbuttons[tk_name].grid(row=row, column=0, sticky="w", padx=10)
        # display all potential solvers
        sorted_solvers = sorted(self.solver_full_name_to_class)
        for solver_name in sorted_solvers:
            row = self.tk_frames["ntbk.ps_adding.quick_add.solvers_frame"].grid_size()[
                1
            ]
            shortened_name = solver_name.split(" ")[0]
            tk_name = f"ntbk.ps_adding.quick_add.problems_frame.{shortened_name}"
            self.tk_var_bools[tk_name] = tk.BooleanVar()
            self.tk_checkbuttons[tk_name] = ttk.Checkbutton(
                master=self.tk_frames["ntbk.ps_adding.quick_add.solvers_frame"],
                text=solver_name,
                variable=self.tk_var_bools[tk_name],
                command=self.cross_design_problem_compatibility,
            )
            self.tk_checkbuttons[tk_name].grid(row=row, column=0, sticky="w", padx=10)
        # Update the scroll region
        self.__update_quick_add_problems_scroll_region()
        self.__update_quick_add_solvers_scroll_region()
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
            dict_name = f"ntbk.ps_adding.quick_add.problems_frame.{problem_name}"
            state = "disabled" if error else "normal"
            self.tk_checkbuttons[dict_name].configure(state=state)

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
            state = "disabled" if error else "normal"
            self.tk_checkbuttons[dict_name].configure(state=state)

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
            solver_save_name = self.get_unique_solver_name(solver.name)
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
            problem_save_name = self.get_unique_problem_name(problem_name)
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
        self.__initialize_quick_add()
        # Reset all the booleans
        for key in self.tk_var_bools:
            if "ntbk.ps_adding.quick_add.problems_frame" in key:
                self.tk_var_bools[key].set(False)
            if "ntbk.ps_adding.quick_add.solvers_frame" in key:
                self.tk_var_bools[key].set(False)

    def raise_not_yet_implemented_error(self) -> None:
        error_msg = "This feature has not yet been implemented."
        messagebox.showerror("Not Yet Implemented", error_msg)

    def load_design(self) -> None:
        # Open file dialog to select design file
        # CSV files only, but all files can be selected (in case someone forgets to change file type)
        design_file = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        # Exit w/o message if no file selected
        if design_file == "" or not design_file:
            return
        # Exit w/ message if file does not exist
        if not os.path.exists(design_file):
            messagebox.showerror(
                "File Not Found",
                "The selected file does not exist. Please select a different file.",
            )
            return
        # Open the file with Pandas
        try:
            design_df = pd.read_csv(design_file, sep="\t")
        except Exception as e:
            messagebox.showerror(
                "Error Reading File",
                f"An error occurred while reading the file. Please ensure the file is a CSV file and try again. Error: {e}",
            )
            return
        # Grab the values for 'name', 'design_type', and 'num_stacks' from the first row
        name = design_df["name"].iloc[0]
        design_type = design_df["design_type"].iloc[0]
        num_stacks = design_df["num_stacks"].iloc[0]
        # Set the name, design type, and number of stacks in the GUI
        self.design_name.set(name)
        self.design_type.set(design_type)
        self.design_num_stacks.set(num_stacks)
        # Check if the first line is a problem or a solver
        if name in problem_directory:
            # Select the correct tab
            self.tk_notebooks["ntbk.ps_adding"].select(0)
            self.update()
            # Find the unabbreviated name and set the combobox
            for unabbreviated_name in problem_unabbreviated_directory:
                if problem_unabbreviated_directory[unabbreviated_name]().name == name:
                    name = name + "  --  " + unabbreviated_name
                    break
            self.tk_comboboxes["ntbk.ps_adding.problem.select"].set(name)
            self.update()
            # Create the frame
            self._create_gen_design_frame(design_file, "Problem")
        elif name in solver_directory:
            # Select the correct tab
            self.tk_notebooks["ntbk.ps_adding"].select(1)
            self.update()
            # Find the unabbreviated name and set the combobox
            for unabbreviated_name in solver_unabbreviated_directory:
                if solver_unabbreviated_directory[unabbreviated_name]().name == name:
                    name = name + "  --  " + unabbreviated_name
                    break
            self.tk_comboboxes["ntbk.ps_adding.solver.select"].set(name)
            self.update()
            # Create the frame
            self._create_gen_design_frame(design_file, "Solver")
        else:
            messagebox.showerror(
                "Invalid Design File",
                f"The name variable in the design file ({name}) is not recognized as a problem or solver. Please ensure the file is a valid design file.",
            )

    def load_experiment(self) -> None:
        # Open file dialog to select design file
        # Pickle files only, but all files can be selected (in case someone
        # forgets to change file type)
        # NOTE: Trying to accept both .pickle and .pkl files using
        # "*.pickle;*.pkl" causes Python to crash on MacOS but works fine on
        # Windows. As long as we only have one pickle file extension, we
        # should be fine.
        # TODO: standardize Pickle file extension (see GitHub issue #71)
        experiment_file = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pickle"), ("All files", "*.*")]
        )
        # Exit w/o message if no file selected
        if experiment_file == "" or not experiment_file:
            return
        # Exit w/ message if file does not exist
        if not os.path.exists(experiment_file):
            messagebox.showerror(
                "File Not Found",
                "The selected file does not exist. Please select a different file.",
            )
            return
        # Open the file with pickle
        try:
            with open(experiment_file, "rb") as f:
                experiment = pickle.load(f)
        except Exception as e:
            messagebox.showerror(
                "Error Reading File",
                f"An error occurred while reading the file. Please ensure the file is a pickled experiment and try again. Error: {e}",
            )
            return

        # Make sure the contents of the file are a valid experiment
        if not isinstance(experiment, ProblemsSolvers):
            messagebox.showerror(
                "Invalid File",
                "The file selected is not a valid experiment file. Please select a different file.",
            )
            return

        # Grab the name from the experiment
        loaded_name = experiment.experiment_name

        # Get a unique name for the experiment
        unique_name = self.get_unique_experiment_name(loaded_name)
        # If the name already exists, make the user change it
        if unique_name != loaded_name:
            msg = f"The experiment name '{loaded_name}' already exists."
            msg += f" Would you like to rename the experiment to '{unique_name}'?"
            msg += "\n\nIf you select 'No', the experiment will not be added."
            response = messagebox.askyesno("Name Conflict", msg)
            if not response:
                return

        self.root_experiment_dict[unique_name] = experiment
        self.add_exp_row(unique_name, is_imported=True)

    def _destroy_widget_children(self, widget: tk.Widget) -> None:
        """_Destroy all children of a widget._.

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
        """Insert the headers for the factors into the given frame.

        Args:
            frame (ttk.Frame): The frame to display factor headers in.
            first_row (int, optional): The row index at which to start inserting
                headers. Defaults to 0.

        Returns:
            int: The index of the last row inserted.
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
            column_idx = header_columns.index(heading)
            frame.grid_columnconfigure(column_idx, weight=1)
            label = ttk.Label(
                master=frame,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=first_row,
                column=column_idx,
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
        """Insert the factors into the given frame.

        Args:
            frame (ttk.Frame): The frame to display the factors in.
            factor_dict (dict[str, DFFactor]): Dictionary mapping factor names to
                `DFFactor` objects.
            first_row (int, optional): The row index at which to start inserting
                factors. Defaults to 2.

        Returns:
            int: The index of the last row displayed.
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
            last_row_idx = len(factor_dict) - 1
            if factor_index != last_row_idx:
                ttk.Separator(frame, orient="horizontal").grid(
                    row=row_index + 1,
                    column=0,
                    columnspan=len(column_functions),
                    sticky="ew",
                )

            # Loop through and insert the factor data into the frame
            for column_index, function in enumerate(column_functions):
                widget = function(frame)
                # Stop if we're out of widgets
                if widget is None:
                    break
                # Add the widget if it isn't none
                widget.grid(
                    row=row_index,
                    column=column_index,
                    padx=10,
                    pady=3,
                    sticky="ew",
                )
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
        self.tk_canvases["ntbk.ps_adding.problem.factors"].create_window(
            (0, 0),
            window=self.tk_frames["ntbk.ps_adding.problem.factors.problems"],
            anchor="nw",
        )

        # show problem factors and store default widgets to this dict
        self.__show_data_farming_core(
            problem,
            frame=self.tk_frames["ntbk.ps_adding.problem.factors.problems"],
        )
        # Update the scroll region
        self.__update_problem_factor_scroll_region()

        # Update the design name to be unique
        unique_name = self.get_unique_problem_name(problem.name)
        self.design_name.set(unique_name)
        self.tk_entries["design_opts.name"].delete(0, tk.END)
        self.tk_entries["design_opts.name"].insert(0, unique_name)

    def _create_solver_factors_canvas(self, solver: Solver) -> None:
        # Clear the canvas
        self._destroy_widget_children(self.tk_canvases["ntbk.ps_adding.solver.factors"])

        # Initialize the frames and headers
        self.tk_frames["ntbk.ps_adding.solver.factors.solvers"] = ttk.Frame(
            master=self.tk_canvases["ntbk.ps_adding.solver.factors"],
        )
        self.tk_canvases["ntbk.ps_adding.solver.factors"].create_window(
            (0, 0),
            window=self.tk_frames["ntbk.ps_adding.solver.factors.solvers"],
            anchor="nw",
        )

        # show problem factors and store default widgets to this dict
        self.__show_data_farming_core(
            solver,
            frame=self.tk_frames["ntbk.ps_adding.solver.factors.solvers"],
        )
        # Update the scroll region
        self.__update_solver_factor_scroll_region()

        # Update the design name to be unique
        unique_name = self.get_unique_solver_name(solver.name)
        self.design_name.set(unique_name)
        self.tk_entries["design_opts.name"].delete(0, tk.END)
        self.tk_entries["design_opts.name"].insert(0, unique_name)

    def __get_unique_name(self, dict_lookup: dict, base_name: str) -> str:
        """Generate a unique name by appending a number to a base name if needed.

        Args:
            dict_lookup (dict): Dictionary to check existing names against.
            base_name (str): Desired base name to make unique.

        Returns:
            str: A unique name not present in `dict_lookup`.
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
            return test_name
        return base_name

    def get_unique_experiment_name(self, base_name: str) -> str:
        """Generate a unique experiment name.

        Args:
            base_name (str): Desired base name to make unique.

        Returns:
            str: A unique name not present in `root_experiment_dict`.
        """
        return self.__get_unique_name(self.root_experiment_dict, base_name)

    def get_unique_problem_name(self, base_name: str) -> str:
        """Generate a unique problem name.

        Args:
            base_name (str): Desired base name to make unique.

        Returns:
            str: A unique name not present in `root_problem_dict`.
        """
        return self.__get_unique_name(self.root_problem_dict, base_name)

    def get_unique_solver_name(self, base_name: str) -> str:
        """Generate a unique solver name.

        Args:
            base_name (str): Desired base name to make unique.

        Returns:
            str: A unique name not present in `root_solver_dict`.
        """
        return self.__get_unique_name(self.root_solver_dict, base_name)

    def __show_data_farming_core(
        self, base_object: Solver | Problem, frame: ttk.Frame
    ) -> None:
        """Show data farming options for a solver or problem.

        Args:
            base_object (Solver | Problem): The solver or problem object to display
                options for.
            frame (ttk.Frame): The frame in which to display the data farming options.
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
            for factor in model_specifications:
                specifications[factor] = model_specifications[factor]
        # Convert the specifications to a dictionary of DFFactor objects

        # TODO: This is a hack to remove the step_type and search_direction factors 
        # because str type is not currently supported in the GUI.
        if isinstance(base_object, Solver) and base_object.class_name_abbr == "FCSA":
            del specifications["step_type"]
            del specifications["search_direction"]

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

        if base_object == "Problem":
            base_dropdown = self.selected_problem_name.get()
            root_dict = self.root_problem_dict
            generate_unique_name_func = self.get_unique_problem_name
        else:
            base_dropdown = self.selected_solver_name.get()
            root_dict = self.root_solver_dict
            generate_unique_name_func = self.get_unique_solver_name

        # Check to see if the user has selected a problem or solver
        if base_dropdown == "":
            messagebox.showerror(
                "Error",
                f"Please select a {base_object} from the dropdown list.",
            )
            return

        # Get the design name
        design_name = self.design_name.get()
        # Check to see if the design name already exists
        if design_name in root_dict:
            # Get a unique name
            new_name = generate_unique_name_func(design_name)
            # Ask the user if they want to use the new name
            prompt_text = f"A {base_object} with the name {design_name}"
            prompt_text += " already exists. Would you like to use the name "
            prompt_text += f"{new_name} instead?\nNote: If you choose 'No',"
            prompt_text += " you will need to choose a different name."
            use_new_name = messagebox.askyesno(
                "Name Exists",
                prompt_text,
            )
            if use_new_name:
                self.design_name.set(new_name)
                design_name = new_name
            else:
                return

        # Get the number of stacks and the type of design
        num_stacks = self.design_num_stacks.get()
        design_type = self.design_type.get()
        # Extract the name of the problem or solver from the dropdown box
        base_name = base_dropdown.split(" ")[0]

        """ Determine factors included in design """
        # List of names of factors included in the design
        design_factors: list[str] = []
        # Dict of cross design factors w/ lists of possible values
        cross_design_factors: dict[str, list[str]] = {}
        # Dict of factors not included in the design
        # Key is the factor name, value is the default value
        fixed_factors: dict[str, object] = {}
        for factor in self.factor_dict:
            # If the factor is not included in the design, it's a fixed factor
            if (
                self.factor_dict[factor].include is None
                or not self.factor_dict[factor].include.get()  # type: ignore
            ):
                fixed_val = self.factor_dict[factor].default_eval
                fixed_factors[factor] = fixed_val
            # If the factor is included in the design, add it to the list of factors
            else:
                if self.factor_dict[factor].type.get() in ("int", "float"):
                    design_factors.append(factor)
                elif self.factor_dict[factor].type.get() == "bool":
                    cross_design_factors[factor] = ["True", "False"]

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
            for factor_name in design_factors:
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
            create_design(
                name=base_name,
                factor_headers=design_factors,
                factor_settings=Path(design_name),
                fixed_factors=fixed_factors,
                cross_design_factors=cross_design_factors,
                n_stacks=num_stacks,
                design_type=design_type,  # type: ignore
            )

        except Exception as e:
            # Strip all ANSI codes from the error message
            error_msg = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(e))
            messagebox.showerror(
                "Error",
                f"An error occurred while creating the design: {error_msg}",
            )
            return

        # Display the design tree
        filename = os.path.join(DATA_FARMING_DIR, f"{design_name}_design.csv")
        self._create_gen_design_frame(filename, base_object)

    def _create_gen_design_frame(
        self, filename: str, base_object: Literal["Problem", "Solver"]
    ) -> None:
        self.display_design_tree(
            csv_filename=filename,
        )
        # Button to add the design to the experiment
        command = (
            self.add_problem_design_to_experiment
            if base_object == "Problem"
            else self.add_solver_design_to_experiment
        )
        self.tk_buttons["gen_design.add"].config(
            text=f"Add this {base_object} design to experiment",
            command=command,
        )
        self.tk_buttons["gen_design.add"].grid()

    def create_solver_design(self) -> None:
        """Create a design for the solver."""
        self.__create_design_core("Solver")

    def create_problem_design(self) -> None:
        """Create a design for the problem."""
        self.__create_design_core("Problem")

    def display_design_tree(
        self,
        csv_filename: str | None = None,
        design_table: pd.DataFrame | None = None,
        master_frame: ttk.Frame | None = None,
    ) -> None:
        """Display the design tree in the GUI.

        Displays a Treeview widget populated with design points from either a
        provided CSV file or a DataFrame. Automatically handles formatting,
        scrollbar setup, and label configuration.

        Args:
            csv_filename (str | None): Optional path to a CSV file containing
                the design.
            design_table (pd.DataFrame | None): Optional DataFrame containing
                the design data.
            master_frame (ttk.Frame | None): Optional parent frame to render
                the design tree in. Defaults to the general design display frame.

        Raises:
            ValueError: If neither `csv_filename` nor `design_table` is provided.
        """
        if csv_filename is None and design_table is None:
            error_msg = "Either csv_filename or dataframe must be provided."
            raise ValueError(error_msg)
        # If the CSV filename is provided, read the design table from the CSV
        if csv_filename is not None:
            # Read  the design table from the csv file
            design_table = pd.read_csv(csv_filename, index_col="design_num", sep="\t")
            # Now drop the 'name', 'design_type', and 'num_stacks' columns
            design_table.drop(
                columns=["name", "design_type", "num_stacks"], inplace=True
            )
        assert design_table is not None

        # Set the master frame to the general design display frame if not provided
        if master_frame is None:
            master_frame = self.tk_frames["gen_design.display"]

        # Reset the master frame
        self._destroy_widget_children(master_frame)
        # Unhide the generated design frame
        self._show_gen_design()

        # Modify the header to show the # of design points and # of duplicates
        unique_design_points = design_table.drop_duplicates().shape[0]
        point_plural = "" if unique_design_points == 1 else "s"
        self.tk_labels["gen_design.header"].configure(
            text=f"Generated Design - {len(design_table)} Design Point{point_plural} ({unique_design_points} Unique)"
        )

        self.design_tree = ttk.Treeview(master=master_frame)
        self.design_tree.grid(row=0, column=0, sticky="nsew")
        self.design_tree.heading("#0", text="Design #")

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

        # Add a vertical scrollbar
        # It's scrollable without one, but it isn't intuitive
        self.design_tree_scroll_y = ttk.Scrollbar(
            master=master_frame,
            orient="vertical",
            command=self.design_tree.yview,
        )
        self.design_tree_scroll_y.grid(row=0, column=1, sticky="ns")
        self.design_tree.configure(yscrollcommand=self.design_tree_scroll_y.set)

        # Add a horizontal scrollbar
        self.design_tree_scroll_x = ttk.Scrollbar(
            master=master_frame,
            orient="horizontal",
            command=self.design_tree.xview,
        )
        self.design_tree_scroll_x.grid(row=1, column=0, sticky="ew")
        self.design_tree.configure(xscrollcommand=self.design_tree_scroll_x.set)

        # Add the 'add' button to the frame but hide it for now
        # Anything that wants to show it needs to add a command and text
        self.tk_buttons["gen_design.add"] = ttk.Button(
            master=self.tk_frames["gen_design.display"],
        )
        self.tk_buttons["gen_design.add"].grid(
            row=2, column=0, sticky="nsew", columnspan=2
        )
        self.tk_buttons["gen_design.add"].grid_remove()
        # Button to close the design tree (without adding)
        self.tk_buttons["gen_design.close"] = ttk.Button(
            master=self.tk_frames["gen_design.display"],
            text="Close design tree",
            command=self._hide_gen_design,
        )
        self.tk_buttons["gen_design.close"].grid(
            row=3, column=0, sticky="nsew", columnspan=2
        )

    def __read_in_generated_design(self) -> pd.DataFrame:
        """Extract the design table from the current Treeview widget.

        Returns:
            pd.DataFrame: A DataFrame representing the design as entered or displayed
                in the Treeview.
        """
        design_table = pd.DataFrame(columns=self.design_tree["columns"])
        for child in self.design_tree.get_children():
            values = self.design_tree.item(child)["values"]
            design_table.loc[child] = values
        return design_table

    def add_problem_design_to_experiment(self) -> None:
        """Add a problem design (from the design tree) to the current experiment.

        Reads the current design table, converts it into a list of problem instances,
        checks for name collisions, and updates the experiment with the new problem
        design.

        Raises:
            messagebox.showerror: If the design name is already in use.
        """
        design_name = self.design_name.get()
        if design_name in self.root_problem_dict:
            messagebox.showerror(
                "Error",
                f"The design name {design_name} is already in use. "
                f"Please choose a different name.",
            )
            return
        selected_name = self.selected_problem_name.get()
        selected_name_short = selected_name.split(" ")[0]

        # Create the list of problems by reading the design table
        design_table = self.__read_in_generated_design()
        design_list = create_design_list_from_table(design_table)

        problem_holder_list = []  # used so problem list matches datafarming format
        for dp in design_list:
            problem_list = []  # holds dictionary of dps and solver name
            problem_list.append(dp)
            problem_list.append(selected_name_short)
            problem_holder_list.append(problem_list)

        # Add the problem to the current experiment
        self.add_problem_to_curr_exp(design_name, problem_holder_list)

        # refresh problem design name entry box
        new_problem_name = self.get_unique_problem_name(design_name)
        self.design_name.set(new_problem_name)

        # Hide the design tree
        self._hide_gen_design()

    def add_solver_design_to_experiment(self) -> None:
        """Add a solver design (from the design tree) to the current experiment.

        Reads the current design table, converts it into a list of solver instances,
        checks for name collisions, and updates the experiment with the new solver
        design.

        Raises:
            messagebox.showerror: If the design name is already in use.
        """
        design_name = self.design_name.get()
        if design_name in self.root_solver_dict:
            messagebox.showerror(
                "Error",
                f"The design name {design_name} is already in use. "
                f"Please choose a different name.",
            )
            return
        selected_name = self.selected_solver_name.get()
        selected_name_short = selected_name.split(" ")[0]

        # Create the list of problems by reading the design table
        design_table = self.__read_in_generated_design()
        design_list = create_design_list_from_table(design_table)

        solver_holder_list = []  # used so solver list matches datafarming format
        for dp in design_list:
            solver_list = []  # holds dictionary of dps and solver name
            solver_list.append(dp)
            solver_list.append(selected_name_short)
            solver_holder_list.append(solver_list)

        # Add solver row to list display
        self.add_solver_to_curr_exp(design_name, solver_holder_list)

        # refresh solver design name entry box
        new_solver_name = self.get_unique_solver_name(design_name)
        self.design_name.set(new_solver_name)

        # Hide the design tree
        self._hide_gen_design()

    def __view_design(self, design_list: list[list]) -> None:
        """Display a design in the GUI from a list of design points.

        Converts a list of design dictionaries into a DataFrame, formats it
        for display, and passes it to the design tree viewer.

        Args:
            design_list (list[list]): A nested list where each item contains a
                dictionary representing a design point.
        """
        # Create an empty dataframe to display the design tree
        column_names = list(design_list[0][0].keys())
        num_rows = len(design_list)
        dataframe = pd.DataFrame(columns=column_names, index=range(num_rows))
        # Populate the design tree
        for index, dp in enumerate(design_list):
            dataframe.loc[index] = dp[0]
        # Convert to a string for display
        dataframe_string = dataframe.astype(str)
        # Display the design tree
        self.display_design_tree(
            design_table=dataframe_string,
        )

    def view_problem_design(self, problem_save_name: str) -> None:
        """Display a saved problem design in the design tree view.

        Args:
            problem_save_name (str): The name associated with the saved problem design.
        """
        problem = self.root_problem_dict[problem_save_name]
        self.__view_design(problem)

    def view_solver_design(self, solver_save_name: str) -> None:
        """Display a saved solver design in the design tree view.

        Args:
            solver_save_name (str): The name associated with the saved solver design.
        """
        solver = self.root_solver_dict[solver_save_name]
        self.__view_design(solver)

    def __delete_from_current_experiment(
        self, root_dict: dict, list_name: str, save_name: str
    ) -> None:
        """Delete a saved item from the current experiment and update the GUI.

        Args:
            root_dict (dict): Dictionary containing saved designs (problems or solvers).
            list_name (str): GUI key prefix for the display widgets.
            save_name (str): The name of the item to delete.
        """
        # Delete from root dict
        del root_dict[save_name]
        # Delete from GUI
        base_name = f"{list_name}.{save_name}"
        lbl_name = f"{base_name}.name"
        edit_bttn_name = f"{base_name}.view"
        del_bttn_name = f"{base_name}.del"
        self.tk_labels[lbl_name].destroy()
        self.tk_buttons[edit_bttn_name].destroy()
        self.tk_buttons[del_bttn_name].destroy()
        del self.tk_labels[lbl_name]
        del self.tk_buttons[edit_bttn_name]
        del self.tk_buttons[del_bttn_name]

    def delete_problem(self, problem_name: str) -> None:
        """Delete a saved problem design from the current experiment.

        Args:
            problem_name (str): The name of the problem design to delete.
        """
        self.__delete_from_current_experiment(
            self.root_problem_dict,
            "curr_exp.lists.problems",
            problem_name,
        )
        self.__update_problem_list_scroll_region()
        # Rerun compatibility check
        self.cross_design_solver_compatibility()
        self.__update_solver_dropdown()

    def delete_solver(self, solver_name: str) -> None:
        """Delete a saved solver design from the current experiment.

        Args:
            solver_name (str): The name of the solver design to delete.
        """
        self.__delete_from_current_experiment(
            self.root_solver_dict,
            "curr_exp.lists.solvers",
            solver_name,
        )
        self.__update_solver_list_scroll_region()
        # Rerun compatibility check
        self.cross_design_problem_compatibility()
        self.__update_problem_dropdown()

    def create_experiment(self) -> None:
        # Check to make sure theres at least one problem and solver
        if len(self.root_solver_dict) == 0 or len(self.root_problem_dict) == 0:
            messagebox.showerror(
                "Error",
                "Please add at least one solver and one problem to the experiment.",
            )
            return

        # get unique experiment name
        entered_name = self.curr_exp_name.get()
        unique_name = self.get_unique_experiment_name(entered_name)
        # If the name already exists, make the user change it
        if unique_name != entered_name:
            msg = f"The experiment name '{entered_name}' already exists."
            msg += f" Would you like to rename the experiment to '{unique_name}'?"
            msg += "\n\nIf you select 'No', the experiment will not be added."
            response = messagebox.askyesno("Name Conflict", msg)
            if not response:
                return

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
        experiment = ProblemsSolvers(
            solver_factors=master_solver_factor_list,
            problem_factors=master_problem_factor_list,
            solver_names=master_solver_name_list,
            problem_names=master_problem_name_list,
            solver_renames=solver_renames,
            problem_renames=problem_renames,
            experiment_name=unique_name,
            create_pair_pickles=pickle_checkstate,
        )

        # run check on solver/problem compatibility
        experiment.check_compatibility()

        # add to master experiment list
        self.root_experiment_dict[unique_name] = experiment

        # add exp to row
        self.add_exp_row(unique_name)

        # Clear the current experiment
        self.clear_experiment()

    def clear_experiment(self) -> None:
        # Delete all problems and solvers
        problem_names = list(self.root_problem_dict.keys())
        for name in problem_names:
            self.delete_problem(name)
        solver_names = list(self.root_solver_dict.keys())
        for name in solver_names:
            self.delete_solver(name)
        # Reset the experiment name
        new_name = self.get_unique_experiment_name(self.DEFAULT_EXP_NAME)
        self.curr_exp_name.set(new_name)
        # Set default pickle checkstate
        self.curr_exp_is_pickled.set(self.DEFAULT_EXP_CHECK)
        # Run all update functions for ensuring compatible options now that the
        # experiment is empty
        self.cross_design_problem_compatibility()
        self.cross_design_solver_compatibility()
        self.__update_problem_dropdown()
        self.__update_solver_dropdown()

    def __spawn_new_thread(self, function: Callable) -> threading.Thread:
        thread = threading.Thread(target=function)
        thread.start()
        return thread

    def __disable_exp_buttons(self, experiment_name: str) -> None:
        name_base: Final[str] = "exp." + experiment_name
        # Get all the buttons
        buttons = [bttn for bttn in self.tk_buttons if name_base in bttn]
        # Disable all buttons (except view)
        for button in buttons:
            if not button.endswith(".view"):
                self.tk_buttons[button].configure(state="disabled")

    def __enable_exp_buttons(self, experiment_name: str) -> None:
        name_base: Final[str] = "exp." + experiment_name
        # Get all the buttons
        buttons = [bttn for bttn in self.tk_buttons if name_base in bttn]
        # Enable all buttons
        for button in buttons:
            self.tk_buttons[button].configure(state="normal")

    def __update_action_button(
        self,
        experiment_name: str,
        text: str,
        command: Callable | None = None,
    ) -> None:
        name_base: Final[str] = "exp." + experiment_name
        action_bttn_name: Final[str] = name_base + ".action"
        if command is None:
            self.tk_buttons[action_bttn_name].configure(text=text)
        else:
            self.tk_buttons[action_bttn_name].configure(text=text, command=command)

    def __update_experiment_label(self, experiment_name: str, status: str) -> None:
        name_base: Final[str] = "exp." + experiment_name
        lbl_name: Final[str] = name_base + ".name"
        text = f"{experiment_name}\n({status})"
        self.tk_labels[lbl_name].configure(text=text)

    def __run_experiment_gui(self, experiment_name: str) -> None:
        # Setup
        self.__disable_exp_buttons(experiment_name)
        self.__update_experiment_label(experiment_name, "Running")
        # Try to run the experiment
        try:
            self.run_experiment(experiment_name)
            # If successful, update the label and button
            self.__update_experiment_label(experiment_name, "Ran")
            self.__update_action_button(
                experiment_name,
                "Post-Replicate",
                lambda exp_name=experiment_name: self.__post_process_gui_thread(
                    exp_name
                ),
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.__update_experiment_label(experiment_name, "Initialized")
        # Enable the buttons
        self.__enable_exp_buttons(experiment_name)

    def __run_experiment_gui_thread(self, experiment_name: str) -> threading.Thread:
        return self.__spawn_new_thread(
            lambda: self.__run_experiment_gui(experiment_name)
        )

    def __post_process_gui(self, experiment_name: str) -> None:
        # Setup
        self.__disable_exp_buttons(experiment_name)
        self.__update_experiment_label(experiment_name, "Post-Processing")
        # Try to run the post-processing
        try:
            self.post_process(experiment_name)
            # If successful, update the label and button
            self.__update_experiment_label(experiment_name, "Post-Processed")
            self.__update_action_button(
                experiment_name,
                "Post-Replicated",
                lambda exp_name=experiment_name: self.__post_normalize_gui_thread(
                    exp_name
                ),
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.__update_experiment_label(experiment_name, "Ran")
        # Enable the buttons
        self.__enable_exp_buttons(experiment_name)

    def __post_process_gui_thread(self, experiment_name: str) -> threading.Thread:
        return self.__spawn_new_thread(lambda: self.__post_process_gui(experiment_name))

    def __post_normalize_gui(self, experiment_name: str) -> None:
        # Setup
        self.__disable_exp_buttons(experiment_name)
        self.__update_experiment_label(experiment_name, "Post-Normalizing")
        # Try to run the post-normalization
        try:
            self.post_normalize(experiment_name)
            # If successful, update the label and button
            self.__update_experiment_label(experiment_name, "Post-Normalized")
            self.__update_action_button(
                experiment_name,
                "Log Results",
                lambda exp_name=experiment_name: self.__log_results_gui_thread(
                    exp_name
                ),
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            # If the post-normalization fails, revert to the post-process button
            self.__update_experiment_label(experiment_name, "Post-Processed")
        # Enable the buttons
        self.__enable_exp_buttons(experiment_name)

    def __post_normalize_gui_thread(self, experiment_name: str) -> threading.Thread:
        return self.__spawn_new_thread(
            lambda: self.__post_normalize_gui(experiment_name)
        )

    def __log_results_gui(self, experiment_name: str) -> None:
        # Setup
        self.__disable_exp_buttons(experiment_name)
        self.__update_experiment_label(experiment_name, "Logging")
        # Try to log the experiment
        try:
            self.log_results(experiment_name)
            # If successful, update the label and button
            self.__update_experiment_label(experiment_name, "Logged")
            self.__update_action_button(experiment_name, "All Steps\nComplete")
            # Update the all button to reflect that all steps are done
            all_bttn_name: Final[str] = "exp." + experiment_name + ".all"
            self.tk_buttons[all_bttn_name].configure(text="All Steps\nComplete")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.__update_experiment_label(experiment_name, "Post-Normalized")
        # Enable the buttons that don't relate to running the experiment
        name_base: Final[str] = "exp." + experiment_name
        buttons = [bttn for bttn in self.tk_buttons if name_base in bttn]
        for button in buttons:
            if not button.endswith(".action") and not button.endswith(".all"):
                self.tk_buttons[button].configure(state="normal")

    def __log_results_gui_thread(self, experiment_name: str) -> threading.Thread:
        return self.__spawn_new_thread(lambda: self.__log_results_gui(experiment_name))

    def __all_action_gui(self, experiment_name: str) -> None:
        # None of these steps do anything if they've already been done
        self.__run_experiment_gui(experiment_name)
        self.__post_process_gui(experiment_name)
        self.__post_normalize_gui(experiment_name)
        self.__log_results_gui(experiment_name)

    def __all_actions_gui_thread(self, experiment_name: str) -> threading.Thread:
        return self.__spawn_new_thread(lambda: self.__all_action_gui(experiment_name))

    def add_exp_row(self, experiment_name: str, is_imported: bool = False) -> None:
        """Display experiment in list."""
        list_frame = self.tk_frames["exps.list_canvas.list"]
        row_idx = list_frame.grid_size()[1]

        # Format:
        # (past_tense, present_tense, future_tense, function)
        name_base: Final[str] = "exp." + experiment_name
        lbl_name: Final[str] = name_base + ".name"
        action_bttn_name: Final[str] = name_base + ".action"
        all_bttn_name: Final[str] = name_base + ".all"
        opt_bttn_name: Final[str] = name_base + ".options"
        view_bttn_name: Final[str] = name_base + ".view"
        del_bttn_name: Final[str] = name_base + ".delete"
        # Put all tk names in a list for easy access
        tk_names = [
            lbl_name,
            action_bttn_name,
            all_bttn_name,
            opt_bttn_name,
            view_bttn_name,
            del_bttn_name,
        ]

        self.tk_labels[lbl_name] = ttk.Label(
            master=list_frame,
            text=experiment_name + "\n(Initialized)",
            justify="center",
            anchor="center",
        )
        self.tk_labels[lbl_name].grid(
            row=row_idx, column=0, padx=5, pady=5, sticky="nsew"
        )

        bttn_text_run_all = "Run All\nRemaining Steps"
        bttn_text_run = "Run\nExperiment"

        def view(experiment_name: str) -> None:
            # Check if there's an experiment in progress
            if len(self.root_problem_dict) > 0 or len(self.root_solver_dict) > 0:
                messagebox.showerror(
                    "Error",
                    "Please clear the current experiment before viewing another.",
                )
                return
            experiment = self.root_experiment_dict[experiment_name]
            # Loop through the problems and solvers and combine any that were
            # datafarmed
            problem_dict: dict[str, list[list]] = {}
            for problem in experiment.problems:
                assert isinstance(problem, Problem)
                # Create the list of factors in the right format
                problem_factors = problem.factors
                model_factors = problem.model.factors
                factors = {**problem_factors, **model_factors}
                # Reverse lookup the string for the class in the dictionary
                key = None
                for key, value in problem_directory.items():
                    if value == problem.__class__:
                        name = key
                        break
                factor_list = [factors, key]
                # Check for datafarming
                if "_dp_" in problem.name:
                    name = problem.name.split("_dp_")[0]
                else:
                    name = problem.name
                # If the name is already in the dictionary, append the factors
                if name in problem_dict:
                    problem_dict[name].append(factor_list)
                # Otherwise, create a new entry
                else:
                    problem_dict[name] = [factor_list]
            solver_dict: dict[str, list[list]] = {}
            for solver in experiment.solvers:
                assert isinstance(solver, Solver)
                # Create the list of factors in the right format
                factors = solver.factors
                # Reverse lookup the string for the class in the dictionary
                key = None
                for key, value in solver_directory.items():
                    if value == solver.__class__:
                        name = key
                        break
                factor_list = [factors, key]
                # Check for datafarming
                if "_dp_" in solver.name:
                    name = solver.name.split("_dp_")[0]
                else:
                    name = solver.name
                # If the name is already in the dictionary, append the factors
                if name in solver_dict:
                    solver_dict[name].append(factor_list)
                # Otherwise, create a new entry
                else:
                    solver_dict[name] = [factor_list]

            # Add the problems and solvers to the GUI
            for name, factors in problem_dict.items():
                self.add_problem_to_curr_exp(name, factors)
            for name, factors in solver_dict.items():
                self.add_solver_to_curr_exp(name, factors)

            # Set all the options
            self.curr_exp_name.set(experiment_name)
            self.curr_exp_is_pickled.set(experiment.create_pair_pickles)

        def delete_experiment(experiment_name: str) -> None:
            for name in tk_names:
                # If it's a label, delete it from the label dict
                if name in self.tk_labels:
                    self.tk_labels[name].destroy()
                    del self.tk_labels[name]
                # if it's a button, delete it from the button dict
                elif name in self.tk_buttons:
                    self.tk_buttons[name].destroy()
                    del self.tk_buttons[name]
            del self.root_experiment_dict[experiment_name]
            # Make sure we can't scroll past the end of the canvas
            self.__update_exp_list_scroll_region()

        # Action button (changes based on step)
        self.tk_buttons[action_bttn_name] = ttk.Button(
            master=list_frame,
            text=bttn_text_run,
            command=lambda name=experiment_name: self.__run_experiment_gui_thread(name),
        )
        self.tk_buttons[action_bttn_name].grid(
            row=row_idx, column=1, padx=5, pady=5, sticky="nsew"
        )
        # All button (complete all remaining steps)
        self.tk_buttons[all_bttn_name] = ttk.Button(
            master=list_frame,
            text=bttn_text_run_all,
            command=lambda name=experiment_name: self.__all_actions_gui_thread(name),
        )
        self.tk_buttons[all_bttn_name].grid(
            row=row_idx, column=2, padx=5, pady=5, sticky="nsew"
        )

        # If the experiment was loaded, assume it's already completed
        if is_imported:
            name_label = self.tk_labels[lbl_name]
            action_button = self.tk_buttons[action_bttn_name]
            all_button = self.tk_buttons[all_bttn_name]
            name_label.configure(text=experiment_name + "\n(Imported)")
            action_button.configure(text="Done", state="disabled")
            all_button.configure(state="disabled")

        # Open the options window
        self.tk_buttons[opt_bttn_name] = ttk.Button(
            master=list_frame,
            text="Options",
            command=lambda name=experiment_name: self.open_post_processing_window(name),
        )
        self.tk_buttons[opt_bttn_name].grid(
            row=row_idx, column=3, padx=5, pady=5, sticky="nsew"
        )

        # View the experiment
        self.tk_buttons[view_bttn_name] = ttk.Button(
            master=list_frame,
            text="View",
            command=lambda name=experiment_name: view(name),
        )
        self.tk_buttons[view_bttn_name].grid(
            row=row_idx, column=4, padx=5, pady=5, sticky="nsew"
        )

        # Delete the experiment
        self.tk_buttons[del_bttn_name] = ttk.Button(
            master=list_frame,
            text="Delete",
            command=lambda name=experiment_name: delete_experiment(name),
        )
        self.tk_buttons[del_bttn_name].grid(
            row=row_idx, column=5, padx=5, pady=5, sticky="nsew"
        )

        # Update the scroll region
        self.__update_exp_list_scroll_region()

    def run_experiment(self, experiment_name: str) -> None:
        # get experiment object from master dict
        experiment = self.root_experiment_dict[experiment_name]

        # get specified number of macro reps
        if experiment_name in self.custom_macro_reps:
            n_macroreps = int(self.custom_macro_reps[experiment_name].get())
        else:
            n_macroreps = self.macro_default
        # use ProblemsSolvers run
        experiment.run(n_macroreps=n_macroreps)

    def open_defaults_window(self) -> None:
        # Create a new window
        default_window_title = (
            "Simopt Graphical User Interface - Experiment Options Defaults"
        )
        self.experiment_defaults_window = Toplevel(
            root=self.root, title=default_window_title
        )
        self.experiment_defaults_window.center_window(0.4)

        # Configure the main frame
        self.experiment_defaults_window.columnconfigure(0, weight=1)
        self.main_frame = ttk.Frame(master=self.experiment_defaults_window)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(1, weight=1)

        # Title label
        title_text = "Default experiment options for all experiments."
        title_text += "\nAny changes made will affect all future and current un-run or processed experiments."
        self.title_label = ttk.Label(
            master=self.main_frame,
            text=title_text,
            font=nametofont("TkHeadingFont"),
            justify="center",
            anchor="center",
        )
        self.title_label.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Divider
        self.divider = ttk.Separator(master=self.main_frame, orient="horizontal")
        self.divider.grid(row=1, column=0, columnspan=2, padx=10, sticky="ew")

        # Macro replication number input
        self.macro_rep_label = ttk.Label(
            master=self.main_frame,
            text="Number of macro-replications of the solver run on the problem",
        )
        self.macro_rep_var = tk.IntVar()
        self.macro_rep_var.set(self.macro_default)
        self.macro_rep_entry = ttk.Entry(
            master=self.main_frame,
            textvariable=self.macro_rep_var,
            width=10,
            justify="center",
        )
        self.macro_rep_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.macro_rep_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Post replication number input
        self.post_rep_label = ttk.Label(
            master=self.main_frame,
            text="Number of post-replications",
        )
        self.post_rep_var = tk.IntVar()
        self.post_rep_var.set(self.post_default)
        self.post_rep_entry = ttk.Entry(
            master=self.main_frame,
            textvariable=self.post_rep_var,
            width=10,
            justify="center",
        )
        self.post_rep_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.post_rep_entry.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

        # CRN across budget
        self.crn_budget_label = ttk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for solutions recommended at different times?",
        )
        self.crn_budget_var = tk.StringVar()
        crn_budget_str = "yes" if self.crn_budget_default else "no"
        self.crn_budget_opt = ttk.OptionMenu(
            self.main_frame,
            self.crn_budget_var,
            crn_budget_str,
            "yes",
            "no",
        )
        self.crn_budget_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.crn_budget_opt.grid(row=4, column=1, padx=10, pady=10, sticky="ew")

        # CRN across macroreps
        self.crn_macro_label = ttk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for solutions recommended on different macro-replications?",
        )
        self.crn_macro_var = tk.StringVar()
        crn_macro_default_str = "yes" if self.crn_macro_default else "no"
        self.crn_macro_opt = ttk.OptionMenu(
            self.main_frame,
            self.crn_macro_var,
            crn_macro_default_str,
            "yes",
            "no",
        )
        self.crn_macro_label.grid(row=5, column=0, padx=10, pady=10, sticky="e")
        self.crn_macro_opt.grid(row=5, column=1, padx=10, pady=10, sticky="ew")

        # Post reps at inital & optimal solution input
        self.init_post_rep_label = ttk.Label(
            master=self.main_frame,
            text="Number of post-replications at initial and optimal solutions",
        )
        self.init_post_rep_var = tk.IntVar()
        self.init_post_rep_var.set(self.init_default)
        self.init_post_rep_entry = ttk.Entry(
            master=self.main_frame,
            textvariable=self.init_post_rep_var,
            width=10,
            justify="center",
        )
        self.init_post_rep_label.grid(row=6, column=0, padx=10, pady=10, sticky="e")
        self.init_post_rep_entry.grid(row=6, column=1, padx=10, pady=10, sticky="ew")

        # CRN across init solutions
        self.crn_init_label = ttk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for initial and optimal solution?",
        )
        self.crn_init_var = tk.StringVar()
        crn_init_default_str = "yes" if self.crn_init_default else "no"
        self.crn_init_opt = ttk.OptionMenu(
            self.main_frame,
            self.crn_init_var,
            crn_init_default_str,
            "yes",
            "no",
        )
        self.crn_init_label.grid(row=7, column=0, padx=10, pady=10, sticky="e")
        self.crn_init_opt.grid(row=7, column=1, padx=10, pady=10, sticky="ew")

        # Solve tol frame
        solve_tol_text = (
            "Relative optimality gap(s) definining when a problem is solved"
        )
        solve_tol_text += "\n(Must be between 0 & 1, list in increasing order)"
        self.solve_tols_label = ttk.Label(
            master=self.main_frame,
            text=solve_tol_text,
            justify="center",
        )
        self.solve_tols_frame = ttk.Frame(master=self.main_frame)
        self.solve_tols_label.grid(row=7, column=0, padx=10, pady=10, sticky="e")
        self.solve_tols_frame.grid(row=7, column=1, padx=5, pady=10, sticky="ew")
        self.solve_tols_frame.columnconfigure(0, weight=1)
        self.solve_tols_frame.columnconfigure(1, weight=1)
        self.solve_tols_frame.columnconfigure(2, weight=1)
        self.solve_tols_frame.columnconfigure(3, weight=1)
        # Solve tol entries
        self.solve_tol_1_var = tk.StringVar()
        self.solve_tol_2_var = tk.StringVar()
        self.solve_tol_3_var = tk.StringVar()
        self.solve_tol_4_var = tk.StringVar()
        self.solve_tol_1_var.set(str(self.solve_tols_default[0]))
        self.solve_tol_2_var.set(str(self.solve_tols_default[1]))
        self.solve_tol_3_var.set(str(self.solve_tols_default[2]))
        self.solve_tol_4_var.set(str(self.solve_tols_default[3]))
        self.solve_tol_1_entry = ttk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_1_var,
            justify="center",
        )
        self.solve_tol_2_entry = ttk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_2_var,
            justify="center",
        )
        self.solve_tol_3_entry = ttk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_3_var,
            justify="center",
        )
        self.solve_tol_4_entry = ttk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_4_var,
            justify="center",
        )
        self.solve_tol_1_entry.grid(row=0, column=0, padx=2, sticky="ew")
        self.solve_tol_2_entry.grid(row=0, column=1, padx=2, sticky="ew")
        self.solve_tol_3_entry.grid(row=0, column=2, padx=2, sticky="ew")
        self.solve_tol_4_entry.grid(row=0, column=3, padx=2, sticky="ew")

        # set options as default for future experiments
        self.set_as_default_button = ttk.Button(
            master=self.main_frame,
            text="Set options as default for all experiments",
            command=self.change_experiment_defaults,
        )
        self.set_as_default_button.grid(
            row=8, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )

    def change_experiment_defaults(self) -> None:
        # Change default values to user input
        self.macro_default = self.macro_rep_var.get()
        self.post_default = self.post_rep_var.get()
        self.init_default = self.init_post_rep_var.get()

        crn_budget_str = self.crn_budget_var.get()
        self.crn_budget_default = crn_budget_str.lower() == "yes"
        crn_macro_str = self.crn_macro_var.get()
        self.crn_macro_default = crn_macro_str.lower() == "yes"
        crn_init_str = self.crn_init_var.get()
        self.crn_init_default = crn_init_str.lower() == "yes"
        solve_tol_1 = self.solve_tol_1_var.get()
        solve_tol_2 = self.solve_tol_2_var.get()
        solve_tol_3 = self.solve_tol_3_var.get()
        solve_tol_4 = self.solve_tol_4_var.get()
        tols = [
            solve_tol_1,
            solve_tol_2,
            solve_tol_3,
            solve_tol_4,
        ]
        self.solve_tols_default = [float(tol) for tol in tols]

        # Close the window
        self.experiment_defaults_window.destroy()

    # Functionally the same as the below function, but for boolean values
    def _find_option_setting_bool(
        self,
        exp_name: str,
        search_dict: dict[str, tk.StringVar],
        default_val: bool,
    ) -> bool:
        if exp_name in search_dict:
            value = search_dict[exp_name].get()
            true_vals = ["yes", "true", "1"]
            return value.lower() in true_vals
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
            experiment_name, self.custom_macro_reps, self.macro_default
        )
        n_postreps: int = self._find_option_setting_int(
            experiment_name, self.custom_post_reps, self.post_default
        )
        crn_budget = self._find_option_setting_bool(
            experiment_name, self.custom_crn_budgets, self.crn_budget_default
        )
        crn_macro = self._find_option_setting_bool(
            experiment_name, self.custom_crn_macros, self.crn_macro_default
        )
        n_initreps = self._find_option_setting_int(
            experiment_name, self.custom_init_post_reps, self.init_default
        )
        crn_init = self._find_option_setting_bool(
            experiment_name, self.custom_crn_inits, self.crn_init_default
        )
        if experiment_name in self.custom_solve_tols:
            solve_tols = []
            for tol in self.custom_solve_tols[experiment_name]:
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
        self.title_label = ttk.Label(
            master=self.main_frame,
            text=f"Options for {experiment_name}.",
            font=nametofont("TkHeadingFont"),
        )
        self.title_label.grid(row=0, column=0, sticky="nsew")

        # Macro replication number input
        self.macro_rep_label = ttk.Label(
            master=self.main_frame,
            text="Number of macro-replications of the solver run on the problem",
        )
        self.macro_rep_label.grid(row=1, column=0)
        self.macro_rep_var = tk.IntVar()
        self.macro_rep_var.set(n_macroreps)
        self.macro_rep_entry = ttk.Entry(
            master=self.main_frame,
            textvariable=self.macro_rep_var,
            width=10,
            justify="right",
        )
        self.macro_rep_entry.grid(row=1, column=1)

        # Post replication number input
        self.post_rep_label = ttk.Label(
            master=self.main_frame,
            text="Number of post-replications",
        )
        self.post_rep_label.grid(row=2, column=0)
        self.post_rep_var = tk.IntVar()
        self.post_rep_var.set(n_postreps)
        self.post_rep_entry = ttk.Entry(
            master=self.main_frame,
            textvariable=self.post_rep_var,
            width=10,
            justify="right",
        )
        self.post_rep_entry.grid(row=2, column=1)

        # CRN across budget
        self.crn_budget_label = ttk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for solutions recommended at different times?",
        )
        self.crn_budget_label.grid(row=3, column=0)
        self.crn_budget_var = tk.StringVar()
        crn_budget_str = "yes" if crn_budget else "no"
        self.crn_budget_opt = ttk.OptionMenu(
            self.main_frame, self.crn_budget_var, crn_budget_str, "yes", "no"
        )
        self.crn_budget_opt.grid(row=3, column=1)

        # CRN across macroreps
        self.crn_macro_label = ttk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for solutions recommended on different macro-replications?",
        )
        self.crn_macro_label.grid(row=4, column=0)
        self.crn_macro_var = tk.StringVar()
        crn_macro_str = "yes" if crn_macro else "no"
        self.crn_macro_opt = ttk.OptionMenu(
            self.main_frame, self.crn_macro_var, crn_macro_str, "yes", "no"
        )
        self.crn_macro_opt.grid(row=4, column=1)

        # Post reps at inital & optimal solution input
        self.init_post_rep_label = ttk.Label(
            master=self.main_frame,
            text="Number of post-replications at initial and optimal solutions",
        )
        self.init_post_rep_label.grid(row=5, column=0)
        self.init_post_rep_var = tk.IntVar()
        self.init_post_rep_var.set(n_initreps)
        self.init_post_rep_entry = ttk.Entry(
            master=self.main_frame,
            textvariable=self.init_post_rep_var,
            width=10,
            justify="right",
        )
        self.init_post_rep_entry.grid(row=5, column=1)

        # CRN across init solutions
        self.crn_init_label = ttk.Label(
            master=self.main_frame,
            text="Use CRN on post-replications for initial and optimal solution?",
        )
        self.crn_init_label.grid(row=6, column=0)
        self.crn_init_var = tk.StringVar()
        crn_init_str = "yes" if crn_init else "no"
        self.crn_init_opt = ttk.OptionMenu(
            self.main_frame, self.crn_init_var, crn_init_str, "yes", "no"
        )
        self.crn_init_opt.grid(row=6, column=1)

        # solve tols
        self.solve_tols_label = ttk.Label(
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
        self.solve_tol_1_entry = ttk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_1_var,
            width=5,
            justify="right",
        )
        self.solve_tol_2_entry = ttk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_2_var,
            width=5,
            justify="right",
        )
        self.solve_tol_3_entry = ttk.Entry(
            master=self.solve_tols_frame,
            textvariable=self.solve_tol_3_var,
            width=5,
            justify="right",
        )
        self.solve_tol_4_entry = ttk.Entry(
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
        self.custom_post_reps[experiment_name] = self.post_rep_var
        self.custom_init_post_reps[experiment_name] = self.init_post_rep_var

        self.custom_crn_budgets[experiment_name] = self.crn_budget_var

        self.custom_macro_reps[experiment_name] = self.macro_rep_var

        self.custom_crn_macros[experiment_name] = self.crn_macro_var

        self.custom_crn_inits[experiment_name] = self.crn_init_var

        self.custom_solve_tols[experiment_name] = [
            self.solve_tol_1_var,
            self.solve_tol_2_var,
            self.solve_tol_3_var,
            self.solve_tol_4_var,
        ]

    def post_process(self, experiment_name: str) -> None:
        # get experiment object from master dict
        experiment = self.root_experiment_dict[experiment_name]

        # get user specified options
        if experiment_name in self.custom_post_reps:
            post_reps = self.custom_post_reps[experiment_name].get()
        else:
            post_reps = self.post_default

        if experiment_name in self.custom_crn_budgets:
            crn_budget_str = self.custom_crn_budgets[experiment_name].get()
            crn_budget = crn_budget_str.lower() == "yes"
        else:
            crn_budget = self.crn_budget_default

        if experiment_name in self.custom_crn_macros:
            crn_macro_str = self.custom_crn_macros[experiment_name].get()
            crn_macro = crn_macro_str.lower() == "yes"
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
        if experiment_name in self.custom_init_post_reps:
            reps = self.custom_init_post_reps[experiment_name].get()
        else:
            reps = self.init_default
        if experiment_name in self.custom_crn_inits:
            crn_str = self.custom_crn_inits[experiment_name].get()
            crn = crn_str.lower() == "yes"
        else:
            crn = self.crn_init_default

        # run post normalization
        experiment.post_normalize(n_postreps_init_opt=reps, crn_across_init_opt=crn)

    def log_results(self, experiment_name: str) -> None:
        # get experiment object from master dict
        experiment = self.root_experiment_dict[experiment_name]

        # get user specified options
        if experiment_name in self.custom_solve_tols:
            tol_1 = self.custom_solve_tols[experiment_name][0].get()
            tol_2 = self.custom_solve_tols[experiment_name][1].get()
            tol_3 = self.custom_solve_tols[experiment_name][2].get()
            tol_4 = self.custom_solve_tols[experiment_name][3].get()
            solve_tols_str = [tol_1, tol_2, tol_3, tol_4]
            solve_tols = [float(tol) for tol in solve_tols_str]
        else:
            solve_tols = self.solve_tols_default

        # log results
        experiment.log_group_experiment_results()
        experiment.report_group_statistics(solve_tols=solve_tols)

    def open_plotting_window(self) -> None:
        # create new window
        self.plotting_window = Toplevel(self.root)
        self.plotting_window.center_window(0.8)

        self.plotting_window.title("Plot Experiments")

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
        self.plot_options_frame = tk.Frame(self.plot_main_frame)
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
        self.title_frame.grid(row=0, column=0, columnspan=5, sticky="nsew")
        self.title_frame.grid_columnconfigure(0, weight=1)
        self.title_label = ttk.Label(
            master=self.title_frame,
            text="Welcome to the Plotting Page of SimOpt.",
            font=nametofont("TkHeadingFont"),
            anchor="center",
            justify="center",
        )
        self.title_label.grid(row=0, column=0, sticky="nsew")
        subtitle_lines = [
            "Select Solvers and Problems to Plot from Experiments that have been Post-Normalized.",
            "Solver/Problem factors will only be displayed if all solvers/problems within the experiment are the same.",
        ]
        subtitle = "\n".join(subtitle_lines)
        self.subtitle_label = ttk.Label(
            master=self.title_frame,
            text=subtitle,
            anchor="center",
            justify="center",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="nsew")

        self.plot_header_divider = ttk.Separator(
            master=self.title_frame, orient="horizontal"
        )
        self.plot_header_divider.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        # experiment selection
        self.plot_selection_frame = tk.Frame(master=self.plot_main_frame, width=10)
        self.plot_selection_frame.grid_columnconfigure(0, weight=0)
        self.plot_selection_frame.grid_columnconfigure(1, weight=0)
        self.plot_selection_frame.grid_columnconfigure(3, weight=0)
        self.plot_selection_frame.grid_columnconfigure(4, weight=0)
        self.plot_selection_frame.grid_rowconfigure(3, weight=1)
        self.plot_selection_frame.grid(row=1, column=0, sticky="nsew")

        self.experiment_selection_frame = tk.Frame(master=self.plot_selection_frame)
        self.experiment_selection_frame.grid(
            row=0, column=0, columnspan=5, sticky="nsew"
        )
        self.experiment_selection_frame.grid_columnconfigure(1, weight=1)
        self.experiment_selection_label = ttk.Label(
            self.experiment_selection_frame,
            text="Selected Experiment:",
            justify="right",
            anchor="e",
        )
        self.experiment_selection_label.grid(row=0, column=0, sticky="ew", padx=10)
        # find experiments that have been postnormalized
        postnorm_experiments = []  # list to hold names of all experiments that have been postnormalized
        for exp_name in self.root_experiment_dict:
            experiment = self.root_experiment_dict[exp_name]
            status = experiment.check_postnormalize()
            if status:
                postnorm_experiments.append(exp_name)
        self.experiment_var = tk.StringVar()
        self.experiment_menu = ttk.OptionMenu(
            self.experiment_selection_frame,
            self.experiment_var,
            "[Select an Experiment]",
            *postnorm_experiments,
            command=self.update_plot_menu,
        )
        self.experiment_menu.grid(row=0, column=1, sticky="ew", padx=10)
        # refresh experiment button
        self.refresh_button = ttk.Button(
            self.experiment_selection_frame,
            text="Refresh Dropdown",
            command=self.refresh_experiments,
        )
        self.refresh_button.grid(row=0, column=2, sticky="ew", padx=10)

        self.select_plot_solvers_label = ttk.Label(
            master=self.plot_selection_frame,
            text="Solver Selection",
            anchor="center",
        )
        self.select_plot_solvers_label.grid(row=2, column=0, sticky="ew", columnspan=2)

        # solver selection (treeview)
        self.solver_tree_frame = tk.Frame(
            master=self.plot_selection_frame, width=300, height=300
        )  # frame just to hold solver tree
        self.solver_tree_frame.grid(
            row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew"
        )
        self.solver_tree_frame.grid_rowconfigure(0, weight=1)
        self.solver_tree_frame.grid_columnconfigure(0, weight=1)
        self.solver_tree_frame.grid_propagate(False)
        self.solver_tree = ttk.Treeview(master=self.solver_tree_frame)
        self.solver_tree.grid(row=0, column=0, sticky="nsew")
        self.style = ttk.Style()
        self.style.configure("Treeview.Heading", font=nametofont("TkHeadingFont"))
        self.style.configure(
            "Treeview", foreground="black", font=nametofont("TkDefaultFont")
        )

        self.solver_tree.bind("<<TreeviewSelect>>", self.select_solver)

        # Create a horizontal scrollbar
        solver_xscrollbar = ttk.Scrollbar(
            master=self.solver_tree_frame,
            orient="horizontal",
            command=self.solver_tree.xview,
        )
        self.solver_tree.configure(xscrollcommand=solver_xscrollbar.set)
        solver_xscrollbar.grid(row=1, column=0, sticky="ew")

        # Select all button
        self.select_all_solvers_button = ttk.Button(
            master=self.plot_selection_frame,
            text="Select All Solvers",
            command=self.select_all_solvers,
        )
        self.select_all_solvers_button.grid(
            row=4, column=0, padx=10, pady=10, sticky="ew"
        )
        # Deselect all button
        self.deselect_all_solvers_button = ttk.Button(
            master=self.plot_selection_frame,
            text="Deselect All Solvers",
            command=self.deselect_all_solvers,
        )
        self.deselect_all_solvers_button.grid(
            row=4, column=1, padx=10, pady=10, sticky="ew"
        )

        # Problem/Solver divider
        self.problem_solver_divider = ttk.Separator(
            master=self.plot_selection_frame, orient="vertical"
        )
        self.problem_solver_divider.grid(
            row=2, column=2, rowspan=3, sticky="ns", padx=10
        )

        self.select_plot_problems_label = ttk.Label(
            master=self.plot_selection_frame,
            text="Problem Selection",
            anchor="center",
        )
        self.select_plot_problems_label.grid(row=2, column=3, sticky="ew", columnspan=2)

        # problem selection (treeview)
        self.problem_tree_frame = tk.Frame(
            master=self.plot_selection_frame, width=300, height=300
        )
        self.problem_tree_frame.grid(
            row=3, column=3, columnspan=2, padx=10, pady=10, sticky="nsew"
        )
        self.problem_tree_frame.grid_rowconfigure(0, weight=1)
        self.problem_tree_frame.grid_columnconfigure(0, weight=1)
        self.problem_tree_frame.grid_propagate(False)
        self.problem_tree = ttk.Treeview(master=self.problem_tree_frame)
        self.problem_tree.grid(row=0, column=0, sticky="nsew")
        self.style = ttk.Style()
        self.style.configure("Treeview.Heading", font=nametofont("TkHeadingFont"))
        self.style.configure(
            "Treeview", foreground="black", font=nametofont("TkDefaultFont")
        )
        self.problem_tree.bind("<<TreeviewSelect>>", self.select_problem)

        # Create a horizontal scrollbar
        problem_xscrollbar = ttk.Scrollbar(
            master=self.problem_tree_frame,
            orient="horizontal",
            command=self.problem_tree.xview,
        )
        self.problem_tree.configure(xscrollcommand=problem_xscrollbar.set)
        problem_xscrollbar.grid(row=1, column=0, sticky="ew")

        # Select all button
        self.select_all_problems_button = ttk.Button(
            master=self.plot_selection_frame,
            text="Select All Problems",
            command=self.select_all_problems,
        )
        self.select_all_problems_button.grid(
            row=4, column=3, padx=10, pady=10, sticky="ew"
        )
        # Deselect all button
        self.deselect_all_problems_button = ttk.Button(
            master=self.plot_selection_frame,
            text="Deselect All Problems",
            command=self.deselect_all_problems,
        )
        self.deselect_all_problems_button.grid(
            row=4, column=4, padx=10, pady=10, sticky="ew"
        )

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

        self.experiment_plotting_divider = ttk.Separator(
            master=self.plot_main_frame, orient="vertical"
        )
        self.experiment_plotting_divider.grid(row=1, column=1, sticky="ns", padx=10)

        # plot options
        self.plot_options_frame.grid(row=1, column=2, sticky="nsew")
        self.plot_type_label = ttk.Label(
            master=self.plot_options_frame,
            text="Select Plot Type",
        )
        self.plot_type_label.grid(row=0, column=0, sticky="n", padx=10)
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

        self.plotting_workspace_divider = ttk.Separator(
            master=self.plot_main_frame, orient="horizontal"
        )
        self.plotting_workspace_divider.grid(
            row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=10
        )

        # blank plotting workspace
        self.plotting_workspace_frame.grid(row=3, column=0, columnspan=3, sticky="nsew")
        self.plotting_workspace_frame.grid_rowconfigure(1, weight=1)
        self.plotting_workspace_frame.grid_columnconfigure(0, weight=1)
        self.workspace_label = ttk.Label(
            master=self.plotting_workspace_frame,
            text="Created Plots by Experiment",
            font=nametofont("TkHeadingFont"),
        )
        self.workspace_label.grid(row=0, column=0, padx=20, pady=10)
        # load plot button
        self.load_plot_button = ttk.Button(
            master=self.plotting_workspace_frame,
            text="Load Plot from Pickle",
            command=self.load_plot,
        )
        self.load_plot_button.grid(row=0, column=1, padx=20, pady=10)
        # view selected plots button
        self.view_selected_plots_button = ttk.Button(
            master=self.plotting_workspace_frame,
            text="View Selected Plots",
            command=self.view_selected_plots,
        )
        self.view_selected_plots_button.grid(row=0, column=2, padx=20, pady=10)
        # view all plots button
        self.view_all_plots_button = ttk.Button(
            master=self.plotting_workspace_frame,
            text="View All Created Plots",
            command=self.view_all_plots,
        )
        self.view_all_plots_button.grid(row=0, column=3, padx=20)
        # empty notebook to hold plots
        self.plot_notebook = ttk.Notebook(self.plotting_workspace_frame)
        self.plot_notebook.grid(row=1, column=0, columnspan=4, sticky="nsew")

        # loaded plots tab
        self.loaded_plots_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.loaded_plots_frame, text="Loaded Plots & Copies")

        self.select_header = ttk.Label(
            master=self.loaded_plots_frame,
            text="Select Plot(s)",
            font=nametofont("TkHeadingFont"),
        )
        self.select_header.grid(row=0, column=0)
        self.plot_name_header = ttk.Label(
            master=self.loaded_plots_frame,
            text="Plot Name",
            font=nametofont("TkHeadingFont"),
        )
        self.plot_name_header.grid(row=0, column=1)
        self.view_header = ttk.Label(
            master=self.loaded_plots_frame,
            text="View/Edit",
            font=nametofont("TkHeadingFont"),
        )
        self.view_header.grid(row=0, column=2, pady=10)
        self.file_path_header = ttk.Label(
            master=self.loaded_plots_frame,
            text="File Location",
            font=nametofont("TkHeadingFont"),
        )
        self.file_path_header.grid(row=0, column=3)
        self.del_header = ttk.Label(
            master=self.loaded_plots_frame,
            text="Delete Plot",
            font=nametofont("TkHeadingFont"),
        )
        self.del_header.grid(row=0, column=4)

        # If there's only one experiment, load it
        if len(postnorm_experiments) == 1:
            self.update_plot_menu(postnorm_experiments[0])

    def refresh_experiments(self) -> None:
        # find experiments that have been postnormalized
        postnorm_experiments = []  # list to hold names of all experiments that have been postnormalized
        for exp_name in self.root_experiment_dict:
            experiment = self.root_experiment_dict[exp_name]
            status = experiment.check_postnormalize()
            if status:
                postnorm_experiments.append(exp_name)
        self.experiment_menu["menu"].delete(0, "end")
        for exp in postnorm_experiments:
            self.experiment_menu["menu"].add_command(
                label=exp,
                command=lambda value=exp: self.update_plot_menu(value),
            )
        # If there's only one experiment, load it
        if len(postnorm_experiments) == 1:
            self.update_plot_menu(postnorm_experiments[0])

    def update_plot_window_scroll(self, event: tk.Event) -> None:
        self.plotting_canvas.configure(scrollregion=self.plotting_canvas.bbox("all"))

    def update_plot_menu(self, tk_experiment_name: tk.StringVar) -> None:
        # If we somehow get a string instead of a variable, just use the string
        if isinstance(tk_experiment_name, str):
            experiment_name = tk_experiment_name
        else:
            experiment_name = tk_experiment_name.get()
        # Set the dropdown to the selected experiment
        self.experiment_var.set(experiment_name)

        self.plot_solver_options = ["All"]  # holds names of potential solvers to plot
        self.plot_problem_options = ["All"]  # holds names of potential problems to plot
        self.plot_experiment = self.root_experiment_dict[experiment_name]
        solver_factor_set = set()  # holds names of solver factors
        problem_factor_set = set()  # holds names of problem factors
        for solver in self.plot_experiment.solvers:
            self.plot_solver_options.append(solver.name)
            for factor in solver.factors:
                solver_factor_set.add(factor)  # append factor names to list

        for problem in self.plot_experiment.problems:
            self.plot_problem_options.append(problem.name)
            for factor in problem.factors:
                problem_factor_set.add(factor)
            for factor in problem.model.factors:
                problem_factor_set.add(factor)

        # determine if all solvers in experiment have the same factor options
        solver_set_len = len(solver_factor_set)
        plot_exp_len = len(self.plot_experiment.solvers)
        self.all_same_solver = solver_set_len == plot_exp_len

        # determine if all problems in experiment have the same factor options
        problem_set_len = len(problem_factor_set)
        plot_exp_prob_len = len(self.plot_experiment.problems[0].factors)
        plot_exp_model_len = len(self.plot_experiment.problems[0].model.factors)
        total_prob_len = plot_exp_prob_len + plot_exp_model_len
        self.all_same_problem = problem_set_len == total_prob_len

        # clear previous values in the solver tree
        for child in self.solver_tree.get_children():
            self.solver_tree.delete(child)

        # create first column of solver tree view
        self.solver_tree.heading("#0", text="#")
        self.solver_tree.column("#0", width=75)
        if self.all_same_solver:
            columns = [
                "Solver Name",
                *list(self.plot_experiment.solvers[0].factors.keys()),
            ]
            self.solver_tree["columns"] = columns  # set column names to factor names
            self.solver_tree.heading(
                "Solver Name", text="Solver Name"
            )  # set heading for name column
            for factor in self.plot_experiment.solvers[0].factors:
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

        # Clear the problem tree
        for child in self.problem_tree.get_children():
            self.problem_tree.delete(child)

        # create first column of problem tree view
        self.problem_tree.heading("#0", text="#")
        self.problem_tree.column("#0", width=75)
        if self.all_same_problem:
            factors = list(self.plot_experiment.problems[0].factors.keys()) + list(
                self.plot_experiment.problems[0].model.factors.keys()
            )
            columns = ["Problem Name", *factors]
            self.problem_tree["columns"] = columns  # set column names to factor names
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

    def show_plot_options(self, plot_type_tk: tk.StringVar) -> None:
        if isinstance(plot_type_tk, tk.StringVar):
            plot_type = plot_type_tk.get()
        else:
            plot_type = plot_type_tk

        self._destroy_widget_children(self.more_options_frame)
        self.more_options_frame.grid(row=1, column=0, columnspan=2)

        self.plot_type = plot_type

        # all in one entry (option is present for all plot types)
        self.all_label = ttk.Label(
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

        description_wrap_length = 500

        if plot_type == "Progress Curve":
            description = "Plot individual or aggregate progress curves for one or more solvers on a single problem."
            self.plot_description = ttk.Label(
                master=self.more_options_frame,
                text=description,
                wraplength=description_wrap_length,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2, pady=20)

            # select subplot type
            self.subplot_type_label = ttk.Label(
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
            self.beta_label = ttk.Label(
                master=self.more_options_frame,
                text="Quantile Probability (0.0-1.0)",
            )
            self.beta_label.grid(row=4, column=0)
            self.beta_var = tk.DoubleVar()
            self.beta_var.set(0.5)
            self.beta_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.beta_var
            )
            self.beta_entry.grid(row=4, column=1)
            self.beta_entry.configure(state="disabled")

            # normalize entry
            self.normalize_label = ttk.Label(
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
            self.boot_label = ttk.Label(
                master=self.more_options_frame,
                text="Number Bootstrap Samples",
            )
            self.boot_label.grid(row=5, column=0)
            self.boot_var = tk.IntVar()
            self.boot_var.set(100)
            self.boot_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.boot_var
            )
            self.boot_entry.grid(row=5, column=1)
            self.boot_entry.configure(state="disabled")

            # confidence level entry
            self.con_level_label = ttk.Label(
                master=self.more_options_frame,
                text="Confidence Level (0.0-1.0)",
            )
            self.con_level_label.grid(row=6, column=0)
            self.con_level_var = tk.DoubleVar()
            self.con_level_var.set(0.95)
            self.con_level_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.con_level_var
            )
            self.con_level_entry.grid(row=6, column=1)
            self.con_level_entry.configure(state="disabled")

            # plot CIs entry
            self.plot_CI_label = ttk.Label(
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
            self.plot_hw_label = ttk.Label(
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
            self.legend_label = ttk.Label(
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
            description = (
                "Plot the solvability cdf for one or more solvers on a single problem."
            )
            self.plot_description = ttk.Label(
                master=self.more_options_frame,
                text=description,
                wraplength=description_wrap_length,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2, pady=20)

            # solve tol entry
            self.solve_tol_label = ttk.Label(
                master=self.more_options_frame,
                text="Solve Tolerance",
            )
            self.solve_tol_label.grid(row=2, column=0)
            self.solve_tol_var = tk.DoubleVar()
            self.solve_tol_var.set(0.1)
            self.solve_tol_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.solve_tol_var
            )
            self.solve_tol_entry.grid(row=2, column=1)

            # num bootstraps entry
            self.boot_label = ttk.Label(
                master=self.more_options_frame,
                text="Number Bootstrap Samples",
            )
            self.boot_label.grid(row=3, column=0)
            self.boot_var = tk.IntVar()
            self.boot_var.set(100)
            self.boot_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.boot_var
            )
            self.boot_entry.grid(row=3, column=1)

            # confidence level entry
            self.con_level_label = ttk.Label(
                master=self.more_options_frame,
                text="Confidence Level (0.0-1.0)",
            )
            self.con_level_label.grid(row=4, column=0)
            self.con_level_var = tk.DoubleVar()
            self.con_level_var.set(0.95)
            self.con_level_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.con_level_var
            )
            self.con_level_entry.grid(row=4, column=1)

            # plot CIs entry
            self.plot_CI_label = ttk.Label(
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
            self.plot_hw_label = ttk.Label(
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
            self.legend_label = ttk.Label(
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
            description = "Plot a scatter plot of mean and standard deviation of area under progress curves."
            self.plot_description = ttk.Label(
                master=self.more_options_frame,
                text=description,
                wraplength=description_wrap_length,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2, pady=20)

            # num bootstraps entry
            self.boot_label = ttk.Label(
                master=self.more_options_frame,
                text="Number Bootstrap Samples",
            )
            self.boot_label.grid(row=2, column=0)
            self.boot_var = tk.IntVar()
            self.boot_var.set(100)
            self.boot_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.boot_var
            )
            self.boot_entry.grid(row=2, column=1)

            # confidence level entry
            self.con_level_label = ttk.Label(
                master=self.more_options_frame,
                text="Confidence Level (0.0-1.0)",
            )
            self.con_level_label.grid(row=3, column=0)
            self.con_level_var = tk.DoubleVar()
            self.con_level_var.set(0.95)
            self.con_level_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.con_level_var
            )
            self.con_level_entry.grid(row=3, column=1)

            # plot CIs entry
            self.plot_CI_label = ttk.Label(
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
            self.plot_hw_label = ttk.Label(
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
            self.legend_label = ttk.Label(
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
            description = "Plot the (difference of) solvability profiles for each solver on a set of problems."
            self.plot_description = ttk.Label(
                master=self.more_options_frame,
                text=description,
                wraplength=description_wrap_length,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2, pady=20)

            # reference solver
            self.ref_solver_label = ttk.Label(
                master=self.more_options_frame,
                text="Solver to use for difference benchmark",
            )
            self.ref_solver_label.grid(row=1, column=0)
            # set none if no solvers selected yet
            self.ref_solver_var = tk.StringVar()
            solver_options = []
            if len(self.selected_solvers) == 0:
                solver_display = "No solvers selected"
            else:
                solver_display = None
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
            self.ref_solver_menu.grid(row=1, column=1)
            self.ref_solver_menu.configure(state="disabled")

            # select subplot type
            self.subplot_type_label = ttk.Label(
                master=self.more_options_frame,
                text="Type",
            )
            self.subplot_type_label.grid(row=2, column=0)
            subplot_type_options = [
                "CDF Solvability",
                "Quantile Solvability",
                "Difference of CDF Solvability",
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
            self.boot_label = ttk.Label(
                master=self.more_options_frame,
                text="Number Bootstrap Samples",
            )
            self.boot_label.grid(row=3, column=0)
            self.boot_var = tk.IntVar()
            self.boot_var.set(100)
            self.boot_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.boot_var
            )
            self.boot_entry.grid(row=3, column=1)

            # confidence level entry
            self.con_level_label = ttk.Label(
                master=self.more_options_frame,
                text="Confidence Level (0.0-1.0)",
            )
            self.con_level_label.grid(row=4, column=0)
            self.con_level_var = tk.DoubleVar()
            self.con_level_var.set(0.95)
            self.con_level_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.con_level_var
            )
            self.con_level_entry.grid(row=4, column=1)

            # plot CIs entry
            self.plot_CI_label = ttk.Label(
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
            self.plot_hw_label = ttk.Label(
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
            self.solve_tol_label = ttk.Label(
                master=self.more_options_frame,
                text="Solve Tolerance",
            )
            self.solve_tol_label.grid(row=7, column=0)
            self.solve_tol_var = tk.DoubleVar()
            self.solve_tol_var.set(0.1)
            self.solve_tol_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.solve_tol_var
            )
            self.solve_tol_entry.grid(row=7, column=1)

            # beta entry (quantile size)
            self.beta_label = ttk.Label(
                master=self.more_options_frame,
                text="Quantile Probability (0.0-1.0)",
            )
            self.beta_label.grid(row=8, column=0)
            self.beta_var = tk.StringVar()
            self.beta_var.set("0.5")  # default value
            self.beta_entry = ttk.Entry(
                master=self.more_options_frame, textvariable=self.beta_var
            )
            self.beta_entry.grid(row=8, column=1)
            self.beta_entry.configure(state="disabled")

            # legend location
            self.legend_label = ttk.Label(
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
            description = "Plot individual or aggregate terminal progress for one or more solvers on a single problem. Each unique selected problem will produce its own plot."
            self.plot_description = ttk.Label(
                master=self.more_options_frame,
                text=description,
                wraplength=description_wrap_length,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2, pady=20)

            # select subplot type
            self.subplot_type_label = ttk.Label(
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
            self.normalize_label = ttk.Label(
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
            description = "Plot a scatter plot of mean and standard deviation of terminal progress."
            self.plot_description = ttk.Label(
                master=self.more_options_frame,
                text=description,
                wraplength=description_wrap_length,
            )
            self.plot_description.grid(row=0, column=0, columnspan=2, pady=20)

            # legend location
            self.legend_label = ttk.Label(
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
        self.solver_set_label = ttk.Label(
            master=self.more_options_frame,
            text="Solver Group Name to be Used in Title",
        )
        self.solver_set_label.grid(row=new_row, column=0)
        self.solver_set_var = tk.StringVar()
        self.solver_set_var.set("SOLVER_SET")
        self.solver_set_entry = ttk.Entry(
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
            self.problem_set_label = ttk.Label(
                master=self.more_options_frame,
                text="Problem Group Name to be Used in Title",
            )
            self.problem_set_label.grid(row=new_row + 1, column=0)
            self.problem_set_var = tk.StringVar()
            self.problem_set_var.set("PROBLEM_SET")
            self.problem_set_entry = ttk.Entry(
                master=self.more_options_frame,
                textvariable=self.problem_set_var,
            )
            self.problem_set_entry.grid(row=new_row + 1, column=1)
            self.problem_set_entry.configure(
                state="normal"
            )  # set disabled unlass all in is true

        # file extension
        self.ext_label = ttk.Label(
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
        self.plot_button.grid(row=2, column=0, sticky="nsew")

    def disable_legend(
        self, is_enabled_tk: tk.StringVar
    ) -> None:  # also enables/disables solver & problem group names
        if isinstance(is_enabled_tk, tk.StringVar):
            is_enabled_str = is_enabled_tk.get()
        else:
            is_enabled_str = is_enabled_tk
        is_enabled = is_enabled_str.lower() == "yes"
        is_term_prog = self.plot_type == "Terminal Progress"

        if is_enabled and not is_term_prog:
            self.legend_menu.configure(state="normal")
            self.solver_set_entry.configure(state="normal")
            if self.plot_type in [
                "Terminal Scatter Plot",
                "Solvability Profile",
                "Area Scatter Plot",
            ]:
                self.problem_set_entry.configure(state="normal")
        else:
            # The code will error if this option is changed while terminal
            # progress is selected, so for now we'll just do nothing
            if is_term_prog:
                return
            self.legend_menu.configure(state="disabled")
            self.solver_set_entry.configure(state="disabled")

    def enable_ref_solver(self, plot_type_tk: tk.StringVar) -> None:
        if isinstance(plot_type_tk, tk.StringVar):
            plot_type = plot_type_tk.get()
        else:
            plot_type = plot_type_tk
        # enable reference solver option
        if plot_type in ["CDF Solvability", "Quantile Solvability"]:
            self.ref_solver_menu.configure(state="disabled")
        elif plot_type in [
            "Difference of CDF Solvability",
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

    def select_all_solvers(self) -> None:
        self.solver_tree.selection_set(self.solver_tree.get_children())

    def deselect_all_solvers(self) -> None:
        self.solver_tree.selection_remove(self.solver_tree.get_children())

    def select_all_problems(self) -> None:
        self.problem_tree.selection_set(self.problem_tree.get_children())

    def deselect_all_problems(self) -> None:
        self.problem_tree.selection_remove(self.problem_tree.get_children())

    def select_solver(
        self, _: tk.Event
    ) -> (
        None
    ):  # upddates solver list and options menu for reference solver when relevant
        selected_items = self.solver_tree.selection()
        self.selected_solvers = []
        for item in selected_items:
            solver_index = int(self.solver_tree.item(item, "text"))
            solver = self.plot_experiment.solvers[
                solver_index
            ]  # get corresponding solver from experiment
            self.selected_solvers.append(solver)

        if self.ref_menu_created:  # if reference solver menu exists update menu
            self.update_ref_solver()

    def select_problem(self, _: tk.Event) -> None:
        selected_items = self.problem_tree.selection()
        self.selected_problems = []
        for item in selected_items:
            problem_index = int(self.problem_tree.item(item, "text"))
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
            for solver in self.selected_solvers:  # append solver names to options list
                solver_options.append(solver.name)
            if (
                saved_solver not in solver_options
            ):  # set new default if previous option was deselected
                saved_solver = solver_options[0]
        else:
            solver_options = ["No solvers selected"]
            saved_solver = "No solvers selected"
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

    def __get_plot_experiment_sublist(self) -> list[list[ProblemSolver]]:
        # get selected solvers & problems
        exp_sublist = []  # sublist of experiments to be plotted (each index represents a group of problems over a single solver)
        for solver in self.selected_solvers:
            solver_list = []
            for problem in self.selected_problems:
                for solver_group in self.plot_experiment.experiments:
                    for dp in solver_group:
                        if id(dp.solver) == id(solver) and id(dp.problem) == id(
                            problem
                        ):
                            solver_list.append(dp)
            exp_sublist.append(solver_list)
        return exp_sublist

    def __plot_progress_curve(self) -> None:
        exp_sublist = self.__get_plot_experiment_sublist()
        n_problems = len(exp_sublist[0])
        all_str = self.all_var.get()
        all_in = all_str.lower() == "yes"
        legend = self.legend_var.get() if all_in else None
        ext = self.ext_var.get()
        solver_set_name = self.solver_set_var.get()
        # get user input
        subplot_type = self.subplot_type_var.get()
        assert subplot_type in ["all", "mean", "quantile"]
        beta = float(self.beta_var.get())
        normalize_str = self.normalize_var.get()
        norm = normalize_str.lower() == "yes"
        n_boot = int(self.boot_var.get())
        con_level = float(self.con_level_var.get())
        plot_ci_str = self.plot_CI_var.get()
        plot_ci = plot_ci_str.lower() == "yes"
        plot_hw_str = self.plot_hw_var.get()
        plot_hw = plot_hw_str.lower() == "yes"
        parameters = {}  # holds relevant parameter info for display
        parameters["Plot Type"] = subplot_type
        parameters["Normalize Optimality Gaps"] = normalize_str
        if subplot_type == "quantile":
            parameters["Quantile Probability"] = beta
        parameters["Number Bootstrap Samples"] = n_boot
        parameters["Confidence Level"] = con_level
        # Lookup plot type enum for passing to plotting function
        subplot_type_enum: PlotType = PlotType.from_str(
            subplot_type.lower()
        )
        # create new plot for each problem
        for i in range(n_problems):
            prob_list = []
            for solver_group in exp_sublist:
                prob_list.append(solver_group[i])
            returned_path = plot_progress_curves(
                experiments=prob_list,
                plot_type=subplot_type_enum,
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
                str(item) for item in returned_path if item is not None
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

            self.add_plot_to_notebook(
                file_paths=file_path,
                solver_names=solver_names,
                problem_names=problem_names,
                parameters=parameters,
            )

    def __plot_solvability_cdf(self) -> None:
        exp_sublist = self.__get_plot_experiment_sublist()
        n_problems = len(exp_sublist[0])
        # Fetch user input
        all_str = self.all_var.get()
        all_in = all_str.lower() == "yes"
        # only get legend location if all in one is selected
        legend = self.legend_var.get() if all_in else None
        ext = self.ext_var.get()
        solver_set_name = self.solver_set_var.get()
        solve_tol = float(self.solve_tol_var.get())
        n_boot = int(self.boot_var.get())
        con_level = float(self.con_level_var.get())
        plot_ci_str = self.plot_CI_var.get()
        plot_ci = plot_ci_str.lower() == "yes"
        plot_hw_str = self.plot_hw_var.get()
        plot_hw = plot_hw_str.lower() == "yes"

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
                str(item) for item in returned_path if item is not None
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

            self.add_plot_to_notebook(
                file_paths=file_path,
                solver_names=solver_names,
                problem_names=problem_names,
                parameters=parameters,
            )

    def __plot_area_scatterplot(self) -> None:
        all_str = self.all_var.get()
        all_in = all_str.lower() == "yes"
        # Ensure that the number of selected solvers is less than or equal to 7
        num_selected_solvers = len(self.selected_solvers)
        if num_selected_solvers > 7 and all_in:
            error_msg = "Area scatter plot can plot at most 7 solvers at one time."
            error_msg += " Please select fewer solvers and plot again."
            messagebox.showerror("Error", error_msg)
            return
        exp_sublist = self.__get_plot_experiment_sublist()
        # Fetch user input
        ext = self.ext_var.get()
        solver_set_name = self.solver_set_var.get()
        problem_set_name = self.problem_set_var.get()
        # get user input
        n_boot = int(self.boot_var.get())
        con_level = float(self.con_level_var.get())
        plot_ci_str = self.plot_CI_var.get()
        plot_ci = plot_ci_str.lower() == "yes"
        plot_hw_str = self.plot_hw_var.get()
        plot_hw = plot_hw_str.lower() == "yes"
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
            str(item) for item in returned_path if item is not None
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

        self.add_plot_to_notebook(
            file_paths=file_path,
            solver_names=solver_names,
            problem_names=problem_names,
            parameters=parameters,
        )

    def __plot_terminal_progress(self) -> None:
        exp_sublist = self.__get_plot_experiment_sublist()
        n_problems = len(exp_sublist[0])
        # Fetch user input
        all_str = self.all_var.get()
        all_in = all_str.lower() == "yes"
        ext = self.ext_var.get()
        solver_set_name = self.solver_set_var.get()
        # get user input
        subplot_type = self.subplot_type_var.get()
        normalize_str = self.normalize_var.get()
        norm = normalize_str.lower() == "yes"
        parameters = {}  # holds relevant parameter info for display
        parameters["Plot Type"] = subplot_type
        assert subplot_type in ["box", "violin"]
        parameters["Normalize Optimality Gaps"] = normalize_str
        # Lookup plot type enum for passing to plotting function
        subplot_type_enum: PlotType = PlotType.from_str(
            subplot_type.lower()
        )
        # create a new plot for each problem
        for i in range(n_problems):
            prob_list = []
            for solver_group in exp_sublist:
                prob_list.append(solver_group[i])
            returned_path = plot_terminal_progress(
                experiments=prob_list,
                plot_type=subplot_type_enum,
                all_in_one=all_in,
                normalize=norm,
                save_as_pickle=True,
                ext=ext,
                solver_set_name=solver_set_name,
            )
            # get plot info and call add plot
            file_path = [
                str(item) for item in returned_path if item is not None
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

            self.add_plot_to_notebook(
                file_paths=file_path,
                solver_names=solver_names,
                problem_names=problem_names,
                parameters=parameters,
            )

    def __plot_terminal_scatterplot(self) -> None:
        exp_sublist = self.__get_plot_experiment_sublist()
        # Fetch user input
        all_str = self.all_var.get()
        all_in = all_str.lower() == "yes"
        # only get legend location if all in one is selected
        legend = self.legend_var.get() if all_in else None
        ext = self.ext_var.get()
        solver_set_name = self.solver_set_var.get()
        problem_set_name = self.problem_set_var.get()
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
            str(item) for item in returned_path if item is not None
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
        self.add_plot_to_notebook(
            file_paths=file_path,
            solver_names=solver_names,
            problem_names=problem_names,
            parameters=None,
        )

    def __plot_solvability_profile(self) -> None:
        exp_sublist = self.__get_plot_experiment_sublist()
        # Fetch user input
        all_str = self.all_var.get()
        all_in = all_str.lower() == "yes"
        # only get legend location if all in one is selected
        legend = self.legend_var.get() if all_in else None
        ext = self.ext_var.get()
        solver_set_name = self.solver_set_var.get()
        problem_set_name = self.problem_set_var.get()
        # Select the correct subplot type
        subplot_types = {
            "CDF Solvability": "cdf_solvability",
            "Quantile Solvability": "quantile_solvability",
            "Difference of CDF Solvability": "diff_cdf_solvability",
            "Difference of Quantile Solvability": "diff_quantile_solvability",
        }
        subplot_type = self.subplot_type_var.get()
        if subplot_type not in subplot_types:
            messagebox.showerror(
                "Error",
                "Invalid plot type selected. Please select a valid plot type.",
            )
            return
        plot_input = subplot_types[subplot_type]

        # Get user input
        beta = float(self.beta_var.get())
        n_boot = int(self.boot_var.get())
        con_level = float(self.con_level_var.get())
        plot_ci_str = self.plot_CI_var.get()
        plot_ci = plot_ci_str.lower() == "yes"
        plot_hw_str = self.plot_hw_var.get()
        plot_hw = plot_hw_str.lower() == "yes"
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
        # Lookup plot type enum for passing to plotting function
        subplot_type_enum = PlotType.from_str(subplot_type)
        if subplot_type in ["CDF Solvability", "Quantile Solvability"]:
            returned_path = plot_solvability_profiles(
                experiments=exp_sublist,
                plot_type=subplot_type_enum,
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
                solve_tol=solve_tol,
            )

        else:  # performing a difference solvability profile
            ref_solver = self.ref_solver_var.get()
            parameters["Reference Solver"] = ref_solver
            returned_path = plot_solvability_profiles(
                experiments=exp_sublist,
                plot_type=subplot_type_enum,
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
                solve_tol=solve_tol,
            )
        # get plot info and call add plot
        file_path = [
            str(item) for item in returned_path if item is not None
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

        self.add_plot_to_notebook(
            file_paths=file_path,
            solver_names=solver_names,
            problem_names=problem_names,
            parameters=parameters,
        )

    def plot(self) -> None:
        # Ensure that at least one solver and one problem are selected
        if len(self.selected_problems) == 0:
            error_msg = "Please select problems to plot."
            messagebox.showerror("Error", error_msg)
            return
        if len(self.selected_solvers) == 0:
            error_msg = "Please select solvers to plot."
            messagebox.showerror("Error", error_msg)
            return

        plot_types: dict[str, Callable[[], None]] = {
            "Progress Curve": self.__plot_progress_curve,
            "Solvability CDF": self.__plot_solvability_cdf,
            "Area Scatter Plot": self.__plot_area_scatterplot,
            "Terminal Progress": self.__plot_terminal_progress,
            "Terminal Scatter Plot": self.__plot_terminal_scatterplot,
            "Solvability Profile": self.__plot_solvability_profile,
        }

        # Ensure that the selected plot type is valid
        if self.plot_type not in plot_types:
            error_msg = "Invalid plot type selected. Please select a valid plot type."
            messagebox.showerror("Error", error_msg)
            return
        # Call the appropriate plot function
        plot_types[self.plot_type]()

    def add_plot_to_notebook(
        self,
        file_paths: list[str],
        solver_names: list[str],
        problem_names: list[str],
        parameters: dict | None = None,
    ) -> None:
        # add new tab for exp if applicable
        exp_name = self.experiment_var.get()
        if exp_name not in self.experiment_tabs:
            tab_frame = tk.Frame(self.plot_notebook)
            self.plot_notebook.add(tab_frame, text=exp_name)
            self.experiment_tabs[exp_name] = tab_frame  # save tab frame to dictionary

            # set up tab first time it is created
            select_header = ttk.Label(
                master=tab_frame,
                text="Select Plot(s)",
                font=nametofont("TkDefaultFont"),
            )
            select_header.grid(row=0, column=0)
            solver_header = ttk.Label(
                master=tab_frame,
                text="Solver(s)",
                font=nametofont("TkDefaultFont"),
            )
            solver_header.grid(row=0, column=1)
            problem_header = ttk.Label(
                master=tab_frame,
                text="Problem(s)",
                font=nametofont("TkDefaultFont"),
            )
            problem_header.grid(row=0, column=2)
            type_header = ttk.Label(
                master=tab_frame,
                text="Plot Type",
                font=nametofont("TkDefaultFont"),
            )
            type_header.grid(row=0, column=3)
            view_header = ttk.Label(
                master=tab_frame,
                text="View/Edit",
                font=nametofont("TkDefaultFont"),
            )
            view_header.grid(row=0, column=4, pady=10)
            file_path_header = ttk.Label(
                master=tab_frame,
                text="File Location",
                font=nametofont("TkDefaultFont"),
            )
            file_path_header.grid(row=0, column=5)
            parameters_header = ttk.Label(
                master=tab_frame,
                text="Plot Parameters",
                font=nametofont("TkDefaultFont"),
            )
            parameters_header.grid(row=0, column=6)
            del_header = ttk.Label(
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

        parameter_list = []
        if parameters is not None:
            for parameter in parameters:
                text = f"{parameter} = {parameters[parameter]}"
                parameter_list.append(text)
        para_display = "\n".join(parameter_list)

        # add plots to display
        for index, file_path in enumerate(file_paths):
            row = tab_frame.grid_size()[1]
            self.plot_check_var = tk.BooleanVar()
            check = tk.Checkbutton(master=tab_frame, variable=self.plot_check_var)
            check.grid(row=row, column=0, padx=5)
            self.plot_check_vars[file_path] = self.plot_check_var
            solver_label = ttk.Label(
                master=tab_frame,
                text=solver_names[index],
            )
            solver_label.grid(row=row, column=1, padx=5)
            problem_label = ttk.Label(
                master=tab_frame,
                text=problem_names[index],
            )
            problem_label.grid(row=row, column=2, padx=5)
            type_label = ttk.Label(
                master=tab_frame,
                text=self.plot_type,
            )
            type_label.grid(row=row, column=3, padx=5)
            view_button = ttk.Button(
                master=tab_frame,
                text="View/Edit",
                command=lambda fp=file_path: self.view_plot(fp),
            )
            view_button.grid(row=row, column=4, padx=5)
            screen_width = self.winfo_screenwidth()
            wrap_length = screen_width // 5
            path_label = ttk.Label(
                master=tab_frame, text=file_path, wraplength=wrap_length
            )
            path_label.grid(row=row, column=5, padx=5)
            para_label = ttk.Label(
                master=tab_frame,
                text=para_display,
            )
            para_label.grid(row=row, column=6, padx=5)
            del_button = ttk.Button(
                master=tab_frame,
                text="Delete",
                command=lambda r=row, frame=tab_frame, fp=file_path: self.delete_plot(
                    r, frame, fp
                ),
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
        del self.plot_check_vars[file_path]  # remove check variable from dictionary

    def load_plot(self) -> None:
        # ask user for pickle file location
        file_path = filedialog.askopenfilename()

        # if no file selected, return
        if not file_path or file_path == "":
            return

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
        plot_name_label = ttk.Label(
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
        path_label = ttk.Label(
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
        self.view_single_window.title("Simopt Graphical User Interface - View Plot")
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
        plot_display = ttk.Label(master=self.image_frame, image=plot_photo)
        plot_display.image = plot_photo
        plot_display.grid(row=0, column=0, padx=10, pady=10)

        # menu options supported by matplotlib
        # TODO: import this from somewhere instead of hardcoding
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
            command=lambda frame=self.image_frame, fp=file_path: self.edit_plot_title(
                fp, frame
            ),
        )
        self.edit_title_button.grid(row=0, column=0, padx=10, pady=10)
        self.edit_axes_button = tk.Button(
            master=self.edit_frame,
            text="Edit Plot Axes",
            command=lambda frame=self.image_frame, fp=file_path: self.edit_plot_x_axis(
                fp, frame
            ),
        )
        self.edit_axes_button.grid(row=1, column=0, padx=10, pady=10)
        self.edit_text_button = tk.Button(
            master=self.edit_frame,
            text="Edit Plot Caption",
            command=lambda frame=self.image_frame, fp=file_path: self.edit_plot_text(
                fp, frame
            ),
        )
        self.edit_text_button.grid(row=2, column=0)
        self.edit_image_button = tk.Button(
            master=self.edit_frame,
            text="Edit Image File",
            command=lambda frame=self.image_frame, fp=file_path: self.edit_plot_image(
                fp, frame
            ),
        )
        self.edit_image_button.grid(row=3, column=0, pady=10)

    def save_plot_changes(
        self,
        fig: plt.Figure,
        pickle_path: str | os.PathLike,
        file_path: str | os.PathLike,
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
            plot_display = ttk.Label(master=image_frame, image=plot_photo)
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
            plot_name_label = ttk.Label(
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
            path_label = ttk.Label(
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
        self.title_label = ttk.Label(
            master=self.edit_title_window,
            text="Plot Title",
        )
        self.title_label.grid(row=0, column=0)
        self.title_var = tk.StringVar()
        self.title_var.set(str(title))
        self.title_entry = ttk.Entry(
            master=self.edit_title_window, textvariable=self.title_var, width=50
        )
        self.title_entry.grid(row=0, column=1, padx=10)
        description = r"Use \n to represent a new line in the title"
        self.title_description_label = ttk.Label(
            master=self.edit_title_window,
            text=description,
        )
        self.title_description_label.grid(row=0, column=2, padx=10)

        self.font_label = ttk.Label(
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

        self.font_size_label = ttk.Label(
            master=self.edit_title_window,
            text="Font Size",
        )
        self.font_size_label.grid(row=2, column=0)
        self.font_size_var = tk.StringVar()
        self.font_size_var.set(font_size)
        self.font_size_entry = ttk.Entry(
            master=self.edit_title_window, textvariable=self.font_size_var
        )
        self.font_size_entry.grid(row=2, column=1, padx=10)

        self.font_style_label = ttk.Label(
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

        self.font_weight_label = ttk.Label(
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

        self.font_color_label = ttk.Label(
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

        self.position_x_label = ttk.Label(
            master=self.edit_title_window,
            text="X Position \n (determines centerpoint of title)",
        )
        self.position_x_label.grid(row=7, column=0)
        self.position_x_var = tk.StringVar()
        self.position_x_var.set(title_position[0])
        self.position_x_entry = ttk.Entry(
            master=self.edit_title_window, textvariable=self.position_x_var
        )
        self.position_x_entry.grid(row=7, column=1, padx=10)

        self.align_label = ttk.Label(
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
        pickle_path: str | os.PathLike,
        file_path: str | os.PathLike,
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

        ax.set_title(f"{title_text}", **font_specs, position=title_pos, loc=alignment)

        self.save_plot_changes(
            fig, pickle_path, file_path, image_frame, copy
        )  # save changes and display new image
        self.edit_title_window.destroy()  # close editing window

    def edit_plot_x_axis(
        self, file_path: str | os.PathLike, image_frame: tk.Frame
    ) -> None:  # actualy edits both axes
        # create new window
        self.edit_x_axis_window = tk.Toplevel(self)
        self.edit_x_axis_window.title(
            "Simopt Graphical User Interface - Edit Plot Axes"
        )
        self.edit_x_axis_window.geometry("800x500")

        # select axis
        self.select_axis_label = ttk.Label(
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
                str(axis), file_path, image_frame
            ),
        )
        self.select_axis_menu.grid(row=0, column=1)
        self.edit_x_axis_frame = tk.Frame(
            self.edit_x_axis_window
        )  # create editing frame

    def show_axis_options(
        self,
        axis: Literal["X-Axis", "Y-Axis"],
        file_path: str | os.PathLike,
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
        space = tick_pos[1] - tick_pos[0] if len(tick_pos) > 1 else "none"

        # display current information in entry widgets
        self.x_title_label = ttk.Label(
            master=self.edit_x_axis_frame,
            text=f"{axis_display}-Axis Title",
        )
        self.x_title_label.grid(row=0, column=0)
        self.x_title_var = tk.StringVar()
        self.x_title_var.set(label)
        self.x_title_entry = ttk.Entry(
            master=self.edit_x_axis_frame,
            textvariable=self.x_title_var,
            width=50,
        )
        self.x_title_entry.grid(row=0, column=1)
        description = r"Use \n to represent a new line in the title"
        self.x_title_description_label = ttk.Label(
            master=self.edit_x_axis_frame,
            text=description,
        )
        self.x_title_description_label.grid(row=0, column=2, padx=10)

        self.x_font_label = ttk.Label(
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

        self.x_font_size_label = ttk.Label(
            master=self.edit_x_axis_frame,
            text="Font Size",
        )
        self.x_font_size_label.grid(row=2, column=0)
        self.x_font_size_var = tk.StringVar()
        self.x_font_size_var.set(font_size)
        self.x_font_size_entry = ttk.Entry(
            master=self.edit_x_axis_frame, textvariable=self.x_font_size_var
        )
        self.x_font_size_entry.grid(row=2, column=1, padx=10)

        self.x_font_style_label = ttk.Label(
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

        self.x_font_weight_label = ttk.Label(
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

        self.x_font_color_label = ttk.Label(
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

        self.align_label = ttk.Label(
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
            self.x_scale_label = ttk.Label(
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

        self.min_x_label = ttk.Label(
            master=self.edit_x_axis_frame,
            text=f"Min {axis_display} Value",
        )
        self.min_x_label.grid(row=8, column=0)
        self.min_x_var = tk.StringVar()
        self.min_x_var.set(limits[0])
        self.min_x_entry = ttk.Entry(
            master=self.edit_x_axis_frame, textvariable=self.min_x_var
        )
        self.min_x_entry.grid(row=8, column=1, padx=10)

        self.max_x_label = ttk.Label(
            master=self.edit_x_axis_frame,
            text=f"Max {axis_display} Value",
        )
        self.max_x_label.grid(row=9, column=0)
        self.max_x_var = tk.StringVar()
        self.max_x_var.set(limits[1])
        self.max_x_entry = ttk.Entry(
            master=self.edit_x_axis_frame, textvariable=self.max_x_var
        )
        self.max_x_entry.grid(row=9, column=1, padx=10)

        self.x_space_label = ttk.Label(
            master=self.edit_x_axis_frame,
            text=f"Space Between {axis_display} Ticks",
        )
        self.x_space_label.grid(row=10, column=0)
        self.x_space_var = tk.StringVar()
        self.x_space_var.set(space)
        self.x_space_entry = ttk.Entry(
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
        pickle_path: str | os.PathLike,
        file_path: str | os.PathLike,
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
        self, file_path: str | os.PathLike, image_frame: tk.Frame
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
        root, _ = os.path.splitext(file_path)
        pickle_path = f"{root}.pkl"
        with open(pickle_path, "rb") as f:
            fig = pickle.load(f)
        ax = fig.axes[0]
        # test to make sure not editing title or axes

        # get current text info
        text_objects = [i for i in ax.get_children() if isinstance(i, Text)]
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

        self.text_label = ttk.Label(
            master=self.edit_text_frame,
            text="Plot Caption",
        )
        self.text_label.grid(row=0, column=0)
        self.text_entry = tk.Text(self.edit_text_frame, height=5, width=75)
        self.text_entry.grid(row=0, column=1, padx=10)
        self.text_entry.insert(tk.END, description)

        self.text_font_label = ttk.Label(
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

        self.text_font_size_label = ttk.Label(
            master=self.edit_text_frame,
            text="Font Size",
        )
        self.text_font_size_label.grid(row=2, column=0)
        self.text_font_size_var = tk.StringVar()
        font_size_str = str(font_size)
        self.text_font_size_var.set(font_size_str)
        self.text_font_size_entry = ttk.Entry(
            master=self.edit_text_frame, textvariable=self.text_font_size_var
        )
        self.text_font_size_entry.grid(row=2, column=1, padx=10)

        self.text_font_style_label = ttk.Label(
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

        self.text_font_weight_label = ttk.Label(
            master=self.edit_text_frame,
            text="Font Weight",
        )
        self.text_font_weight_label.grid(row=4, column=0)
        self.text_font_weight_var = tk.StringVar()
        font_weight_str = str(font_weight)
        self.text_font_weight_var.set(font_weight_str)
        self.text_font_weight_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_font_weight_var,
            font_weight_str,
            *self.font_weight_options,
        )
        self.text_font_weight_menu.grid(row=4, column=1, padx=10)

        self.text_font_color_label = ttk.Label(
            master=self.edit_text_frame,
            text="Font Color",
        )
        self.text_font_color_label.grid(row=5, column=0)
        self.text_font_color_var = tk.StringVar()
        color_str = str(color)
        self.text_font_color_var.set(color_str)
        self.text_font_color_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_font_color_var,
            color_str,
            *self.color_options,
        )
        self.text_font_color_menu.grid(row=5, column=1, padx=10)

        self.text_align_label = ttk.Label(
            master=self.edit_text_frame,
            text="Horizontal Alignment",
        )
        self.text_align_label.grid(row=6, column=0)
        self.text_align_var = tk.StringVar()
        # TODO: check if this is supposed to be alignment since color doesn't
        # make a ton of sense in this context
        self.text_align_var.set(color)
        self.text_align_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_align_var,
            h_alignment,
            *["left", "right", "center"],
        )
        self.text_align_menu.grid(row=6, column=1, padx=10)

        self.text_valign_label = ttk.Label(
            master=self.edit_text_frame,
            text="Vertical Alignment",
        )
        self.text_valign_label.grid(row=7, column=0)
        self.text_valign_var = tk.StringVar()
        # TODO: check if this is supposed to be alignment since color doesn't
        # make a ton of sense in this context
        self.text_valign_var.set(color)
        self.text_valign_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.text_valign_var,
            v_alignment,
            *["top", "bottom", "center", "baseline"],
        )
        self.text_valign_menu.grid(row=7, column=1, padx=10)

        self.text_position_x_label = ttk.Label(
            master=self.edit_text_frame,
            text="Description X Position \n (can be + or -)",
        )
        self.text_position_x_label.grid(row=8, column=0)
        self.text_position_x_var = tk.StringVar()
        position_x_str = str(position[0])
        self.text_position_x_var.set(position_x_str)
        self.text_position_x_entry = ttk.Entry(
            master=self.edit_text_frame, textvariable=self.text_position_x_var
        )
        self.text_position_x_entry.grid(row=8, column=1, padx=10)

        self.text_position_y_label = ttk.Label(
            master=self.edit_text_frame,
            text="Description Y Position \n (can be + or -)",
        )
        self.text_position_y_label.grid(row=9, column=0)
        self.text_position_y_var = tk.StringVar()
        position_y_str = str(position[1])
        self.text_position_y_var.set(position_y_str)
        self.text_position_y_entry = ttk.Entry(
            master=self.edit_text_frame, textvariable=self.text_position_y_var
        )
        self.text_position_y_entry.grid(row=9, column=1, padx=10)

        self.background_color_label = ttk.Label(
            master=self.edit_text_frame,
            text="Background Color",
        )
        self.background_color_label.grid(row=10, column=0)
        self.background_color_var = tk.StringVar()
        face_color_str = str(face_color)
        self.background_color_var.set(face_color_str)
        self.background_color_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.background_color_var,
            face_color_str,
            *([*self.color_options, "none"]),
        )
        self.background_color_menu.grid(row=10, column=1, padx=10)

        self.border_color_label = ttk.Label(
            master=self.edit_text_frame,
            text="Border Color",
        )
        self.border_color_label.grid(row=11, column=0)
        self.border_color_var = tk.StringVar()
        # TODO: check if this is supposed to be face color since the color menu
        # uses edge color instead
        self.border_color_var.set(face_color_str)
        edge_color_str = str(edge_color)
        self.border_color_menu = ttk.OptionMenu(
            self.edit_text_frame,
            self.border_color_var,
            edge_color_str,
            *([*self.color_options, "none"]),
        )
        self.border_color_menu.grid(row=11, column=1, padx=10)

        self.border_weight_label = ttk.Label(
            master=self.edit_text_frame,
            text="Border Weight",
        )
        self.border_weight_label.grid(row=12, column=0)
        self.border_weight_var = tk.StringVar()
        line_width_str = str(line_width)
        self.border_weight_var.set(line_width_str)
        self.border_weight_menu = ttk.Entry(
            master=self.edit_text_frame, textvariable=self.border_weight_var
        )
        self.border_weight_menu.grid(row=12, column=1, padx=10)

        self.alpha_label = ttk.Label(
            master=self.edit_text_frame,
            text="Transparency",
        )
        self.alpha_label.grid(row=13, column=0)
        self.alpha_var = tk.StringVar()
        alpha_str = str(alpha)
        self.alpha_var.set(alpha_str)
        self.alpha_menu = ttk.Entry(
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
        pickle_path: str | os.PathLike,
        file_path: str | os.PathLike,
        image_frame: tk.Frame,
        text: Text,
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
        self, file_path: str | os.PathLike, image_frame: tk.Frame
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

        self.dpi_label = ttk.Label(
            master=self.edit_image_frame,
            text="DPI (dots per square inch)",
        )
        self.dpi_label.grid(row=0, column=0)
        self.dpi_var = tk.StringVar()
        self.dpi_var.set(dpi)
        self.dpi_entry = ttk.Entry(
            master=self.edit_image_frame, textvariable=self.dpi_var
        )
        self.dpi_entry.grid(row=0, column=1, padx=10)

        self.ext_label = ttk.Label(
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
        pickle_path: str | os.PathLike,
        file_path: str | os.PathLike,
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
            plot_display = ttk.Label(master=image_frame, image=plot_photo)
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
            plt.savefig(extended_path_name, bbox_inches="tight", dpi=dpi)  # save image
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
            plot_name_label = ttk.Label(
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
            path_label = ttk.Label(
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
        self.view_all_frame.bind("<Configure>", self.update_view_all_window_scroll)

        # open plot images
        row = 0
        col = 0
        for image_path in self.plot_check_vars:  # get file path of all created plots
            plot_image = Image.open(image_path)
            plot_photo = ImageTk.PhotoImage(plot_image)
            plot_display = ttk.Label(master=self.view_all_frame, image=plot_photo)
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
            self.view_canvas.create_window((0, 0), window=self.view_frame, anchor="nw")

            # Bind the configure event to update the scroll region
            self.view_frame.bind("<Configure>", self.update_view_window_scroll)

            # open plot images
            row = 0
            col = 0
            for image_path in selected_plots:
                plot_image = Image.open(image_path)
                plot_photo = ImageTk.PhotoImage(plot_image)
                plot_display = ttk.Label(master=self.view_frame, image=plot_photo)
                plot_display.image = plot_photo
                plot_display.grid(row=row, column=col, padx=10, pady=10)
                col += 1
                if col == 3:  # reset col val and move down one row
                    row += 1
                    col = 0

    def update_view_window_scroll(self, event: tk.Event) -> None:
        self.view_canvas.configure(scrollregion=self.view_canvas.bbox("all"))

    def update_view_all_window_scroll(self, event: tk.Event) -> None:
        self.view_all_canvas.configure(scrollregion=self.view_all_canvas.bbox("all"))

    def update_view_single_window_scroll(self, event: tk.Event) -> None:
        self.view_single_canvas.configure(
            scrollregion=self.view_single_canvas.bbox("all")
        )
