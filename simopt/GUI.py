"""GUI for SimOpt Library."""  # noqa: N999

import ast
import os
import pickle
import re
import sys
import time
import tkinter as tk
from abc import ABC, abstractmethod
from functools import partial
from tkinter import Listbox, Scrollbar, filedialog, simpledialog, ttk
from tkinter.constants import MULTIPLE
from tkinter.font import nametofont
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageTk

from simopt.base import Model, Problem, Solver
from simopt.data_farming_base import DATA_FARMING_DIR, DataFarmingExperiment
from simopt.directory import (
    model_directory,
    model_problem_unabbreviated_directory,
    model_unabbreviated_directory,
    problem_directory,
    problem_unabbreviated_directory,
    solver_directory,
    solver_unabbreviated_directory,
)
from simopt.experiment_base import (
    EXPERIMENT_DIR,
    ProblemSolver,
    ProblemsSolvers,
    create_design,
    find_missing_experiments,
    make_full_metaexperiment,
    plot_area_scatterplots,
    plot_progress_curves,
    plot_solvability_cdfs,
    plot_solvability_profiles,
    plot_terminal_progress,
    plot_terminal_scatterplots,
    post_normalize,
)

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")
    )
)


def center_window(screen: tk.Tk, scale: float) -> str:
    """Centers the window to the main display/monitor.

    Example Usage
    -------------
    position = center_window(self.master, 0.8)

    self.master.geometry(position)

    Parameters
    ----------
    screen : tk.Tk
        The main window of the GUI
    scale : float
        The scale of the window

    Returns
    -------
    str
        The string representation of the window size and position

    """
    screen_width = screen.winfo_screenwidth()
    screen_height = screen.winfo_screenheight()
    width = int(screen_width * scale)
    height = int(screen_height * scale)
    x = int((screen_width / 2) - (width / 2))
    y = int(
        (screen_height / 2) - (height / 1.9)
    )  # Slight adjustment for taskbar
    return f"{width}x{height}+{x}+{y}"


class DFFactor(ABC):
    """Class to store factors for problems and solvers."""

    @property
    def name(self) -> tk.StringVar:
        """The name of the factor."""
        return self.__name

    @property
    def description(self) -> tk.StringVar:
        """The description of the factor."""
        return self.__description

    @property
    @abstractmethod
    def type(self) -> tk.StringVar:
        """The type of the factor."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default(self) -> tk.BooleanVar | tk.IntVar | tk.DoubleVar:
        """The default value of the factor."""
        raise NotImplementedError

    @default.setter
    @abstractmethod
    def default(
        self, default: tk.BooleanVar | tk.IntVar | tk.DoubleVar
    ) -> None:
        raise NotImplementedError

    @property
    def include(self) -> tk.BooleanVar | None:
        """Whether to include the factor in the experiment."""
        return None

    @property
    def include_default_state(self) -> Literal["normal", "disabled", None]:
        """Whether or not the default field is enabled."""
        if self.include is None:
            return None
        if self.include.get():
            return "disabled"
        return "normal"

    @property
    def include_datafarm_state(self) -> Literal["normal", "disabled", None]:
        """Whether or not the datafarm fields are enabled."""
        if self.include is None:
            return None
        if self.include.get():
            return "normal"
        return "disabled"

    @property
    def minimum(self) -> tk.IntVar | tk.DoubleVar | None:
        """The minimum value of the factor."""
        return None

    @property
    def maximum(self) -> tk.IntVar | tk.DoubleVar | None:
        """The maximum value of the factor."""
        return None

    @property
    def num_decimals(self) -> tk.IntVar | None:
        """The number of decimals of the factor."""
        return None

    def __init__(self, name: str, description: str) -> None:
        """Initialize the factor class.

        Parameters
        ----------
        name : str
            The name of the factor
        description : str
            The description of the factor
        include : bool
            Whether to include the factor in the experiment

        """
        self.__name = tk.StringVar(value=name)
        self.__description = tk.StringVar(value=description)
        self.__include = None
        self.__minimum = None
        self.__maximum = None
        self.__num_decimals = None

    def get_name_label(self, master: tk.Tk) -> tk.Label:
        """Get the name label of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        tk.Label
            The name label of the factor

        """
        if not hasattr(self, "lbl_name"):
            self.lbl_name = tk.Label(
                master=master,
                text=self.name.get(),
                justify=tk.LEFT,
            )
        return self.lbl_name

    def get_description_label(self, master: tk.Tk) -> tk.Label:
        """Get the description label of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        tk.Label
            The description label of the factor

        """
        if not hasattr(self, "lbl_description"):
            self.lbl_description = tk.Label(
                master=master,
                text=self.description.get(),
                justify=tk.LEFT,
                wraplength=300,
            )
        return self.lbl_description

    def get_type_label(self, master: tk.Tk) -> tk.Label:
        """Get the type label of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        tk.Label
            The type label of the factor

        """
        if not hasattr(self, "lbl_type"):
            self.lbl_type = tk.Label(
                master=master,
                text=self.type.get(),
                justify=tk.LEFT,
            )
        return self.lbl_type

    def get_default_entry(self, master: tk.Tk) -> ttk.Entry:
        """Get the default entry of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        ttk.Entry
            The default entry of the factor

        """
        if not hasattr(self, "ent_default"):
            self.ent_default = ttk.Entry(
                master=master,
                state=self.include_default_state,
                textvariable=self.default,
                justify=tk.RIGHT,
            )
        return self.ent_default

    def get_include_checkbutton(self, master: tk.Tk) -> tk.Checkbutton | None:
        """Get the include checkbutton of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        tk.Checkbutton | None
            The include checkbutton of the factor (if applicable)

        """
        if self.include is None:
            return None
        if not hasattr(self, "chk_include"):
            self.chk_include = tk.Checkbutton(
                master=master,
                variable=self.include,
                command=self._toggle_fields,
            )
        return self.chk_include

    def get_minimum_entry(self, master: tk.Tk) -> ttk.Entry | None:
        """Get the minimum entry of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        ttk.Entry | None
            The minimum entry of the factor (if applicable)

        """
        if self.minimum is None:
            return None
        if not hasattr(self, "ent_minimum"):
            self.ent_minimum = ttk.Entry(
                master=master,
                state=self.include_datafarm_state,
                textvariable=self.minimum,
                justify=tk.RIGHT,
            )
        return self.ent_minimum

    def get_maximum_entry(self, master: tk.Tk) -> ttk.Entry | None:
        """Get the maximum entry of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        ttk.Entry | None
            The maximum entry of the factor (if applicable)

        """
        if self.maximum is None:
            return None
        if not hasattr(self, "ent_maximum"):
            self.ent_maximum = ttk.Entry(
                master=master,
                state=self.include_datafarm_state,
                textvariable=self.maximum,
                justify=tk.RIGHT,
            )
        return self.ent_maximum

    def get_num_decimals_entry(self, master: tk.Tk) -> ttk.Entry | None:
        """Get the number of decimals entry of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        ttk.Entry | None
            The number of decimals entry of the factor (if applicable)

        """
        if self.num_decimals is None:
            return None
        if not hasattr(self, "ent_num_decimals"):
            self.ent_num_decimals = ttk.Entry(
                master=master,
                state=self.include_datafarm_state,
                textvariable=self.num_decimals,
                justify=tk.RIGHT,
            )
        return self.ent_num_decimals

    def _toggle_fields(self) -> None:
        """Toggle the states of the datafarm fields."""
        if self.include.get():
            # Disable the default field
            self.ent_default.configure(state="disabled")
            # Enable the datafarm fields
            if self.minimum is not None:
                self.ent_minimum.configure(state="normal")
                self.ent_maximum.configure(state="normal")
            if self.num_decimals is not None:
                self.ent_num_decimals.configure(state="normal")
        else:
            # Enable the default field
            self.ent_default.configure(state="normal")
            # Disable the datafarm fields
            if self.minimum is not None:
                self.ent_minimum.configure(state="disabled")
                self.ent_maximum.configure(state="disabled")
            if self.num_decimals is not None:
                self.ent_num_decimals.configure(state="disabled")


class DFBoolean(DFFactor):
    """Class to store boolean factors for problems and solvers."""

    @property
    def include(self) -> tk.BooleanVar:
        """Whether to include the factor in the experiment."""
        return self.__include

    @property
    def type(self) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="bool")

    @property
    def default(self) -> tk.BooleanVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.BooleanVar) -> None:
        self.__default = default

    def __init__(self, name: str, description: str, default: bool) -> None:
        """Initialize the boolean factor class.

        Parameters
        ----------
        name : str
            The name of the factor
        description : str
            The description of the factor
        default : bool
            The default value of the factor

        """
        super().__init__(name, description)
        self.__default = tk.BooleanVar(value=default)
        self.__include = tk.BooleanVar(value=False)

    def get_default_entry(self, master: tk.Tk) -> ttk.Entry:
        """Get the default entry of the factor.

        Parameters
        ----------
        master : tk.Tk
            The main window of the GUI

        Returns
        -------
        ttk.Entry
            The default entry of the factor

        """
        if not hasattr(self, "ent_default"):
            # Create a dropdown menu for boolean values
            self.ent_default = ttk.Combobox(
                master=master,
                state=self.include_default_state,
                textvariable=self.default,
                values=["True", "False"],
                justify=tk.LEFT,
            )
            self.ent_default.current(0 if self.default.get() else 1)
        return self.ent_default


class DFInteger(DFFactor):
    """Class to store integer factors for problems and solvers."""

    @property
    def type(self) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="int")

    @property
    def default(self) -> tk.IntVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.IntVar) -> None:
        self.__default = default

    @property
    def include(self) -> tk.BooleanVar:
        """Whether to include the factor in the experiment."""
        return self.__include

    @property
    def minimum(self) -> tk.IntVar:
        """The minimum value of the factor."""
        return self.__minimum

    @minimum.setter
    def minimum(self, minimum: tk.IntVar) -> None:
        self.__minimum = minimum

    @property
    def maximum(self) -> tk.IntVar:
        """The maximum value of the factor."""
        return self.__maximum

    @maximum.setter
    def maximum(self, maximum: tk.IntVar) -> None:
        self.__maximum = maximum

    def __init__(self, name: str, description: str, default: int) -> None:
        """Initialize the integer factor class.

        Parameters
        ----------
        name : str
            The name of the factor
        description : str
            The description of the factor
        default : int
            The default value of the factor

        """
        super().__init__(name, description)
        self.__default = tk.IntVar(value=default)
        self.__include = tk.BooleanVar(value=False)
        self.__minimum = tk.IntVar(value=default)
        self.__maximum = tk.IntVar(value=default)


class DFFloat(DFFactor):
    """Class to store float factors for problems and solvers."""

    @property
    def type(self) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="float")

    @property
    def default(self) -> tk.DoubleVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.DoubleVar) -> None:
        self.__default = default

    @property
    def include(self) -> tk.BooleanVar:
        """Whether to include the factor in the experiment."""
        return self.__include

    @property
    def minimum(self) -> tk.DoubleVar:
        """The minimum value of the factor."""
        return self.__minimum

    @minimum.setter
    def minimum(self, minimum: tk.DoubleVar) -> None:
        self.__minimum = minimum

    @property
    def maximum(self) -> tk.DoubleVar:
        """The maximum value of the factor."""
        return self.__maximum

    @maximum.setter
    def maximum(self, maximum: tk.DoubleVar) -> None:
        self.__maximum = maximum

    @property
    def num_decimals(self) -> tk.IntVar:
        """The number of decimals of the factor."""
        return self.__num_decimals

    @num_decimals.setter
    def num_decimals(self, num_decimals: tk.IntVar) -> None:
        self.__num_decimals = num_decimals

    def __init__(self, name: str, description: str, default: float) -> None:
        """Initialize the float factor class.

        Parameters
        ----------
        name : str
            The name of the factor
        description : str
            The description of the factor
        default : float
            The default value of the factor

        """
        super().__init__(name, description)
        self.__default = tk.DoubleVar(value=default)
        self.__include = tk.BooleanVar(value=False)
        self.__minimum = tk.DoubleVar(value=default)
        self.__maximum = tk.DoubleVar(value=default)
        num_decimals = str(default)[::-1].find(".")
        self.__num_decimals = tk.IntVar(value=num_decimals)


def set_theme(window: tk.Tk) -> tk.Tk:
    """Set the theme of the GUI.

    Parameters
    ----------
    window : tk.Tk
        GUI object without a theme set

    Returns
    -------
    tk.Tk
        GUI object with the theme set

    """
    # Configure the theme of the GUI
    window.style = ttk.Style()
    window.style.theme_use("alt")
    # Configure the default fonts based on screen size
    # https://tkinter-docs.readthedocs.io/en/latest/generic/fonts.html
    # Scale by width because it's easy to scroll vertically, but scrolling
    # horizontally is a pain. This way, the text will always fit on
    # the screen.
    width = window.master.winfo_screenwidth()
    font_medium = int(width / 200)
    font_large = int(font_medium * 1.2)
    font_small = int(font_medium * 0.8)

    # Adjust the default fonts
    nametofont("TkDefaultFont").configure(size=font_medium)
    nametofont("TkTextFont").configure(size=font_medium)
    nametofont("TkHeadingFont").configure(
        size=font_large, weight="bold"
    )  # Default header
    nametofont("TkCaptionFont").configure(size=font_large)
    nametofont("TkTooltipFont").configure(
        size=font_small, slant="italic"
    )  # Default small italics
    nametofont("TkFixedFont").configure(size=font_medium)
    nametofont("TkIconFont").configure(size=font_medium)
    nametofont("TkMenuFont").configure(size=font_medium)
    nametofont("TkSmallCaptionFont").configure(size=font_small)

    return window


class MainMenuWindow(tk.Tk):
    """Main menu window of the GUI."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the main menu window of the GUI.

        Parameters
        ----------
        root : tk.Tk
            The main window of the GUI

        """
        self.master = root
        self = set_theme(self)

        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        position = center_window(self.master, 0.8)
        self.master.geometry(position)

        self.title_label = tk.Label(
            master=self.master,
            text="Welcome to SimOpt Library Graphic User Interface",
            font=nametofont("TkHeadingFont"),
            justify="center",
            width=50,
        )
        self.title_label.place(relx=0.1, rely=0.05)

        # Button to open original main window to run experiments across solvers & problems
        self.experiment_button = tk.Button(
            master=self.master,
            text="Run Single Problem-Solver Experiment",
            width=50,
            command=self.open_experiment_window,
        )
        self.experiment_button.place(relx=0.15, rely=0.2)
        self.experiment_button.configure(background="light gray")

        # Button to open model data farming window
        self.datafarm_model_button = tk.Button(
            master=self.master,
            text="Data Farm Models",
            width=50,
            command=self.open_model_datafarming,
        )
        self.datafarm_model_button.place(relx=0.15, rely=0.35)
        self.datafarm_model_button.configure(background="light gray")

        # # Button to open solver & problem data farming window
        # self.datafarm_prob_sol_button = tk.Button(
        #     master=self.master,
        #     text="Solver Data Farming",
        #     width=50,
        #     command=self.open_prob_sol_datafarming,
        # )
        # self.datafarm_prob_sol_button.place(relx=0.15, rely=0.4)
        # self.datafarm_prob_sol_button.configure(background="light gray")

        # Button to open new experiment window
        self.new_experiment_button = tk.Button(
            master=self.master,
            text="Data Farm Solvers, Problems, and Models",
            width=50,
            command=self.open_new_experiment,
        )
        self.new_experiment_button.place(relx=0.15, rely=0.5)
        self.new_experiment_button.configure(background="light gray")

        # Open the new experiment window and hide the main menu window
        # self.open_new_experiment()
        # self.master.withdraw()

    def open_experiment_window(self) -> None:
        """Open the experiment window."""
        self.create_experiment_window = tk.Toplevel(self.master)
        self.create_experiment_window.title(
            "SimOpt Library Graphical User Interface - Problem-Solver Experiment"
        )
        self.experiment_app = ExperimentWindow(self.create_experiment_window)
        self.master.withdraw()

    def open_model_datafarming(self) -> None:
        """Open the model data farming window."""
        self.datafarming_window = tk.Toplevel(self.master)
        self.datafarming_window.title(
            "SimOpt Library Graphical User Interface - Model Data Farming"
        )
        self.datafarming_app = DataFarmingWindow(self.datafarming_window, self)
        self.master.withdraw()

    def open_new_experiment(self) -> None:
        """Open the new experiment window."""
        self.new_experiment_window = tk.Toplevel(self.master)
        self.new_experiment_window.title(
            "SimOpt Library Graphical User Interface - New Experiment Window"
        )
        self.new_experiment_app = NewExperimentWindow(
            self.new_experiment_window
        )
        self.master.withdraw()


class ExperimentWindow(tk.Toplevel):
    """Main window of the GUI.

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

    """

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the main window of the GUI."""
        self.master = root
        self = set_theme(self)

        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        position = center_window(self.master, 0.8)
        self.master.geometry(position)

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

        self.instruction_label = tk.Label(
            master=self.master,  # window label is used in
            text="Welcome to SimOpt Library Graphic User Interface\n Please Load or Add Your Problem-Solver Pair(s): ",
            font=nametofont("TkHeadingFont"),
            justify="center",
        )

        self.problem_label = tk.Label(
            master=self.master,  # window label is used in
            text="Select Problem:",
        )

        self.or_label = tk.Label(
            master=self.master,  # window label is used in
            text=" OR ",
        )
        self.or_label2 = tk.Label(
            master=self.master,  # window label is used in
            text=" OR Select Problem and Solver from Below:",
        )
        self.or_label22 = tk.Label(
            master=self.master,  # window label is used in
            text="Select from Below:",
        )

        # from experiments.inputs.all_factors.py:
        self.problem_list = problem_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.problem_var = tk.StringVar(master=self.master)
        # sets the default OptionMenu value

        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.problem_menu = ttk.OptionMenu(
            self.master,
            self.problem_var,
            "Problem",
            *self.problem_list,
            command=self.show_problem_factors,
        )

        self.solver_label = tk.Label(
            master=self.master,  # window label is used in
            text="Select Solver:",
        )

        # from experiments.inputs.all_factors.py:
        self.solver_list = solver_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.solver_var = tk.StringVar(master=self.master)
        # sets the default OptionMenu value

        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.solver_menu = ttk.OptionMenu(
            self.master,
            self.solver_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_factors,
        )

        # self.macro_label = tk.Label(master=self.master,
        #                text = "Number of Macroreplications:",
        #              font = f"{TEXT_FAMILY} 13")

        self.macro_definition = tk.Label(
            master=self.master,
            text="",
        )

        self.macro_definition_label = tk.Label(
            master=self.master,
            text="Number of Macroreplications:",
            width=25,
        )

        self.macro_var = tk.StringVar(self.master)
        self.macro_entry = ttk.Entry(
            master=self.master,
            textvariable=self.macro_var,
            justify=tk.LEFT,
            width=10,
        )
        self.macro_entry.insert(index=tk.END, string="10")

        self.add_button = ttk.Button(
            master=self.master,
            text="Add Problem-Solver Pair",
            width=15,
            command=self.add_experiment,
        )

        self.clear_queue_button = ttk.Button(
            master=self.master,
            text="Clear All Problem-Solver Pairs",
            width=15,
            command=self.clear_queue,
        )  # (self.experiment_added, self.problem_added, self.solver_added, self.macros_added, self.run_button_added))

        self.crossdesign_button = ttk.Button(
            master=self.master,
            text="Create Problem-Solver Group",
            width=50,
            command=self.crossdesign_function,
        )

        self.pickle_file_load_button = ttk.Button(
            master=self.master,
            text="Load Problem-Solver Pair",
            width=50,
            command=self.load_pickle_file_function,
        )

        self.attribute_description_label = tk.Label(
            master=self.master,
            text="Attribute Description Label for Problems:\n Objective (Single [S] or Multiple [M])\n Constraint (Unconstrained [U], Box[B], Determinisitic [D], Stochastic [S])\n Variable (Discrete [D], Continuous [C], Mixed [M])\n Gradient Available (True [G] or False [N])",
            font=nametofont("TkTooltipFont"),
        )
        self.attribute_description_label.place(x=450, rely=0.478)

        self.post_normal_all_button = ttk.Button(
            master=self.master,
            text="Post-Normalize Selected",
            width=20,
            state="normal",
            command=self.post_normal_all_function,
        )

        self.make_meta_experiment = ttk.Button(
            master=self.master,
            text="Create Problem-Solver Group from Selected",
            width=35,
            state="normal",
            command=self.make_meta_experiment_func,
        )

        self.pickle_file_pathname_label = tk.Label(
            master=self.master,
            text="File Selected:",
        )

        self.pickle_file_pathname_show = tk.Label(
            master=self.master,
            text="No File Selected!",
            foreground="red",
            wraplength="500",
        )

        self.style.configure("Bold.TLabel", font=nametofont("TkHeadingFont"))
        self.label_Workspace = ttk.Label(
            master=self.master, text="Workspace", style="Bold.TLabel"
        )
        self.queue_label_frame = ttk.LabelFrame(
            master=self.master, labelwidget=self.label_Workspace
        )

        self.queue_canvas = tk.Canvas(
            master=self.queue_label_frame, borderwidth=0
        )

        self.queue_frame = ttk.Frame(master=self.queue_canvas)
        self.vert_scroll_bar = Scrollbar(
            self.queue_label_frame,
            orient="vertical",
            command=self.queue_canvas.yview,
        )
        self.horiz_scroll_bar = Scrollbar(
            self.queue_label_frame,
            orient="horizontal",
            command=self.queue_canvas.xview,
        )
        self.queue_canvas.configure(
            xscrollcommand=self.horiz_scroll_bar.set,
            yscrollcommand=self.vert_scroll_bar.set,
        )

        self.vert_scroll_bar.pack(side="right", fill="y")
        self.horiz_scroll_bar.pack(side="bottom", fill="x")

        self.queue_canvas.pack(side="left", fill="both", expand=True)
        self.queue_canvas.create_window(
            (0, 0),
            window=self.queue_frame,
            anchor="nw",
            tags="self.queue_frame",
        )

        self.queue_frame.bind("<Configure>", self.on_frame_configure_queue)

        self.notebook = ttk.Notebook(master=self.queue_frame)
        self.notebook.pack(fill="both")

        self.tab_one = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_one, text="Queue of Problem-Solver Pairs")

        self.tab_one.grid_rowconfigure(0)

        self.heading_list = [
            "Selected",
            "Pair #",
            "Problem",
            "Solver",
            "Macroreps",
            "",
            "",
            "",
            "",
            "",
        ]

        for heading in self.heading_list:
            self.tab_one.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(
                master=self.tab_one,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=0, column=self.heading_list.index(heading), padx=10, pady=3
            )

        self.tab_two = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_two, text="Queue of Problem-Solver Groups")
        self.tab_two.grid_rowconfigure(0)
        self.heading_list = [
            "Problems",
            "Solvers",
            "Macroreps",
            "",
            "",
            "",
            "",
            "",
        ]

        for heading in self.heading_list:
            self.tab_two.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(
                master=self.tab_two,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=0, column=self.heading_list.index(heading), padx=10, pady=3
            )

        self.tab_three = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_three, text="Post-Normalize by Problem")
        self.tab_three.grid_rowconfigure(0)
        self.heading_list = [
            "Problem",
            "Solvers",
            "Selected",
            "",
            "",
            "",
            "",
            "",
        ]

        for heading in self.heading_list:
            self.tab_three.grid_columnconfigure(
                self.heading_list.index(heading)
            )
            label = tk.Label(
                master=self.tab_three,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=0, column=self.heading_list.index(heading), padx=10, pady=3
            )

        def on_tab_change(event: tk.Event) -> None:
            """Handle tab change event.

            Parameters
            ----------
            event : tk.Event
                The event object

            """
            tab = event.widget.tab("current")["text"]
            if tab == "Post-Normalize by Problem":
                self.post_norm_setup()
                self.post_normal_all_button.place(x=10, rely=0.92)
            else:
                self.post_normal_all_button.place_forget()
            if tab == "Queue of Problem-Solver Pairs":
                # My code starts here
                # Make meta experiment button wider & releative x
                self.make_meta_experiment.place(relx=0.02, rely=0.92, width=300)
                # My code ends here
            else:
                self.make_meta_experiment.place_forget()

        self.notebook.bind("<<NotebookTabChanged>>", on_tab_change)

        self.instruction_label.place(relx=0.4, y=0)

        self.solver_label.place(relx=0.01, rely=0.1)
        self.solver_menu.place(relx=0.1, rely=0.1)

        self.problem_label.place(relx=0.3, rely=0.1)
        self.problem_menu.place(relx=0.4, rely=0.1)

        # self.macro_label.place(relx=.7, rely=.1)
        self.macro_entry.place(relx=0.89, rely=0.1, width=100)

        self.macro_definition.place(relx=0.73, rely=0.05)
        self.macro_definition_label.place(relx=0.7, rely=0.1)

        # self.macro_definition_label.bind("<Enter>",self.on_enter)
        # self.macro_definition_label.bind("<Leave>",self.on_leave)

        self.or_label.place(x=215, rely=0.06)
        self.crossdesign_button.place(x=255, rely=0.06, width=220)

        y_place = 0.06
        self.pickle_file_load_button.place(x=10, rely=y_place, width=195)
        self.or_label2.place(x=480, rely=0.06)
        # self.or_label22.place(x=435, rely=.06)

        self.queue_label_frame.place(
            x=10, rely=0.56, relheight=0.35, relwidth=0.99
        )
        # self.post_normal_all_button.place(x=400,rely=.95)

        self.frame.pack(fill="both")

        # self.l1.bind("<Enter>", self.on_enter)
        # self.l1.bind("<Leave>", self.on_leave)

    # def on_enter(self, event):
    # self.l2(text="Hover Works :)")
    # def on_leave(self, enter):
    # self.l2.configure(text="")

    # def on_enter(self, event):
    # self.macro_definition.configure(text="Definition of MacroReplication")

    # def on_leave(self, enter):
    # self.macro_definition.configure(text="")

    def show_problem_factors(self, *args: tuple) -> None:
        """Show the problem factors.

        Parameters
        ----------
        *args : tuple
            The arguments

        """
        # if args and len(args) == 2:
        #     print("ARGS: ", args[1])
        # ("arg length:", len(args))

        self.problem_factors_list = []
        self.problem_factors_types = []

        self.factor_label_frame_problem = ttk.LabelFrame(
            master=self.master, text="Problem Factors"
        )

        self.factor_canvas_problem = tk.Canvas(
            master=self.factor_label_frame_problem, borderwidth=0
        )

        self.factor_frame_problem = ttk.Frame(master=self.factor_canvas_problem)
        self.vert_scroll_bar_factor_problem = Scrollbar(
            self.factor_label_frame_problem,
            orient="vertical",
            command=self.factor_canvas_problem.yview,
        )
        self.horiz_scroll_bar_factor_problem = Scrollbar(
            self.factor_label_frame_problem,
            orient="horizontal",
            command=self.factor_canvas_problem.xview,
        )
        self.factor_canvas_problem.configure(
            xscrollcommand=self.horiz_scroll_bar_factor_problem.set,
            yscrollcommand=self.vert_scroll_bar_factor_problem.set,
        )

        self.vert_scroll_bar_factor_problem.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_problem.pack(side="bottom", fill="x")

        self.factor_canvas_problem.pack(side="left", fill="both", expand=True)
        self.factor_canvas_problem.create_window(
            (0, 0),
            window=self.factor_frame_problem,
            anchor="nw",
            tags="self.factor_frame_problem",
        )

        self.factor_frame_problem.bind(
            "<Configure>", self.on_frame_configure_factor_problem
        )

        self.factor_notebook_problem = ttk.Notebook(
            master=self.factor_frame_problem
        )
        self.factor_notebook_problem.pack(fill="both")

        self.factor_tab_one_problem = tk.Frame(
            master=self.factor_notebook_problem
        )

        self.factor_notebook_problem.add(
            self.factor_tab_one_problem,
            text=str(self.problem_var.get()) + " Factors",
        )

        self.factor_tab_one_problem.grid_rowconfigure(0)

        self.factor_heading_list_problem = ["Description", "Input"]

        for heading in self.factor_heading_list_problem:
            self.factor_tab_one_problem.grid_columnconfigure(
                self.factor_heading_list_problem.index(heading)
            )
            label_problem = tk.Label(
                master=self.factor_tab_one_problem,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label_problem.grid(
                row=0,
                column=self.factor_heading_list_problem.index(heading),
                padx=10,
                pady=3,
            )

        self.problem_object = problem_unabbreviated_directory[
            self.problem_var.get()
        ]

        count_factors_problem = 1

        if args and len(args) == 2 and args[0]:
            oldname = args[1][3][1]

        else:
            problem_object = problem_unabbreviated_directory[
                self.problem_var.get()
            ]
            oldname = problem_object().name

        self.save_label_problem = tk.Label(
            master=self.factor_tab_one_problem,
            text="save problem as",
        )

        self.save_var_problem = tk.StringVar(self.factor_tab_one_problem)
        self.save_entry_problem = ttk.Entry(
            master=self.factor_tab_one_problem,
            textvariable=self.save_var_problem,
            justify=tk.LEFT,
            width=15,
        )

        self.save_entry_problem.insert(index=tk.END, string=oldname)

        self.save_label_problem.grid(
            row=count_factors_problem, column=0, sticky="nsew"
        )
        self.save_entry_problem.grid(
            row=count_factors_problem, column=1, sticky="nsew"
        )

        self.problem_factors_list.append(self.save_var_problem)
        self.problem_factors_types.append(str)

        count_factors_problem += 1

        for _, factor_type in enumerate(
            self.problem_object().specifications, start=0
        ):
            # (factor_type, len(self.problem_object().specifications[factor_type]['default']) )

            self.dictionary_size_problem = len(
                self.problem_object().specifications[factor_type]
            )
            datatype = (
                self.problem_object()
                .specifications[factor_type]
                .get("datatype")
            )
            description = (
                self.problem_object()
                .specifications[factor_type]
                .get("description")
            )
            default = (
                self.problem_object().specifications[factor_type].get("default")
            )

            if datatype is not bool:
                self.int_float_description_problem = tk.Label(
                    master=self.factor_tab_one_problem,
                    text=str(description),
                    wraplength=150,
                )

                self.int_float_var_problem = tk.StringVar(
                    self.factor_tab_one_problem
                )
                self.int_float_entry_problem = ttk.Entry(
                    master=self.factor_tab_one_problem,
                    textvariable=self.int_float_var_problem,
                    justify=tk.LEFT,
                    width=15,
                )
                if args and len(args) == 2 and args[0]:
                    self.int_float_entry_problem.insert(
                        index=tk.END, string=str(args[1][3][0][factor_type])
                    )
                elif datatype is tuple and len(default) == 1:
                    # (factor_type, len(self.problem_object().specifications[factor_type]['default']) )
                    # self.int_float_entry_problem.insert(index=tk.END, string=str(self.problem_object().specifications[factor_type].get("default")))
                    self.int_float_entry_problem.insert(
                        index=tk.END, string=str(default[0])
                    )
                else:
                    self.int_float_entry_problem.insert(
                        index=tk.END, string=str(default)
                    )

                self.int_float_description_problem.grid(
                    row=count_factors_problem, column=0, sticky="nsew"
                )
                self.int_float_entry_problem.grid(
                    row=count_factors_problem, column=1, sticky="nsew"
                )

                self.problem_factors_list.append(self.int_float_var_problem)
                if datatype is not tuple:
                    self.problem_factors_types.append(datatype)
                else:
                    self.problem_factors_types.append(str)

                count_factors_problem += 1

            if datatype is bool:
                self.boolean_description_problem = tk.Label(
                    master=self.factor_tab_one_problem,
                    text=str(description),
                    wraplength=150,
                )
                self.boolean_var_problem = tk.BooleanVar(
                    self.factor_tab_one_problem, value=bool(default)
                )
                self.boolean_menu_problem = tk.Checkbutton(
                    self.factor_tab_one_problem,
                    variable=self.boolean_var_problem.get(),
                    onvalue=True,
                    offvalue=False,
                )
                self.boolean_description_problem.grid(
                    row=count_factors_problem, column=0, sticky="nsew"
                )
                self.boolean_menu_problem.grid(
                    row=count_factors_problem, column=1, sticky="nsew"
                )
                self.problem_factors_list.append(self.boolean_var_problem)
                self.problem_factors_types.append(datatype)

                count_factors_problem += 1

        # self.factor_label_frame_problem.place(x=400, y=70, height=300, width=475)
        self.factor_label_frame_problem.place(
            relx=0.35, rely=0.15, relheight=0.33, relwidth=0.34
        )

        # Switching from Problems to Oracles

        self.oracle_factors_list = []
        self.oracle_factors_types = []

        ## Rina Adding After this
        problem = str(self.problem_var.get())
        self.oracle = model_problem_unabbreviated_directory[
            problem
        ]  # returns model string
        self.oracle_object = model_directory[self.oracle]
        ##self.oracle = problem.split("-")
        ##self.oracle = self.oracle[0]
        ##self.oracle_object = model_directory[self.oracle]

        ## Stop adding for Rina

        self.factor_label_frame_oracle = ttk.LabelFrame(
            master=self.master, text="Model Factors"
        )

        self.factor_canvas_oracle = tk.Canvas(
            master=self.factor_label_frame_oracle, borderwidth=0
        )

        self.factor_frame_oracle = ttk.Frame(master=self.factor_canvas_oracle)
        self.vert_scroll_bar_factor_oracle = Scrollbar(
            self.factor_label_frame_oracle,
            orient="vertical",
            command=self.factor_canvas_oracle.yview,
        )
        self.horiz_scroll_bar_factor_oracle = Scrollbar(
            self.factor_label_frame_oracle,
            orient="horizontal",
            command=self.factor_canvas_oracle.xview,
        )
        self.factor_canvas_oracle.configure(
            xscrollcommand=self.horiz_scroll_bar_factor_oracle.set,
            yscrollcommand=self.vert_scroll_bar_factor_oracle.set,
        )

        self.vert_scroll_bar_factor_oracle.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_oracle.pack(side="bottom", fill="x")

        self.factor_canvas_oracle.pack(side="left", fill="both", expand=True)
        self.factor_canvas_oracle.create_window(
            (0, 0),
            window=self.factor_frame_oracle,
            anchor="nw",
            tags="self.factor_frame_oracle",
        )

        self.factor_frame_oracle.bind(
            "<Configure>", self.on_frame_configure_factor_oracle
        )

        self.factor_notebook_oracle = ttk.Notebook(
            master=self.factor_frame_oracle
        )
        self.factor_notebook_oracle.pack(fill="both")

        self.factor_tab_one_oracle = tk.Frame(
            master=self.factor_notebook_oracle
        )

        self.factor_notebook_oracle.add(
            self.factor_tab_one_oracle, text=str(self.oracle + " Factors")
        )

        self.factor_tab_one_oracle.grid_rowconfigure(0)

        self.factor_heading_list_oracle = ["Description", "Input"]

        for heading in self.factor_heading_list_oracle:
            self.factor_tab_one_oracle.grid_columnconfigure(
                self.factor_heading_list_oracle.index(heading)
            )
            label_oracle = tk.Label(
                master=self.factor_tab_one_oracle,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label_oracle.grid(
                row=0,
                column=self.factor_heading_list_oracle.index(heading),
                padx=10,
                pady=3,
            )

        count_factors_oracle = 1
        for factor_type in self.oracle_object().specifications:
            self.dictionary_size_oracle = len(
                self.oracle_object().specifications[factor_type]
            )
            datatype = (
                self.oracle_object().specifications[factor_type].get("datatype")
            )
            description = (
                self.oracle_object()
                .specifications[factor_type]
                .get("description")
            )
            default = (
                self.oracle_object().specifications[factor_type].get("default")
            )

            if datatype is not bool:
                # ("yes?")
                self.int_float_description_oracle = tk.Label(
                    master=self.factor_tab_one_oracle,
                    text=str(description),
                    wraplength=150,
                )

                self.int_float_var_oracle = tk.StringVar(
                    self.factor_tab_one_oracle
                )
                self.int_float_entry_oracle = ttk.Entry(
                    master=self.factor_tab_one_oracle,
                    textvariable=self.int_float_var_oracle,
                    justify=tk.LEFT,
                    width=15,
                )

                if args and len(args) == 2 and args[0]:
                    self.int_float_entry_oracle.insert(
                        index=tk.END, string=str(args[1][4][0][factor_type])
                    )
                else:
                    self.int_float_entry_oracle.insert(
                        index=tk.END, string=str(default)
                    )

                self.int_float_description_oracle.grid(
                    row=count_factors_oracle, column=0, sticky="nsew"
                )
                self.int_float_entry_oracle.grid(
                    row=count_factors_oracle, column=1, sticky="nsew"
                )

                self.oracle_factors_list.append(self.int_float_var_oracle)
                if not datatype is not tuple:
                    self.oracle_factors_types.append(datatype)
                else:
                    self.oracle_factors_types.append(str)

                count_factors_oracle += 1

            if datatype is bool:
                self.boolean_description_oracle = tk.Label(
                    master=self.factor_tab_one_oracle,
                    text=str(description),
                    wraplength=150,
                )
                self.boolean_var_oracle = tk.BooleanVar(
                    self.factor_tab_one_oracle, value=bool(default)
                )
                self.boolean_menu_oracle = tk.Checkbutton(
                    self.factor_tab_one_oracle,
                    variable=self.boolean_var_oracle.get(),
                    onvalue=True,
                    offvalue=False,
                )
                self.boolean_description_oracle.grid(
                    row=count_factors_oracle, column=0, sticky="nsew"
                )
                self.boolean_menu_oracle.grid(
                    row=count_factors_oracle, column=1, sticky="nsew"
                )
                self.oracle_factors_list.append(self.boolean_var_oracle)
                self.oracle_factors_types.append(datatype)

                count_factors_oracle += 1

        self.factor_label_frame_oracle.place(
            relx=0.7, rely=0.15, relheight=0.33, relwidth=0.3
        )
        if str(self.solver_var.get()) != "Solver":
            self.add_button.place(x=10, rely=0.48, width=200, height=30)

    def show_solver_factors(self, *args: tuple) -> None:
        """Show the solver factors.

        Parameters
        ----------
        *args : tuple
            The arguments

        """
        if args and len(args) == 3 and not args[2]:
            pass
        else:
            self.update_problem_list_compatability()

        self.solver_factors_list = []
        self.solver_factors_types = []

        self.factor_label_frame_solver = ttk.LabelFrame(
            master=self.master, text="Solver Factors"
        )

        self.factor_canvas_solver = tk.Canvas(
            master=self.factor_label_frame_solver, borderwidth=0
        )

        self.factor_frame_solver = ttk.Frame(master=self.factor_canvas_solver)
        self.vert_scroll_bar_factor_solver = Scrollbar(
            self.factor_label_frame_solver,
            orient="vertical",
            command=self.factor_canvas_solver.yview,
        )
        self.horiz_scroll_bar_factor_solver = Scrollbar(
            self.factor_label_frame_solver,
            orient="horizontal",
            command=self.factor_canvas_solver.xview,
        )
        self.factor_canvas_solver.configure(
            xscrollcommand=self.horiz_scroll_bar_factor_solver.set,
            yscrollcommand=self.vert_scroll_bar_factor_solver.set,
        )

        self.vert_scroll_bar_factor_solver.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_solver.pack(side="bottom", fill="x")

        self.factor_canvas_solver.pack(side="left", fill="both", expand=True)
        self.factor_canvas_solver.create_window(
            (0, 0),
            window=self.factor_frame_solver,
            anchor="nw",
            tags="self.factor_frame_solver",
        )

        self.factor_frame_solver.bind(
            "<Configure>", self.on_frame_configure_factor_solver
        )

        self.factor_notebook_solver = ttk.Notebook(
            master=self.factor_frame_solver
        )
        self.factor_notebook_solver.pack(fill="both")

        self.factor_tab_one_solver = tk.Frame(
            master=self.factor_notebook_solver
        )

        self.factor_notebook_solver.add(
            self.factor_tab_one_solver,
            text=str(self.solver_var.get()) + " Factors",
        )

        self.factor_tab_one_solver.grid_rowconfigure(0)

        self.factor_heading_list_solver = ["Description", "Input"]

        for heading in self.factor_heading_list_solver:
            self.factor_tab_one_solver.grid_columnconfigure(
                self.factor_heading_list_solver.index(heading)
            )
            label = tk.Label(
                master=self.factor_tab_one_solver,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=0,
                column=self.factor_heading_list_solver.index(heading),
                padx=10,
                pady=3,
            )

        self.solver_object = solver_unabbreviated_directory[
            self.solver_var.get()
        ]

        count_factors_solver = 1

        self.save_label_solver = tk.Label(
            master=self.factor_tab_one_solver,
            text="save solver as",
        )

        if args and len(args) == 3 and args[0]:
            oldname = args[1][5][1]

        else:
            solver_object = solver_unabbreviated_directory[
                self.solver_var.get()
            ]
            oldname = solver_object().name

        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(
            master=self.factor_tab_one_solver,
            textvariable=self.save_var_solver,
            justify=tk.LEFT,
            width=15,
        )

        self.save_entry_solver.insert(index=tk.END, string=oldname)

        self.save_label_solver.grid(
            row=count_factors_solver, column=0, sticky="nsew"
        )
        self.save_entry_solver.grid(
            row=count_factors_solver, column=1, sticky="nsew"
        )

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)

        count_factors_solver += 1

        for factor_type in self.solver_object().specifications:
            # ("size of dictionary", len(self.solver_object().specifications[factor_type]))
            # ("first", factor_type)
            # ("second", self.solver_object().specifications[factor_type].get("description"))
            # ("third", self.solver_object().specifications[factor_type].get("datatype"))
            # ("fourth", self.solver_object().specifications[factor_type].get("default"))

            self.dictionary_size_solver = len(
                self.solver_object().specifications[factor_type]
            )
            datatype = (
                self.solver_object().specifications[factor_type].get("datatype")
            )
            description = (
                self.solver_object()
                .specifications[factor_type]
                .get("description")
            )
            default = (
                self.solver_object().specifications[factor_type].get("default")
            )
            if datatype is not bool:
                self.int_float_description = tk.Label(
                    master=self.factor_tab_one_solver,
                    text=str(description),
                    wraplength=150,
                )

                self.int_float_var = tk.StringVar(self.factor_tab_one_solver)
                self.int_float_entry = ttk.Entry(
                    master=self.factor_tab_one_solver,
                    textvariable=self.int_float_var,
                    justify=tk.LEFT,
                    width=15,
                )

                if args and len(args) == 3 and args[0]:
                    self.int_float_entry.insert(
                        index=tk.END, string=str(args[1][5][0][factor_type])
                    )
                else:
                    self.int_float_entry.insert(
                        index=tk.END, string=str(default)
                    )

                self.int_float_description.grid(
                    row=count_factors_solver, column=0, sticky="nsew"
                )
                self.int_float_entry.grid(
                    row=count_factors_solver, column=1, sticky="nsew"
                )
                self.solver_factors_list.append(self.int_float_var)

                if datatype is not tuple:
                    self.solver_factors_types.append(datatype)
                else:
                    self.solver_factors_types.append(str)

                count_factors_solver += 1

            if datatype is bool:
                self.boolean_description = tk.Label(
                    master=self.factor_tab_one_solver,
                    text=str(description),
                    wraplength=150,
                )
                self.boolean_var = tk.BooleanVar(
                    self.factor_tab_one_solver, value=bool(default)
                )
                self.boolean_menu = tk.Checkbutton(
                    self.factor_tab_one_solver,
                    variable=self.boolean_var.get(),
                    onvalue=True,
                    offvalue=False,
                )
                self.boolean_description.grid(
                    row=count_factors_solver, column=0, sticky="nsew"
                )
                self.boolean_menu.grid(
                    row=count_factors_solver, column=1, sticky="nsew"
                )
                self.solver_factors_list.append(self.boolean_var)
                self.solver_factors_types.append(datatype)

                count_factors_solver += 1

        # self.factor_label_frame_problem.place(relx=.32, y=70, height=150, relwidth=.34)
        self.factor_label_frame_solver.place(
            x=10, rely=0.15, relheight=0.33, relwidth=0.34
        )
        if str(self.problem_var.get()) != "Problem":
            self.add_button.place(x=10, rely=0.48, width=200, height=30)

    # Creates a function that checks the compatibility of the solver picked with the list of problems and adds
    # the compatible problems to a new list
    def update_problem_list_compatability(self) -> None:
        """Update the problem list compatibility."""
        if self.solver_var.get() != "Solver":
            self.problem_menu.destroy()
            temp_problem_list = []

            for problem in problem_unabbreviated_directory:
                temp_problem = problem_unabbreviated_directory[
                    problem
                ]  # problem object
                temp_problem_name = temp_problem().name

                temp_solver = solver_unabbreviated_directory[
                    self.solver_var.get()
                ]
                temp_solver_name = temp_solver().name

                temp_experiment = ProblemSolver(
                    solver_name=temp_solver_name, problem_name=temp_problem_name
                )
                comp = temp_experiment.check_compatibility()

                if comp == "":
                    temp_problem_list.append(problem)

            # from experiments.inputs.all_factors.py:
            self.problem_list = temp_problem_list
            # stays the same, has to change into a special type of variable via tkinter function
            self.problem_var = tk.StringVar(master=self.master)
            # sets the default OptionMenu value

            # creates drop down menu, for tkinter, it is called "OptionMenu"
            self.problem_menu = ttk.OptionMenu(
                self.master,
                self.problem_var,
                "Problem",
                *self.problem_list,
                command=self.show_problem_factors,
            )
            self.problem_menu.place(relx=0.4, rely=0.1)

    def clear_row_function(self, row_index: int) -> None:
        """Clear the row.

        Parameters
        ----------
        row_index : int
            Row to clear

        """
        for widget in self.widget_list[row_index - 1]:
            widget.grid_remove()

        self.experiment_master_list.pop(row_index - 1)
        self.experiment_object_list.pop(row_index - 1)
        self.widget_list.pop(row_index - 1)

        self.check_box_list[row_index - 1].grid_remove()

        self.check_box_list.pop(row_index - 1)
        self.check_box_list_var.pop(row_index - 1)

        # if (integer - 1) in self.normalize_list:
        #     self.normalize_list.remove(integer - 1)
        # for i in range(len(self.normalize_list)):
        #     if i < self.normalize_list[i]:
        #         self.normalize_list[i] = self.normalize_list[i] - 1

        for row_of_widgets in self.widget_list:
            row_index = self.widget_list.index(row_of_widgets)
            row_of_widgets[7]["text"] = str(row_index + 1)

            run_button_added = row_of_widgets[3]
            text_on_run = run_button_added["text"]
            split_text = text_on_run.split(" ")
            split_text[len(split_text) - 1] = str(row_index + 1)
            # new_text = " ".join(split_text)
            # run_button_added["text"] = new_text
            run_button_added["command"] = partial(
                self.run_row_function, row_index + 1
            )

            row_of_widgets[3] = run_button_added

            view_edit_button_added = row_of_widgets[4]
            text_on_view_edit = view_edit_button_added["text"]
            split_text = text_on_view_edit.split(" ")
            split_text[len(split_text) - 1] = str(row_index + 1)
            # new_text = " ".join(split_text)
            # viewEdit_button_added["text"] = new_text
            view_edit_button_added["command"] = partial(
                self.view_edit_function, row_index + 1
            )

            row_of_widgets[4] = view_edit_button_added

            clear_button_added = row_of_widgets[5]
            text_on_clear = clear_button_added["text"]
            split_text = text_on_clear.split(" ")
            split_text[len(split_text) - 1] = str(row_index + 1)
            # new_text = " ".join(split_text)
            # clear_button_added["text"] = new_text
            clear_button_added["command"] = partial(
                self.clear_row_function, row_index + 1
            )

            row_of_widgets[5] = clear_button_added

            postprocess_button_added = row_of_widgets[6]
            postprocess_button_added["command"] = partial(
                self.post_rep_function, row_index + 1
            )

            row_of_widgets[6] = postprocess_button_added

            current_check_box = self.check_box_list[row_index]
            current_check_box.grid(
                row=(row_index + 1), column=0, sticky="nsew", padx=10, pady=3
            )
            # TODO: figure out why ihdex 7 maps to column 1
            row_of_widgets[7].grid(
                row=(row_index + 1), column=1, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[0].grid(
                row=(row_index + 1), column=2, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[1].grid(
                row=(row_index + 1), column=3, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[2].grid(
                row=(row_index + 1), column=4, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[3].grid(
                row=(row_index + 1), column=5, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[4].grid(
                row=(row_index + 1), column=6, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[5].grid(
                row=(row_index + 1), column=7, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[6].grid(
                row=(row_index + 1), column=8, sticky="nsew", padx=10, pady=3
            )

        self.count_experiment_queue = len(self.widget_list) + 1

    def clear_meta_function(self, row_index: int) -> None:
        """Clear the meta function.

        Parameters
        ----------
        row_index : int
            The integer

        """
        for widget in self.widget_meta_list[row_index - 1]:
            widget.grid_remove()

        self.meta_experiment_master_list.pop(row_index - 1)

        self.widget_meta_list.pop(row_index - 1)

        for row_of_widgets in self.widget_meta_list:
            row_index = self.widget_meta_list.index(row_of_widgets)

            run_button_added = row_of_widgets[3]
            text_on_run = run_button_added["text"]
            split_text = text_on_run.split(" ")
            split_text[len(split_text) - 1] = str(row_index + 1)
            new_text = " ".join(split_text)
            run_button_added["text"] = new_text
            run_button_added["command"] = partial(
                self.run_meta_function, row_index + 1
            )
            row_of_widgets[3] = run_button_added

            clear_button_added = row_of_widgets[4]
            text_on_clear = clear_button_added["text"]
            split_text = text_on_clear.split(" ")
            split_text[len(split_text) - 1] = str(row_index + 1)
            new_text = " ".join(split_text)
            clear_button_added["text"] = new_text
            clear_button_added["command"] = partial(
                self.clear_meta_function, row_index + 1
            )
            row_of_widgets[4] = clear_button_added

            postprocess_button_added = row_of_widgets[5]
            postprocess_button_added["command"] = partial(
                self.post_rep_meta_function, row_index + 1
            )
            row_of_widgets[5] = postprocess_button_added

            plot_button_added = row_of_widgets[6]
            plot_button_added["command"] = partial(
                self.plot_meta_function, row_index + 1
            )
            row_of_widgets[6] = plot_button_added

            view_button_added = row_of_widgets[7]
            view_button_added["command"] = partial(
                self.view_meta_function, row_index + 1
            )
            row_of_widgets[7] = view_button_added

            row_of_widgets[0].grid(
                row=(row_index + 1), column=0, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[1].grid(
                row=(row_index + 1), column=1, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[2].grid(
                row=(row_index + 1), column=2, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[3].grid(
                row=(row_index + 1), column=3, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[4].grid(
                row=(row_index + 1), column=4, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[5].grid(
                row=(row_index + 1), column=5, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[6].grid(
                row=(row_index + 1), column=6, sticky="nsew", padx=10, pady=3
            )
            row_of_widgets[7].grid(
                row=(row_index + 1), column=6, sticky="nsew", padx=10, pady=3
            )

        # self.count_meta_experiment_queue = len(self.widget_meta_list) + 1
        self.count_meta_experiment_queue = self.count_meta_experiment_queue - 1

        # resets problem_var to default value
        self.problem_var.set("Problem")
        # resets solver_var to default value
        self.solver_var.set("Solver")

    def view_edit_function(self, row_index: int) -> None:
        """View the edit function.

        Parameters
        ----------
        row_index : int
            The integer

        """
        self.experiment_object_list[row_index - 1]
        # (current_experiment)
        current_experiment_arguments = self.experiment_master_list[
            row_index - 1
        ]

        self.problem_var.set(current_experiment_arguments[0])
        # self.problem_var.set(problem_solver_abbreviated_name_to_unabbreviated(current_experiment_arguments[0], problem_directory, problem_unabbreviated_directory))

        self.solver_var.set(current_experiment_arguments[1])
        # self.solver_var.set(problem_solver_abbreviated_name_to_unabbreviated(current_experiment_arguments[1], solver_directory, solver_unabbreviated_directory))'

        self.macro_var.set(current_experiment_arguments[2])
        self.show_problem_factors(True, current_experiment_arguments)
        # print(" self.show_problem_factors", self.show_problem_factors(True, current_experiment_arguments))
        # self.my_experiment[1][3][1]
        self.show_solver_factors(True, current_experiment_arguments, False)
        # print("self.show_solver_factors", self. show_solver_factors(True, current_experiment_arguments))
        view_edit_button_added = self.widget_list[row_index - 1][5]
        view_edit_button_added["text"] = "Save Changes"
        view_edit_button_added["command"] = partial(
            self.save_edit_function, row_index
        )
        view_edit_button_added.grid(
            row=(row_index), column=5, sticky="nsew", padx=10, pady=3
        )

    def clear_queue(self) -> None:
        """Clear the queue."""
        # for row in self.widget_list:
        #     for widget in row:
        #         widget.grid_remove()
        for row in range(len(self.widget_list), 0, -1):
            self.clear_row_function(row)

        self.experiment_master_list.clear()
        self.experiment_object_list.clear()
        self.widget_list.clear()

    # TODO: change away from *args
    def add_experiment(self, *args: tuple) -> None:
        """Add an experiment.

        Parameters
        ----------
        *args : tuple
            Arguments D:

        """
        if len(args) == 1 and args[0] is int:
            place = args[0] - 1
        else:
            place = len(self.experiment_object_list)

        if (
            self.problem_var.get() in problem_unabbreviated_directory
            and self.solver_var.get() in solver_unabbreviated_directory
            and self.macro_entry.get().isnumeric()
        ):
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

                solver_object, self.solver_name = (
                    problem_solver_unabbreviated_to_object(
                        self.solver_name, solver_unabbreviated_directory
                    )
                )
                problem_object, self.problem_name = (
                    problem_solver_unabbreviated_to_object(
                        self.problem_name, problem_unabbreviated_directory
                    )
                )

                # self.selected[0] = self.problem_name

                self.my_experiment = ProblemSolver(
                    solver_name=self.solver_name,
                    problem_name=self.problem_name,
                    solver_rename=self.solver_rename,
                    problem_rename=self.problem_rename,
                    solver_fixed_factors=self.solver_factors,
                    problem_fixed_factors=self.problem_factors,
                    model_fixed_factors=self.oracle_factors,
                )
                # print("type", type(self.selected[2]))
                self.my_experiment.n_macroreps = self.selected[2]
                self.my_experiment.post_norm_ready = False

                compatibility_result = self.my_experiment.check_compatibility()
                for exp in self.experiment_object_list:
                    if (
                        exp.problem.name == self.my_experiment.problem.name
                        and exp.solver.name == self.my_experiment.solver.name
                    ):
                        if exp.problem != self.my_experiment.problem:
                            message = "Please Save the Problem for Unique Factors with a Unique Name"
                            tk.messagebox.showerror(
                                title="Error Window", message=message
                            )
                            return False

                if compatibility_result == "":
                    self.experiment_object_list.insert(
                        place, self.my_experiment
                    )
                    self.experiment_master_list.insert(place, self.selected)
                    # this option list doesnt autoupdate - not sure why but this will force it to update
                    self.experiment_master_list[place][5][0][
                        "crn_across_solns"
                    ] = self.boolean_var.get()

                    self.rows = 5

                    self.problem_added = tk.Label(
                        master=self.tab_one,
                        text=self.selected[3][1],
                        justify="center",
                    )
                    self.problem_added.grid(
                        row=self.count_experiment_queue,
                        column=2,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.checkbox_select_var = tk.BooleanVar(
                        self.tab_one, value=False
                    )
                    self.checkbox_select = tk.Checkbutton(
                        master=self.tab_one,
                        text="",
                        state="normal",
                        variable=self.checkbox_select_var,
                    )
                    self.checkbox_select.deselect()
                    self.checkbox_select.grid(
                        row=self.count_experiment_queue,
                        column=0,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.exp_num = tk.Label(
                        master=self.tab_one,
                        text=str(self.count_experiment_queue),
                        justify="center",
                    )
                    self.exp_num.grid(
                        row=self.count_experiment_queue,
                        column=1,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.solver_added = tk.Label(
                        master=self.tab_one,
                        text=self.selected[5][1],
                        justify="center",
                    )
                    self.solver_added.grid(
                        row=self.count_experiment_queue,
                        column=3,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.macros_added = tk.Label(
                        master=self.tab_one,
                        text=self.selected[2],
                        justify="center",
                    )
                    self.macros_added.grid(
                        row=self.count_experiment_queue,
                        column=4,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.run_button_added = ttk.Button(
                        master=self.tab_one,
                        text="Run",
                        command=partial(
                            self.run_row_function, self.count_experiment_queue
                        ),
                    )
                    self.run_button_added.grid(
                        row=self.count_experiment_queue,
                        column=5,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.viewEdit_button_added = ttk.Button(
                        master=self.tab_one,
                        text="View / Edit",
                        command=partial(
                            self.view_edit_function, self.count_experiment_queue
                        ),
                    )
                    self.viewEdit_button_added.grid(
                        row=self.count_experiment_queue,
                        column=6,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.clear_button_added = ttk.Button(
                        master=self.tab_one,
                        text="Remove",
                        command=partial(
                            self.clear_row_function, self.count_experiment_queue
                        ),
                    )
                    self.clear_button_added.grid(
                        row=self.count_experiment_queue,
                        column=7,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.postprocess_button_added = ttk.Button(
                        master=self.tab_one,
                        text="Post-Process",
                        command=partial(
                            self.post_rep_function, self.count_experiment_queue
                        ),
                        state="disabled",
                    )
                    self.postprocess_button_added.grid(
                        row=self.count_experiment_queue,
                        column=8,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.widget_row = [
                        self.problem_added,
                        self.solver_added,
                        self.macros_added,
                        self.run_button_added,
                        self.viewEdit_button_added,
                        self.clear_button_added,
                        self.postprocess_button_added,
                        self.exp_num,
                    ]
                    self.check_box_list.append(self.checkbox_select)
                    self.check_box_list_var.append(self.checkbox_select_var)

                    self.widget_list.insert(place, self.widget_row)

                    # separator = ttk.Separator(master=self.tab_one, orient='horizontal')

                    # separator.place(x=0.1, y=self.prev, relwidth=1)
                    # self.prev += 32

                    self.count_experiment_queue += 1

                else:
                    tk.messagebox.showerror(
                        title="Error Window", message=compatibility_result
                    )
                    self.selected.clear()

            else:
                # reset macro_entry to "10"
                self.macro_entry.delete(0, len(self.macro_entry.get()))
                # resets macro_entry textbox
                self.macro_entry.insert(index=tk.END, string="10")

                message = "Please enter a postivie (non zero) integer for the number of Macroreplications, example: 10"
                tk.messagebox.showerror(title="Error Window", message=message)

            # s selected (list) in console/terminal
            # ("it works", self.experiment_master_list)
            self.notebook.select(self.tab_one)
            return self.experiment_master_list

        # problem selected, but solver NOT selected
        elif (
            self.problem_var.get() in problem_unabbreviated_directory
            and self.solver_var.get() not in solver_unabbreviated_directory
        ):
            message = "You have not selected a Solver!"
            tk.messagebox.showerror(title="Error Window", message=message)

        # problem NOT selected, but solver selected
        elif (
            self.problem_var.get() not in problem_unabbreviated_directory
            and self.solver_var.get() in solver_unabbreviated_directory
        ):
            message = "You have not selected a Problem!"
            tk.messagebox.showerror(title="Error Window", message=message)

        # macro_entry not numeric or negative
        elif not self.macro_entry.get().isnumeric():
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

    def confirm_problem_factors(self) -> list:
        """Confirm the problem factors.

        Returns
        -------
        list
            The problem factors

        """
        self.problem_factors_return = []
        self.problem_factors_dictionary = dict()

        keys = list(self.problem_object().specifications.keys())
        # ("keys ->", keys)
        # ("self.problem_factors_types -> ", self.problem_factors_types)

        for problem_factor in self.problem_factors_list:
            # (problem_factor.get() + " " + str(type(problem_factor.get())))
            index = self.problem_factors_list.index(problem_factor)

            # (problem_factor.get())
            if index == 0:
                if problem_factor.get() == self.problem_var.get():
                    # self.problem_object().specifications[factor_type].get("default")
                    # self.problem_factors_return.append(None)
                    self.problem_factors_return.append(problem_factor.get())
                else:
                    self.problem_factors_return.append(problem_factor.get())
                    # self.problem_factors_dictionary["rename"] = problem_factor.get()

            if index > 0:
                # (self.problem_factors_types[index])
                # datatype = self.problem_factors_types[index]

                # if the data type is tuple update data
                # self.problem_factors_dictionary[keys[index]] = datatype(nextVal)
                # (ast.literal_eval(problem_factor.get()) , keys[index])
                if keys[index - 1] == "initial_solution" and isinstance(
                    type(ast.literal_eval(problem_factor.get())), int
                ):
                    t = (ast.literal_eval(problem_factor.get()),)
                    # (t)
                    self.problem_factors_dictionary[keys[index - 1]] = t
                else:
                    self.problem_factors_dictionary[keys[index - 1]] = (
                        ast.literal_eval(problem_factor.get())
                    )
                # ("datatype of factor -> ", type(datatype(problem_factor.get())))

        self.problem_factors_return.insert(0, self.problem_factors_dictionary)
        return self.problem_factors_return

    def confirm_oracle_factors(self) -> list:
        """Confirm the oracle factors.

        Returns
        -------
        list
            The oracle factors

        """
        self.oracle_factors_return = []
        self.oracle_factors_dictionary = dict()

        keys = list(self.oracle_object().specifications.keys())
        # ("keys ->", keys)
        # ("self.oracle_factors_types -> ", self.oracle_factors_types)

        keys = list(self.oracle_object().specifications.keys())

        for oracle_factor in self.oracle_factors_list:
            index = self.oracle_factors_list.index(oracle_factor)
            self.oracle_factors_dictionary[keys[index]] = oracle_factor.get()
            # (self.oracle_factors_types[index])

            datatype = self.oracle_factors_types[index]
            if str(datatype) == "<class 'list'>":
                new_list = ast.literal_eval(oracle_factor.get())

                self.oracle_factors_dictionary[keys[index]] = new_list
            else:
                self.oracle_factors_dictionary[keys[index]] = datatype(
                    oracle_factor.get()
                )
            # (str(datatype(oracle_factor.get())) + " " + str(datatype))
            # ("datatype of factor -> ", type(datatype(oracle_factor.get())))

        self.oracle_factors_return.append(self.oracle_factors_dictionary)
        return self.oracle_factors_return

    def confirm_solver_factors(self) -> list:
        """Confirm the solver factors.

        Returns
        -------
        list
            The solver factors

        """
        self.solver_factors_return = []
        self.solver_factors_dictionary = dict()

        keys = list(self.solver_object().specifications.keys())
        # ("keys ->", keys)
        # ("self.solver_factors_types -> ", self.solver_factors_types)

        for solver_factor in self.solver_factors_list:
            index = self.solver_factors_list.index(solver_factor)
            # (solver_factor.get())
            if index == 0:
                if solver_factor.get() == self.solver_var.get():
                    # self.solver_factors_return.append(None)
                    self.solver_factors_return.append(solver_factor.get())
                else:
                    self.solver_factors_return.append(solver_factor.get())
                    # self.solver_factors_dictionary["rename"] = solver_factor.get()
            if index > 0:
                # (self.solver_factors_types[index])
                datatype = self.solver_factors_types[index]
                self.solver_factors_dictionary[keys[index - 1]] = datatype(
                    solver_factor.get()
                )
                # ("datatype of factor -> ", type(datatype(solver_factor.get())))

        self.solver_factors_return.insert(0, self.solver_factors_dictionary)
        return self.solver_factors_return

    def on_frame_configure_queue(self, event: tk.Event) -> None:
        """Configure the queue.

        Parameters
        ----------
        event : tk.Event
            Event triggering the function

        """
        self.queue_canvas.configure(scrollregion=self.queue_canvas.bbox("all"))

    def on_frame_configure_factor_problem(self, event: tk.Event) -> None:
        """Configure the problem factors.

        Parameters
        ----------
        event : tk.Event
            Event triggering the function

        """
        self.factor_canvas_problem.configure(
            scrollregion=self.factor_canvas_problem.bbox("all")
        )

    def on_frame_configure_factor_solver(self, event: tk.Event) -> None:
        """Configure the solver factors.

        Parameters
        ----------
        event : tk.Event
            Event triggering the function

        """
        self.factor_canvas_solver.configure(
            scrollregion=self.factor_canvas_solver.bbox("all")
        )

    def on_frame_configure_factor_oracle(self, event: tk.Event) -> None:
        """Configure the oracle factors.

        Parameters
        ----------
        event : tk.Event
            Event triggering the function

        """
        self.factor_canvas_oracle.configure(
            scrollregion=self.factor_canvas_oracle.bbox("all")
        )

    def save_edit_function(self, row_index: int) -> None:
        """Save the edit function.

        Parameters
        ----------
        row_index : int
            Index of the row

        """
        self.experiment_master_list[row_index - 1]
        self.experiment_master_list[row_index - 1][5][0]["crn_across_solns"] = (
            self.boolean_var.get()
        )

        if self.add_experiment(row_index):
            self.clear_row_function(row_index + 1)

            # resets problem_var to default value
            self.problem_var.set("Problem")
            # resets solver_var to default value
            self.solver_var.set("Solver")

            self.factor_label_frame_problem.destroy()
            self.factor_label_frame_oracle.destroy()
            self.factor_label_frame_solver.destroy()

    def select_pickle_file_fuction(self, *args: tuple) -> None:
        """Load a pickle file.

        Parameters
        ----------
        *args : tuple
            Arguments

        """
        filename = filedialog.askopenfilename(
            parent=self.master,
            initialdir=EXPERIMENT_DIR,
            title="Select Pickle File",
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

    def load_pickle_file_function(self) -> None:
        """Load data from a pickle file."""
        self.select_pickle_file_fuction()

        filename = self.pickle_file_pathname_show["text"]
        acceptable_types = ["pickle", "pck", "pcl", "pkl", "db"]

        if filename != "No file selected":
            filetype = filename.split(".")
            filetype = filetype[len(filetype) - 1]
            if filetype in acceptable_types:
                experiment_pathname = filename[filename.index(EXPERIMENT_DIR) :]

                pickle_file = experiment_pathname
                infile = open(pickle_file, "rb")
                new_dict = pickle.load(infile)
                infile.close()

                self.my_experiment = new_dict
                compatibility_result = self.my_experiment.check_compatibility()
                place = len(self.experiment_object_list)
                self.my_experiment.post_norm_ready = True

                if compatibility_result == "":
                    self.experiment_object_list.insert(
                        place, self.my_experiment
                    )

                    # filler in master list so that placement stays correct
                    self.experiment_master_list.insert(place, None)

                    self.rows = 5

                    self.checkbox_select_var = tk.BooleanVar(
                        self.tab_one, value=False
                    )
                    self.checkbox_select = tk.Checkbutton(
                        master=self.tab_one,
                        text="",
                        state="normal",
                        variable=self.checkbox_select_var,
                    )
                    self.checkbox_select.deselect()
                    self.checkbox_select.grid(
                        row=self.count_experiment_queue,
                        column=0,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.exp_num = tk.Label(
                        master=self.tab_one,
                        text=str(self.count_experiment_queue),
                        justify="center",
                    )
                    self.exp_num.grid(
                        row=self.count_experiment_queue,
                        column=1,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.problem_added = tk.Label(
                        master=self.tab_one,
                        text=self.my_experiment.problem.name,
                        justify="center",
                    )
                    self.problem_added.grid(
                        row=self.count_experiment_queue,
                        column=2,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.solver_added = tk.Label(
                        master=self.tab_one,
                        text=self.my_experiment.solver.name,
                        justify="center",
                    )
                    self.solver_added.grid(
                        row=self.count_experiment_queue,
                        column=3,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.macros_added = tk.Label(
                        master=self.tab_one,
                        text=self.my_experiment.n_macroreps,
                        justify="center",
                    )
                    self.macros_added.grid(
                        row=self.count_experiment_queue,
                        column=4,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.run_button_added = ttk.Button(
                        master=self.tab_one,
                        text="Run",
                        command=partial(
                            self.run_row_function, self.count_experiment_queue
                        ),
                    )
                    self.run_button_added.grid(
                        row=self.count_experiment_queue,
                        column=5,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.viewEdit_button_added = ttk.Button(
                        master=self.tab_one,
                        text="View / Edit",
                        command=partial(
                            self.view_edit_function, self.count_experiment_queue
                        ),
                    )
                    self.viewEdit_button_added.grid(
                        row=self.count_experiment_queue,
                        column=6,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.clear_button_added = ttk.Button(
                        master=self.tab_one,
                        text="Remove  ",
                        command=partial(
                            self.clear_row_function, self.count_experiment_queue
                        ),
                    )
                    self.clear_button_added.grid(
                        row=self.count_experiment_queue,
                        column=7,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.postprocess_button_added = ttk.Button(
                        master=self.tab_one,
                        text="Post-Process",
                        command=partial(
                            self.post_rep_function, self.count_experiment_queue
                        ),
                        state="disabled",
                    )
                    self.postprocess_button_added.grid(
                        row=self.count_experiment_queue,
                        column=8,
                        sticky="nsew",
                        padx=10,
                        pady=3,
                    )

                    self.widget_row = [
                        self.problem_added,
                        self.solver_added,
                        self.macros_added,
                        self.run_button_added,
                        self.viewEdit_button_added,
                        self.clear_button_added,
                        self.postprocess_button_added,
                        self.exp_num,
                    ]
                    self.widget_list.insert(place, self.widget_row)
                    self.check_box_list.append(self.checkbox_select)
                    self.check_box_list_var.append(self.checkbox_select_var)

                    row_of_widgets = self.widget_list[len(self.widget_list) - 1]
                    if self.my_experiment.has_run:
                        run_button = row_of_widgets[3]
                        run_button["state"] = "disabled"
                        run_button["text"] = "Run Complete"
                        run_button = row_of_widgets[4]
                        run_button["state"] = "disabled"
                        run_button = row_of_widgets[6]
                        run_button["state"] = "normal"
                        self.my_experiment.post_norm_ready = False
                        if self.my_experiment.check_postreplicate():
                            self.experiment_object_list[
                                place
                            ].post_norm_ready = True
                            self.widget_list[place][6]["text"] = (
                                "Post-Processing Complete"
                            )
                            self.widget_list[place][6]["state"] = "disabled"

                        # separator = ttk.Separator(master=self.tab_one, orient='horizontal')

                        # separator.place(x=0.1, y=self.prev, relwidth=1)
                        # self.prev += 32

                    self.count_experiment_queue += 1
                    if self.notebook.index("current") == 2:
                        self.post_norm_setup()

            else:
                message = f"You have loaded a file, but {filetype} files are not acceptable!\nPlease try again."
                tk.messagebox.showwarning(
                    master=self.master, title=" Warning", message=message
                )
        # else:
        #     message = "You are attempting to load a file, but haven't selected one yet.\nPlease select a file first."
        #     tk.messagebox.showwarning(master=self.master, title=" Warning", message=message)

    def run_row_function(self, row_to_run: int) -> None:
        """Run the specified row.

        Parameters
        ----------
        row_to_run : int
            The row to run

        """
        # stringtuple[1:-1].split(separator=",")
        row_index = row_to_run - 1

        # run_button = row_of_widgets[3]
        self.widget_list[row_index][3]["state"] = "disabled"
        self.widget_list[row_index][3]["text"] = "Run Complete"
        self.widget_list[row_index][4]["state"] = "disabled"
        self.widget_list[row_index][6]["state"] = "normal"
        # run_button["state"] = "disabled"
        # run_button = row_of_widgets[4]
        # run_button["state"] = "disabled"
        # row_of_widgets[6]["state"] = "normal"
        # run_button.grid(row=integer, column=3, sticky='nsew', padx=10, pady=3)

        # widget_row = [row_of_widgets[0], row_of_widgets[1], row_of_widgets[2], row_of_widgets[3], run_button, row_of_widgets[4], row_of_widgets[5], row_of_widgets[6],row_of_widgets[7] ]
        # self.widget_list[row_index] = widget_row

        self.my_experiment = self.experiment_object_list[row_index]

        self.selected = self.experiment_master_list[row_index]
        self.macro_reps = self.selected[2]
        self.my_experiment.run(n_macroreps=self.macro_reps)

    def post_rep_function(self, selected_row: int) -> None:
        """Post replicate function.

        Parameters
        ----------
        selected_row : int
            The selected row

        """
        row_index = selected_row - 1
        self.my_experiment = self.experiment_object_list[row_index]
        self.selected = self.experiment_object_list[row_index]
        self.post_rep_function_row_index = selected_row
        # calls postprocessing window

        self.postrep_window = tk.Tk()
        position = center_window(self.master, 0.8)
        self.postrep_window.geometry(position)
        self.postrep_window.title("Post-Processing Page")
        self.app = PostProcessingWindow(
            self.postrep_window, self.my_experiment, self.selected, self
        )

    def post_process_disable_button(self, meta: bool = False) -> None:
        """Disable the post process button in the GUI.

        Parameters
        ----------
        meta : bool, optional
            The boolean, by default False

        """
        if meta:
            row_index = self.post_rep_function_row_index - 1
            self.widget_meta_list[row_index][5]["text"] = (
                "Post-Processed & Post-Normalized"
            )
            self.widget_meta_list[row_index][5]["state"] = "disabled"
            self.widget_meta_list[row_index][6]["state"] = "normal"
            # self.normalize_button_added["state"] = "normal"
        else:
            row_index = self.post_rep_function_row_index - 1
            self.experiment_object_list[row_index].post_norm_ready = True
            self.widget_list[row_index][6]["text"] = "Post-Processing Complete"
            self.widget_list[row_index][6]["state"] = "disabled"
            # self.widget_list[row_index][7]["state"] = "normal"

    def checkbox_function2(self, exp: ProblemSolver, row_num: int) -> None:
        newlist = sorted(
            self.experiment_object_list, key=lambda x: x.problem.name
        )
        prob_name = newlist[row_num].problem.name
        if row_num in self.normalize_list2:
            self.normalize_list2.remove(row_num)
            self.post_norm_exp_list.remove(exp)

            if len(self.normalize_list2) == 0:
                for i in self.widget_norm_list:
                    i[2]["state"] = "normal"
        else:
            self.normalize_list2.append(row_num)
            self.post_norm_exp_list.append(exp)
            for i in self.widget_norm_list:
                if i[0]["text"] != prob_name:
                    i[2]["state"] = "disable"

    def crossdesign_function(self) -> None:
        # self.crossdesign_window = tk.Tk()
        self.crossdesign_window = tk.Toplevel(self.master)
        position = center_window(self.master, 0.8)
        self.crossdesign_window.geometry(position)
        self.crossdesign_window.title("Cross-Design Problem-Solver Group")
        self.cross_app = CrossDesignWindow(self.crossdesign_window, self)

    # My code starts here
    # Open data farming window
    def datafarming_function(self) -> None:
        self.datafarming_window = tk.Toplevel(self.master)
        position = center_window(self.master, 0.8)
        self.datafarming_window.geometry(position)
        self.datafarming_window.title("Data Farming")
        self.datafarming_app = DataFarmingWindow(self.datafarming_window, self)

    # My code ends here

    def add_meta_exp_to_frame(
        self,
        n_macroreps: int | None = None,
        input_meta_experiment: ProblemsSolvers | None = None,
    ) -> None:
        if n_macroreps is None and input_meta_experiment is not None:
            self.cross_app = CrossDesignWindow(
                root=None, main_widow=None, forced_creation=True
            )
            self.cross_app.crossdesign_MetaExperiment = input_meta_experiment
            self.meta_experiment_macro_reps.append("mixed")
            text_macros_added = "mixed"
        elif n_macroreps is not None and input_meta_experiment is None:
            self.meta_experiment_macro_reps.append(int(n_macroreps.get()))
            text_macros_added = n_macroreps.get()

        row_num = self.count_meta_experiment_queue + 1

        self.macros_added = tk.Label(
            master=self.tab_two,
            text=text_macros_added,
            justify="center",
        )
        self.macros_added.grid(
            row=row_num, column=2, sticky="nsew", padx=10, pady=3
        )

        self.problem_added = tk.Label(
            master=self.tab_two,
            text=self.cross_app.crossdesign_MetaExperiment.problem_names,
            justify="center",
        )
        self.problem_added.grid(
            row=row_num, column=0, sticky="nsew", padx=10, pady=3
        )

        self.solver_added = tk.Label(
            master=self.tab_two,
            text=self.cross_app.crossdesign_MetaExperiment.solver_names,
            justify="center",
        )
        self.solver_added.grid(
            row=row_num, column=1, sticky="nsew", padx=10, pady=3
        )

        self.run_button_added = ttk.Button(
            master=self.tab_two,
            text="Run",
            command=partial(self.run_meta_function, row_num),
        )
        self.run_button_added.grid(
            row=row_num, column=3, sticky="nsew", padx=10, pady=3
        )

        self.clear_button_added = ttk.Button(
            master=self.tab_two,
            text="Remove",
            command=partial(self.clear_meta_function, row_num),
        )
        self.clear_button_added.grid(
            row=row_num, column=4, sticky="nsew", padx=10, pady=3
        )

        self.postprocess_button_added = ttk.Button(
            master=self.tab_two,
            text="Post-Process and Post-Normalize",
            command=partial(self.post_rep_meta_function, row_num),
            state="disabled",
        )
        self.postprocess_button_added.grid(
            row=row_num, column=5, sticky="nsew", padx=10, pady=3
        )

        self.plot_button_added = ttk.Button(
            master=self.tab_two,
            text="Plot",
            command=partial(self.plot_meta_function, row_num),
            state="disabled",
        )
        self.plot_button_added.grid(
            row=row_num, column=6, sticky="nsew", padx=10, pady=3
        )

        self.view_button_added = ttk.Button(
            master=self.tab_two,
            text="View Problem-Solver Group",
            command=partial(self.view_meta_function, row_num),
        )
        self.view_button_added.grid(
            row=row_num, column=7, sticky="nsew", padx=10, pady=3
        )

        # self.select_checkbox = tk.Checkbutton(self.tab_one,text="",state="disabled",command=partial(self.checkbox_function, self.count_experiment_queue - 1))
        # self.select_checkbox.grid(row=self.count_experiment_queue, column=7, sticky='nsew', padx=10, pady=3)

        self.widget_row_meta = [
            self.problem_added,
            self.solver_added,
            self.macros_added,
            self.run_button_added,
            self.clear_button_added,
            self.postprocess_button_added,
            self.plot_button_added,
            self.view_button_added,
        ]
        self.widget_meta_list.insert(row_num - 1, self.widget_row_meta)
        self.meta_experiment_master_list.insert(
            row_num - 1, self.cross_app.crossdesign_MetaExperiment
        )
        # self.select_checkbox.deselect()

        self.count_meta_experiment_queue += 1
        self.notebook.select(self.tab_two)

    def plot_meta_function(self, integer: int) -> None:
        row_index = integer - 1
        self.my_experiment = self.meta_experiment_master_list[row_index]
        # (self.my_experiment.experiments)
        exps = []
        for ex in self.my_experiment.experiments:
            for e in ex:
                exps.append(e)

        self.postrep_window = tk.Toplevel()
        position = center_window(self.master, 0.8)
        self.postrep_window.geometry(position)
        self.postrep_window.title("Plotting Page")
        PlotWindow(
            self.postrep_window,
            self,
            experiment_list=exps,
            meta_list=self.my_experiment,
        )

    def run_meta_function(self, integer: int) -> None:
        row_index = integer - 1
        self.widget_meta_list[row_index][5]["state"] = "normal"
        self.widget_meta_list[row_index][3]["state"] = "disabled"

        self.my_experiment = self.meta_experiment_master_list[row_index]
        # self.macro_reps = self.selected[2]
        self.macro_reps = self.meta_experiment_macro_reps[row_index]

        # (self.my_experiment.n_solvers)
        # (self.my_experiment.n_problems)
        # (self.macro_reps)

        if self.macro_reps == "mixed":
            ask_for_macro_rep = simpledialog.askinteger(
                "Macroreplication",
                "To make a Problem-Solver Group a common macroreplication is needed:",
            )
            self.my_experiment.run(n_macroreps=ask_for_macro_rep)
        else:
            self.my_experiment.run(n_macroreps=int(self.macro_reps))

    def post_rep_meta_function(self, integer: int) -> None:
        row_index = integer - 1
        self.selected = self.meta_experiment_master_list[row_index]
        # (self.selected)
        self.post_rep_function_row_index = integer
        # calls postprocessing window
        self.postrep_window = tk.Tk()
        position = center_window(self.master, 0.8)
        self.postrep_window.geometry(position)
        self.postrep_window.title("Post-Processing and Post-Normalization Page")
        self.app = PostProcessingWindow(
            self.postrep_window, self.selected, self.selected, self, True
        )

    def progress_bar_test(self) -> None:
        root = tk.Tk()
        progress = ttk.Progressbar(
            root, orient="horizontal", length=100, mode="determinate"
        )
        progress["value"] = 20
        root.update_idletasks()
        time.sleep(1)

        progress["value"] = 40
        root.update_idletasks()
        time.sleep(1)

        progress["value"] = 50
        root.update_idletasks()
        time.sleep(1)

        progress["value"] = 60
        root.update_idletasks()
        time.sleep(1)

        progress["value"] = 80
        root.update_idletasks()
        time.sleep(1)
        progress["value"] = 100

        progress.pack(pady=10)

    def post_norm_setup(self) -> None:
        newlist = sorted(
            self.experiment_object_list, key=lambda x: x.problem.name
        )
        for widget in self.tab_three.winfo_children():
            widget.destroy()

        self.heading_list = [
            "Problem",
            "Solvers",
            "Selected",
            "",
            "",
            "",
            "",
            "",
        ]
        for heading in self.heading_list:
            self.tab_three.grid_columnconfigure(
                self.heading_list.index(heading)
            )
            label = tk.Label(
                master=self.tab_three,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=0, column=self.heading_list.index(heading), padx=10, pady=3
            )

        self.widget_norm_list = []
        self.normalize_list2 = []
        self.post_norm_exp_list = []

        for i, exp in enumerate(newlist):
            if exp.post_norm_ready:
                row_num = i + 1
                self.problem_added = tk.Label(
                    master=self.tab_three,
                    text=exp.problem.name,
                    justify="center",
                )
                self.problem_added.grid(
                    row=row_num, column=0, sticky="nsew", padx=10, pady=3
                )

                self.solver_added = tk.Label(
                    master=self.tab_three,
                    text=exp.solver.name,
                    justify="center",
                )
                self.solver_added.grid(
                    row=row_num, column=1, sticky="nsew", padx=10, pady=3
                )

                self.select_checkbox = tk.Checkbutton(
                    self.tab_three,
                    text="",
                    command=partial(self.checkbox_function2, exp, row_num - 1),
                )
                self.select_checkbox.grid(
                    row=row_num, column=2, sticky="nsew", padx=10, pady=3
                )
                self.select_checkbox.deselect()

                self.widget_norm_list.append(
                    [
                        self.problem_added,
                        self.solver_added,
                        self.select_checkbox,
                    ]
                )

    def post_normal_all_function(self) -> None:
        self.postrep_window = tk.Toplevel()
        position = center_window(self.master, 0.8)
        self.postrep_window.geometry(position)
        self.postrep_window.title("Post-Normalization Page")
        self.app = PostNormalWindow(
            self.postrep_window, self.post_norm_exp_list, self
        )
        # post_normalize(self.post_norm_exp_list, n_postreps_init_opt, crn_across_init_opt=True, proxy_init_val=None, proxy_opt_val=None, proxy_opt_x=None)

    def post_norm_return_func(self) -> None:
        # ('IN post_process_disable_button ', self.post_rep_function_row_index)
        # print("youve returned")
        pass

    def make_meta_experiment_func(self) -> None:
        self.list_checked_experiments = []
        self.list_unique_solver = []
        self.list_unique_problems = []
        self.list_missing_experiments = []

        message2 = "There are experiments missing, would you like to add them?"
        response = tk.messagebox.askyesno(
            title="Make ProblemsSolvers Experiemnts", message=message2
        )

        if response:
            for index, checkbox in enumerate(self.check_box_list_var):
                if checkbox.get():
                    index = self.check_box_list_var.index(checkbox)
                    experiment_checked = self.experiment_object_list[
                        index
                    ]  ## Is this right?
                    self.list_checked_experiments.append(experiment_checked)
                    # print("checkbox",checkbox.get())
                    # print("experiment_checked:", experiment_checked )
                    # Making the checkbox in the Queue of Porblem-Solver Groups disabled
                    check_box_object = self.check_box_list[index]
                    check_box_object["state"] = "disabled"
            (
                self.list_unique_solver,
                self.list_unique_problems,
                self.list_missing_experiments,
            ) = find_missing_experiments(self.list_checked_experiments)
            self.meta_experiment_created = make_full_metaexperiment(
                self.list_checked_experiments,
                self.list_unique_solver,
                self.list_unique_problems,
                self.list_missing_experiments,
            )

            self.add_meta_exp_to_frame(
                n_macroreps=None,
                input_meta_experiment=self.meta_experiment_created,
            )
            self.meta_experiment_problem_solver_list(
                self.meta_experiment_created
            )
            self.meta_experiment_master_list.append(
                self.meta_experiment_created
            )

    def meta_experiment_problem_solver_list(
        self, meta_experiment: ProblemsSolvers
    ) -> None:
        self.list_meta_experiment_problems = []
        self.list_meta_experiment_solvers = []

        self.list_meta_experiment_problems = meta_experiment.problem_names
        # print("self.list_meta_experiment_problems", self.list_meta_experiment_problems)
        self.list_meta_experiment_solvers = meta_experiment.solver_names
        # print("self.list_meta_experiment_solvers", self.list_meta_experiment_solvers)

    def view_meta_function(self, row_num: int) -> None:
        self.factor_label_frame_solvers.destroy()
        self.factor_label_frame_oracle.destroy()
        self.factor_label_frame_problems.destroy()

        row_index = row_num - 1
        self.problem_menu.destroy()
        self.problem_label.destroy()
        self.solver_menu.destroy()
        self.solver_label.destroy()

        self.problem_label2 = tk.Label(
            master=self.master,
            text="Group Problem(s):*",
        )
        self.problem_var2 = tk.StringVar(master=self.master)

        self.problem_menu2 = ttk.OptionMenu(
            self.master,
            self.problem_var2,
            "Problem",
            *self.list_meta_experiment_problems,
            command=partial(self.show_problem_factors2, row_index),
        )

        self.problem_label2.place(relx=0.35, rely=0.1)
        self.problem_menu2.place(relx=0.45, rely=0.1)
        self.solver_label2 = tk.Label(
            master=self.master,
            text="Group Solver(s):*",
        )
        self.solver_var2 = tk.StringVar(master=self.master)
        self.solver_menu2 = ttk.OptionMenu(
            self.master,
            self.solver_var2,
            "Solver",
            *self.list_meta_experiment_solvers,
            command=partial(self.show_solver_factors2, row_index),
        )

        self.solver_label2.place(relx=0.01, rely=0.1)
        self.solver_menu2.place(relx=0.1, rely=0.1)

        view_button_added = self.widget_meta_list[row_index][7]
        view_button_added["text"] = "Exit View Problem-Solver Group"
        view_button_added["command"] = partial(self.exit_meta_view, row_num)
        view_button_added.grid(
            row=(row_num), column=7, sticky="nsew", padx=10, pady=3
        )

        self.add_button["state"] = "disabled"

        for i in range(self.count_meta_experiment_queue):
            self.clear_button_added = self.widget_meta_list[i][4]
            self.clear_button_added["state"] = "disabled"

            self.run_button = self.widget_meta_list[i][3]
            self.run_button["state"] = "disabled"

            if i != (row_index):
                view_button_added = self.widget_meta_list[i][7]
                view_button_added["state"] = "disabled"

        for i in range(self.count_experiment_queue - 1):
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

    def exit_meta_view(self, row_num: int) -> None:
        row_index = row_num - 1
        self.add_button["state"] = "normal"
        self.problem_menu2.destroy()
        self.problem_label2.destroy()
        self.solver_menu2.destroy()
        self.solver_label2.destroy()
        self.factor_label_frame_solver.destroy()
        self.factor_label_frame_oracle.destroy()
        self.factor_label_frame_problem.destroy()
        self.problem_label = tk.Label(
            master=self.master,  # window label is used in
            text="Select Problem:",
        )
        self.problem_var = tk.StringVar(master=self.master)
        self.problem_menu = ttk.OptionMenu(
            self.master,
            self.problem_var,
            "Problem",
            *self.problem_list,
            command=self.show_problem_factors,
        )

        self.problem_label.place(relx=0.3, rely=0.1)
        self.problem_menu.place(relx=0.4, rely=0.1)
        self.solver_label = tk.Label(
            master=self.master,  # window label is used in
            text="Select Solver(s):*",
        )
        self.solver_var = tk.StringVar(master=self.master)
        self.solver_menu = ttk.OptionMenu(
            self.master,
            self.solver_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_factors,
        )

        self.solver_label.place(relx=0.01, rely=0.1)
        self.solver_menu.place(relx=0.1, rely=0.1)

        view_button_added = self.widget_meta_list[row_index][7]
        view_button_added["text"] = "View Problem-Solver Group"
        view_button_added["command"] = partial(self.view_meta_function, row_num)
        view_button_added.grid(
            row=(row_num), column=7, sticky="nsew", padx=10, pady=3
        )

        for i in range(self.count_meta_experiment_queue):
            self.clear_button_added = self.widget_meta_list[i][4]
            self.clear_button_added["state"] = "normal"

            self.run_button = self.widget_meta_list[i][3]
            self.run_button["state"] = "normal"

            if i != (row_index):
                view_button_added = self.widget_meta_list[i][7]
                view_button_added["state"] = "normal"

        for i in range(self.count_experiment_queue - 1):
            self.run_button_added = self.widget_list[i][3]
            self.run_button_added["state"] = "normal"

            self.viewEdit_button_added = self.widget_list[i][4]
            self.viewEdit_button_added["state"] = "normal"

            self.clear_button_added = self.widget_list[i][5]
            self.clear_button_added["state"] = "normal"

        self.pickle_file_load_button["state"] = "normal"
        self.crossdesign_button["state"] = "normal"
        self.macro_entry["state"] = "normal"

    def show_solver_factors2(self, row_index: int, *args: tuple) -> None:
        self.factor_label_frame_solver.destroy()

        self.solver_factors_list = []
        self.solver_factors_types = []

        self.factor_label_frame_solver = ttk.LabelFrame(
            master=self.master, text="Solver Factors"
        )

        self.factor_canvas_solver = tk.Canvas(
            master=self.factor_label_frame_solver, borderwidth=0
        )

        self.factor_frame_solver = ttk.Frame(master=self.factor_canvas_solver)
        self.vert_scroll_bar_factor_solver = Scrollbar(
            self.factor_label_frame_solver,
            orient="vertical",
            command=self.factor_canvas_solver.yview,
        )
        self.horiz_scroll_bar_factor_solver = Scrollbar(
            self.factor_label_frame_solver,
            orient="horizontal",
            command=self.factor_canvas_solver.xview,
        )
        self.factor_canvas_solver.configure(
            xscrollcommand=self.horiz_scroll_bar_factor_solver.set,
            yscrollcommand=self.vert_scroll_bar_factor_solver.set,
        )

        self.vert_scroll_bar_factor_solver.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_solver.pack(side="bottom", fill="x")

        self.factor_canvas_solver.pack(side="left", fill="both", expand=True)
        self.factor_canvas_solver.create_window(
            (0, 0),
            window=self.factor_frame_solver,
            anchor="nw",
            tags="self.factor_frame_solver",
        )

        self.factor_frame_solver.bind(
            "<Configure>", self.on_frame_configure_factor_solver
        )

        self.factor_notebook_solver = ttk.Notebook(
            master=self.factor_frame_solver
        )
        self.factor_notebook_solver.pack(fill="both")

        self.factor_tab_one_solver = tk.Frame(
            master=self.factor_notebook_solver
        )

        self.factor_notebook_solver.add(
            self.factor_tab_one_solver,
            text=str(self.solver_var2.get()) + " Factors",
        )

        self.factor_tab_one_solver.grid_rowconfigure(0)

        self.factor_heading_list_solver = ["Description", "Input"]

        for heading in self.factor_heading_list_solver:
            self.factor_tab_one_solver.grid_columnconfigure(
                self.factor_heading_list_solver.index(heading)
            )
            label = tk.Label(
                master=self.factor_tab_one_solver,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=0,
                column=self.factor_heading_list_solver.index(heading),
                padx=10,
                pady=3,
            )

        meta_experiment = self.meta_experiment_master_list[row_index]
        solver_name = self.solver_var2.get()
        solver_index = meta_experiment.solver_names.index(str(solver_name))
        self.solver_object = meta_experiment.solvers[solver_index]

        meta_experiment = self.meta_experiment_master_list[row_index]
        solver_name = self.solver_var2.get()
        solver_index = meta_experiment.solver_names.index(str(solver_name))
        self.custom_solver_object = meta_experiment.solvers[solver_index]
        # explanation: https://stackoverflow.com/questions/5924879/how-to-create-a-new-instance-from-a-class-object-in-python
        default_solver_class = self.custom_solver_object.__class__
        self.default_solver_object = default_solver_class()

        count_factors_solver = 1

        self.save_label_solver = tk.Label(
            master=self.factor_tab_one_solver,
            text="save solver as",
        )

        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(
            master=self.factor_tab_one_solver,
            textvariable=self.save_var_solver,
            justify=tk.LEFT,
            width=15,
        )

        self.save_entry_solver.insert(index=tk.END, string=solver_name)
        self.save_entry_solver["state"] = "disabled"
        self.save_label_solver.grid(
            row=count_factors_solver, column=0, sticky="nsew"
        )
        self.save_entry_solver.grid(
            row=count_factors_solver, column=1, sticky="nsew"
        )

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)

        count_factors_solver += 1

        for factor_type in self.default_solver_object.specifications:
            self.dictionary_size_solver = len(
                self.default_solver_object.specifications[factor_type]
            )
            datatype = self.default_solver_object.specifications[
                factor_type
            ].get("datatype")
            description = self.default_solver_object.specifications[
                factor_type
            ].get("description")
            default = self.default_solver_object.specifications[
                factor_type
            ].get("default")

            if datatype is not bool:
                self.int_float_description = tk.Label(
                    master=self.factor_tab_one_solver,
                    text=str(description),
                    wraplength=150,
                )

                self.int_float_var = tk.StringVar(self.factor_tab_one_solver)
                self.int_float_entry = ttk.Entry(
                    master=self.factor_tab_one_solver,
                    textvariable=self.int_float_var,
                    justify=tk.LEFT,
                    width=15,
                )
                self.int_float_entry.insert(
                    index=tk.END,
                    string=str(self.custom_solver_object.factors[factor_type]),
                )
                self.int_float_entry["state"] = "disabled"
                self.int_float_description.grid(
                    row=count_factors_solver, column=0, sticky="nsew"
                )
                self.int_float_entry.grid(
                    row=count_factors_solver, column=1, sticky="nsew"
                )
                self.solver_factors_list.append(self.int_float_var)

                if datatype is not tuple:
                    self.solver_factors_types.append(datatype)
                else:
                    self.solver_factors_types.append(str)

                count_factors_solver += 1

            if datatype is bool:
                self.boolean_description = tk.Label(
                    master=self.factor_tab_one_solver,
                    text=str(description),
                    wraplength=150,
                )

                self.boolean_var = tk.BooleanVar(
                    self.factor_tab_one_solver, value=bool(default)
                )
                self.boolean_menu = tk.Checkbutton(
                    self.factor_tab_one_solver,
                    variable=self.boolean_var,
                    onvalue=True,
                    offvalue=False,
                )

                # self.boolean_menu.configure(state = "disabled")
                self.boolean_description.grid(
                    row=count_factors_solver, column=0, sticky="nsew"
                )
                self.boolean_menu.grid(
                    row=count_factors_solver, column=1, sticky="nsew"
                )
                self.solver_factors_list.append(self.boolean_var)
                self.solver_factors_types.append(datatype)

                count_factors_solver += 1

        self.factor_label_frame_solver.place(
            x=10, rely=0.15, relheight=0.33, relwidth=0.34
        )
        if str(self.problem_var.get()) != "Problem":
            self.add_button.place(x=10, rely=0.48, width=200, height=30)

    def show_problem_factors2(self, row_index: int, *args: tuple) -> None:
        self.factor_label_frame_problem.destroy()
        self.factor_label_frame_oracle.destroy()
        self.problem_factors_list = []
        self.problem_factors_types = []

        self.factor_label_frame_problem = ttk.LabelFrame(
            master=self.master, text="Problem Factors"
        )

        self.factor_canvas_problem = tk.Canvas(
            master=self.factor_label_frame_problem, borderwidth=0
        )

        self.factor_frame_problem = ttk.Frame(master=self.factor_canvas_problem)
        self.vert_scroll_bar_factor_problem = Scrollbar(
            self.factor_label_frame_problem,
            orient="vertical",
            command=self.factor_canvas_problem.yview,
        )
        self.horiz_scroll_bar_factor_problem = Scrollbar(
            self.factor_label_frame_problem,
            orient="horizontal",
            command=self.factor_canvas_problem.xview,
        )
        self.factor_canvas_problem.configure(
            xscrollcommand=self.horiz_scroll_bar_factor_problem.set,
            yscrollcommand=self.vert_scroll_bar_factor_problem.set,
        )

        self.vert_scroll_bar_factor_problem.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_problem.pack(side="bottom", fill="x")

        self.factor_canvas_problem.pack(side="left", fill="both", expand=True)
        self.factor_canvas_problem.create_window(
            (0, 0),
            window=self.factor_frame_problem,
            anchor="nw",
            tags="self.factor_frame_problem",
        )

        self.factor_frame_problem.bind(
            "<Configure>", self.on_frame_configure_factor_problem
        )

        self.factor_notebook_problem = ttk.Notebook(
            master=self.factor_frame_problem
        )
        self.factor_notebook_problem.pack(fill="both")

        self.factor_tab_one_problem = tk.Frame(
            master=self.factor_notebook_problem
        )

        self.factor_notebook_problem.add(
            self.factor_tab_one_problem,
            text=str(self.problem_var2.get()) + " Factors",
        )

        self.factor_tab_one_problem.grid_rowconfigure(0)

        self.factor_heading_list_problem = ["Description", "Input"]

        for heading in self.factor_heading_list_problem:
            self.factor_tab_one_problem.grid_columnconfigure(
                self.factor_heading_list_problem.index(heading)
            )
            label_problem = tk.Label(
                master=self.factor_tab_one_problem,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label_problem.grid(
                row=0,
                column=self.factor_heading_list_problem.index(heading),
                padx=10,
                pady=3,
            )

        meta_experiment = self.meta_experiment_master_list[row_index]
        problem_name = self.problem_var2.get()
        problem_index = meta_experiment.problem_names.index(str(problem_name))
        self.custom_problem_object = meta_experiment.problems[problem_index]
        # explanation: https://stackoverflow.com/questions/5924879/how-to-create-a-new-instance-from-a-class-object-in-python
        default_problem_class = self.custom_problem_object.__class__
        self.default_problem_object = default_problem_class()

        count_factors_problem = 1

        self.save_label_problem = tk.Label(
            master=self.factor_tab_one_problem,
            text="save problem as",
        )

        self.save_var_problem = tk.StringVar(self.factor_tab_one_problem)
        self.save_entry_problem = ttk.Entry(
            master=self.factor_tab_one_problem,
            textvariable=self.save_var_problem,
            justify=tk.LEFT,
            width=15,
        )

        self.save_entry_problem.insert(index=tk.END, string=problem_name)
        self.save_entry_problem["state"] = "disabled"
        self.save_label_problem.grid(
            row=count_factors_problem, column=0, sticky="nsew"
        )
        self.save_entry_problem.grid(
            row=count_factors_problem, column=1, sticky="nsew"
        )

        self.problem_factors_list.append(self.save_var_problem)
        self.problem_factors_types.append(str)

        count_factors_problem += 1

        for _, factor_type in enumerate(
            self.default_problem_object.specifications, start=0
        ):
            self.dictionary_size_problem = len(
                self.default_problem_object.specifications[factor_type]
            )
            datatype = self.default_problem_object.specifications[
                factor_type
            ].get("datatype")
            description = self.default_problem_object.specifications[
                factor_type
            ].get("description")
            default = self.default_problem_object.specifications[factor_type][
                "default"
            ]

            if datatype is not bool:
                self.int_float_description_problem = tk.Label(
                    master=self.factor_tab_one_problem,
                    text=str(description),
                    wraplength=150,
                )

                self.int_float_var_problem = tk.StringVar(
                    self.factor_tab_one_problem
                )
                self.int_float_entry_problem = ttk.Entry(
                    master=self.factor_tab_one_problem,
                    textvariable=self.int_float_var_problem,
                    justify=tk.LEFT,
                    width=15,
                )
                if datatype is tuple and len(default) == 1:
                    self.int_float_entry_problem.insert(
                        index=tk.END,
                        string=str(
                            self.custom_problem_object.factors[factor_type][0]
                        ),
                    )
                else:
                    self.int_float_entry_problem.insert(
                        index=tk.END,
                        string=str(
                            self.custom_problem_object.factors[factor_type]
                        ),
                    )

                self.int_float_entry_problem["state"] = "disabled"
                self.int_float_description_problem.grid(
                    row=count_factors_problem, column=0, sticky="nsew"
                )
                self.int_float_entry_problem.grid(
                    row=count_factors_problem, column=1, sticky="nsew"
                )

                self.problem_factors_list.append(self.int_float_var_problem)
                datatype = self.default_problem_object.specifications[
                    factor_type
                ].get("datatype")

                if datatype is not tuple:
                    self.problem_factors_types.append(datatype)
                else:
                    self.problem_factors_types.append(str)

                count_factors_problem += 1

            if datatype is bool:
                self.boolean_description_problem = tk.Label(
                    master=self.factor_tab_one_problem,
                    text=str(description),
                    wraplength=150,
                )
                self.boolean_var_problem = tk.BooleanVar(
                    self.factor_tab_one_problem, value=bool(default)
                )
                self.boolean_menu_problem = tk.Checkbutton(
                    self.factor_tab_one_problem,
                    variable=self.boolean_var_problem,
                    onvalue=True,
                    offvalue=False,
                )
                self.boolean_description_problem.grid(
                    row=count_factors_problem, column=0, sticky="nsew"
                )
                self.boolean_menu_problem.grid(
                    row=count_factors_problem, column=1, sticky="nsew"
                )

                self.problem_factors_list.append(self.boolean_var_problem)
                self.problem_factors_types.append(datatype)

                count_factors_problem += 1

        self.factor_label_frame_problem.place(
            relx=0.35, rely=0.15, relheight=0.33, relwidth=0.34
        )

        # Switching from Problems to Oracles

        self.oracle_factors_list = []
        self.oracle_factors_types = []

        self.factor_label_frame_oracle = ttk.LabelFrame(
            master=self.master, text="Model Factors"
        )

        self.factor_canvas_oracle = tk.Canvas(
            master=self.factor_label_frame_oracle, borderwidth=0
        )

        self.factor_frame_oracle = ttk.Frame(master=self.factor_canvas_oracle)
        self.vert_scroll_bar_factor_oracle = Scrollbar(
            self.factor_label_frame_oracle,
            orient="vertical",
            command=self.factor_canvas_oracle.yview,
        )
        self.horiz_scroll_bar_factor_oracle = Scrollbar(
            self.factor_label_frame_oracle,
            orient="horizontal",
            command=self.factor_canvas_oracle.xview,
        )
        self.factor_canvas_oracle.configure(
            xscrollcommand=self.horiz_scroll_bar_factor_oracle.set,
            yscrollcommand=self.vert_scroll_bar_factor_oracle.set,
        )

        self.vert_scroll_bar_factor_oracle.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_oracle.pack(side="bottom", fill="x")

        self.factor_canvas_oracle.pack(side="left", fill="both", expand=True)
        self.factor_canvas_oracle.create_window(
            (0, 0),
            window=self.factor_frame_oracle,
            anchor="nw",
            tags="self.factor_frame_oracle",
        )

        self.factor_frame_oracle.bind(
            "<Configure>", self.on_frame_configure_factor_oracle
        )

        self.factor_notebook_oracle = ttk.Notebook(
            master=self.factor_frame_oracle
        )
        self.factor_notebook_oracle.pack(fill="both")

        self.factor_tab_one_oracle = tk.Frame(
            master=self.factor_notebook_oracle
        )

        self.factor_notebook_oracle.add(
            self.factor_tab_one_oracle, text=str(self.oracle + " Factors")
        )

        self.factor_tab_one_oracle.grid_rowconfigure(0)

        self.factor_heading_list_oracle = ["Description", "Input"]

        for heading in self.factor_heading_list_oracle:
            self.factor_tab_one_oracle.grid_columnconfigure(
                self.factor_heading_list_oracle.index(heading)
            )
            label_oracle = tk.Label(
                master=self.factor_tab_one_oracle,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label_oracle.grid(
                row=0,
                column=self.factor_heading_list_oracle.index(heading),
                padx=10,
                pady=3,
            )

        self.default_oracle_object = self.default_problem_object.model
        self.custom_oracle_object = self.custom_problem_object.model

        count_factors_oracle = 1
        for factor_type in self.default_oracle_object.specifications:
            self.dictionary_size_oracle = len(
                self.default_oracle_object.specifications[factor_type]
            )
            datatype = self.default_oracle_object.specifications[
                factor_type
            ].get("datatype")
            description = self.default_oracle_object.specifications[
                factor_type
            ].get("description")
            default = self.default_oracle_object.specifications[
                factor_type
            ].get("default")

            if datatype is bool:
                # ("yes?")
                self.int_float_description_oracle = tk.Label(
                    master=self.factor_tab_one_oracle,
                    text=str(description),
                    wraplength=150,
                )
                self.int_float_var_oracle = tk.StringVar(
                    self.factor_tab_one_oracle
                )
                self.int_float_entry_oracle = ttk.Entry(
                    master=self.factor_tab_one_oracle,
                    textvariable=self.int_float_var_oracle,
                    justify=tk.LEFT,
                    width=15,
                )
                self.int_float_entry_oracle.insert(
                    index=tk.END,
                    string=str(self.custom_oracle_object.factors[factor_type]),
                )
                self.int_float_entry_oracle["state"] = "disabled"
                self.int_float_description_oracle.grid(
                    row=count_factors_oracle, column=0, sticky="nsew"
                )
                self.int_float_entry_oracle.grid(
                    row=count_factors_oracle, column=1, sticky="nsew"
                )

                self.oracle_factors_list.append(self.int_float_var_oracle)

                if datatype is not tuple:
                    self.oracle_factors_types.append(datatype)
                else:
                    self.oracle_factors_types.append(str)

                count_factors_oracle += 1

            if datatype is bool:
                # ("yes!")
                self.boolean_description_oracle = tk.Label(
                    master=self.factor_tab_one_oracle,
                    text=str(description),
                    wraplength=150,
                )
                self.boolean_var_oracle = tk.BooleanVar(
                    self.factor_tab_one_oracle, value=bool(default)
                )
                self.boolean_menu_oracle = tk.Checkbutton(
                    self.factor_tab_one_oracle,
                    variable=self.boolean_var_oracle,
                    onvalue=True,
                    offvalue=False,
                )
                self.boolean_description_oracle.grid(
                    row=count_factors_oracle, column=0, sticky="nsew"
                )
                self.boolean_menu_oracle.grid(
                    row=count_factors_oracle, column=1, sticky="nsew"
                )
                self.oracle_factors_list.append(self.boolean_var_oracle)
                self.oracle_factors_types.append(datatype)

                count_factors_oracle += 1

        self.factor_label_frame_oracle.place(
            relx=0.7, rely=0.15, relheight=0.33, relwidth=0.3
        )
        if str(self.solver_var.get()) != "Solver":
            self.add_button.place(x=10, rely=0.48, width=200, height=30)

    def show_solver_factors(self, *args: tuple) -> None:
        """Show the solver factors in the GUI.

        Parameters
        ----------
        args : tuple
            The arguments passed to the function.

        """
        if args and len(args) == 3 and not args[2]:
            pass
        else:
            self.update_problem_list_compatability()

        self.solver_factors_list = []
        self.solver_factors_types = []

        self.factor_label_frame_solver = ttk.LabelFrame(
            master=self.master, text="Solver Factors"
        )

        self.factor_canvas_solver = tk.Canvas(
            master=self.factor_label_frame_solver, borderwidth=0
        )

        self.factor_frame_solver = ttk.Frame(master=self.factor_canvas_solver)
        self.vert_scroll_bar_factor_solver = Scrollbar(
            self.factor_label_frame_solver,
            orient="vertical",
            command=self.factor_canvas_solver.yview,
        )
        self.horiz_scroll_bar_factor_solver = Scrollbar(
            self.factor_label_frame_solver,
            orient="horizontal",
            command=self.factor_canvas_solver.xview,
        )
        self.factor_canvas_solver.configure(
            xscrollcommand=self.horiz_scroll_bar_factor_solver.set,
            yscrollcommand=self.vert_scroll_bar_factor_solver.set,
        )

        self.vert_scroll_bar_factor_solver.pack(side="right", fill="y")
        self.horiz_scroll_bar_factor_solver.pack(side="bottom", fill="x")

        self.factor_canvas_solver.pack(side="left", fill="both", expand=True)
        self.factor_canvas_solver.create_window(
            (0, 0),
            window=self.factor_frame_solver,
            anchor="nw",
            tags="self.factor_frame_solver",
        )

        self.factor_frame_solver.bind(
            "<Configure>", self.on_frame_configure_factor_solver
        )

        self.factor_notebook_solver = ttk.Notebook(
            master=self.factor_frame_solver
        )
        self.factor_notebook_solver.pack(fill="both")

        self.factor_tab_one_solver = tk.Frame(
            master=self.factor_notebook_solver
        )

        self.factor_notebook_solver.add(
            self.factor_tab_one_solver,
            text=str(self.solver_var.get()) + " Factors",
        )

        self.factor_tab_one_solver.grid_rowconfigure(0)

        self.factor_heading_list_solver = ["Description", "Input"]

        for heading in self.factor_heading_list_solver:
            self.factor_tab_one_solver.grid_columnconfigure(
                self.factor_heading_list_solver.index(heading)
            )
            label = tk.Label(
                master=self.factor_tab_one_solver,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=0,
                column=self.factor_heading_list_solver.index(heading),
                padx=10,
                pady=3,
            )

        self.solver_object = solver_unabbreviated_directory[
            self.solver_var.get()
        ]

        count_factors_solver = 1

        self.save_label_solver = tk.Label(
            master=self.factor_tab_one_solver,
            text="save solver as",
        )

        if args and len(args) == 3 and args[0]:
            oldname = args[1][5][1]

        else:
            solver_object = solver_unabbreviated_directory[
                self.solver_var.get()
            ]
            oldname = solver_object().name

        self.save_var_solver = tk.StringVar(self.factor_tab_one_solver)
        self.save_entry_solver = ttk.Entry(
            master=self.factor_tab_one_solver,
            textvariable=self.save_var_solver,
            justify=tk.LEFT,
            width=15,
        )

        self.save_entry_solver.insert(index=tk.END, string=oldname)

        self.save_label_solver.grid(
            row=count_factors_solver, column=0, sticky="nsew"
        )
        self.save_entry_solver.grid(
            row=count_factors_solver, column=1, sticky="nsew"
        )

        self.solver_factors_list.append(self.save_var_solver)

        self.solver_factors_types.append(str)

        count_factors_solver += 1

        for factor_type in self.solver_object().specifications:
            # ("size of dictionary", len(self.solver_object().specifications[factor_type]))
            # ("first", factor_type)
            # ("second", self.solver_object().specifications[factor_type].get("description"))
            # ("third", self.solver_object().specifications[factor_type].get("datatype"))
            # ("fourth", self.solver_object().specifications[factor_type].get("default"))

            self.dictionary_size_solver = len(
                self.solver_object().specifications[factor_type]
            )
            datatype = (
                self.solver_object().specifications[factor_type].get("datatype")
            )
            description = (
                self.solver_object()
                .specifications[factor_type]
                .get("description")
            )
            default = (
                self.solver_object().specifications[factor_type].get("default")
            )

            if datatype is not bool:
                self.int_float_description = tk.Label(
                    master=self.factor_tab_one_solver,
                    text=str(description),
                    wraplength=150,
                )

                self.int_float_var = tk.StringVar(self.factor_tab_one_solver)
                self.int_float_entry = ttk.Entry(
                    master=self.factor_tab_one_solver,
                    textvariable=self.int_float_var,
                    justify=tk.LEFT,
                    width=15,
                )

                if args and len(args) == 3 and args[0]:
                    self.int_float_entry.insert(
                        index=tk.END, string=str(args[1][5][0][factor_type])
                    )
                else:
                    self.int_float_entry.insert(
                        index=tk.END, string=str(default)
                    )

                self.int_float_description.grid(
                    row=count_factors_solver, column=0, sticky="nsew"
                )
                self.int_float_entry.grid(
                    row=count_factors_solver, column=1, sticky="nsew"
                )
                self.solver_factors_list.append(self.int_float_var)

                if datatype is not tuple:
                    self.solver_factors_types.append(datatype)
                else:
                    self.solver_factors_types.append(str)

                count_factors_solver += 1

            if datatype is bool:
                self.boolean_description = tk.Label(
                    master=self.factor_tab_one_solver,
                    text=str(description),
                    wraplength=150,
                )
                self.boolean_var = tk.BooleanVar(
                    self.factor_tab_one_solver, value=bool(default)
                )
                self.boolean_menu = tk.Checkbutton(
                    self.factor_tab_one_solver,
                    variable=self.boolean_var,
                    onvalue=True,
                    offvalue=False,
                )
                self.boolean_description.grid(
                    row=count_factors_solver, column=0, sticky="nsew"
                )
                self.boolean_menu.grid(
                    row=count_factors_solver, column=1, sticky="nsew"
                )
                self.solver_factors_list.append(self.boolean_var)
                self.solver_factors_types.append(datatype)

                count_factors_solver += 1

        # self.factor_label_frame_problem.place(relx=.32, y=70, height=150, relwidth=.34)
        self.factor_label_frame_solver.place(
            x=10, rely=0.15, relheight=0.33, relwidth=0.34
        )
        if str(self.problem_var.get()) != "Problem":
            self.add_button.place(x=10, rely=0.48, width=200, height=30)


class NewExperimentWindow(tk.Toplevel):
    """New Experiment Window."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize New Experiment Window."""
        self.master = root
        self = set_theme(self)

        self.master.title("New Experiment")

        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        position = center_window(self.master, 0.8)
        self.master.geometry(position)

        # self.main_window = main_widow
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # master row numbers
        self.notebook_row = 1
        self.load_design_button_row = 2
        self.sol_prob_list_display_row = 3
        self.create_experiment_row = 4
        self.experiment_button_row = 5
        self.experiment_list_display_row = 6
        # master list variables
        self.master_solver_dict = {}  # for each name of solver or solver design has list that includes: list of dps, solver name
        self.master_problem_dict = {}  # for each name of solver or solver design has list that includes: [[problem factors], [model factors], problem name]
        self.master_experiment_dict = {}  # dictionary of experiment name and related solver/problem lists (solver_factor_list, problem_factor_list, solver_name_list, problem_name_list)
        self.ran_experiments_dict = {}  # dictionary of experiments that have been run orgainized by experiment name
        self.design_types_list = [
            "nolhs"
        ]  # available design types that can be used during datafarming
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

        # create master canvas
        self.master_canvas = tk.Canvas(self.master)
        self.master_canvas.grid(row=0, column=0, sticky="nsew")

        # create master frame
        self.main_frame = tk.Frame(self.master_canvas)
        self.main_frame.grid(row=0, column=0)
        self.main_frame.bind(
            "<Configure>", self.update_main_window_scroll
        )  # bind main frame to scroll bar
        self.main_frame.bind_all("<MouseWheel>", self.on_mousewheel)

        # create window scrollbars
        vert_scroll = ttk.Scrollbar(
            self.master, orient=tk.VERTICAL, command=self.master_canvas.yview
        )
        vert_scroll.grid(row=0, column=1, sticky="ns")
        horiz_scroll = ttk.Scrollbar(
            self.master, orient=tk.HORIZONTAL, command=self.master_canvas.xview
        )
        horiz_scroll.grid(row=1, column=0, sticky="ew")
        self.master_canvas.create_window(
            (0, 0), window=self.main_frame, anchor="nw"
        )  # add main frame as window to canvas
        self.master_canvas.configure(
            yscrollcommand=vert_scroll.set, xscrollcommand=horiz_scroll.set
        )

        # empty frames so clear frames fn can execute
        self.solver_selection_frame = tk.Frame(master=self.main_frame)
        self.problem_selection_frame = tk.Frame(master=self.main_frame)
        self.prob_mod_frame = tk.Frame(master=self.main_frame)
        self.solver_frame = tk.Frame(master=self.main_frame)
        self.problem_frame = tk.Frame(master=self.main_frame)
        self.model_frame = tk.Frame(master=self.main_frame)
        self.solver_datafarm_frame = tk.Frame(master=self.main_frame)
        self.problem_datafarm_frame = tk.Frame(master=self.main_frame)
        self.model_datafarm_frame = tk.Frame(master=self.main_frame)
        self.design_display_frame = tk.Frame(master=self.main_frame)

        """Title"""

        self.title_label = tk.Label(
            master=self.main_frame,
            text="New Experiment Page",
            font=nametofont("TkHeadingFont"),
        )
        self.title_label.grid(row=0, column=0)

        # self.add_buttons_frame = tk.Frame(master = self.main_frame)
        # self.add_buttons_frame.grid(row = self.add_buttons_row, column = 0)

        """Solver/Problem Notebook & Selection Menus"""

        self.sol_prob_book = ttk.Notebook(master=self.main_frame)
        self.sol_prob_book.grid(row=self.notebook_row, column=0)

        self.solver_notebook_frame = ttk.Frame(master=self.sol_prob_book)
        self.problem_notebook_frame = ttk.Frame(master=self.sol_prob_book)
        self.solver_datafarm_notebook_frame = ttk.Frame(
            master=self.sol_prob_book
        )
        self.problem_datafarm_notebook_frame = ttk.Frame(
            master=self.sol_prob_book
        )

        self.sol_prob_book.add(self.solver_notebook_frame, text="Add Solver")
        self.sol_prob_book.add(self.problem_notebook_frame, text="Add Problem")
        self.sol_prob_book.add(
            self.solver_datafarm_notebook_frame,
            text="Add Solver w/ Data Farming",
        )
        self.sol_prob_book.add(
            self.problem_datafarm_notebook_frame,
            text="Add Problem w/ Data Farming",
        )

        # Solver selection menu frames
        self.solver_selection_frame = tk.Frame(
            master=self.solver_notebook_frame
        )
        self.solver_selection_frame.grid(row=0, column=0)

        # Option menu to select solver
        self.solver_select_label = tk.Label(
            master=self.solver_selection_frame,
            text="Select Solver:",
            width=20,
        )
        self.solver_select_label.grid(row=0, column=0, sticky=tk.N + tk.W)

        # Variable to store selected solver
        self.solver_var = tk.StringVar()

        # Directory of solver names
        self.solver_list = solver_unabbreviated_directory

        self.solver_select_menu = ttk.OptionMenu(
            self.solver_selection_frame,
            self.solver_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_factors,
        )
        self.solver_select_menu.grid(row=0, column=1)

        # problem selection frames
        self.problem_selection_frame = tk.Frame(
            master=self.problem_notebook_frame
        )
        self.problem_selection_frame.grid(row=0, column=0)

        # Option menu to select problem
        self.problem_select_label = tk.Label(
            master=self.problem_selection_frame,
            text="Select Problem",
            width=20,
        )
        self.problem_select_label.grid(row=0, column=0)

        # Variable to store selected problem
        self.problem_var = tk.StringVar()

        # Directory of problem names
        self.problem_list = problem_unabbreviated_directory

        self.problem_select_menu = ttk.OptionMenu(
            self.problem_selection_frame,
            self.problem_var,
            "Problem",
            *self.problem_list,
            command=self.show_problem_factors,
        )
        self.problem_select_menu.grid(row=0, column=1)

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
        self.solver_datafarm_var = tk.StringVar()

        # Directory of solver names
        self.solver_list = solver_unabbreviated_directory

        self.solver_datafarm_select_menu = ttk.OptionMenu(
            self.solver_datafarm_selection_frame,
            self.solver_datafarm_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_datafarm,
        )
        self.solver_datafarm_select_menu.grid(row=0, column=1)

        # Problem selection w/ data farming
        self.problem_datafarm_selection_frame = tk.Frame(
            master=self.problem_datafarm_notebook_frame
        )
        self.problem_datafarm_selection_frame.grid(row=0, column=0)

        # Option menu to select problem
        self.problem_select_label = tk.Label(
            master=self.problem_datafarm_selection_frame,
            text="Select Problem",
            width=20,
        )
        self.problem_select_label.grid(row=0, column=0)

        # Variable to store selected problem
        self.problem_datafarm_var = tk.StringVar()

        # Directory of problem names
        self.problem_list = problem_unabbreviated_directory

        self.problem_datafarm_select_menu = ttk.OptionMenu(
            self.problem_datafarm_selection_frame,
            self.problem_datafarm_var,
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
        
        # cross-design window button
        self.cross_design_button = tk.Button(
            master=self.main_frame,
            text="Create Cross Design Using Default Factor Settings",
            command=self.cross_design_window,
        )
        self.cross_design_button.grid(row=self.load_design_button_row, column=1, padx=10)

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
        self.clear_experiment_button.grid(row=0, column=0)
        self.experiment_name_label = tk.Label(
            master=self.experiment_button_frame,
            text="Experiment Name",
        )
        self.experiment_name_label.grid(row=0, column=0)
        self.experiment_name_var = tk.StringVar()
        self.experiment_name_var.set("experiment")
        self.experiment_name_entry = tk.Entry(
            master=self.experiment_button_frame,
            textvariable=self.experiment_name_var,
            width=30,
            justify="left",
        )
        self.experiment_name_entry.grid(row=0, column=1)
        self.run_experiment_button = tk.Button(
            master=self.experiment_button_frame,
            text="Create experiment with listed solvers & problems",
            command=self.create_experiment,
        )
        self.run_experiment_button.grid(row=2, column=0)

        # ind pair pickle checkbox
        self.pickle_label = tk.Label(
            master=self.experiment_button_frame,
            text="Create pickles for each problem-solver pair?",
        )
        self.pickle_label.grid(row=1, column=0)
        self.pickle_checkstate = tk.BooleanVar()
        self.pickle_checkbox = tk.Checkbutton(
            master=self.experiment_button_frame,
            variable=self.pickle_checkstate,
            width=5,
        )
        self.pickle_checkbox.grid(row=1, column=1)

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
        self.master_canvas.configure(
            scrollregion=self.master_canvas.bbox("all")
        )

    def on_mousewheel(self, event: tk.Event) -> None:
        self.master_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def check_problem_compatibility(self) -> None: 
        # create temp objects for current selected solvers and all possilble problems
        temp_solvers = []
        for solver_group in self.master_solver_dict:
            dp_0 = self.master_solver_dict[solver_group][0] # frist design point if design, only design pt if no design
            solver_name = dp_0[1]
            temp_solver = solver_directory[solver_name]()
            temp_solvers.append(temp_solver)
        # check solver selections based on which tab is open
        current_tab = self.sol_prob_book.index('current')
        if current_tab == 0:
            selected_solver = self.solver_var.get()
        if current_tab == 2:
            selected_solver = self.solver_datafarm_var.get()
        if selected_solver != "Solver":
            temp_solver = solver_unabbreviated_directory[selected_solver]()
            temp_solvers.append(temp_solver)
        all_problems = problem_unabbreviated_directory
        self.problem_list = {} #clear current problem selection options
        for problem_name in all_problems:
            temp_problem = [all_problems[problem_name]()]
            temp_exp = ProblemsSolvers(solvers=temp_solvers, problems=temp_problem) #temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            if not error:
                self.problem_list[problem_name] = all_problems[problem_name]
        
        #update problem & problem datafarming selections
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
            self.problem_datafarm_var,
            "Problem",
            *self.problem_list,
            command=self.show_problem_datafarm,
        )
        self.problem_datafarm_select_menu.grid(row=0, column=1)
        
    def check_solver_compatibility(self) -> None: 
        # create temp objects for current selected solvers and all possilble problems
        temp_problems = []
        for problem_group in self.master_problem_dict:
            dp_0 = self.master_problem_dict[problem_group][0] # frist design point if design, only design pt if no design
            problem_name = dp_0[1]
            temp_problem = problem_directory[problem_name]()
            temp_problems.append(temp_problem)
        # check problem selections based on which tab is open
        current_tab = self.sol_prob_book.index('current')
        if current_tab == 1:
            selected_problem = self.problem_var.get()
        if current_tab == 3:
            selected_problem = self.problem_datafarm_var.get()
        if selected_problem != "Problem":
            temp_problem = problem_unabbreviated_directory[selected_problem]()
            temp_problems.append(temp_problem)
        all_solvers = solver_unabbreviated_directory
        self.solver_list = {} #clear current problem selection options
        for solver_name in all_solvers:
            temp_solver = [all_solvers[solver_name]()]
            temp_exp = ProblemsSolvers(solvers=temp_solver, problems=temp_problems) #temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            if not error:
                self.solver_list[solver_name] = all_solvers[solver_name]
        
        #update solver & solver datafarming selections
        self.solver_select_menu.destroy()
        self.solver_select_menu = ttk.OptionMenu(
            self.solver_selection_frame,
            self.solver_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_factors,
        )
        self.solver_select_menu.grid(row=0, column=1)
        self.solver_datafarm_select_menu.destroy()
        self.solver_datafarm_select_menu = ttk.OptionMenu(
            self.solver_datafarm_selection_frame,
            self.solver_datafarm_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_datafarm,
        )
        self.solver_datafarm_select_menu.grid(row=0, column=1)
    
    def cross_design_window(self) -> None:
        self.cross_design_window = tk.Toplevel(self.master)
        self.cross_design_window.title(
            "Simopt Graphical User Interface - Cross Design"
        )
        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        position = center_window(self.master, 0.8)
        self.cross_design_window.geometry(position)

        # Configure the grid layout to expand properly
        self.cross_design_window.grid_rowconfigure(0, weight=1)
        self.cross_design_window.grid_columnconfigure(0, weight=1)
        self.cross_design_window.grid_rowconfigure(1, weight=1)
        self.cross_design_window.grid_columnconfigure(1, weight=1)
        self.cross_design_window.grid_rowconfigure(2, weight=1)
        self.cross_design_window.grid_columnconfigure(2, weight=1)
        
        self.solvers_canvas = tk.Canvas(
            master = self.cross_design_window
        )
        self.solvers_canvas.grid(row=2, column=0, sticky="nsew")
        self.problems_canvas = tk.Canvas(
            master = self.cross_design_window
        )
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
        self.solvers_canvas.configure(
            yscrollcommand=solver_scroll.set 
        )
        self.problems_canvas.configure(
            yscrollcommand=problem_scroll.set 
        )

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
        self.solvers_frame.bind("<Configure>", self.update_solvers_canvas_scroll)
        self.problems_frame.bind("<Configure>", self.update_problems_canvas_scroll)
        
        self.cross_design_title = tk.Label(
            master=self.cross_design_window,
            text= 'Select solvers and problems to be included in cross-design. \n Solvers and problems will be run with default factor settings.',
            font=nametofont("TkHeadingFont")
        )
        self.cross_design_title.grid(row=0, column=0, columnspan=4, sticky="n")
        self.solvers_label = tk.Label(
            master=self.cross_design_window,
            text= 'Select Solvers:',
        )
        self.solvers_label.grid(row=1, column=0, sticky='nw')
        self.problems_label = tk.Label(
            master=self.cross_design_window,
            text= 'Select Problems:',
        )
        self.problems_label.grid(row=1, column=2, sticky='nw')
        self.solver_checkboxes = {} # holds checkbutton widgets, store as dictonary for now 
        self.solver_check_vars = {} # holds check boolvars, store as dictonary for now
        # display all potential solvers
        for solver in solver_unabbreviated_directory:
            row=self.solvers_frame.grid_size()[1]
            checkstate= tk.BooleanVar()
            solver_checkbox = tk.Checkbutton(
                master= self.solvers_frame,
                text= solver,
                variable = checkstate,
                command= self.cross_design_problem_compatibility
                ) 
            solver_checkbox.grid(row=row, column=0, sticky='w', padx=10)
            self.solver_checkboxes[solver] = solver_checkbox
            self.solver_check_vars[solver] = checkstate
        self.problem_checkboxes = {} # holds checkbutton widgets, store as dictonary for now 
        self.problem_check_vars = {} # holds check boolvars, store as dictonary for now
        # display all potential problems
        for problem in problem_unabbreviated_directory:
            row=self.problems_frame.grid_size()[1]
            checkstate= tk.BooleanVar()
            problem_checkbox = tk.Checkbutton(
                master= self.problems_frame,
                text= problem,
                variable = checkstate,
                command= self.cross_design_solver_compatibility
                ) 
            problem_checkbox.grid(row=row, column=0, sticky='w', padx=10)
            self.problem_checkboxes[problem] = problem_checkbox
            self.problem_check_vars[problem] = checkstate
        self.create_cross_button = tk.Button(
            master=self.cross_design_window,
            text= 'Add Cross Design to Experiment',
            command = self.create_cross_design)
        self.create_cross_button.grid(row=3,column=0)
        self.cross_design_problem_compatibility() # run to check solvers already in experiment
        self.cross_design_solver_compatibility() # run to check problems already in experiment

    def cross_design_problem_compatibility(self) -> None:
        # create temp objects for current selected solvers and all possilble problems
        temp_solvers = []
        # solvers previously added to experiment
        for solver_group in self.master_solver_dict:
            dp_0 = self.master_solver_dict[solver_group][0] # frist design point if design, only design pt if no design
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
            temp_exp = ProblemsSolvers(solvers=temp_solvers, problems=temp_problem) #temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            if error:
                self.problem_checkboxes[problem_name].configure(state='disabled')
            else:
                self.problem_checkboxes[problem_name].configure(state='normal')
        
        #update problem & problem datafarming selections
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
            self.problem_datafarm_var,
            "Problem",
            *self.problem_list,
            command=self.show_problem_datafarm,
        )
        self.problem_datafarm_select_menu.grid(row=0, column=1)
    
    def cross_design_solver_compatibility(self) -> None:
        # create temp objects for current selected solvers and all possilble problems
        temp_problems = []
        # solvers previously added to experiment
        for problem_group in self.master_problem_dict:
            dp_0 = self.master_problem_dict[problem_group][0] # frist design point if design, only design pt if no design
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
            temp_exp = ProblemsSolvers(solvers=temp_solver, problems=temp_problems) #temp experiment to run check compatibility
            error = temp_exp.check_compatibility()
            if error:
                self.solver_checkboxes[solver_name].configure(state='disabled')
            else:
                self.solver_checkboxes[solver_name].configure(state='normal')
        
        #update solver & solver datafarming selections
        self.solver_select_menu.destroy()
        self.solver_select_menu = ttk.OptionMenu(
            self.solver_selection_frame,
            self.solver_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_factors,
        )
        self.solver_select_menu.grid(row=0, column=1)
        self.solver_datafarm_select_menu.destroy()
        self.solver_datafarm_select_menu = ttk.OptionMenu(
            self.solver_datafarm_selection_frame,
            self.solver_datafarm_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_datafarm,
        )
        self.solver_datafarm_select_menu.grid(row=0, column=1)
    def create_cross_design(self) -> None:
        for solver in self.solver_check_vars:
            checkstate = self.solver_check_vars[solver].get()
            if checkstate: #add solver with default factor settings to master dict
                temp_solver = solver_unabbreviated_directory[solver]()
                factors = {factor: value["default"] for factor, value in temp_solver.specifications.items()}
                solver_name = temp_solver.name
                solver_save_name = self.get_unique_name(
                    self.master_solver_dict, solver_name
                )
                self.master_solver_dict[solver_save_name] = [[factors, solver_name]]
                # add solver row to list display
                solver_row = len(self.master_solver_dict) - 1
                self.solver_list_label = tk.Label(
                    master=self.solver_list_canvas,
                    text=solver_save_name,
                )
                self.solver_list_label.grid(row=solver_row, column=1)
                self.solver_list_labels[solver_save_name] = self.solver_list_label

                # add delete and view/edit buttons
                self.solver_edit_button = tk.Button(
                    master=self.solver_list_canvas,
                    text="View/Edit",
                    command=lambda: self.edit_solver(solver_save_name),
                )
                self.solver_edit_button.grid(row=solver_row, column=2)
                self.solver_edit_buttons[solver_save_name] = self.solver_edit_button
                self.solver_del_button = tk.Button(
                    master=self.solver_list_canvas,
                    text="Delete",
                    command=lambda: self.delete_solver(solver_save_name),
                )
                self.solver_del_button.grid(row=solver_row, column=3)
                self.solver_del_buttons[solver_save_name] = self.solver_del_button
                
        for problem in self.problem_check_vars:
            checkstate = self.problem_check_vars[problem].get()
            if checkstate: #add problem with default factor settings to master dict, ignore disabled boxes
                temp_problem = problem_unabbreviated_directory[problem]()
                factors = {factor: value["default"] for factor, value in temp_problem.specifications.items()}
                model_factors = {factor: value["default"] for factor, value in temp_problem.model.specifications.items()}
                factors.update(model_factors)
                problem_name = temp_problem.name
                problem_save_name = self.get_unique_name(
                    self.master_problem_dict, problem_name
                )
                self.master_problem_dict[problem_save_name] = [[factors, problem_name]]
                # add problem row to list display
                problem_row = len(self.master_problem_dict) - 1
                self.problem_list_label = tk.Label(
                    master=self.problem_list_canvas,
                    text=problem_save_name,
                )
                self.problem_list_label.grid(row=problem_row, column=1)
                self.problem_list_labels[problem_save_name] = self.problem_list_label

                # add delete and view/edit buttons
                self.problem_edit_button = tk.Button(
                    master=self.problem_list_canvas,
                    text="View/Edit",
                    command=lambda: self.edit_problem(problem_save_name),
                )
                self.problem_edit_button.grid(row=problem_row, column=2)
                self.problem_edit_buttons[problem_save_name] = self.problem_edit_button
                self.problem_del_button = tk.Button(
                    master=self.problem_list_canvas,
                    text="Delete",
                    command=lambda: self.delete_problem(problem_save_name),
                )
                self.problem_del_button.grid(row=problem_row, column=3)
                self.problem_del_buttons[problem_save_name] = self.problem_del_button
        
    
    def update_solvers_canvas_scroll(self, event: tk.Event) -> None:
        self.solvers_canvas.configure(
            scrollregion=self.solvers_canvas.bbox("all")
        )
    def update_problems_canvas_scroll(self, event: tk.Event) -> None:
        self.problems_canvas.configure(
            scrollregion=self.problems_canvas.bbox("all")
        )
    def load_design(self) -> None:
        # get csv file location and convert to dataframe
        design_file = filedialog.askopenfilename()
        self.design_df = pd.read_csv(design_file, index_col=False)
        file_name = os.path.splitext(os.path.basename(design_file))[0]
        self.dir_path = os.path.dirname(
            design_file
        )  # used to save updated version
        # Get design information from table
        name = self.design_df.at[1, "Name"]
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
            self.design_name_var = tk.StringVar()
            # get unique problem name
            problem_name = self.get_unique_name(
                self.master_problem_dict, file_name
            )
            self.design_name_var.set(problem_name)
            self.problem_name_entry = tk.Entry(
                master=self.model_frame,
                textvariable=self.design_name_var,
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
            self.design_name_var = tk.StringVar()
            # get unique solver name
            solver_name = self.get_unique_name(
                self.master_solver_dict, file_name
            )
            self.design_name_var.set(solver_name)
            self.solver_name_entry = tk.Entry(
                master=self.solver_frame,
                textvariable=self.design_name_var,
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
        csv_filename = f"{self.dir_path}/{self.design_name_var.get()}.csv"
        self.design_df.to_csv(csv_filename, mode="w", header=True, index=False)

        # update design tree
        self.design_tree.destroy()
        self.display_design_tree(csv_filename, frame=self.tree_frame, row=4)

    def add_loaded_solver_to_experiment(self) -> None:
        # convert df to list of dictionaries
        self.design_list = self.filtered_design_df.to_dict(orient="records")

        design_name = self.design_name_var.get()

        solver_holder_list = []  # used so solver list matches datafarming format
        for dp in self.design_list:
            converted_dp = self.convert_proper_datatype(dp, self.obj, var=False)
            solver_list = []  # holds dictionary of dps and solver name
            solver_list.append(converted_dp)
            solver_list.append(self.obj.name)
            solver_holder_list.append(solver_list)

        self.master_solver_dict[design_name] = solver_holder_list

        # add solver name to solver index
        solver_row = len(self.master_solver_dict) - 1
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
        self.solver_design_name_var.set(
            self.get_unique_name(self.master_solver_dict, design_name)
        )

    def add_loaded_problem_to_experiment(self) -> None:
        # convert df to list of dictionaries
        self.design_list = self.filtered_design_df.to_dict(orient="records")

        design_name = self.design_name_var.get()

        problem_holder_list = []  # holds all problem lists within design name
        for dp in self.design_list:
            dp_list = []  # holds dictionary of factors for current dp
            converted_dp = self.convert_proper_datatype(dp, self.obj, var=False)
            dp_list.append(converted_dp)  # append problem factors
            dp_list.append(self.obj.name)  # append name of problem
            problem_holder_list.append(
                dp_list
            )  # add current dp information to holder list

        self.master_problem_dict[design_name] = problem_holder_list

        # add solver name to solver index
        problem_row = len(self.master_problem_dict) - 1
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
            self.get_unique_name(self.master_problem_dict, design_name)
        )

    def load_experiment(self) -> None:
        # ask user for pickle file location
        file_path = filedialog.askopenfilename()
        base = os.path.basename(file_path)
        exp_name = os.path.splitext(base)[0]

        # make sure name is unique
        self.experiment_name = self.get_unique_name(
            self.master_experiment_dict, exp_name
        )

        # load pickle
        tk.messagebox.showinfo(
            "Loading",
            "Loading pickle file. This may take a few minutes. Experiment will appear within created experiments list once loaded.",
        )
        with open(file_path, "rb") as f:
            exp = pickle.load(f)

        self.master_experiment_dict[self.experiment_name] = exp
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

    def show_factor_headers(
        self,
        frame: tk.Frame,
        first_row: int = 0,
    ) -> None:
        """Show factor headers in the GUI.

        Parameters
        ----------
        frame : tk.Frame
            Frame to display factors.
        factor_heading_list : list[str]
            List of factor headings.
        first_row : int, optional
            First row to display factors.

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

    def show_datafarming_options(
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
        #check solver compatibility
        self.check_solver_compatibility()

        """ Initialize frames, headers, and data farming buttons"""

        # self.prob_mod_frame = tk.Frame(master = self.master)
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
            self.master_problem_dict, self.problem_object.name
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
            self.master_solver_dict, self.solver_object.name
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

        return new_name

    def show_problem_datafarm(self, event: tk.Event) -> None:
        # clear previous selections
        self.clear_frame(self.problem_datafarm_frame)
        self.clear_frame(self.model_datafarm_frame)
        self.clear_frame(self.design_display_frame)
        #check solver compatibility
        self.check_solver_compatibility()

        """ Initialize frames, headers, and data farming buttons"""

        self.problem_datafarm_frame = tk.Frame(
            master=self.problem_datafarm_notebook_frame, bg="Red"
        )
        self.problem_datafarm_frame.grid(row=1, column=0)
        self.problem_factor_display_canvas = tk.Canvas(
            master=self.problem_datafarm_frame, bg="Orange"
        )
        self.problem_factor_display_canvas.grid(row=1, column=0)
        self.model_datafarm_frame = tk.Frame(
            master=self.problem_datafarm_notebook_frame, bg="Yellow"
        )
        self.model_datafarm_frame.grid(row=2, column=0)
        self.model_factor_display_canvas = tk.Canvas(
            master=self.model_datafarm_frame, bg="Green"
        )
        self.model_factor_display_canvas.grid(row=1, column=0)

        # Create column for problem factor names
        self.problem_headername_label = tk.Label(
            master=self.problem_datafarm_frame,
            text="Problem Factors",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.problem_headername_label.grid(row=0, column=0, sticky=tk.N + tk.W)

        # Create column for factor type
        self.header_lbl_type = tk.Label(
            master=self.problem_datafarm_frame,
            text="Factor Type",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.header_lbl_type.grid(row=0, column=1, sticky=tk.N + tk.W)

        # Create column for factor default values
        self.header_lbl_include = tk.Label(
            master=self.problem_datafarm_frame,
            text="Default Value",
            font=nametofont("TkHeadingFont"),
            width=20,
        )
        self.header_lbl_include.grid(row=0, column=2, sticky=tk.N + tk.W)

        """ Get problem information from dicrectory and display"""
        # Get problem info from directory
        self.selected_datafarm_problem = self.problem_datafarm_var.get()
        self.problem_datafarm_object = self.problem_list[
            self.selected_datafarm_problem
        ]()

        # show problem factors and store default widgets and values to this dict
        self.problem_datafarm_defaults, last_row = self.show_factor_defaults(
            self.problem_datafarm_object, self.problem_factor_display_canvas
        )
        (
            self.problem_checkstates,
            self.problem_min_vals,
            self.problem_max_vals,
            self.problem_dec_vals,
            self.problem_datafarm_widgets,
            last_row,
        ) = self.show_datafarming_options(
            self.problem_datafarm_object, self.problem_factor_display_canvas
        )

        """ Get model information from dicrectory and display"""
        # self.model_problem_dict = model_problem_class_directory # directory that relates problem name to model class
        # self.model_datafarm_object = self.model_problem_dict[self.selected_datafarm_problem]()
        # show model factors and store default widgets and default values to these
        self.model_datafarm_defaults, new_last_row = self.show_factor_defaults(
            base_object=self.problem_datafarm_object,
            frame=self.problem_factor_display_canvas,
            is_model=True,
            first_row=last_row + 1,
        )
        (
            self.model_checkstates,
            self.model_min_vals,
            self.model_max_vals,
            self.model_dec_vals,
            self.model_datafarm_widgets,
            new_last_row,
        ) = self.show_datafarming_options(
            self.problem_datafarm_object,
            self.problem_factor_display_canvas,
            True,
            last_row + 1,
        )

        # Update problem values with model values
        self.problem_datafarm_defaults.update(self.model_datafarm_defaults)
        self.problem_checkstates.update(self.model_checkstates)
        self.problem_min_vals.update(self.model_min_vals)
        self.problem_max_vals.update(self.model_max_vals)
        self.problem_dec_vals.update(self.model_dec_vals)
        self.problem_datafarm_widgets.update(self.model_datafarm_widgets)

        """Options for creaing design"""
        # Design type for problem
        self.design_type_label = tk.Label(
            master=self.problem_datafarm_frame,
            text="Select Design Type",
            width=30,
        )
        self.design_type_label.grid(row=2, column=0)

        self.problem_design_var = tk.StringVar()
        self.problem_design_var.set("nolhs")
        self.problem_design_type_menu = ttk.OptionMenu(
            self.problem_datafarm_frame,
            self.problem_design_var,
            "nolhs",
            *self.design_types_list,
        )
        self.problem_design_type_menu.grid(row=2, column=1, padx=30)

        # Stack selection for problem
        self.problem_stack_label = tk.Label(
            master=self.problem_datafarm_frame,
            text="Number of Stacks for Problem",
            width=30,
        )
        self.problem_stack_label.grid(row=3, column=0)
        self.problem_stack_var = tk.StringVar()
        self.problem_stack_var.set("1")
        self.problem_stack_entry = ttk.Entry(
            master=self.problem_datafarm_frame,
            width=10,
            textvariable=self.problem_stack_var,
            justify="right",
        )
        self.problem_stack_entry.grid(row=3, column=1)

        # design name entry
        self.problem_design_name_label = tk.Label(
            master=self.model_datafarm_frame,
            text="Name of Design",
            width=20,
        )
        self.problem_design_name_label.grid(row=4, column=0)
        self.problem_design_name_var = tk.StringVar()
        # get unique problem design name
        problem_name = self.get_unique_name(
            self.master_problem_dict,
            f"{self.problem_datafarm_object.name}_design",
        )
        self.problem_design_name_var.set(problem_name)
        self.problem_design_name_entry = tk.Entry(
            master=self.model_datafarm_frame,
            textvariable=self.problem_design_name_var,
            width=20,
        )
        self.problem_design_name_entry.grid(row=4, column=1)
        # create design button
        self.create_problem_design_button = tk.Button(
            master=self.model_datafarm_frame,
            text="Create Design",
            command=self.create_problem_design,
        )
        self.create_problem_design_button.grid(row=2, column=2)

    def show_solver_datafarm(self, event: tk.Event) -> None:
        """Show solver data farming options.

        Parameters
        ----------
        event : tk.Event, optional
            Event that triggered the function.

        """
        # clear previous selections
        self.clear_frame(self.solver_datafarm_frame)
        # check compatibility of problem selections
        self.check_problem_compatibility()

        # Initialize the frame for the solver data farming factor options
        self.solver_datafarm_frame = tk.Frame(
            master=self.solver_datafarm_notebook_frame,
        )
        self.solver_datafarm_frame.grid(row=1, column=0, sticky=tk.N + tk.W)

        # Add all the column headers
        self.show_factor_headers(self.solver_datafarm_frame)

        # Get solver info from dictionary
        selected_solver_name = self.solver_datafarm_var.get()
        self.solver_datafarm_object = self.solver_list[selected_solver_name]()

        # Create a dictionary of factor objects
        # Factor name maps to the factor object
        self.factor_dict: dict[str, DFFactor] = {}

        factors = self.solver_datafarm_object.specifications
        for factor in factors:
            # Get the factor's datatype, description, and default value
            f_type = factors[factor].get("datatype")
            f_description = factors[factor].get("description")
            f_default = factors[factor].get("default")

            if f_type is bool:
                self.factor_dict[factor] = DFBoolean(
                    factor, f_description, f_default
                )
            elif f_type is int:
                self.factor_dict[factor] = DFInteger(
                    factor, f_description, f_default
                )
            elif f_type is float:
                self.factor_dict[factor] = DFFloat(
                    factor, f_description, f_default
                )
            else:
                raise NotImplementedError("Factor type not yet implemented.")

        # Loop through and add everything to the frame
        for index, factor in enumerate(self.factor_dict):
            # Skip every other row to allow for the separator
            row_index = index * 2 + 2

            # Add the line separator
            ttk.Separator(self.solver_datafarm_frame, orient="horizontal").grid(
                row=row_index + 1, column=0, columnspan=8, sticky=tk.E + tk.W
            )

            # Get the factor object
            factor_obj = self.factor_dict[factor]
            # Get the name label
            name_label = factor_obj.get_name_label(self.solver_datafarm_frame)
            name_label.grid(
                row=row_index, column=0, padx=10, pady=3, sticky=tk.W
            )
            # Get the description label
            desc_label = factor_obj.get_description_label(
                self.solver_datafarm_frame
            )
            desc_label.grid(
                row=row_index, column=1, padx=10, pady=3, sticky=tk.W
            )
            # Get the type label
            type_label = factor_obj.get_type_label(self.solver_datafarm_frame)
            type_label.grid(
                row=row_index, column=2, padx=10, pady=3, sticky=tk.W
            )
            # Get the default value entry
            default_entry = factor_obj.get_default_entry(
                self.solver_datafarm_frame
            )
            default_entry.grid(
                row=row_index, column=3, padx=10, pady=3, sticky=tk.W
            )
            # Check if the factor is not a bool/int/float
            if factor_obj.include is None:
                continue
            # Get the include in design checkbutton
            include_checkbutton = factor_obj.get_include_checkbutton(
                self.solver_datafarm_frame
            )
            include_checkbutton.grid(row=row_index, column=4, padx=10, pady=3)
            # Check if the factor is not an int/float
            if factor_obj.minimum is None:  # Max entry also works
                continue
            # Get the min value entry
            min_entry = factor_obj.get_minimum_entry(self.solver_datafarm_frame)
            min_entry.grid(
                row=row_index, column=5, padx=10, pady=3, sticky=tk.W
            )
            # Get the max value entry
            max_entry = factor_obj.get_maximum_entry(self.solver_datafarm_frame)
            max_entry.grid(
                row=row_index, column=6, padx=10, pady=3, sticky=tk.W
            )
            # Check if the factor is not a float
            if factor_obj.num_decimals is None:
                continue
            # Get the decimal value entry
            dec_entry = factor_obj.get_num_decimals_entry(
                self.solver_datafarm_frame
            )
            dec_entry.grid(
                row=row_index, column=7, padx=10, pady=3, sticky=tk.W
            )

        new_last_row = row_index * 2 + 2

        """Options for creaing design"""

        self.design_frame = tk.Frame(master=self.solver_datafarm_frame)
        self.design_frame.grid(row=new_last_row, column=0, columnspan=8)

        # Design type
        self.design_type_label = tk.Label(
            master=self.design_frame,
            text="Design Type",
            width=20,
        )
        self.design_type_label.grid(row=0, column=0)

        self.solver_design_var = tk.StringVar()
        self.solver_design_var.set("nolhs")
        self.design_type_menu = ttk.OptionMenu(
            self.design_frame,
            self.solver_design_var,
            "nolhs",
            *self.design_types_list,
        )
        self.design_type_menu.grid(row=0, column=1, padx=30)

        # Stack selection menu
        self.stack_label = tk.Label(
            self.design_frame,
            text="Number of Stacks",
            width=20,
        )
        self.stack_label.grid(row=1, column=0)
        self.solver_stack_var = tk.StringVar()
        self.solver_stack_var.set("1")
        self.stack_menu = ttk.Entry(
            master=self.design_frame,
            width=10,
            textvariable=self.solver_stack_var,
            justify="right",
        )
        self.stack_menu.grid(row=1, column=1)

        # design name entry
        self.solver_design_name_label = tk.Label(
            master=self.design_frame,
            text="Name of Design",
            width=20,
        )
        self.solver_design_name_label.grid(row=2, column=0)
        self.solver_design_name_var = tk.StringVar()
        # get unique solver design name
        solver_name = self.get_unique_name(
            self.master_solver_dict,
            f"{self.solver_datafarm_object.name}_design",
        )
        self.solver_design_name_var.set(solver_name)
        self.solver_design_name_entry = tk.Entry(
            master=self.design_frame,
            textvariable=self.solver_design_name_var,
            width=20,
        )
        self.solver_design_name_entry.grid(row=2, column=1)
        # create design button
        self.create_solver_design_button = tk.Button(
            master=self.design_frame,
            text="Create Design",
            command=self.create_solver_design,
        )
        self.create_solver_design_button.grid(row=4, column=0, columnspan=2)

    def enable_datafarm_entry(self, class_type: Problem) -> None:
        # enable datafarming options for factors selected to be included in design
        if class_type == Problem:
            for factor in self.problem_checkstates:
                checkstate = self.problem_checkstates[factor].get()
                if factor in self.problem_datafarm_widgets:
                    datafarm_widget_list = self.problem_datafarm_widgets[factor]
                    if checkstate:
                        for widget in datafarm_widget_list:
                            widget.configure(state="normal")
                    else:
                        for widget in datafarm_widget_list:
                            widget.delete(0, tk.END)
                            widget.configure(state="disabled")

    def create_solver_design(self) -> None:
        # Get unique solver design name
        self.solver_design_name = self.get_unique_name(
            self.master_solver_dict, self.solver_design_name_var.get()
        )

        # get n stacks and design type from user input
        n_stacks = int(self.solver_stack_var.get())
        design_type = self.solver_design_var.get()

        """ Determine factors included in design """
        self.solver_design_factors = []  # list of names of factors included in design
        self.solver_cross_design_factors = {}  # dict of cross design factors w/ lists of possible values
        self.solver_fixed_factors = {}  # contains fixed values for factors not in design
        for factor in self.factor_dict:
            # If the factor is not included in the design, it's a fixed factor
            if (
                self.factor_dict[factor].include is None
                or not self.factor_dict[factor].include.get()
            ):
                fixed_val = self.factor_dict[factor].default.get()
                self.solver_fixed_factors[factor] = [fixed_val]
            # If the factor is included in the design, add it to the list of factors
            else:
                if self.factor_dict[factor].type.get() in ("int", "float"):
                    self.solver_design_factors.append(factor)
                elif self.factor_dict[factor].type.get() == "bool":
                    self.solver_cross_design_factors[factor] = ["True", "False"]

        """ Check if there are any factors included in the design """
        if (
            not self.solver_design_factors
            and not self.solver_cross_design_factors
        ):
            error_msg = "No factors included in design\nAdding regular solver to experiment via this button is not yet implemented"
            tk.messagebox.showerror("Error Creating Design", error_msg)
            return

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
            for factor_name in self.solver_design_factors:
                # Lookup the factor in the dictionary
                factor = self.factor_dict[factor_name]
                min_val = factor.minimum.get()
                max_val = factor.maximum.get()
                if factor.type.get() == "float":
                    dec_val = factor.num_decimals.get()
                else:
                    dec_val = "0"

                # Write the values to the file
                data_insert = f"{min_val} {max_val} {dec_val}\n"
                settings_file.write(data_insert)

        try:
            self.solver_design_list = create_design(
                name=self.solver_datafarm_object.name,
                factor_headers=self.solver_design_factors,
                factor_settings_filename=self.solver_design_name,
                fixed_factors=self.solver_fixed_factors,
                cross_design_factors=self.solver_cross_design_factors,
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
            self.solver_datafarm_frame,
            row=999,
            column=0,
            columnspan=8,
        )
        # button to add solver design to experiment
        self.add_solver_design_button = tk.Button(
            master=self.solver_datafarm_frame,
            text="Add this solver to experiment",
            command=self.add_solver_design_to_experiment,
        )
        self.add_solver_design_button.grid(row=1000, column=0, columnspan=8)
        # disable design name entry
        self.solver_design_name_entry.configure(state="disabled")

    def create_problem_design(self) -> None:
        # Get unique solver design name
        self.problem_design_name = self.get_unique_name(
            self.master_problem_dict, self.problem_design_name_var.get()
        )

        # get n stacks and design type from user input
        n_stacks = self.problem_stack_var.get()
        design_type = self.problem_design_var.get()
        # combine model and problem specifications dictionaries
        specifications = {
            **self.problem_datafarm_object.specifications,
            **self.problem_datafarm_object.model.specifications,
        }
        print("specifications", specifications)

        """ Determine factors included in design """
        self.problem_design_factors = []  # list of names of factors included in design
        self.problem_cross_design_factors = {}  # dict of cross design factors w/ lists of possible values
        for factor in self.problem_checkstates:
            checkstate = self.problem_checkstates[factor].get()
            factor_datatype = specifications[factor].get("datatype")

            if checkstate:
                if factor_datatype in (int, float):
                    self.problem_design_factors.append(factor)
                elif factor_datatype is bool:
                    self.problem_cross_design_factors[factor] = [
                        "True",
                        "False",
                    ]

        # if no cross design factors, set dict to None
        if self.problem_cross_design_factors == {}:
            self.problem_cross_design_factors = None

        """ Determine values of fixed factors """
        problem_fixed_factors = {}  # contains fixed values for factors not in design
        for factor in self.problem_datafarm_defaults:
            if factor not in self.problem_design_factors:
                factor_value = self.problem_datafarm_defaults[factor].get()
                problem_fixed_factors[factor] = ast.literal_eval(factor_value)
        self.problem_fixed_factors = problem_fixed_factors
        # # convert fixed factors to proper datatype
        # self.problem_fixed_factors = self.convert_proper_datatype(
        #     problem_fixed_factors, self.problem_datafarm_object
        # )

        """ Create factor settings txt file"""
        settings_filename = f"{self.problem_design_name}_problem_factors"
        settings_filepath = os.path.join(
            DATA_FARMING_DIR, f"{settings_filename}.txt"
        )
        with open(
            settings_filepath,
            "w",
        ) as settings_file:
            settings_file.write("")
        for factor in self.problem_design_factors:
            factor_datatype = specifications[factor].get("datatype")
            min_val = self.problem_min_vals[factor].get()
            max_val = self.problem_max_vals[factor].get()
            if factor_datatype is float:
                dec_val = self.problem_dec_vals[factor].get()
            else:
                dec_val = "0"
            data_insert = f"{min_val} {max_val} {dec_val}\n"
            with open(
                settings_filepath,
                "a",
            ) as settings_file:
                settings_file.write(data_insert)

        self.problem_design_list = create_design(
            name=self.problem_datafarm_object.name,
            factor_headers=self.problem_design_factors,
            factor_settings_filename=settings_filename,
            fixed_factors=self.problem_fixed_factors,
            cross_design_factors=self.problem_cross_design_factors,
            n_stacks=n_stacks,
            design_type=design_type,
            is_problem=True,
        )

        # display design tree for problem, model, or both depending on design options
        self.design_display_frame = tk.Frame(
            master=self.problem_datafarm_notebook_frame
        )
        self.design_display_frame.grid(row=3, column=0)

        # display only problem design if no model design created
        # if all_false_model:
        self.problem_design_tree_label = tk.Label(
            master=self.design_display_frame,
            text="Generated Design over Problem Factors",
            font=nametofont("TkHeadingFont"),
        )
        self.problem_design_tree_label.grid(row=0, column=0)
        self.display_design_tree(
            os.path.join(
                DATA_FARMING_DIR,
                f"{self.problem_design_name}_problem_factors_design.csv",
            ),
            self.design_display_frame,
            row=1,
        )
        # button to add problem design to experiment
        self.add_problem_design_button = tk.Button(
            master=self.design_display_frame,
            text="Add this problem design to experiment",
            command=self.add_problem_design_to_experiment,
        )
        self.add_problem_design_button.grid(row=2, column=0)

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
            self.master_solver_dict, self.solver_name_var.get()
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

        self.master_solver_dict[solver_name] = solver_holder_list
        # add solver name to solver index
        solver_row = len(self.master_solver_dict) - 1
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
            self.get_unique_name(self.master_solver_dict, solver_name)
        )

    def add_problem_to_experiment(self) -> None:
        # Convect problem and model factor values to proper data type
        prob_fixed_factors = self.convert_proper_datatype(
            self.problem_defaults, self.problem_object
        )
        # mod_fixed_factors = self.convert_proper_datatype(self.model_defaults, self.model_object.specifications)

        # get problem name and ensure it is unique
        problem_name = self.get_unique_name(
            self.master_problem_dict, self.problem_name_var.get()
        )

        problem_list = []  # holds dictionary of dps and solver name
        problem_list.append(prob_fixed_factors)
        problem_list.append(self.problem_object.name)

        problem_holder_list = []  # used so solver list matches datafarming format
        problem_holder_list.append(problem_list)

        self.master_problem_dict[problem_name] = problem_holder_list

        # add problem name to problem index
        problem_row = len(self.master_problem_dict) - 1
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
            self.get_unique_name(self.master_problem_dict, problem_name)
        )

    def add_problem_design_to_experiment(self) -> None:
        problem_design_name = self.problem_design_name

        problem_holder_list = []  # holds all problem lists within design name
        for _, dp in enumerate(self.problem_design_list):
            dp_list = []  # holds dictionary of factors for current dp
            dp_list.append(dp)  # append problem factors
            dp_list.append(
                self.problem_datafarm_object.name
            )  # append name of problem
            problem_holder_list.append(
                dp_list
            )  # add current dp information to holder list

        self.master_problem_dict[problem_design_name] = problem_holder_list
        self.add_problem_design_to_list()

    def add_problem_design_to_list(self) -> None:
        problem_design_name = self.problem_design_name

        # add solver name to solver index
        problem_row = len(self.master_problem_dict) - 1
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
            solver_list.append(self.solver_datafarm_object.name)
            solver_holder_list.append(solver_list)

        self.master_solver_dict[solver_design_name] = solver_holder_list
        # add solver name to solver index
        solver_row = len(self.master_solver_dict) - 1
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
        self.solver_design_name_var.set(
            self.get_unique_name(self.master_solver_dict, solver_design_name)
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
        solver = self.master_solver_dict[solver_save_name][0][1]

        self.solver_object = solver_directory[solver]()

        # show problem factors and store default widgets to this dict
        self.solver_defaults = self.show_factor_defaults(
            self.solver_object,
            self.factor_display_canvas,
            factor_dict=self.master_solver_dict[solver_save_name][0][0],
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
        self.master_solver_dict[self.solver_prev_name][0][0] = fixed_factors

        # Change solver save name if applicable
        new_solver_name = self.solver_name_var.get()
        if new_solver_name != self.solver_prev_name:
            self.master_solver_dict[new_solver_name] = self.master_solver_dict[
                self.solver_prev_name
            ]
            del self.master_solver_dict[self.solver_prev_name]
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
        del self.master_solver_dict[solver_name]

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
        del self.master_problem_dict[problem_name]

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
            self.master_experiment_dict, self.experiment_name_var.get()
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

        for solver_group_name in self.master_solver_dict:
            solver_group = self.master_solver_dict[solver_group_name]
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

        for problem_group_name in self.master_problem_dict:
            problem_group = self.master_problem_dict[problem_group_name]
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
        self.master_experiment_dict[self.experiment_name] = self.experiment

        # reset default experiment name for next experiment
        self.experiment_name_var.set(
            self.get_unique_name(
                self.master_experiment_dict, self.experiment_name
            )
        )

        # add exp to row
        self.add_exp_row()

    def clear_experiment(self) -> None:
        # clear solver and problem lists
        self.master_solver_factor_list = []
        self.master_solver_name_list = []
        self.master_problem_factor_list = []
        self.master_problem_name_list = []
        self.master_solver_dict = {}
        self.master_problem_dict = {}
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
        del self.master_experiment_dict[experiment_name]
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
        experiment = self.master_experiment_dict[experiment_name]

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
        self.experiment_defaults_window = tk.Toplevel(self.master)
        self.experiment_defaults_window.title(
            "Simopt Graphical User Interface - Experiment Options Defaults"
        )
        position = center_window(self.master, 0.8)
        self.experiment_defaults_window.geometry(position)

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
        self.post_processing_window = tk.Toplevel(self.master)
        self.post_processing_window.title(
            "Simopt Graphical User Interface - Experiment Options"
        )
        position = center_window(self.master, 0.8)
        self.post_processing_window.geometry(position)

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
        experiment = self.master_experiment_dict[experiment_name]

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
        experiment = self.master_experiment_dict[experiment_name]

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
        experiment = self.master_experiment_dict[experiment_name]

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
        self.plotting_window = tk.Toplevel(self.master)
        self.plotting_window.title(
            "Simopt Graphical User Interface - Experiment Plots"
        )
        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        position = center_window(self.master, 0.8)
        self.plotting_window.geometry(position)

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
        for exp_name in self.master_experiment_dict:
            experiment = self.master_experiment_dict[exp_name]
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
        for exp_name in self.master_experiment_dict:
            experiment = self.master_experiment_dict[exp_name]
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
        self.plot_experiment = self.master_experiment_dict[experiment_name]
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
                        plot_CIs=plot_ci,
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
                        plot_CIs=plot_ci,
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
                        plot_CIs=plot_ci,
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
        self.view_single_window = tk.Toplevel(self.master)
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
        self.edit_title_window = tk.Toplevel(self.master)
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
        self.edit_x_axis_window = tk.Toplevel(self.master)
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
        self.edit_text_window = tk.Toplevel(self.master)
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
        self.edit_image_window = tk.Toplevel(self.master)
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
        self.view_all_window = tk.Toplevel(self.master)
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
            self.view_window = tk.Toplevel(self.master)
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


# Create data farming window class
class DataFarmingWindow:
    """Class to create the data farming window."""

    def __init__(
        self, root: tk.Tk, main_widow: tk.Frame, forced_creation: bool = False
    ) -> None:
        """Initialize the data farming window.

        Parameters
        ----------
        root : tk.Tk
            The root window of the application.
        main_widow : tk.Frame
            The main window of the application.
        forced_creation : bool, optional
            If True, the window will be created even if it already exists, by default False.

        """
        if not forced_creation:
            self.master = root
            self = set_theme(self)

            self.master = root
            # Set the screen width and height
            # Scaled down slightly so the whole window fits on the screen
            position = center_window(self.master, 0.8)
            self.master.geometry(position)

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
            self.design_frame = tk.Frame(master=self.master)
            self.design_frame.grid(row=5, column=0)

            self.create_design_frame = tk.Frame(master=self.master)
            self.run_frame = tk.Frame(master=self.master)
            self.factor_canvas = tk.Canvas(master=self.master)
            self.factors_frame = tk.Frame(master=self.factor_canvas)

            # Initial variable values
            self.factor_que_length = 1
            self.default_values_list = []
            self.checkstate_list = []
            self.min_list = []
            self.max_list = []
            self.dec_list = []

            # Create main window title
            self.title_frame = tk.Frame(master=self.master)
            self.title_frame.grid_rowconfigure(0, weight=1)
            self.title_frame.grid_columnconfigure(0, weight=1)
            self.title_frame.grid(row=0, column=0, sticky=tk.N)
            self.datafarming_title_label = tk.Label(
                master=self.title_frame,
                text="Model Data Farming",
                font=nametofont("TkHeadingFont"),
            )
            self.datafarming_title_label.grid(row=0, column=0)

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

            self.modelselect_frame.grid(row=2, column=0, sticky=tk.W)
            self.model_label = tk.Label(
                master=self.modelselect_frame,  # window label is used in
                text="Select Model:",
                width=20,
            )
            self.model_label.grid(row=0, column=0, sticky=tk.W)
            self.model_var = tk.StringVar()
            self.model_menu = ttk.OptionMenu(
                self.modelselect_frame,
                self.model_var,
                "Model",
                *self.model_list,
                command=self.show_model_factors,
            )
            self.model_menu.grid(row=0, column=1, sticky=tk.W)

            # Create load design button

            self.or_label = tk.Label(
                master=self.modelselect_frame,
                text="OR",
                width=20,
            )
            self.or_label.grid(row=0, column=2, sticky=tk.W)

            self.load_design_button = tk.Button(
                master=self.modelselect_frame,
                text="Load Design CSV",
                width=20,
                command=self.load_design,
            )
            self.load_design_button.grid(row=0, column=3, sticky=tk.W)

    def clear_frame(self, frame: tk.Frame) -> None:
        """Clear all widgets from a frame.

        Parameters
        ----------
        frame : tk.Frame
            Name of frame that you wish to delete all widgets from.

        """
        for widget in frame.winfo_children():
            widget.destroy()

    def load_design(self) -> None:
        # Clear previous selections
        self.clear_frame(frame=self.factors_frame)
        self.clear_frame(frame=self.create_design_frame)
        self.clear_frame(frame=self.run_frame)
        self.clear_frame(frame=self.design_frame)

        # Initialize frame canvas
        self.factor_canvas = tk.Canvas(master=self.master)
        self.factor_canvas.grid_rowconfigure(0, weight=1)
        self.factor_canvas.grid_columnconfigure(0, weight=1)
        self.factor_canvas.grid(row=4, column=0, sticky="nsew")

        self.factors_title_frame = tk.Frame(master=self.master)
        self.factors_title_frame.grid(row=3, column=0, sticky=tk.N + tk.W)
        self.factors_title_frame.grid_rowconfigure(0, weight=0)
        self.factors_title_frame.grid_columnconfigure(0, weight=1)
        self.factors_title_frame.grid_columnconfigure(1, weight=1)
        self.factors_title_frame.grid_columnconfigure(2, weight=1)

        self.factors_frame = tk.Frame(master=self.factor_canvas)
        self.factors_frame.grid(row=0, column=0, sticky=tk.W + tk.N)
        self.factors_frame.grid_rowconfigure(0, weight=1)
        self.factors_frame.grid_columnconfigure(0, weight=1)
        self.factors_frame.grid_columnconfigure(1, weight=1)
        self.factors_frame.grid_columnconfigure(2, weight=1)
        self.factors_frame.grid_columnconfigure(3, weight=1)

        self.loaded_design = True  # Design was loaded by user

        # Create column for model factor names
        self.headername_label = tk.Label(
            master=self.factors_frame,
            text="Default Factors",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.headername_label.grid(row=0, column=0, sticky=tk.N + tk.W, padx=10)

        # Create column for factor type
        self.headertype_label = tk.Label(
            master=self.factors_frame,
            text="Factor Type",
            font=nametofont("TkHeadingFont"),
            width=20,
            anchor="w",
        )
        self.headertype_label.grid(row=0, column=1, sticky=tk.N + tk.W)

        # Values to help with formatting
        entry_width = 20

        # List to hold default values
        self.default_values_list = []
        self.fixed_str = {}

        # Create column for factor default values
        self.headerdefault_label = tk.Label(
            master=self.factors_frame,
            text="Default Value",
            font=nametofont("TkHeadingFont"),
            width=20,
        )
        self.headerdefault_label.grid(row=0, column=2, sticky=tk.N + tk.W)

        # Name of design csv file
        self.csv_filename = filedialog.askopenfilename()

        # get experiment name
        filename = os.path.basename(self.csv_filename)
        name, ext = os.path.splitext(filename)
        # remove design from name if present
        self.experiment_name = name.replace("_design", "")

        # convert loaded design to data frame
        self.design_table = pd.read_csv(self.csv_filename, index_col=False)

        # Get design information from table
        self.model_name = self.design_table.at[1, "Name"]
        self.design_type = self.design_table.at[1, "Design Type"]
        self.n_stacks = self.design_table.at[1, "Number Stacks"]
        self.model_var.set(self.model_name)

        all_factor_names = [col for col in self.design_table.columns[1:-3]]
        self.factor_names = []  # names of factors included in design
        # determine what factors are included in design
        self.factor_status = {}  # dictionary that contains true/false for wheither factor is in design
        for col in self.design_table.columns[
            1:-3
        ]:  # col correspond to factor names, exclude index and information cols
            factor_set = set(self.design_table[col])

            if len(factor_set) > 1:
                design_factor = True
            else:
                design_factor = False

            self.factor_status[col] = design_factor

        # get default values for fixed factors
        self.default_factors = {}  # contains only factors not in design, factor default vals input as str
        for factor in self.factor_status:
            if not self.factor_status[factor]:
                self.default_factors[factor] = self.design_table.at[1, factor]
            else:
                self.factor_names.append(factor)

        self.model_object = model_directory[self.model_name]()

        # Allow user to change default values
        for _, factor in enumerate(all_factor_names):
            self.factor_datatype = self.model_object.specifications[factor].get(
                "datatype"
            )
            self.factor_description = self.model_object.specifications[
                factor
            ].get("description")

            if not self.factor_status[factor]:
                self.factor_default = self.default_factors[factor]

            else:
                self.factor_default = "Cannot Edit Design Factor"

            self.factors_frame.grid_rowconfigure(
                self.factor_que_length, weight=1
            )

            if self.factor_datatype is int:
                self.str_type = "int"
            elif self.factor_datatype is float:
                self.str_type = "float"
            elif self.factor_datatype is list:
                self.str_type = "list"
            elif self.factor_datatype is tuple:
                self.str_type = "tuple"

            # Add label for factor names
            self.factorname_label = tk.Label(
                master=self.factors_frame,
                text=f"{factor} - {self.factor_description}",
                width=40,
                anchor="w",
            )
            self.factorname_label.grid(
                row=self.factor_que_length,
                column=0,
                sticky=tk.N + tk.W,
                padx=10,
            )

            # Add label for factor type
            self.factortype_label = tk.Label(
                master=self.factors_frame,
                text=self.str_type,
                width=20,
                anchor="w",
            )
            self.factortype_label.grid(
                row=self.factor_que_length, column=1, sticky=tk.N + tk.W
            )

            # Add entry box for default value
            default_len = len(str(self.factor_default))
            if default_len > entry_width:
                entry_width = default_len
                if default_len > 150:
                    entry_width = 150
            self.default_value = tk.StringVar()
            self.default_value.set(self.factor_default)
            self.default_entry = tk.Entry(
                master=self.factors_frame,
                width=entry_width,
                textvariable=self.default_value,
                justify="right",
            )
            self.default_entry.grid(
                row=self.factor_que_length,
                column=2,
                sticky=tk.N + tk.W,
                columnspan=5,
            )
            # Display original default value
            # self.default_entry.insert(0, str(self.factor_default))
            # self.default_values_list.append(self.default_value)

            if self.factor_status[factor]:
                self.default_entry.configure(state="disabled")
            else:
                self.default_values_list.append(self.default_value)

            self.factor_que_length += 1

        self.show_design_options()
        self.display_design_tree()

        # disable run until either continue button is selected
        self.run_button.configure(state="disabled")

    def enable_run_button(self) -> None:
        self.run_button.configure(state="normal")

    def show_design_options(self) -> None:
        # Design type selection menu
        self.design_frame = tk.Frame(master=self.master)
        self.design_frame.grid(row=5, column=0)

        # Input options from loaded designs
        if self.loaded_design:
            stack_display = (
                self.n_stacks
            )  # same num of stacks as original loaded design
            design_display = self.design_type
        else:
            stack_display = "1"
            design_display = "nolhs"

        self.design_type_label = tk.Label(
            master=self.design_frame,
            text="Select Design Type",
            width=20,
        )
        self.design_type_label.grid(row=0, column=0)

        self.design_types_list = ["nolhs"]
        self.design_var = tk.StringVar()
        self.design_var.set(design_display)
        self.design_type_menu = ttk.OptionMenu(
            self.design_frame,
            self.design_var,
            design_display,
            *self.design_types_list,
        )
        self.design_type_menu.grid(row=0, column=1, padx=30)

        # Stack selection menu
        self.stack_label = tk.Label(
            master=self.design_frame,
            text="Number of Stacks",
            width=20,
        )
        self.stack_label.grid(row=1, column=0)
        self.stack_var = tk.StringVar()
        self.stack_var.set(stack_display)
        self.stack_menu = ttk.Entry(
            master=self.design_frame,
            width=10,
            textvariable=self.stack_var,
            justify="right",
        )
        self.stack_menu.grid(row=1, column=1)

        # Disable selections for loaded designs
        if self.loaded_design:
            self.design_type_menu.configure(state="disabled")
            self.stack_menu.configure(state="disabled")

        # Name of design file entry
        self.design_filename_label = tk.Label(
            master=self.design_frame,
            text="Name of design:",
            width=20,
        )
        self.design_filename_label.grid(row=0, column=2)
        self.design_filename_var = (
            tk.StringVar()
        )  # variable to hold user specification of design file name
        self.design_filename_var.set(self.experiment_name)
        self.design_filename_entry = tk.Entry(
            master=self.design_frame,
            width=40,
            textvariable=self.design_filename_var,
            justify="right",
        )
        self.design_filename_entry.grid(row=0, column=3)

        # Create design button
        if not self.loaded_design:
            self.create_design_button = tk.Button(
                master=self.design_frame,
                text="Create Design",
                command=self.create_design,
                width=20,
            )
            self.create_design_button.grid(row=0, column=4)

        # Modify and continue design button for loaded designs
        if self.loaded_design:
            self.mod_design_button = tk.Button(
                master=self.design_frame,
                text="Modify Design",
                command=self.mod_design,
                width=20,
            )
            self.mod_design_button.grid(row=0, column=4)
            self.con_design_button = tk.Button(
                master=self.design_frame,
                text="Continue w/o Modifications",
                command=self.con_design,
                width=25,
            )
            self.con_design_button.grid(row=1, column=4)

    def mod_design(self) -> None:
        self.default_values = [
            self.default_value.get()
            for self.default_value in self.default_values_list
        ]  # default value of each factor
        factor_index = 0
        for factor in self.default_factors:
            # self.default_values = [self.default_value.get() for self.default_value in self.default_values_list] # default value of each factor
            new_val = self.default_values[factor_index]
            self.design_table[factor] = new_val
            self.default_factors[factor] = new_val
            factor_index += 1

        self.experiment_name = (
            self.design_filename_var.get()
        )  # name of design file specified by user

        self.csv_filename = os.path.join(
            DATA_FARMING_DIR, f"{self.experiment_name}_design.csv"
        )

        self.design_table.to_csv(self.csv_filename, index=False)

        # read new design csv and convert to df
        self.design_table = pd.read_csv(self.csv_filename, index_col=False)

        tk.messagebox.showinfo(
            "Information",
            f"Design has been modified. {self.experiment_name}_design.csv has been created in {DATA_FARMING_DIR}. ",
        )

        self.display_design_tree()
        self.con_design()

    def con_design(self) -> None:
        # Create design txt file
        self.experiment_name = (
            self.design_filename_var.get()
        )  # name of design file specified by user
        self.design_table[self.factor_names].to_csv(
            os.path.join(
                DATA_FARMING_DIR,
                f"{self.experiment_name}_design.txt",
            ),
            sep="\t",
            index=False,
            header=False,
        )
        self.design_filename = f"{self.experiment_name}_design"

        # get fixed factors in proper data type
        self.fixed_factors = self.convert_proper_datatype(self.default_factors)

        self.enable_run_button()

    def convert_proper_datatype(self, fixed_factors: dict) -> dict:
        """Convert fixed factors to proper data type.

        Parameters
        ----------
        fixed_factors : dict
            Dictionary containing fixed factor names not included in design and corresponding user selected value as str.

        Returns
        -------
        converted_fixed_factors : dict
            Dictrionary containing fixed factor names and corresponding values converted to proper data type.

        """
        converted_fixed_factors = {}

        for _, factor in enumerate(fixed_factors):
            fixed_val = fixed_factors[factor]
            datatype = self.model_object.specifications[factor].get("datatype")

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

    # Display Model Factors
    def show_model_factors(self, *args: tuple) -> None:
        """Show model factors in GUI.

        Parameters
        ----------
        args : tuple
            Tuple containing model name selected by user.

        """
        self.factor_canvas.destroy()

        # Initialize frame canvas
        self.factor_canvas = tk.Canvas(master=self.master)
        self.factor_canvas.grid_rowconfigure(0, weight=1)
        self.factor_canvas.grid_columnconfigure(0, weight=1)
        self.factor_canvas.grid(row=4, column=0, sticky="nsew")
        self.factors_frame = tk.Frame(master=self.factor_canvas)
        self.factor_canvas.create_window(
            (0, 0), window=self.factors_frame, anchor="nw"
        )

        self.factors_frame.grid_rowconfigure(
            self.factor_que_length + 1, weight=1
        )

        self.factors_title_frame = tk.Frame(master=self.master)
        self.factors_title_frame.grid(row=3, column=0, sticky="nsew")
        self.factors_title_frame.grid_rowconfigure(0, weight=0)
        self.factors_title_frame.grid_columnconfigure(0, weight=0)
        self.factors_title_frame.grid_columnconfigure(1, weight=0)
        self.factors_title_frame.grid_columnconfigure(2, weight=0)
        self.factors_title_frame.grid_columnconfigure(3, weight=0)
        self.factors_title_frame.grid_columnconfigure(4, weight=0)
        self.factors_title_frame.grid_columnconfigure(5, weight=0)
        self.factors_title_frame.grid_columnconfigure(6, weight=0)
        self.factors_title_frame.grid_columnconfigure(7, weight=0)

        # self.factors_frame = tk.Frame( master = self.factor_canvas)
        self.factors_frame.grid(row=0, column=0, sticky="nsew")
        self.factors_frame.grid_rowconfigure(0, weight=0)
        self.factors_frame.grid_columnconfigure(0, weight=0)
        self.factors_frame.grid_columnconfigure(1, weight=0)
        self.factors_frame.grid_columnconfigure(2, weight=0)
        self.factors_frame.grid_columnconfigure(3, weight=0)
        self.factors_frame.grid_columnconfigure(4, weight=0)
        self.factors_frame.grid_columnconfigure(5, weight=0)
        self.factors_frame.grid_columnconfigure(6, weight=0)
        self.factors_frame.grid_columnconfigure(7, weight=0)

        # Clear previous selections
        self.clear_frame(self.factors_frame)
        self.clear_frame(self.create_design_frame)
        self.clear_frame(self.run_frame)
        self.clear_frame(self.run_frame)
        self.clear_frame(self.design_frame)

        # created design not loaded
        self.loaded_design = False

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
        self.checkstate_list = []
        self.min_list = []
        self.max_list = []
        self.dec_list = []

        # Values to help with formatting
        entry_width = 20

        # Create column for model factor names
        self.headername_label = tk.Label(
            master=self.factors_frame,
            text="Model Factors",
            font=nametofont("TkHeadingFont"),
            width=10,
            anchor="w",
        )
        self.headername_label.grid(row=0, column=0, sticky=tk.N + tk.W, padx=10)

        # Create column for factor type
        self.headertype_label = tk.Label(
            master=self.factors_frame,
            text="Factor Type",
            font=nametofont("TkHeadingFont"),
            width=10,
            anchor="w",
        )
        self.headertype_label.grid(row=0, column=1, sticky=tk.N + tk.W)

        # Create column for factor default values
        self.headerdefault_label = tk.Label(
            master=self.factors_frame,
            text="Default Value",
            font=nametofont("TkHeadingFont"),
            width=15,
        )
        self.headerdefault_label.grid(row=0, column=2, sticky=tk.N + tk.W)

        # Create column for factor check box
        self.headercheck_label = tk.Label(
            master=self.factors_frame,
            text="Include in Experiment",
            font=nametofont("TkHeadingFont"),
            width=20,
        )
        self.headercheck_label.grid(row=0, column=3, sticky=tk.N + tk.W)

        # Create header for experiment options
        self.headercheck_label = tk.Label(
            master=self.factors_frame,
            text="Experiment Options",
            font=nametofont("TkHeadingFont"),
            width=60,
        )
        self.headercheck_label.grid(row=0, column=4, columnspan=3)

        # Get model selected from drop down
        self.selected_model = self.model_var.get()

        # Get model info from dictionary
        self.model_object = self.model_list[self.selected_model]()
        self.model_name = self.model_object.name

        for factor in self.model_object.specifications:
            self.factor_datatype = self.model_object.specifications[factor].get(
                "datatype"
            )
            self.factor_description = self.model_object.specifications[
                factor
            ].get("description")
            self.factor_default = self.model_object.specifications[factor].get(
                "default"
            )

            # Values to help with formatting
            entry_width = 10

            self.factors_frame.grid_rowconfigure(
                self.factor_que_length, weight=1
            )

            # Add label for factor names
            self.factorname_label = tk.Label(
                master=self.factors_frame,
                text=f"{factor} - {self.factor_description}",
                width=40,
                anchor="w",
            )
            self.factorname_label.grid(
                row=self.factor_que_length,
                column=0,
                sticky=tk.N + tk.W,
                padx=10,
            )

            if self.factor_datatype is float:
                self.factors_frame.grid_rowconfigure(
                    self.factor_que_length, weight=1
                )

                self.str_type = "float"

                # Add label for factor type
                self.factortype_label = tk.Label(
                    master=self.factors_frame,
                    text=self.str_type,
                    width=10,
                    anchor="w",
                )
                self.factortype_label.grid(
                    row=self.factor_que_length, column=1, sticky=tk.N + tk.W
                )

                # Add entry box for default value
                self.default_value = tk.StringVar()
                self.default_entry = tk.Entry(
                    master=self.factors_frame,
                    width=entry_width,
                    textvariable=self.default_value,
                    justify="right",
                )
                self.default_entry.grid(
                    row=self.factor_que_length, column=2, sticky=tk.N + tk.W
                )
                # Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.default_values_list.append(self.default_value)

                self.default_widgets[factor] = self.default_entry

                # Add check box
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton(
                    master=self.factors_frame,
                    variable=self.checkstate,
                    command=self.include_factor,
                    width=5,
                )
                self.checkbox.grid(
                    row=self.factor_que_length, column=3, sticky="nsew"
                )
                self.checkstate_list.append(self.checkstate)

                self.check_widgets[factor] = self.checkbox

                # Add entry box for min val
                self.min_frame = tk.Frame(master=self.factors_frame)
                self.min_frame.grid(
                    row=self.factor_que_length, column=4, sticky=tk.N + tk.W
                )

                self.min_label = tk.Label(
                    master=self.min_frame,
                    text="Min Value",
                    width=10,
                )
                self.min_label.grid(row=0, column=0)
                self.min_val = tk.StringVar()
                self.min_entry = tk.Entry(
                    master=self.min_frame,
                    width=10,
                    textvariable=self.min_val,
                    justify="right",
                )
                self.min_entry.grid(row=0, column=1, sticky=tk.N + tk.W)

                self.min_list.append(self.min_val)

                self.min_widgets[factor] = self.min_entry

                self.min_entry.configure(state="disabled")

                # Add entry box for max val
                self.max_frame = tk.Frame(master=self.factors_frame)
                self.max_frame.grid(
                    row=self.factor_que_length, column=5, sticky=tk.N + tk.W
                )

                self.max_label = tk.Label(
                    master=self.max_frame,
                    text="Max Value",
                    width=10,
                )
                self.max_label.grid(row=0, column=0)

                self.max_val = tk.StringVar()
                self.max_entry = tk.Entry(
                    master=self.max_frame,
                    width=10,
                    textvariable=self.max_val,
                    justify="right",
                )
                self.max_entry.grid(row=0, column=1, sticky=tk.N + tk.W)

                self.max_list.append(self.max_val)

                self.max_widgets[factor] = self.max_entry

                self.max_entry.configure(state="disabled")

                # Add entry box for editable decimals
                self.dec_frame = tk.Frame(master=self.factors_frame)
                self.dec_frame.grid(
                    row=self.factor_que_length, column=6, sticky=tk.N + tk.W
                )

                self.dec_label = tk.Label(
                    master=self.dec_frame,
                    text="# Decimals",
                    width=10,
                )
                self.dec_label.grid(row=0, column=0)

                self.dec_val = tk.StringVar()
                self.dec_entry = tk.Entry(
                    master=self.dec_frame,
                    width=10,
                    textvariable=self.dec_val,
                    justify="right",
                )
                self.dec_entry.grid(row=0, column=1, sticky=tk.N + tk.W)

                self.dec_list.append(self.dec_val)

                self.dec_widgets[factor] = self.dec_entry

                self.dec_entry.configure(state="disabled")

                self.factor_que_length += 1

            elif self.factor_datatype is int:
                self.factors_frame.grid_rowconfigure(
                    self.factor_que_length, weight=1
                )

                self.str_type = "int"

                # Add label for factor type
                self.factortype_label = tk.Label(
                    master=self.factors_frame,
                    text=self.str_type,
                    width=10,
                    anchor="w",
                )
                self.factortype_label.grid(
                    row=self.factor_que_length, column=1, sticky=tk.N + tk.W
                )

                # Add entry box for default value
                self.default_value = tk.StringVar()
                self.default_entry = tk.Entry(
                    master=self.factors_frame,
                    width=entry_width,
                    textvariable=self.default_value,
                    justify="right",
                )
                self.default_entry.grid(
                    row=self.factor_que_length, column=2, sticky=tk.N + tk.W
                )
                # Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.default_values_list.append(self.default_value)

                self.default_widgets[factor] = self.default_entry

                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton(
                    master=self.factors_frame,
                    variable=self.checkstate,
                    command=self.include_factor,
                )
                self.checkbox.grid(
                    row=self.factor_que_length, column=3, sticky="nsew"
                )
                self.checkstate_list.append(self.checkstate)

                self.check_widgets[factor] = self.checkbox

                # Add entry box for min val
                self.min_frame = tk.Frame(master=self.factors_frame)
                self.min_frame.grid(
                    row=self.factor_que_length, column=4, sticky=tk.N + tk.W
                )

                self.min_label = tk.Label(
                    master=self.min_frame,
                    text="Min Value",
                    width=10,
                )
                self.min_label.grid(row=0, column=0)
                self.min_val = tk.StringVar()
                self.min_entry = tk.Entry(
                    master=self.min_frame,
                    width=10,
                    textvariable=self.min_val,
                    justify="right",
                )
                self.min_entry.grid(row=0, column=1, sticky=tk.N + tk.W)

                self.min_list.append(self.min_val)

                self.min_widgets[factor] = self.min_entry

                self.min_entry.configure(state="disabled")

                # Add entry box for max val
                self.max_frame = tk.Frame(master=self.factors_frame)
                self.max_frame.grid(
                    row=self.factor_que_length, column=5, sticky=tk.N + tk.W
                )

                self.max_label = tk.Label(
                    master=self.max_frame,
                    text="Max Value",
                    width=10,
                )
                self.max_label.grid(row=0, column=0)

                self.max_val = tk.StringVar()
                self.max_entry = tk.Entry(
                    master=self.max_frame,
                    width=10,
                    textvariable=self.max_val,
                    justify="right",
                )
                self.max_entry.grid(row=0, column=1, sticky=tk.N + tk.W)

                self.max_list.append(self.max_val)

                self.max_widgets[factor] = self.max_entry

                self.max_entry.configure(state="disabled")

                self.factor_que_length += 1

            elif self.factor_datatype is list:
                self.factors_frame.grid_rowconfigure(
                    self.factor_que_length, weight=1
                )

                self.str_type = "list"

                # Add label for factor type
                self.factortype_label = tk.Label(
                    master=self.factors_frame,
                    text=self.str_type,
                    width=10,
                    anchor="w",
                )
                self.factortype_label.grid(
                    row=self.factor_que_length, column=1, sticky=tk.N + tk.W
                )

                # Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 25:
                        entry_width = 25
                self.default_value = tk.StringVar()
                self.default_entry = tk.Entry(
                    master=self.factors_frame,
                    width=entry_width,
                    textvariable=self.default_value,
                    justify="right",
                )
                self.default_entry.grid(
                    row=self.factor_que_length,
                    column=2,
                    sticky=tk.N + tk.W,
                    columnspan=5,
                )
                # Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.default_values_list.append(self.default_value)

                # Add checkbox (currently not visible)
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton(
                    master=self.factors_frame,
                    variable=self.checkstate,
                    command=self.include_factor,
                )
                # self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)

                self.check_widgets[factor] = self.checkbox

                self.factor_que_length += 1

            elif self.factor_datatype is tuple:
                self.factors_frame.grid_rowconfigure(
                    self.factor_que_length, weight=1
                )

                self.str_type = "tuple"

                # Add label for factor type
                self.factortype_label = tk.Label(
                    master=self.factors_frame,
                    text=self.str_type,
                    width=10,
                    anchor="w",
                )
                self.factortype_label.grid(
                    row=self.factor_que_length, column=1, sticky=tk.N + tk.W
                )

                # Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 25:
                        entry_width = 25
                self.default_value = tk.StringVar()
                self.default_entry = tk.Entry(
                    master=self.factors_frame,
                    width=entry_width,
                    textvariable=self.default_value,
                    justify="right",
                )
                self.default_entry.grid(
                    row=self.factor_que_length,
                    column=2,
                    sticky=tk.N + tk.W,
                    columnspan=5,
                )
                # Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.default_values_list.append(self.default_value)

                # Add checkbox (currently not visible)
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton(
                    master=self.factors_frame,
                    variable=self.checkstate,
                    command=self.include_factor,
                )
                # self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)

                self.check_widgets[factor] = self.checkbox

                self.factor_que_length += 1

        self.show_design_options()

    # Used to display the design tree for both created and loaded designs
    def display_design_tree(self) -> None:
        # Initialize design tree
        self.create_design_frame = tk.Frame(master=self.master)
        self.create_design_frame.grid(row=6, column=0)
        self.create_design_frame.grid_rowconfigure(0, weight=0)
        self.create_design_frame.grid_rowconfigure(1, weight=1)
        self.create_design_frame.grid_columnconfigure(0, weight=1)
        self.create_design_frame.grid_columnconfigure(1, weight=1)

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
        design_table = pd.read_csv(self.csv_filename, index_col="Design #")
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

        # Create buttons to run experiment
        self.run_frame = tk.Frame(master=self.master)
        self.run_frame.grid(row=7, column=0)
        self.run_frame.grid_columnconfigure(0, weight=1)
        self.run_frame.grid_columnconfigure(1, weight=1)
        self.run_frame.grid_columnconfigure(2, weight=1)
        self.run_frame.grid_rowconfigure(0, weight=0)
        self.run_frame.grid_rowconfigure(1, weight=0)

        self.rep_label = tk.Label(
            master=self.run_frame,
            text="Replications",
            width=20,
        )
        self.rep_label.grid(row=0, column=0, sticky=tk.W)
        self.rep_var = tk.StringVar()
        self.rep_entry = tk.Entry(
            master=self.run_frame, textvariable=self.rep_var, width=10
        )
        self.rep_entry.grid(row=0, column=1, sticky=tk.W)

        self.crn_label = tk.Label(
            master=self.run_frame,
            text="CRN",
            width=20,
        )
        self.crn_label.grid(row=1, column=0, sticky=tk.W)
        self.crn_var = tk.StringVar()
        self.crn_option = ttk.OptionMenu(
            self.run_frame, self.crn_var, "Yes", "Yes", "No"
        )
        self.crn_option.grid(row=1, column=1, sticky=tk.W)

        self.run_button = tk.Button(
            master=self.run_frame,
            text="Run All",
            width=20,
            command=self.run_experiment,
        )
        self.run_button.grid(row=0, column=2, sticky=tk.E, padx=30)

    def create_design(self, *args: tuple) -> None:
        """Create a design txt file and design csv file based on user specified design options.

        Parameters
        ----------
        args : tuple
            Tuple containing the number of stacks and design type.

        """
        self.create_design_frame = tk.Frame(master=self.master)
        self.create_design_frame.grid(row=6, column=0)
        self.create_design_frame.grid_rowconfigure(0, weight=0)
        self.create_design_frame.grid_rowconfigure(1, weight=1)
        self.create_design_frame.grid_columnconfigure(0, weight=1)

        # Dictionary used for tree view display of fixed factors
        self.fixed_str = {}

        # user specified design options
        n_stacks = self.stack_var.get()
        design_type = self.design_var.get()
        # List to hold names of all factors part of model to be displayed in csv
        self.factor_names = []  # names of model factors included in experiment
        self.fixed_factors = {}  # fixed factor names and corresponding value

        # Get user inputs for factor values
        self.experiment_name = self.design_filename_var.get()
        default_values = [
            default_value.get() for default_value in self.default_values_list
        ]
        check_values = [checkstate.get() for checkstate in self.checkstate_list]
        min_values = [min_val.get() for min_val in self.min_list]
        max_values = [max_val.get() for max_val in self.max_list]
        dec_values = [dec_val.get() for dec_val in self.dec_list]

        with open(
            os.path.join(DATA_FARMING_DIR, f"{self.experiment_name}.txt"),
        ) as self.model_design_factors:
            self.model_design_factors.write("")

        # values to index through factors
        maxmin_index = 0
        dec_index = 0

        # List to hold names of all factors part of model to be displayed in csv
        self.factor_names = []
        def_factor_str = {}

        # Get experiment information
        for factor_index, factor in enumerate(self.model_object.specifications):
            factor_datatype = self.model_object.specifications[factor].get(
                "datatype"
            )
            factor_include = check_values[factor_index]

            # get user inputs for design factors

            if factor_include:
                self.factor_names.append(factor)

                if factor_datatype in (float, int):
                    factor_min = str(min_values[maxmin_index])
                    factor_max = str(max_values[maxmin_index])
                    maxmin_index += 1

                    if factor_datatype is float:
                        factor_dec = str(dec_values[dec_index])
                        dec_index += 1

                    elif factor_datatype is int:
                        factor_dec = "0"

                data_insert = f"{factor_min} {factor_max} {factor_dec}\n"

                with open(
                    os.path.join(
                        DATA_FARMING_DIR,
                        f"{self.experiment_name}.txt",
                    ),
                    mode="a",
                ) as self.model_design_factors:
                    self.model_design_factors.write(data_insert)

            # add fixed factors to dictionary and increase index values
            else:
                def_factor_str[factor] = default_values[factor_index]
                if factor_datatype is float:
                    dec_index += 1
                    maxmin_index += 1
                elif factor_datatype is int:
                    maxmin_index += 1

        # convert fixed factors to proper data type
        self.fixed_factors = self.convert_proper_datatype(def_factor_str)
        self.design_filename = f"{self.experiment_name}_design"
        self.csv_filename = os.path.join(
            DATA_FARMING_DIR, f"{self.experiment_name}_design.csv"
        )

        """ Use create_design to create a design txt file & design csv"""
        self.design_list = create_design(
            name=self.model_object.name,
            factor_headers=self.factor_names,
            factor_settings_filename=self.experiment_name,
            fixed_factors=self.fixed_factors,
            n_stacks=n_stacks,
            design_type=design_type,
            class_type="model",
            csv_filename=self.csv_filename,
        )
        # Pop up message that csv design file has been created
        tk.messagebox.showinfo(
            "Information",
            f"Design file {self.experiment_name}_design.csv has been created in {DATA_FARMING_DIR}. ",
        )

        # Display Design Values
        self.display_design_tree()

    def run_experiment(self, *args: tuple) -> None:
        """Run experiment with specified design and experiment options.

        Parameters
        ----------
        args : tuple
            Tuple containing the number of replications and whether to use common random numbers.

        """
        # Specify a common number of replications to run of the model at each
        # design point.
        n_reps = int(self.rep_var.get())

        # Specify whether to use common random numbers across different versions
        # of the model.
        if self.crn_var.get() == "Yes":
            crn_across_design_pts = True
        else:
            crn_across_design_pts = False

        output_filename = os.path.join(
            DATA_FARMING_DIR,
            f"{self.experiment_name}_raw_results",
        )

        # Create DataFarmingExperiment object.
        myexperiment = DataFarmingExperiment(
            model_name=self.model_object.name,
            factor_settings_filename=None,
            factor_headers=self.factor_names,
            design_filepath=self.design_filename,
            model_fixed_factors=self.fixed_factors,
        )

        # Run replications and print results to file.
        myexperiment.run(
            n_reps=n_reps, crn_across_design_pts=crn_across_design_pts
        )
        myexperiment.print_to_csv(csv_filename=output_filename)

        # run confirmation message
        tk.messagebox.showinfo(
            "Run Completed",
            f"Experiment Completed. Output file can be found at {output_filename}",
        )

    def include_factor(self, *args: tuple) -> None:
        """Include factor in experiment and enable experiment options.

        Parameters
        ----------
        args : tuple
            Tuple containing the factor name and checkstate value.

        """
        self.check_values = [
            self.checkstate.get() for self.checkstate in self.checkstate_list
        ]
        self.check_index = 0
        self.cat_index = 0

        # If checkbox to include in experiment checked, enable experiment option buttons
        for factor in self.model_object.specifications:
            # Get current checksate values from include experiment column
            self.current_checkstate = self.check_values[self.check_index]
            # Cross check factor type
            self.factor_datatype = self.model_object.specifications[factor].get(
                "datatype"
            )
            self.factor_description = self.model_object.specifications[
                factor
            ].get("description")
            self.factor_default = self.model_object.specifications[factor].get(
                "default"
            )

            # Disable / enable experiment option widgets depending on factor type
            if self.factor_datatype in (int, float):
                self.current_min_entry = self.min_widgets[factor]
                self.current_max_entry = self.max_widgets[factor]

                if self.current_checkstate:
                    self.current_min_entry.configure(state="normal")
                    self.current_max_entry.configure(state="normal")

                else:
                    # Empty current entries
                    self.current_min_entry.delete(0, tk.END)
                    self.current_max_entry.delete(0, tk.END)

                    self.current_min_entry.configure(state="disabled")
                    self.current_max_entry.configure(state="disabled")

            if self.factor_datatype is float:
                self.current_dec_entry = self.dec_widgets[factor]

                if self.current_checkstate:
                    self.current_dec_entry.configure(state="normal")

                else:
                    self.current_dec_entry.delete(0, tk.END)
                    self.current_dec_entry.configure(state="disabled")

            self.check_index += 1


class CrossDesignWindow:
    """Class to create a window for the user to select problems and solvers for a cross-design problem-solver group."""

    def __init__(
        self, root: tk.Tk, main_widow: tk.Tk, forced_creation: bool = False
    ) -> None:
        """Initialize the CrossDesignWindow class.

        Parameters
        ----------
        root : tk.Tk
            The root window of the application.
        main_widow : tk.Tk
            The main window of the application.
        forced_creation : bool, optional
            Whether the window is being forced to be created, by default False.

        """
        if not forced_creation:
            self.master = root
            self = set_theme(self)

            self.main_window = main_widow
            # Set the screen width and height
            # Scaled down slightly so the whole window fits on the screen
            position = center_window(self.master, 0.8)
            self.master.geometry(position)

            self.crossdesign_title_label = tk.Label(
                master=self.master,
                text="Create Cross-Design Problem-Solver Group",
                font=nametofont("TkHeadingFont"),
            )
            self.crossdesign_title_label.place(x=10, y=25)

            self.crossdesign_problem_label = tk.Label(
                master=self.master,
                text="Select Problems:",
            )
            self.crossdesign_problem_label.place(x=190, y=55)

            self.crossdesign_solver_label = tk.Label(
                master=self.master,
                text="Select Solvers:",
            )
            self.crossdesign_solver_label.place(x=10, y=55)

            self.crossdesign_checkbox_problem_list = []
            self.crossdesign_checkbox_problem_names = []
            self.crossdesign_checkbox_solver_list = []
            self.crossdesign_checkbox_solver_names = []

            solver_cnt = 0

            for solver in solver_unabbreviated_directory:
                self.crossdesign_solver_checkbox_var = tk.BooleanVar(
                    self.master, value=False
                )
                self.crossdesign_solver_checkbox = tk.Checkbutton(
                    master=self.master,
                    text=solver,
                    variable=self.crossdesign_solver_checkbox_var,
                )
                self.crossdesign_solver_checkbox.place(
                    x=10, y=85 + (25 * solver_cnt)
                )

                self.crossdesign_checkbox_solver_list.append(
                    self.crossdesign_solver_checkbox_var
                )
                self.crossdesign_checkbox_solver_names.append(solver)

                solver_cnt += 1

            problem_cnt = 0
            for problem in problem_unabbreviated_directory:
                self.crossdesign_problem_checkbox_var = tk.BooleanVar(
                    self.master, value=False
                )
                self.crossdesign_problem_checkbox = tk.Checkbutton(
                    master=self.master,
                    text=problem,
                    variable=self.crossdesign_problem_checkbox_var,
                )
                self.crossdesign_problem_checkbox.place(
                    x=190, y=85 + (25 * problem_cnt)
                )

                self.crossdesign_checkbox_problem_list.append(
                    self.crossdesign_problem_checkbox_var
                )
                self.crossdesign_checkbox_problem_names.append(problem)

                problem_cnt += 1

            if problem_cnt < solver_cnt:
                solver_cnt += 1
                self.crossdesign_macro_label = tk.Label(
                    master=self.master,
                    text="Number of Macroreplications:",
                )
                self.crossdesign_macro_label.place(
                    x=15, y=80 + (25 * problem_cnt)
                )

                self.crossdesign_macro_var = tk.StringVar(self.master)
                self.crossdesign_macro_entry = ttk.Entry(
                    master=self.master,
                    textvariable=self.crossdesign_macro_var,
                    justify=tk.LEFT,
                    width=15,
                )
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")
                self.crossdesign_macro_entry.place(
                    x=15, y=105 + (25 * solver_cnt)
                )

                self.crossdesign_button = ttk.Button(
                    master=self.master,
                    text="Add Cross-Design Problem-Solver Group",
                    width=65,
                    command=self.confirm_cross_design_function,
                )
                self.crossdesign_button.place(x=15, y=135 + (25 * solver_cnt))

            if problem_cnt > solver_cnt:
                problem_cnt += 1

                self.crossdesign_macro_label = tk.Label(
                    master=self.master,
                    text="Number of Macroreplications:",
                )
                self.crossdesign_macro_label.place(
                    x=15, y=80 + (25 * problem_cnt)
                )

                self.crossdesign_macro_var = tk.StringVar(self.master)
                self.crossdesign_macro_entry = ttk.Entry(
                    master=self.master,
                    textvariable=self.crossdesign_macro_var,
                    justify=tk.LEFT,
                    width=15,
                )
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")

                self.crossdesign_macro_entry.place(
                    x=15, y=105 + (25 * problem_cnt)
                )

                self.crossdesign_button = ttk.Button(
                    master=self.master,
                    text="Add Cross-Design Problem-Solver Group",
                    width=45,
                    command=self.confirm_cross_design_function,
                )
                self.crossdesign_button.place(x=15, y=135 + (25 * problem_cnt))

            if problem_cnt == solver_cnt:
                problem_cnt += 1

                self.crossdesign_macro_label = tk.Label(
                    master=self.master,
                    text="Number of Macroreplications:",
                )
                self.crossdesign_macro_label.place(
                    x=15, y=80 + (25 * problem_cnt)
                )

                self.crossdesign_macro_var = tk.StringVar(self.master)
                self.crossdesign_macro_entry = ttk.Entry(
                    master=self.master,
                    textvariable=self.crossdesign_macro_var,
                    justify=tk.LEFT,
                    width=15,
                )
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")
                self.crossdesign_macro_entry.place(
                    x=15, y=105 + (25 * problem_cnt)
                )

                self.crossdesign_button = ttk.Button(
                    master=self.master,
                    text="Add Cross-Design Problem-Solver Group",
                    width=30,
                    command=self.confirm_cross_design_function,
                )
                self.crossdesign_button.place(x=15, y=135 + (25 * problem_cnt))
            else:
                # print("forced creation of cross design window class")
                pass

    def confirm_cross_design_function(self) -> ProblemsSolvers:
        solver_names_list = list(solver_directory.keys())
        problem_names_list = list(problem_directory.keys())
        problem_list = []
        solver_list = []

        for checkbox in self.crossdesign_checkbox_solver_list:
            if checkbox.get():
                # (self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)] + " was selected (solver)")
                # solver_list.append(solver_directory[self.crossdesign_checkbox_solver_names[self.crossdesign_checkbox_solver_list.index(checkbox)]])
                solver_list.append(
                    solver_names_list[
                        self.crossdesign_checkbox_solver_list.index(checkbox)
                    ]
                )

        for checkbox in self.crossdesign_checkbox_problem_list:
            if checkbox.get():
                # (self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)] + " was selected (problem)")
                # problem_list.append(problem_directory[self.crossdesign_checkbox_problem_names[self.crossdesign_checkbox_problem_list.index(checkbox)]])
                problem_list.append(
                    problem_names_list[
                        self.crossdesign_checkbox_problem_list.index(checkbox)
                    ]
                )

        # Solver can handle upto deterministic constraints, but problem has stochastic constraints.
        stochastic = ["FACSIZE-1", "FACSIZE-2", "RMITD-1"]
        if len(solver_list) == 0 or len(problem_list) == 0:
            self.crossdesign_warning = tk.Label(
                master=self.master,
                text="Select at least one solver and one problem",
                font=nametofont("TkHeadingFont"),
                wraplength=300,
            )
            self.crossdesign_warning.place(x=10, y=345)
            return

        if "ASTRODF" in solver_list and any(
            item in stochastic for item in problem_list
        ):
            self.crossdesign_warning = tk.Label(
                master=self.master,
                text="ASTRODF can handle upto deterministic constraints, but problem has stochastic constraints",
                font=nametofont("TkHeadingFont"),
                wraplength=300,
            )
            self.crossdesign_warning.place(x=10, y=345)
            return
        # macro_reps = self.crossdesign_macro_var.get()
        # (solver_list, problem_list)
        # self.crossdesign_ProblemsSolvers = ProblemsSolvers(solver_names=solver_list, problem_names=problem_list, fixed_factors_filename="all_factors")
        self.crossdesign_MetaExperiment = ProblemsSolvers(
            solver_names=solver_list, problem_names=problem_list
        )

        # if self.count_meta_experiment_queue == 0:
        #     self.create_meta_exp_frame()
        self.master.destroy()
        ExperimentWindow.add_meta_exp_to_frame(
            self.main_window, self.crossdesign_macro_var
        )

        return self.crossdesign_MetaExperiment

        # (self.crossdesign_MetaExperiment)

    def get_crossdesign_meta_experiment(self) -> ProblemsSolvers:
        return self.crossdesign_MetaExperiment


class PostProcessingWindow:
    """Postprocessing Page of the GUI.

    Arguments:
    ---------
    master : tk.Tk
        Tkinter window created from Experiment_Window.run_single_function
    myexperiment : object(Experiment)
        Experiment object created in Experiment_Window.run_single_function
    experiment_list : list
        List of experiment object arguments

    """

    def __init__(
        self,
        root: tk.Tk,
        myexperiment,
        experiment_list: list,
        main_window: tk.Tk,
        meta: bool = False,
    ) -> None:
        """Initialize the PostProcessingWindow class.

        Parameters
        ----------
        root : tk.Tk
            The root window of the application.
        myexperiment :
            Experiment object created in Experiment_Window.run_single_function.
        experiment_list :
            List of experiment object arguments.
        main_window : tk.Tk
            The main window of the application.
        meta : bool, optional
            Whether the window is for a meta experiment, by default False.

        """
        self.master = root
        self = set_theme(self)

        self.meta = meta
        self.main_window = main_window
        self.my_experiment = myexperiment
        # ("my exp post pro ", experiment_list)
        self.selected = experiment_list

        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        position = center_window(self.master, 0.8)
        self.master.geometry(position)

        self.frame = tk.Frame(self.master)

        self.title = tk.Label(
            master=self.master,
            text="Welcome to the Post-Processing Page",
            font=nametofont("TkHeadingFont"),
            justify="center",
        )
        if self.meta:
            self.title = tk.Label(
                master=self.master,
                text="Welcome to the Post-Processing \nand Post-Normalization Page",
                font=nametofont("TkHeadingFont"),
                justify="center",
            )

        self.n_postreps_label = tk.Label(
            master=self.master,
            text="Number of Postreplications at each Recommended Solution:",
            wraplength="250",
        )

        self.n_postreps_var = tk.StringVar(self.master)
        self.n_postreps_entry = ttk.Entry(
            master=self.master,
            textvariable=self.n_postreps_var,
            justify=tk.LEFT,
            width=15,
        )
        self.n_postreps_entry.insert(index=tk.END, string="100")

        self.crn_across_budget_label = tk.Label(
            master=self.master,
            text="Use CRN for Postreplications at Solutions Recommended at Different Times?",
            wraplength="250",
        )

        self.crn_across_budget_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_budget_var = tk.StringVar(self.master)
        # sets the default OptionMenu selection
        # self.crn_across_budget_var.set("True")
        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.crn_across_budget_menu = ttk.OptionMenu(
            self.master,
            self.crn_across_budget_var,
            "True",
            *self.crn_across_budget_list,
        )

        self.crn_across_macroreps_label = tk.Label(
            master=self.master,
            text="Use CRN for Postreplications at Solutions Recommended on Different Macroreplications?",
            wraplength="325",
        )

        self.crn_across_macroreps_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_macroreps_var = tk.StringVar(self.master)

        self.crn_across_macroreps_menu = ttk.OptionMenu(
            self.master,
            self.crn_across_macroreps_var,
            "False",
            *self.crn_across_macroreps_list,
        )

        self.crn_norm_budget_label = tk.Label(
            master=self.master,
            text="Use CRN for Postreplications at x\u2080 and x\u002a?",
            wraplength="325",
        )
        self.crn_norm_across_macroreps_var = tk.StringVar(self.master)
        self.crn_norm_across_macroreps_menu = ttk.OptionMenu(
            self.master,
            self.crn_norm_across_macroreps_var,
            "True",
            *self.crn_across_macroreps_list,
        )

        self.n_norm_label = tk.Label(
            master=self.master,
            text="Post-Normalization Parameters",
            font=nametofont("TkHeadingFont"),
            wraplength="300",
        )

        self.n_proc_label = tk.Label(
            master=self.master,
            text="Post-Processing Parameters",
            font=nametofont("TkHeadingFont"),
            wraplength="300",
        )

        self.n_norm_ostreps_label = tk.Label(
            master=self.master,
            text="Number of Postreplications at x\u2080 and x\u002a:",
            wraplength="300",
        )

        self.n_norm_postreps_var = tk.StringVar(self.master)
        self.n_norm_postreps_entry = ttk.Entry(
            master=self.master,
            textvariable=self.n_norm_postreps_var,
            justify=tk.LEFT,
            width=15,
        )
        self.n_norm_postreps_entry.insert(index=tk.END, string="200")

        self.post_processing_run_label = tk.Label(
            master=self.master,  # window label is used for
            text="Complete Post-Processing of the Problem-Solver Pairs:",
            wraplength="250",
        )

        if self.meta:
            self.post_processing_run_label = tk.Label(
                master=self.master,  # window label is used for
                text="Complete Post-Processing and Post-Normalization of the Problem-Solver Pair(s)",
                wraplength="300",
            )

        self.post_processing_run_button = ttk.Button(
            master=self.master,  # window button is used in
            # aesthetic of button and specific formatting options
            text="Post-Process",
            width=15,  # width of button
            command=self.post_processing_run_function,
        )  # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click

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

            self.crn_norm_budget_label.place(x=10, y=305)
            self.crn_norm_across_macroreps_menu.place(x=300, y=305)

            self.n_norm_ostreps_label.place(x=10, y=355)
            self.n_norm_postreps_entry.place(x=300, y=355)

            self.post_processing_run_label.place(x=10, y=405)
            self.post_processing_run_button.place(x=300, y=405)

        self.frame.pack(side="top", fill="both", expand=True)
        self.run_all = all

    def post_processing_run_function(self) -> list:
        self.experiment_list = []
        # self.experiment_list = [self.selected[3], self.selected[4], self.selected[2]]

        # if self.n_postreps_entry.get().isnumeric() != False and self.n_postreps_init_opt_entry.get().isnumeric() != False and self.crn_across_budget_var.get() in self.crn_across_budget_list and self.crn_across_macroreps_var.get() in self.crn_across_macroreps_list:
        if (
            self.n_postreps_entry.get().isnumeric()
            and self.crn_across_budget_var.get() in self.crn_across_budget_list
            and self.crn_across_macroreps_var.get()
            in self.crn_across_macroreps_list
            and (
                self.meta
                and self.n_norm_postreps_entry.get().isnumeric()
                or not self.meta
            )
        ):
            self.experiment_list.append(int(self.n_postreps_entry.get()))
            # self.experiment_list.append(int(self.n_postreps_init_opt_entry.get()))

            # actually adding a boolean value to the list instead of a string
            if self.crn_across_budget_var.get() == "True":
                self.experiment_list.append(True)
            else:
                self.experiment_list.append(False)

            if self.crn_across_macroreps_var.get() == "True":
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

            self.n_postreps = self.experiment_list[0]  # int
            # print("self.n_prostreps", type(self.n_postreps))
            # self.n_postreps_init_opt = self.experiment_list[4] # int
            self.crn_across_budget = self.experiment_list[1]  # boolean
            # print("self.n_prostreps", type(self.n_postreps))
            self.crn_across_macroreps = self.experiment_list[2]  # boolean

            # print("This is the experiment object", self.my_experiment)
            # print("This is the problem name: ", self.my_experiment.problem.name)
            # print("This is the solver name: ", self.my_experiment.solver.name)
            # print("This is the experiment list", self.selected)
            # print ("This is experiment_list ", self.experiment_list)
            # self, n_postreps, crn_across_budget=True, crn_across_macroreps=False
            self.my_experiment.post_replicate(
                self.n_postreps,
                self.crn_across_budget,
                self.crn_across_macroreps,
            )

            if self.meta:
                self.my_experiment.post_normalize(
                    n_postreps_init_opt=int(self.n_norm_postreps_entry.get()),
                    crn_across_init_opt=norm,
                )

            # (self.experiment_list)
            self.master.destroy()
            self.post_processed_bool = True
            ExperimentWindow.post_process_disable_button(
                self.main_window, self.meta
            )

            return self.experiment_list

        elif not self.n_postreps_entry.get().isnumeric():
            message = "Please enter a valid value for the number of postreplications at each recommended solution."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_entry.delete(0, len(self.n_postreps_entry.get()))
            self.n_postreps_entry.insert(index=tk.END, string="100")

        elif (
            self.crn_across_macroreps_var.get()
            not in self.crn_across_macroreps_list
        ):
            message = "Please answer the following question: 'Use CRN for postreplications at Solutions Recommended at Different Times?' with True or False."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.crn_across_budget_var.set("----")

        elif (
            self.crn_across_budget_var.get() not in self.crn_across_budget_list
        ):
            message = "Please answer the following question: 'Use CRN for Postreplications at Solutions Recommended on Different Macroreplications?' with True or False."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.crn_across_macroreps_var.set("----")

        else:
            message = "You have not selected all required field! Check for '*' signs near required input boxes."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_init_opt_entry.delete(
                0, len(self.n_postreps_init_opt_entry.get())
            )
            self.n_postreps_init_opt_entry.insert(index=tk.END, string="6")

            self.crn_across_budget_var.set("True")

            self.crn_across_macroreps_var.set("False")


class PostNormalWindow:
    """Post-Normalization Page of the GUI.

    Arguments:
    ---------
    master : tk.Tk
        Tkinter window created from Experiment_Window.run_single_function
    myexperiment : object(Experiment)
        Experiment object created in Experiment_Window.run_single_function
    experiment_list : list
        List of experiment object arguments

    """

    def __init__(
        self,
        root: tk.Tk,
        experiment_list: list,
        main_window: tk.Tk,
        meta: bool = False,
    ) -> None:
        """Initialize the PostNormalWindow class.

        Parameters
        ----------
        root : tk.Tk
            The root window of the application.
        experiment_list : list
            List of experiment object arguments.
        main_window : tk.Tk
            The main window of the application.
        meta : bool, optional
            Whether the window is for a meta experiment, by default False.

        """
        self.master = root
        self = set_theme(self)

        self.post_norm_exp_list = experiment_list
        self.meta = meta
        self.main_window = main_window
        self.optimal_var = tk.StringVar(master=self.master)
        self.initial_var = tk.StringVar(master=self.master)
        self.check_var = tk.IntVar(master=self.master)
        self.init_var = tk.StringVar(self.master)
        self.proxy_var = tk.StringVar(self.master)
        self.proxy_sol = tk.StringVar(self.master)

        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        position = center_window(self.master, 0.8)
        self.master.geometry(position)

        self.all_solvers = []
        for solvers in self.post_norm_exp_list:
            if solvers.solver.name not in self.all_solvers:
                self.all_solvers.append(solvers.solver.name)

        # ("my exp post pro ", experiment_list)
        self.selected = experiment_list

        self.frame = tk.Frame(self.master)
        top_lab = (
            "Welcome to the Post-Normalization Page for "
            + self.post_norm_exp_list[0].problem.name
            + " \n with Solvers:"
        )
        if self.post_norm_exp_list[0].problem.minmax[0] == 1:
            minmax = "max"
        else:
            minmax = "min"

        opt = "unknown"
        if self.post_norm_exp_list[0].problem.optimal_solution is not None:
            if len(self.post_norm_exp_list[0].problem.optimal_solution) == 1:
                opt = str(
                    self.post_norm_exp_list[0].problem.optimal_solution[0]
                )
            else:
                opt = str(self.post_norm_exp_list[0].problem.optimal_solution)

        for solv in self.all_solvers:
            top_lab = top_lab + " " + solv

        self.title = tk.Label(
            master=self.master,
            text=top_lab,
            font=nametofont("TkHeadingFont"),
            justify="center",
        )
        initsol = self.post_norm_exp_list[0].problem.factors["initial_solution"]
        if len(initsol) == 1:
            initsol = str(initsol[0])
        else:
            initsol = str(initsol)

        self.n_init_label = tk.Label(
            master=self.master,
            text="The Initial Solution, x\u2080, is " + initsol + ".",
            wraplength="400",
        )

        self.n_opt_label = tk.Label(
            master=self.master,
            text="The Optimal Solution, x\u002a, is "
            + opt
            + " for this "
            + minmax
            + "imization Problem. \nIf the Proxy Optimal Value or the Proxy Optimal Solution is unspecified, SimOpt uses the best Solution found in the selected Problem-Solver Pair experiments as the Proxy Optimal Solution.",
            wraplength="600",
            justify="left",
        )

        self.n_optimal_label = tk.Label(
            master=self.master,
            text="Optimal Solution (optional):",
            wraplength="250",
        )
        self.n_proxy_val_label = tk.Label(
            master=self.master,
            text="Insert Proxy Optimal Value, f(x\u002a):",
            wraplength="250",
        )
        self.n_proxy_sol_label = tk.Label(
            master=self.master,
            text="Insert Proxy Optimal Solution, x\u002a:",
            wraplength="250",
        )

        # t = ["x","f(x)"]
        self.n_proxy_sol_entry = ttk.Entry(
            master=self.master,
            textvariable=self.proxy_sol,
            justify=tk.LEFT,
            width=8,
        )
        self.n_proxy_val_entry = ttk.Entry(
            master=self.master,
            textvariable=self.proxy_var,
            justify=tk.LEFT,
            width=8,
        )
        self.n_initial_entry = ttk.Entry(
            master=self.master,
            textvariable=self.init_var,
            justify=tk.LEFT,
            width=10,
        )

        self.n_crn_label = tk.Label(
            master=self.master,
            text="CRN for x\u2080 and Optimal x\u002a?",
            wraplength="310",
        )
        self.n_crn_checkbox = tk.Checkbutton(
            self.master, text="", variable=self.check_var
        )

        self.n_postreps_init_opt_label = tk.Label(
            master=self.master,
            text="Number of Post-Normalizations at x\u2080 and x\u002a:",
            wraplength="310",
        )

        self.n_postreps_init_opt_var = tk.StringVar(self.master)
        self.n_postreps_init_opt_entry = ttk.Entry(
            master=self.master,
            textvariable=self.n_postreps_init_opt_var,
            justify=tk.LEFT,
            width=15,
        )
        self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

        self.post_processing_run_label = tk.Label(
            master=self.master,  # window label is used for
            text="Click to Post-Normalize the Problem-Solver Pairs",
            wraplength="300",
        )

        self.post_processing_run_button = ttk.Button(
            master=self.master,  # window button is used in
            # aesthetic of button and specific formatting options
            text="Post-Normalize",
            width=15,  # width of button
            command=self.post_norm_run_function,
        )  # if command=function(), it will only work once, so cannot call function, only specify which one, activated by left mouse click

        self.title.place(x=75, y=15)

        self.n_init_label.place(x=10, y=70)

        self.n_opt_label.place(x=10, y=90)

        # self.n_proxy_label.place(x=10, y=200)
        self.n_proxy_val_label.place(x=10, y=190)
        self.n_proxy_sol_label.place(x=325, y=190)
        self.n_proxy_val_entry.place(x=220, y=190)
        self.n_proxy_sol_entry.place(x=530, y=190)

        self.n_crn_label.place(x=10, y=230)
        self.n_crn_checkbox.place(x=325, y=230)
        # default to selected
        self.n_crn_checkbox.select()

        self.n_postreps_init_opt_label.place(x=10, y=270)
        self.n_postreps_init_opt_entry.place(x=325, y=270)

        self.post_processing_run_label.place(x=10, y=310)
        self.post_processing_run_button.place(x=325, y=310)

        self.frame.pack(side="top", fill="both", expand=True)

    def post_norm_run_function(self) -> None:
        self.experiment_list = []

        # if self.n_postreps_entry.get().isnumeric() != False and self.n_postreps_init_opt_entry.get().isnumeric() != False and self.crn_across_budget_var.get() in self.crn_across_budget_list and self.crn_across_macroreps_var.get() in self.crn_across_macroreps_list:
        if self.n_postreps_init_opt_entry.get().isnumeric():
            n_postreps_init_opt = int(self.n_postreps_init_opt_entry.get())
            crn = self.check_var.get()
            proxy_val = None
            proxy_sol = None
            if self.proxy_sol.get() != "":
                proxy_sol = ast.literal_eval(self.proxy_sol.get())
            if self.proxy_var.get() != "":
                proxy_val = ast.literal_eval(self.proxy_var.get())
            post_normalize(
                self.post_norm_exp_list,
                n_postreps_init_opt,
                crn_across_init_opt=crn,
                proxy_init_val=None,
                proxy_opt_val=proxy_val,
                proxy_opt_x=proxy_sol,
            )
            # self.master.destroy()
            self.post_processed_bool = True

            self.postrep_window = tk.Toplevel()
            position = center_window(self.master, 0.8)
            self.postrep_window.geometry(position)
            self.postrep_window.title("Plotting Page")
            self.master.destroy()
            PlotWindow(
                self.postrep_window,
                self.main_window,
                experiment_list=self.post_norm_exp_list,
            )

            return

        else:
            message = "Please enter a valid value for the number of postreplications at each recommended solution."
            tk.messagebox.showerror(title="Error Window", message=message)

            self.n_postreps_entry.delete(0, len(self.n_postreps_entry.get()))
            self.n_postreps_entry.insert(index=tk.END, string="100")


class PlotWindow:
    """Plot Window Page of the GUI.

    Arguments:
    ---------
    master : tk.Tk
        Tkinter window created from Experiment_Window.run_single_function
    myexperiment : object(Experiment)
        Experiment object created in Experiment_Window.run_single_function
    experiment_list : list
        List of experiment object arguments

    """

    def __init__(
        self,
        root: tk.Tk,
        main_window: tk.Tk,
        experiment_list: list,
        meta_list=None,
    ) -> None:
        """Initialize the Plot_Window class.

        Parameters
        ----------
        root : tk.Tk
            The root window of the application.
        main_window : tk.Tk
            The main window of the application.
        experiment_list : list, optional
            List of experiment object arguments, by default None.
        meta_list : object, optional
            MetaList object, by default None.

        """
        self.master = root
        self = set_theme(self)

        self.metaList = meta_list
        self.experiment_list = experiment_list
        self.main_window = main_window
        self.plot_types_inputs = [
            "cdf_solvability",
            "quantile_solvability",
            "diff_cdf_solvability",
            "diff_quantile_solvability",
        ]
        self.plot_type_names = [
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
        self.num_plots = 0
        self.plot_exp_list = []
        self.plot_type_list = []
        self.checkbox_list = []
        self.plot_CI_list = []
        self.plot_param_list = []
        self.all_path_names = []
        self.bad_label = None
        self.plot_var = tk.StringVar(master=self.master)

        # Set the screen width and height
        # Scaled down slightly so the whole window fits on the screen
        position = center_window(self.master, 0.8)
        self.master.geometry(position)

        self.params = [
            tk.StringVar(master=self.master),
            tk.StringVar(master=self.master),
            tk.StringVar(master=self.master),
            tk.StringVar(master=self.master),
            tk.StringVar(master=self.master),
            tk.StringVar(master=self.master),
            tk.StringVar(master=self.master),
        ]

        self.problem_menu = Listbox(
            self.master,
            selectmode=MULTIPLE,
            exportselection=False,
            width=10,
            height=6,
        )
        self.solver_menu = Listbox(
            self.master,
            selectmode=MULTIPLE,
            exportselection=False,
            width=10,
            height=6,
        )

        self.all_problems = []
        i = 0

        # Creating a list of problems from the experiment list
        for problem in self.experiment_list:
            if problem.problem.name not in self.all_problems:
                self.all_problems.append(problem.problem.name)
                self.problem_menu.insert(i, problem.problem.name)
                i += 1

        # ("solvers:",self.all_solvers)
        if self.metaList is not None:
            i = 0
            # Getting the names for the solvers from the metalist and add it to the solver menu
            for name in self.metaList.solver_names:
                self.solver_menu.insert(i, name)
                i += 1
        else:
            self.all_solvers = []
            i = 0
            # Getting the solvers from the experiment list and add it to the solver menu
            for solvers in self.experiment_list:
                if solvers.solver.name not in self.all_solvers:
                    self.all_solvers.append(solvers.solver.name)
                    self.solver_menu.insert(i, solvers.solver.name)
                    i += 1
        # ("exp:",self.experiment_list[0].solver_names)

        self.solver_menu.bind("<<ListboxSelect>>", self.solver_select_function)

        self.instruction_label = tk.Label(
            master=self.master,  # window label is used in
            text="Welcome to the Plotting Page of SimOpt \n Select Problems and Solvers to Plot",
            justify="center",
        )

        self.problem_label = tk.Label(
            master=self.master,  # window label is used in
            text="Select Problem(s):*",
        )
        self.plot_label = tk.Label(
            master=self.master,  # window label is used in
            text="Select Plot Type:*",
        )

        # from experiments.inputs.all_factors.py:
        self.problem_list = problem_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.problem_var = tk.StringVar(master=self.master)

        # self.problem_menu = tk.Listbox(self.master, self.problem_var, "Problem", *self.all_problems, command=self.experiment_list[0].problem.name)
        self.plot_menu = ttk.OptionMenu(
            self.master,
            self.plot_var,
            "Plot",
            *self.plot_type_names,
            command=partial(self.get_parameters_and_settings, self.plot_var),
        )
        self.solver_label = tk.Label(
            master=self.master,  # window label is used in
            text="Select Solver(s):*",
        )

        # from experiments.inputs.all_factors.py:
        self.solver_list = solver_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.solver_var = tk.StringVar(master=self.master)

        self.add_button = ttk.Button(
            master=self.master, text="Add", width=15, command=self.add_plot
        )

        self.post_normal_all_button = ttk.Button(
            master=self.master,
            text="See All Plots",
            width=20,
            state="normal",
            command=self.plot_button,
        )

        self.style = ttk.Style()
        self.style.configure("Bold.TLabel", font=nametofont("TkHeadingFont"))
        workspace_lbl = ttk.Label(
            master=self.master, text="Plotting Workspace", style="Bold.TLabel"
        )

        self.queue_label_frame = ttk.LabelFrame(
            master=self.master, labelwidget=workspace_lbl
        )

        self.queue_canvas = tk.Canvas(
            master=self.queue_label_frame, borderwidth=0
        )

        self.queue_frame = ttk.Frame(master=self.queue_canvas)
        self.vert_scroll_bar = Scrollbar(
            self.queue_label_frame,
            orient="vertical",
            command=self.queue_canvas.yview,
        )
        self.horiz_scroll_bar = Scrollbar(
            self.queue_label_frame,
            orient="horizontal",
            command=self.queue_canvas.xview,
        )
        self.queue_canvas.configure(
            xscrollcommand=self.horiz_scroll_bar.set,
            yscrollcommand=self.vert_scroll_bar.set,
        )

        self.vert_scroll_bar.pack(side="right", fill="y")
        self.horiz_scroll_bar.pack(side="bottom", fill="x")

        self.queue_canvas.pack(side="left", fill="both", expand=True)
        self.queue_canvas.create_window(
            (0, 0),
            window=self.queue_frame,
            anchor="nw",
            tags="self.queue_frame",
        )

        self.notebook = ttk.Notebook(master=self.queue_frame)
        self.notebook.pack(fill="both")
        self.tab_one = tk.Frame(master=self.notebook)
        self.notebook.add(self.tab_one, text="Problem-Solver Pairs to Plots")
        self.tab_one.grid_rowconfigure(0)

        self.heading_list = [
            "Problem",
            "Solver",
            "Plot Type",
            "Remove Row",
            "View Plot",
            "Parameters",
            "PNG File Path",
        ]

        for heading in self.heading_list:
            self.tab_one.grid_columnconfigure(self.heading_list.index(heading))
            label = tk.Label(
                master=self.tab_one,
                text=heading,
                font=nametofont("TkHeadingFont"),
            )
            label.grid(
                row=0, column=self.heading_list.index(heading), padx=10, pady=3
            )

        self.instruction_label.place(relx=0.3, y=0)

        self.problem_label.place(x=10, rely=0.08)
        self.problem_menu.place(x=10, rely=0.11, relwidth=0.3)

        self.solver_label.place(x=10, rely=0.25)
        self.solver_menu.place(x=10, rely=0.28, relwidth=0.3)

        self.plot_label.place(relx=0.4, rely=0.08)
        self.plot_menu.place(relx=0.55, rely=0.08)

        self.add_button.place(relx=0.45, rely=0.45)

        separator = ttk.Separator(master=self.master, orient="horizontal")
        separator.place(relx=0.35, rely=0.08, relheight=0.4)

        self.post_normal_all_button.place(relx=0.01, rely=0.92)

        # self.queue_label_frame.place(x=10, rely=.7, relheight=.3, relwidth=1)
        self.queue_label_frame.place(
            x=10, rely=0.56, relheight=0.35, relwidth=0.99
        )

        self.param_label = []
        self.param_entry = []
        self.factor_label_frame_problem = None

        self.CI_label_frame = ttk.LabelFrame(
            master=self.master, text="Plot Settings and Parameters"
        )
        self.CI_canvas = tk.Canvas(master=self.CI_label_frame, borderwidth=0)
        self.CI_frame = ttk.Frame(master=self.CI_canvas)

        self.CI_canvas.pack(side="left", fill="both", expand=True)
        self.CI_canvas.create_window(
            (0, 0), window=self.CI_frame, anchor="nw", tags="self.queue_frame"
        )

        self.CI_label_frame.place(
            relx=0.4, rely=0.15, relheight=0.2, relwidth=0.3
        )

        self.settings_label_frame = ttk.LabelFrame(
            master=self.master, text="Error Estimation Setting and Parameters"
        )
        self.settings_canvas = tk.Canvas(
            master=self.settings_label_frame, borderwidth=0
        )
        self.settings_frame = ttk.Frame(master=self.settings_canvas)

        self.settings_canvas.pack(side="left", fill="both", expand=True)
        self.settings_canvas.create_window(
            (0, 0),
            window=self.settings_frame,
            anchor="nw",
            tags="self.queue_frame",
        )
        self.settings_canvas.grid_rowconfigure(0)
        self.settings_label_frame.place(
            relx=0.65, rely=0.15, relheight=0.2, relwidth=0.3
        )

        """
            # Confidence Interval Checkbox
            entry1 = tk.Checkbutton(self.settings_canvas, variable=self.params[0], onvalue=True, offvalue=False)
            entry1.select()
            # entry1 = ttk.OptionMenu(self.settings_canvas, self.params[0], "True", *tf_list)
            label1 = tk.Label(master=self.settings_canvas, text="Confidence Intervals", font=f"{TEXT_FAMILY} 14")
            label1.grid(row=0, column=0, padx=10, pady=3)
            entry1.grid(row=0, column=1, padx=10, pady=3)

            # Plot Together Checkbox
            entry = tk.Checkbutton(self.settings_canvas, variable=self.params[1], onvalue=True, offvalue=False)
            entry.select()
            # Creates the Check Mark that checks whether the plots will be plot together
            label = tk.Label(master=self.settings_canvas, text="Plot Together", font=f"{TEXT_FAMILY} 14")
            label.grid(row=1, column=0, padx=10, pady=3)
            entry.grid(row=1, column=1, padx=10, pady=3)

            entry2 = tk.Checkbutton(self.settings_canvas, variable=self.params[2], onvalue=True, offvalue=False)
            entry2.select()
            label2 = tk.Label(master=self.settings_canvas, text="Print Max HW", font=f"{TEXT_FAMILY} 14")
            label2.grid(row=2, column=0, padx=10, pady=3)
            entry2.grid(row=2, column=1, padx=10, pady=3)
            """

    def add_plot(self) -> None:
        self.plot_exp_list = []

        solver_lst = ""
        # Appends experiment that is part of the experiment list if it matches what was chosen in the solver menu
        for i in self.solver_menu.curselection():
            solver_lst = solver_lst + self.solver_menu.get(i) + " "
            for j in self.problem_menu.curselection():
                problem_lst = ""
                if self.metaList is not None:
                    for metaexp in self.metaList.experiments:
                        for exp in metaexp:
                            if exp.solver.name == self.solver_menu.get(
                                i
                            ) and exp.problem.name == self.problem_menu.get(j):
                                self.plot_exp_list.append(exp)
                else:
                    for exp in self.experiment_list:
                        if exp.solver.name == self.solver_menu.get(
                            i
                        ) and exp.problem.name == self.problem_menu.get(j):
                            self.plot_exp_list.append(exp)
                problem_lst = problem_lst + self.problem_menu.get(j) + " "

        plot_type = str(self.plot_var.get())
        if len(self.plot_exp_list) == 0 or str(plot_type) == "Plot":
            txt = "At least 1 Problem, 1 Solver, and 1 Plot Type must be selected."
            self.bad_label = tk.Label(
                master=self.master,
                text=txt,
                justify="center",
            )
            self.bad_label.place(relx=0.45, rely=0.5)
            return
        elif self.bad_label is not None:
            self.bad_label.destroy()
            self.bad_label = None

        self.plot_type_list.append(plot_type)

        param_value_list = []
        for t in self.params:
            new_value = ""
            if t.get() is True:
                new_value = True
            elif t.get() is False:
                new_value = False
            elif t.get() != "":
                try:
                    new_value = float(t.get())
                except ValueError:
                    new_value = t.get()
            param_value_list.append(new_value)

        exp_list = self.plot_exp_list
        if self.metaList is not None:
            list_exp_list = self.metaList.experiments
        else:
            list_exp_list = [[exp] for exp in exp_list]

        if self.plot_type_list[-1] == "All Progress Curves":
            path_name = plot_progress_curves(
                exp_list,
                plot_type="all",
                normalize=bool(param_value_list[1]),
                all_in_one=bool(param_value_list[0]),
            )
            param_list = {"normalize": bool(param_value_list[1])}
        if self.plot_type_list[-1] == "Mean Progress Curve":
            path_name = plot_progress_curves(
                exp_list,
                plot_type="mean",
                normalize=bool(param_value_list[3]),
                all_in_one=bool(param_value_list[1]),
                plot_conf_ints=bool(param_value_list[0]),
                print_max_hw=bool(param_value_list[2]),
                n_bootstraps=int(param_value_list[4]),
                conf_level=param_value_list[5],
            )
            param_list = {
                "plot CIs": bool(param_value_list[0]),
                "print max hw": bool(param_value_list[2]),
                "normalize": bool(param_value_list[3]),
                "n_bootstraps": int(param_value_list[4]),
                "conf_level": param_value_list[5],
            }
        elif self.plot_type_list[-1] == "Quantile Progress Curve":
            path_name = plot_progress_curves(
                exp_list,
                plot_type="quantile",
                beta=param_value_list[3],
                normalize=bool(param_value_list[4]),
                plot_conf_ints=bool(param_value_list[0]),
                all_in_one=bool(param_value_list[1]),
                print_max_hw=bool(param_value_list[2]),
                n_bootstraps=int(param_value_list[5]),
                conf_level=param_value_list[6],
            )
            param_list = {
                "plot CIs": bool(param_value_list[0]),
                "print max hw": bool(param_value_list[2]),
                "normalize": bool(param_value_list[4]),
                "beta": param_value_list[3],
                "n_bootstraps": int(param_value_list[5]),
                "conf_level": param_value_list[6],
            }
        elif self.plot_type_list[-1] == "Solve time CDF":
            path_name = plot_solvability_cdfs(
                exp_list,
                solve_tol=param_value_list[2],
                plot_conf_ints=bool(param_value_list[0]),
                print_max_hw=bool(param_value_list[1]),
                n_bootstraps=int(param_value_list[3]),
                conf_level=param_value_list[4],
            )
            param_list = {
                "plot CIs": bool(param_value_list[0]),
                "print max hw": bool(param_value_list[1]),
                "solve tol": param_value_list[2],
                "n_bootstraps": int(param_value_list[3]),
                "conf_level": param_value_list[4],
            }
        elif self.plot_type_list[-1] == "Area Scatter Plot":
            path_name = plot_area_scatterplots(
                list_exp_list,
                plot_conf_ints=bool(param_value_list[0]),
                print_max_hw=bool(param_value_list[1]),
                n_bootstraps=int(param_value_list[2]),
                conf_level=param_value_list[3],
            )
            param_list = {
                "plot CIs": bool(param_value_list[0]),
                "print max hw": bool(param_value_list[1]),
                "n_bootstraps": int(param_value_list[2]),
                "conf_level": param_value_list[3],
            }
        elif self.plot_type_list[-1] == "CDF Solvability":
            path_name = plot_solvability_profiles(
                list_exp_list,
                plot_type="cdf_solvability",
                plot_conf_ints=bool(param_value_list[0]),
                print_max_hw=bool(param_value_list[1]),
                solve_tol=param_value_list[2],
                ref_solver=None,
                n_bootstraps=int(param_value_list[3]),
                conf_level=param_value_list[4],
            )
            param_list = {
                "plot CIs": bool(param_value_list[0]),
                "print max hw": bool(param_value_list[1]),
                "solve tol": param_value_list[2],
                "n_bootstraps": int(param_value_list[3]),
                "conf_level": param_value_list[4],
            }
        elif self.plot_type_list[-1] == "Quantile Solvability":
            path_name = plot_solvability_profiles(
                list_exp_list,
                plot_type="quantile_solvability",
                plot_conf_ints=bool(param_value_list[0]),
                print_max_hw=bool(param_value_list[1]),
                solve_tol=param_value_list[2],
                beta=param_value_list[3],
                ref_solver=None,
                n_bootstraps=int(param_value_list[4]),
                conf_level=param_value_list[5],
            )
            param_list = {
                "plot CIs": bool(param_value_list[0]),
                "print max hw": bool(param_value_list[1]),
                "solve tol": param_value_list[2],
                "beta": param_value_list[3],
                "n_bootstraps": int(param_value_list[4]),
                "conf_level": param_value_list[5],
            }
        elif self.plot_type_list[-1] == "CDF Difference Plot":
            path_name = plot_solvability_profiles(
                list_exp_list,
                plot_type="diff_cdf_solvability",
                plot_conf_ints=bool(param_value_list[0]),
                print_max_hw=bool(param_value_list[1]),
                solve_tol=param_value_list[2],
                ref_solver=param_value_list[3],
                n_bootstraps=int(param_value_list[4]),
                conf_level=param_value_list[5],
            )
            param_list = {
                "plot CIs": bool(param_value_list[0]),
                "print max hw": bool(param_value_list[1]),
                "solve tol": param_value_list[2],
                "ref solver": param_value_list[3],
                "n_bootstraps": int(param_value_list[4]),
                "conf_level": param_value_list[5],
            }
        elif self.plot_type_list[-1] == "Quantile Difference Plot":
            path_name = plot_solvability_profiles(
                list_exp_list,
                plot_type="diff_quantile_solvability",
                plot_conf_ints=bool(param_value_list[0]),
                print_max_hw=bool(param_value_list[1]),
                solve_tol=param_value_list[2],
                beta=param_value_list[3],
                ref_solver=param_value_list[4],
                n_bootstraps=int(param_value_list[5]),
                conf_level=param_value_list[6],
            )
            param_list = {
                "plot CIs": bool(param_value_list[0]),
                "print max hw": bool(param_value_list[1]),
                "solve tol": param_value_list[2],
                "ref solver": param_value_list[4],
                "beta": param_value_list[3],
                "n_bootstraps": int(param_value_list[5]),
                "conf_level": param_value_list[6],
            }
        elif self.plot_type_list[-1] == "Terminal Progress Plot":
            path_name = plot_terminal_progress(
                exp_list,
                plot_type=param_value_list[1],
                normalize=bool(param_value_list[2]),
                all_in_one=bool(param_value_list[0]),
            )
            param_list = {"normalize": bool(param_value_list[2])}
        elif self.plot_type_list[-1] == "Terminal Scatter Plot":
            path_name = plot_terminal_scatterplots(
                list_exp_list, all_in_one=bool(param_value_list[0])
            )
            param_list = {}
        else:
            print(f"{self.plot_type_list[-1]} is the plot_type_list")

        for i, new_plot in enumerate(path_name):
            place = self.num_plots + 1
            if len(path_name) == 1:
                prob_text = solver_lst
            else:
                prob_text = self.solver_menu.get(i)

            self.problem_button_added = tk.Label(
                master=self.tab_one,
                text=problem_lst,
                justify="center",
            )
            self.problem_button_added.grid(
                row=place, column=0, sticky="nsew", padx=10, pady=3
            )

            self.solver_button_added = tk.Label(
                master=self.tab_one,
                text=prob_text,
                justify="center",
            )
            self.solver_button_added.grid(
                row=place, column=1, sticky="nsew", padx=10, pady=3
            )

            self.plot_type_button_added = tk.Label(
                master=self.tab_one,
                text=plot_type,
                justify="center",
            )
            self.plot_type_button_added.grid(
                row=place, column=2, sticky="nsew", padx=10, pady=3
            )

            param_text = ""
            for key, item in param_list.items():
                param_text = param_text + key + ": " + str(item) + ", "
            param_text = param_text[: len(param_text) - 2]

            self.params_label_added = tk.Label(
                master=self.tab_one,
                text=param_text,
                justify="center",
            )
            self.params_label_added.grid(
                row=place, column=5, sticky="nsew", padx=10, pady=3
            )

            # TODO: remove plot does not work
            self.clear_plot = tk.Button(
                master=self.tab_one,
                text="Remove",
                justify="center",
                command=partial(self.clear_row, place - 1),
            )
            self.clear_plot.grid(
                row=place, column=3, sticky="nsew", padx=10, pady=3
            )

            self.view_plot = tk.Button(
                master=self.tab_one,
                text="View Plot",
                justify="center",
                command=partial(self.view_one_plot, new_plot),
            )
            self.view_plot.grid(
                row=place, column=4, sticky="nsew", padx=10, pady=3
            )

            self.plot_path = tk.Label(
                master=self.tab_one,
                text=new_plot,
                justify="center",
            )
            self.plot_path.grid(
                row=place, column=6, sticky="nsew", padx=10, pady=3
            )
            # self.view_plot.pack()
            self.change_on_hover(self.view_plot, "red", "yellow")
            self.all_path_names.append(new_plot)
            # print("all_path_names",self.all_path_names)
            self.num_plots += 1

    def change_on_hover(
        self, button: tk.Button, color_on_hover: str, color_on_leave: str
    ) -> None:
        """Change the color of the button when the mouse is hovered over it.

        Parameters
        ----------
        button : tk.Button
            The button that will change color when hovered over.
        color_on_hover : str
            The color the button will change to when hovered over.
        color_on_leave : str
            The color the button will change to when the mouse leaves the button.

        """
        # adjusting backgroung of the widget
        # background on entering widget
        button.bind(
            "<Enter>", func=lambda e: button.config(background=color_on_hover)
        )

        # background color on leving widget
        button.bind(
            "<Leave>", func=lambda e: button.config(background=color_on_leave)
        )

    def solver_select_function(self) -> None:
        # if user clicks plot type then a solver, this is update parameters
        if self.plot_var.get() != "Plot" and self.plot_var.get() != "":
            self.get_parameters_and_settings(0, self.plot_var.get())

    def get_parameters_and_settings(
        self,
        plot_choice: Literal[
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
        ],
    ) -> None:
        # ref solver needs to a drop down of solvers that is selected in the problem
        # numbers between 0 and 1
        # checkbox for normalize
        # move CI to parameters
        # checkbox with print_max_hw checkbox
        # remove CI from experiment box

        # beta=0.50, normalize=True
        if plot_choice == "All Progress Curves":
            param_list = {"normalize": True}
        elif plot_choice == "Mean Progress Curve":
            param_list = {
                "normalize": True,
                "n_bootstraps": 100,
                "conf_level": 0.95,
            }
        elif plot_choice == "Quantile Progress Curve":
            param_list = {
                "beta": 0.50,
                "normalize": True,
                "n_bootstraps": 100,
                "conf_level": 0.95,
            }
        elif plot_choice == "Solve time CDF":
            param_list = {
                "solve_tol": 0.1,
                "n_bootstraps": 100,
                "conf_level": 0.95,
            }
        elif plot_choice == "Area Scatter Plot":
            param_list = {"n_bootstraps": 100, "conf_level": 0.95}
        elif plot_choice == "CDF Solvability":
            param_list = {
                "solve_tol": 0.1,
                "n_bootstraps": 100,
                "conf_level": 0.95,
            }
        elif plot_choice == "Quantile Solvability":
            param_list = {
                "solve_tol": 0.1,
                "beta": 0.5,
                "n_bootstraps": 100,
                "conf_level": 0.95,
            }
        elif plot_choice == "CDF Difference Plot":
            param_list = {
                "solve_tol": 0.1,
                "ref_solver": None,
                "n_bootstraps": 100,
                "conf_level": 0.95,
            }
        elif plot_choice == "Quantile Difference Plot":
            param_list = {
                "solve_tol": 0.1,
                "beta": 0.5,
                "ref_solver": None,
                "n_bootstraps": 100,
                "conf_level": 0.95,
            }
        elif plot_choice == "Terminal Progress Plot":
            param_list = {"plot type": "violin", "normalize": True}
        elif plot_choice == "Terminal Scatter Plot":
            param_list = {}
        else:
            error_msg = f"Plot choice {plot_choice} is not a valid plot choice."
            raise ValueError(error_msg)
        self.param_list = param_list

        # self.params = [tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master), tk.StringVar(master=self.master)]

        self.CI_label_frame.destroy()
        self.CI_label_frame = ttk.LabelFrame(
            master=self.master, text="Plot Settings and Parameters"
        )
        self.CI_canvas = tk.Canvas(master=self.CI_label_frame, borderwidth=0)
        self.CI_frame = ttk.Frame(master=self.CI_canvas)

        self.CI_canvas.pack(side="left", fill="both", expand=True)
        self.CI_canvas.create_window(
            (0, 0), window=self.CI_frame, anchor="nw", tags="self.queue_frame"
        )
        self.CI_canvas.grid_rowconfigure(0)

        self.CI_label_frame.place(
            relx=0.4, rely=0.15, relheight=0.3, relwidth=0.25
        )

        self.settings_label_frame.destroy()
        self.settings_label_frame = ttk.LabelFrame(
            master=self.master, text="Error Estimation Settings and Parameters"
        )
        self.settings_canvas = tk.Canvas(
            master=self.settings_label_frame, borderwidth=0
        )
        self.settings_frame = ttk.Frame(master=self.settings_canvas)

        self.settings_canvas.pack(side="left", fill="both", expand=True)
        self.settings_canvas.create_window(
            (0, 0),
            window=self.settings_frame,
            anchor="nw",
            tags="self.queue_frame",
        )
        self.settings_canvas.grid_rowconfigure(0)

        self.settings_label_frame.place(
            relx=0.65, rely=0.15, relheight=0.3, relwidth=0.3
        )

        bp_list = ["violin", "box"]
        self.solvers_names = []
        for i in self.solver_menu.curselection():
            self.solvers_names.append(self.solver_menu.get(i))

        # Plot Settings
        i = 0
        if (
            plot_choice == "Mean Progress Curve"
            or plot_choice == "Quantile Progress Curve"
            or plot_choice == "Solve time CDF"
            or plot_choice == "Area Scatter Plot"
            or plot_choice == "CDF Solvability"
            or plot_choice == "Quantile Solvability"
            or plot_choice == "CDF Difference Plot"
            or plot_choice == "Quantile Difference Plot"
        ):
            # Confidence Intervals
            entry1 = tk.Checkbutton(
                self.settings_canvas,
                variable=self.params[i],
                onvalue=True,
                offvalue=False,
            )
            entry1.select()
            # entry1 = ttk.OptionMenu(self.settings_canvas, self.params[0], "True", *tf_list)
            label1 = tk.Label(
                master=self.settings_canvas,
                text="Show Confidence Intervals",
                wraplength="150",
            )
            label1.grid(row=0, column=0, padx=10, pady=3)
            entry1.grid(row=0, column=1, padx=10, pady=3)
            i += 1

        if (
            plot_choice == "All Progress Curves"
            or plot_choice == "Mean Progress Curve"
            or plot_choice == "Quantile Progress Curve"
            or plot_choice == "Terminal Progress Plot"
            or plot_choice == "Terminal Scatter Plot"
        ):
            # Plot Together Checkbox
            entry = tk.Checkbutton(
                self.CI_canvas,
                variable=self.params[i],
                onvalue=True,
                offvalue=False,
            )
            entry.select()
            # Creates the Check Mark that checks whether the plots will be plot together
            label = tk.Label(
                self.CI_canvas,
                text="Plot Together",
                wraplength="150",
            )
            label.grid(row=i, column=0, padx=10, pady=3)
            entry.grid(row=i, column=1, padx=10, pady=3)
            i += 1

        if (
            plot_choice == "Mean Progress Curve"
            or plot_choice == "Quantile Progress Curve"
            or plot_choice == "Solve time CDF"
            or plot_choice == "Area Scatter Plot"
            or plot_choice == "CDF Solvability"
            or plot_choice == "Quantile Solvability"
            or plot_choice == "CDF Difference Plot"
            or plot_choice == "Quantile Difference Plot"
        ):
            # Show Print Max Halfwidth
            entry2 = tk.Checkbutton(
                self.settings_canvas,
                variable=self.params[i],
                onvalue=True,
                offvalue=False,
            )
            entry2.select()
            label2 = tk.Label(
                master=self.settings_canvas,
                text="Show Max Halfwidth",
                wraplength="150",
            )
            label2.grid(row=1, column=0, padx=10, pady=3)
            entry2.grid(row=1, column=1, padx=10, pady=3)
            i += 1

        for param, param_val in param_list.items():
            if param == "normalize":
                entry = tk.Checkbutton(
                    master=self.CI_canvas,
                    variable=self.params[i],
                    onvalue=True,
                    offvalue=False,
                )
                entry.select()
                label = tk.Label(
                    master=self.CI_canvas,
                    text="Normalize by Relative Optimality Gap",
                    wraplength="150",
                )
                label.grid(row=i, column=0, padx=10, pady=3)
                entry.grid(row=i, column=1, padx=10, pady=3)
            elif param == "ref_solver":
                label = tk.Label(
                    master=self.CI_canvas,
                    text="Select Solver",
                )
                if len(self.solvers_names) != 0:
                    label = tk.Label(
                        master=self.CI_canvas,
                        text="Benchmark Solver",
                        wraplength="150",
                    )
                    entry = ttk.OptionMenu(
                        self.CI_canvas,
                        self.params[i],
                        self.solvers_names[0],
                        *self.solvers_names,
                    )
                    entry.grid(row=i, column=1, padx=10, pady=3)
                label.grid(row=i, column=0, padx=10, pady=3)
            elif param == "solve_tol":
                label = tk.Label(
                    master=self.CI_canvas,
                    text="Optimality Gap Threshold",
                    wraplength="150",
                )
                label.grid(row=i, column=0, padx=10, pady=3)
                entry = ttk.Entry(
                    master=self.CI_canvas,
                    textvariable=self.params[i],
                    justify=tk.LEFT,
                    width=15,
                )
                if param_val is not None:
                    entry.delete(0, "end")
                    entry.insert(index=tk.END, string=param_val)
                entry.grid(row=i, column=1, padx=10, pady=3)
            elif param == "beta":
                label = tk.Label(
                    master=self.CI_canvas,
                    text="Quantile Probability",
                    wraplength="150",
                )
                label.grid(row=i, column=0, padx=10, pady=3)
                entry = ttk.Entry(
                    master=self.CI_canvas,
                    textvariable=self.params[i],
                    justify=tk.LEFT,
                    width=15,
                )
                if param_val is not None:
                    entry.delete(0, "end")
                    entry.insert(index=tk.END, string=param_val)
                entry.grid(row=i, column=1, padx=10, pady=3)
            elif param == "plot type":
                label = tk.Label(
                    master=self.CI_canvas,
                    text="Type of Terminal Progress Plot",
                    wraplength="150",
                )
                entry = ttk.OptionMenu(
                    self.CI_canvas, self.params[i], "violin", *bp_list
                )
                label.grid(row=i, column=0, padx=10, pady=3)
                entry.grid(row=i, column=1, padx=10, pady=3)
            elif param == "n_bootstraps":
                label = tk.Label(
                    master=self.settings_canvas,
                    text="Number of Bootstraps",
                    wraplength="150",
                )
                label.grid(row=3, column=0, padx=10, pady=3)
                entry = ttk.Entry(
                    master=self.settings_canvas,
                    textvariable=self.params[i],
                    justify=tk.LEFT,
                    width=15,
                )
                if param_val is not None:
                    entry.delete(0, "end")
                    entry.insert(index=tk.END, string=param_val)
                entry.grid(row=3, column=1, padx=10, pady=3)
            elif param == "conf_level":
                label = tk.Label(
                    master=self.settings_canvas,
                    text="Confidence Level",
                    wraplength="150",
                )
                label.grid(row=2, column=0, padx=10, pady=3)
                entry = ttk.Entry(
                    master=self.settings_canvas,
                    textvariable=self.params[i],
                    justify=tk.LEFT,
                    width=15,
                )
                if param_val is not None:
                    entry.delete(0, "end")
                    entry.insert(index=tk.END, string=param_val)
                entry.grid(row=2, column=1, padx=10, pady=3)
            else:
                label = tk.Label(
                    master=self.CI_canvas,
                    text=param,
                )
                label.grid(row=i, column=0, padx=10, pady=3)
                entry = ttk.Entry(
                    master=self.CI_canvas,
                    textvariable=self.params[i],
                    justify=tk.LEFT,
                    width=15,
                )
                if param_val is not None:
                    entry.delete(0, "end")
                    entry.insert(index=tk.END, string=param_val)
                entry.grid(row=i, column=1, padx=10, pady=3)
            i += 1

    def clear_row(self, place: int) -> None:
        self.plot_CI_list.pop(place)
        self.plot_exp_list.pop(place)
        print("Clear")

    def plot_button(self) -> None:
        self.postrep_window = tk.Toplevel()
        position = center_window(self.master, 0.8)
        self.postrep_window.geometry(position)
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

        for _, path_name in enumerate(self.all_path_names):
            width = 350
            height = 350
            img = Image.open(path_name)
            img = img.resize((width, height), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            # img = tk.PhotoImage(file=path_name)

            # img = img.resize(200,200)
            self.panel = tk.Label(self.postrep_window, image=img)
            self.panel.photo = img
            self.panel.grid(row=ro, column=c)
            c += 1
            if c == 3:
                c = 0
                ro += 1

            # panel.place(x=10,y=0)

    def view_one_plot(self, path_name: os.PathLike | str) -> None:
        self.postrep_window = tk.Toplevel()
        position = center_window(self.master, 0.8)
        self.postrep_window.geometry(position)
        self.postrep_window.title("View One Plot")

        ro = 0
        c = 0

        width = 400
        height = 400
        print("This is path", path_name)
        img = Image.open(path_name)

        img = img.resize((width, height), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        # img = tk.PhotoImage(file=path_name)

        # img = img.resize(200,200)
        self.panel = tk.Label(self.postrep_window, image=img)
        self.panel.photo = img
        self.panel.grid(row=ro, column=c)
        c += 1
        if c == 4:
            c = 0
            ro += 1


def problem_solver_unabbreviated_to_object(
    problem_or_solver_name: str, unabbreviated_dictionary: dict
) -> tuple[Problem | Solver, str]:
    """Convert the unabbreviated name of a problem or solver to the object of the problem or solver.

    Arguments:
    ---------
    problem_or_solver_name: str
        The unabbreviated name of the problem or solver.
    unabbreviated_dictionary: dict
        The dictionary that maps the unabbreviated name of the problem or solver to the object of the problem or solver.

    Returns:
    -------
    Problem | Solver
        The object of the problem or solver.
    str
        The name of the problem or solver.

    Raises:
    ------
        ValueError: If the problem_or_solver_name is not found in the unabbreviated_dictionary.

    """
    if problem_or_solver_name in unabbreviated_dictionary.keys():
        problem_or_solver_object = unabbreviated_dictionary[
            problem_or_solver_name
        ]
        return problem_or_solver_object, problem_or_solver_object().name
    else:
        error_msg = (
            f"{problem_or_solver_name} not found in {unabbreviated_dictionary}"
        )
        raise ValueError(error_msg)


def problem_solver_abbreviated_name_to_unabbreviated(
    problem_or_solver_name: str,
    abbreviated_dictionary: dict,
    unabbreviated_dictionary: dict,
) -> str:
    """Convert the abbreviated name of a problem or solver to the unabbreviated name of the problem or solver.

    Arguments:
    ---------
    problem_or_solver_name: str
        The abbreviated name of the problem or solver.
    abbreviated_dictionary: dict
        The dictionary that maps the abbreviated name of the problem or solver to the object of the problem or solver.
    unabbreviated_dictionary: dict
        The dictionary that maps the unabbreviated name of the problem or solver to the object of the problem or solver.

    Returns:
    -------
    str
        The unabbreviated name of the problem or solver.

    Raises:
    ------
        ValueError: If the problem_or_solver_name is not found in the abbreviated_dictionary.

    """
    if problem_or_solver_name in abbreviated_dictionary.keys():
        problem_or_solver_object = abbreviated_dictionary[
            problem_or_solver_name
        ]
        for key, value in unabbreviated_dictionary.items():
            if problem_or_solver_object == value:
                return key
    else:
        error_msg = (
            f"{problem_or_solver_name} not found in {abbreviated_dictionary}"
        )
        raise ValueError(error_msg)


def main() -> None:
    """Run the GUI."""
    root = tk.Tk()
    root.title("SimOpt Library Graphical User Interface")
    root.pack_propagate(False)

    # app = Experiment_Window(root)
    MainMenuWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
