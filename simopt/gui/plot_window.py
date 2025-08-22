from __future__ import annotations

import logging
import os
import tkinter as tk
from functools import partial
from tkinter import Listbox, Scrollbar, ttk
from tkinter.constants import MULTIPLE
from tkinter.font import nametofont
from typing import Literal

from PIL import Image, ImageTk

import simopt.directory as directory
from simopt.experiment_base import (
    PlotType,
    ProblemSolver,
    plot_area_scatterplots,
    plot_progress_curves,
    plot_solvability_cdfs,
    plot_solvability_profiles,
    plot_terminal_progress,
    plot_terminal_scatterplots,
)
from simopt.gui.toplevel_custom import Toplevel

problem_unabbreviated_directory = directory.problem_unabbreviated_directory
solver_unabbreviated_directory = directory.solver_unabbreviated_directory


class PlotWindow(Toplevel):
    """Plot Window Page of the GUI."""

    def __init__(
        self,
        root: tk.Tk,
        main_window: tk.Tk,
        experiment_list: list,
        meta_list: list[ProblemSolver] | None = None,
    ) -> None:
        """Initialize the PlotWindow class.

        Args:
            root (tk.Tk): The root window of the application.
            main_window (tk.Tk): The main window of the application.
            experiment_list (list): List of experiment objects.
            meta_list (list[ProblemSolver] | None, optional): List of `ProblemSolver`
                objects used for meta-plotting. Defaults to None.
        """
        super().__init__(root, title="SimOpt GUI - Plotting Page")
        self.center_window(0.8)  # 80% scaling

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
        self.plot_var = tk.StringVar(master=self)

        self.params = [
            tk.StringVar(master=self),
            tk.StringVar(master=self),
            tk.StringVar(master=self),
            tk.StringVar(master=self),
            tk.StringVar(master=self),
            tk.StringVar(master=self),
            tk.StringVar(master=self),
        ]

        self.problem_menu = Listbox(
            self,
            selectmode=MULTIPLE,
            exportselection=False,
            width=10,
            height=6,
        )
        self.solver_menu = Listbox(
            self,
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
            master=self,  # window label is used in
            text=(
                "Welcome to the Plotting Page of SimOpt\n"
                "Select Problems and Solvers to Plot"
            ),
            justify="center",
        )

        self.problem_label = tk.Label(
            master=self,  # window label is used in
            text="Select Problem(s):*",
        )
        self.plot_label = tk.Label(
            master=self,  # window label is used in
            text="Select Plot Type:*",
        )

        # from experiments.inputs.all_factors.py:
        self.problem_list = problem_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter
        # function
        self.problem_var = tk.StringVar(master=self)

        # self.problem_menu = tk.Listbox(
        #     self,
        #     self.problem_var,
        #     "Problem",
        #     *self.all_problems,
        #     command=self.experiment_list[0].problem.name,
        # )
        self.plot_menu = ttk.OptionMenu(
            self,
            self.plot_var,
            "Plot",
            *self.plot_type_names,
            command=partial(self.get_parameters_and_settings, self.plot_var),
        )
        self.solver_label = tk.Label(
            master=self,  # window label is used in
            text="Select Solver(s):*",
        )

        # from experiments.inputs.all_factors.py:
        self.solver_list = solver_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter
        # function
        self.solver_var = tk.StringVar(master=self)

        self.add_button = ttk.Button(
            master=self, text="Add", width=15, command=self.add_plot
        )

        self.post_normal_all_button = ttk.Button(
            master=self,
            text="See All Plots",
            width=20,
            state="normal",
            command=self.plot_button,
        )

        self.style = ttk.Style()
        self.style.configure("Bold.TLabel", font=nametofont("TkHeadingFont"))
        workspace_lbl = ttk.Label(
            master=self, text="Plotting Workspace", style="Bold.TLabel"
        )

        self.queue_label_frame = ttk.LabelFrame(master=self, labelwidget=workspace_lbl)

        self.queue_canvas = tk.Canvas(master=self.queue_label_frame, borderwidth=0)

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
            label.grid(row=0, column=self.heading_list.index(heading), padx=10, pady=3)

        self.instruction_label.place(relx=0.3, y=0)

        self.problem_label.place(x=10, rely=0.08)
        self.problem_menu.place(x=10, rely=0.11, relwidth=0.3)

        self.solver_label.place(x=10, rely=0.25)
        self.solver_menu.place(x=10, rely=0.28, relwidth=0.3)

        self.plot_label.place(relx=0.4, rely=0.08)
        self.plot_menu.place(relx=0.55, rely=0.08)

        self.add_button.place(relx=0.45, rely=0.45)

        separator = ttk.Separator(master=self, orient="horizontal")
        separator.place(relx=0.35, rely=0.08, relheight=0.4)

        self.post_normal_all_button.place(relx=0.01, rely=0.92)

        # self.queue_label_frame.place(x=10, rely=.7, relheight=.3, relwidth=1)
        self.queue_label_frame.place(x=10, rely=0.56, relheight=0.35, relwidth=0.99)

        self.param_label = []
        self.param_entry = []
        self.factor_label_frame_problem = None

        self.CI_label_frame = ttk.LabelFrame(
            master=self, text="Plot Settings and Parameters"
        )
        self.CI_canvas = tk.Canvas(master=self.CI_label_frame, borderwidth=0)
        self.CI_frame = ttk.Frame(master=self.CI_canvas)

        self.CI_canvas.pack(side="left", fill="both", expand=True)
        self.CI_canvas.create_window(
            (0, 0), window=self.CI_frame, anchor="nw", tags="self.queue_frame"
        )

        self.CI_label_frame.place(relx=0.4, rely=0.15, relheight=0.2, relwidth=0.3)

        self.settings_label_frame = ttk.LabelFrame(
            master=self, text="Error Estimation Setting and Parameters"
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

    def add_plot(self) -> None:
        self.plot_exp_list = []

        solver_lst = ""
        # Appends experiment that is part of the experiment list if it matches what was
        # chosen in the solver menu
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
                master=self,
                text=txt,
                justify="center",
            )
            self.bad_label.place(relx=0.45, rely=0.5)
            return
        if self.bad_label is not None:
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
                plot_type=PlotType.ALL,
                normalize=bool(param_value_list[1]),
                all_in_one=bool(param_value_list[0]),
            )
            param_list = {"normalize": bool(param_value_list[1])}
        if self.plot_type_list[-1] == "Mean Progress Curve":
            path_name = plot_progress_curves(
                exp_list,
                plot_type=PlotType.MEAN,
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
                plot_type=PlotType.QUANTILE,
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
                plot_type=PlotType.CDF_SOLVABILITY,
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
                plot_type=PlotType.QUANTILE_SOLVABILITY,
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
                plot_type=PlotType.DIFFERENCE_OF_CDF_SOLVABILITY,
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
                plot_type=PlotType.DIFFERENCE_OF_QUANTILE_SOLVABILITY,
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
            error_msg = f"Plot type {self.plot_type_list[-1]} is not a valid plot type."
            logging.error(error_msg)
            raise ValueError(error_msg)

        for i, new_plot in enumerate(path_name):
            place = self.num_plots + 1
            prob_text = solver_lst if len(path_name) == 1 else self.solver_menu.get(i)

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
            self.clear_plot.grid(row=place, column=3, sticky="nsew", padx=10, pady=3)

            self.view_plot = tk.Button(
                master=self.tab_one,
                text="View Plot",
                justify="center",
                command=partial(self.view_one_plot, new_plot),
            )
            self.view_plot.grid(row=place, column=4, sticky="nsew", padx=10, pady=3)

            self.plot_path = tk.Label(
                master=self.tab_one,
                text=new_plot,
                justify="center",
            )
            self.plot_path.grid(row=place, column=6, sticky="nsew", padx=10, pady=3)
            # self.view_plot.pack()
            self.change_on_hover(self.view_plot, "red", "yellow")
            self.all_path_names.append(new_plot)
            # logging.debug("all_path_names",self.all_path_names)
            self.num_plots += 1

    def change_on_hover(
        self, button: tk.Button, color_on_hover: str, color_on_leave: str
    ) -> None:
        """Change the color of a button when hovered over.

        Args:
            button (tk.Button): The button to apply the hover effect to.
            color_on_hover (str): The color the button changes to when hovered.
            color_on_leave (str): The color the button reverts to when the mouse leaves.
        """
        # adjusting backgroung of the widget
        # background on entering widget
        button.bind("<Enter>", func=lambda _: button.config(background=color_on_hover))

        # background color on leving widget
        button.bind("<Leave>", func=lambda _: button.config(background=color_on_leave))

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

        # self.params = [tk.StringVar(master=self), tk.StringVar(master=self), tk.StringVar(master=self), tk.StringVar(master=self), tk.StringVar(master=self)]

        self.CI_label_frame.destroy()
        self.CI_label_frame = ttk.LabelFrame(
            master=self, text="Plot Settings and Parameters"
        )
        self.CI_canvas = tk.Canvas(master=self.CI_label_frame, borderwidth=0)
        self.CI_frame = ttk.Frame(master=self.CI_canvas)

        self.CI_canvas.pack(side="left", fill="both", expand=True)
        self.CI_canvas.create_window(
            (0, 0), window=self.CI_frame, anchor="nw", tags="self.queue_frame"
        )
        self.CI_canvas.grid_rowconfigure(0)

        self.CI_label_frame.place(relx=0.4, rely=0.15, relheight=0.3, relwidth=0.25)

        self.settings_label_frame.destroy()
        self.settings_label_frame = ttk.LabelFrame(
            master=self, text="Error Estimation Settings and Parameters"
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
        logging.debug("Clear")

    def plot_button(self) -> None:
        self.postrep_window = Toplevel(self)
        self.postrep_window.center_window(0.8)
        self.postrep_window.set_style()
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
        self.postrep_window = Toplevel(self)
        self.postrep_window.center_window(0.8)
        self.postrep_window.set_style()
        self.postrep_window.title("View One Plot")

        ro = 0
        c = 0

        width = 400
        height = 400
        logging.debug("This is path", path_name)
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
