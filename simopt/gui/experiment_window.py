import ast
import pickle
import time
import tkinter as tk
from functools import partial
from tkinter import Scrollbar, filedialog, simpledialog, ttk
from tkinter.font import nametofont

from simopt.base import Problem, Solver
from simopt.directory import (
    model_directory,
    model_problem_unabbreviated_directory,
    problem_directory,
    problem_unabbreviated_directory,
    solver_directory,
    solver_unabbreviated_directory,
)
from simopt.experiment_base import (
    EXPERIMENT_DIR,
    ProblemSolver,
    ProblemsSolvers,
    find_missing_experiments,
    make_full_metaexperiment,
    post_normalize,
)
from simopt.gui.data_farming_window import DataFarmingWindow
from simopt.gui.plot_window import PlotWindow
from simopt.gui.toplevel_custom import Toplevel


class ExperimentWindow(Toplevel):
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
        super().__init__(
            root, title="SimOpt GUI - Experiment", exit_on_close=True
        )
        self.center_window(0.8)  # 80% scaling

        self.frame = tk.Frame(self)
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
            master=self,  # window label is used in
            text="Welcome to SimOpt Library Graphic User Interface\n Please Load or Add Your Problem-Solver Pair(s): ",
            font=nametofont("TkHeadingFont"),
            justify="center",
        )

        self.problem_label = tk.Label(
            master=self,  # window label is used in
            text="Select Problem:",
        )

        self.or_label = tk.Label(
            master=self,  # window label is used in
            text=" OR ",
        )
        self.or_label2 = tk.Label(
            master=self,  # window label is used in
            text=" OR Select Problem and Solver from Below:",
        )
        self.or_label22 = tk.Label(
            master=self,  # window label is used in
            text="Select from Below:",
        )

        # from experiments.inputs.all_factors.py:
        self.problem_list = problem_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.problem_var = tk.StringVar(master=self)
        # sets the default OptionMenu value

        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.problem_menu = ttk.OptionMenu(
            self,
            self.problem_var,
            "Problem",
            *self.problem_list,
            command=self.show_problem_factors,
        )

        self.solver_label = tk.Label(
            master=self,  # window label is used in
            text="Select Solver:",
        )

        # from experiments.inputs.all_factors.py:
        self.solver_list = solver_unabbreviated_directory
        # stays the same, has to change into a special type of variable via tkinter function
        self.solver_var = tk.StringVar(master=self)
        # sets the default OptionMenu value

        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.solver_menu = ttk.OptionMenu(
            self,
            self.solver_var,
            "Solver",
            *self.solver_list,
            command=self.show_solver_factors,
        )

        # self.macro_label = tk.Label(master=self,
        #                text = "Number of Macroreplications:",
        #              font = f"{TEXT_FAMILY} 13")

        self.macro_definition = tk.Label(
            master=self,
            text="",
        )

        self.macro_definition_label = tk.Label(
            master=self,
            text="Number of Macroreplications:",
            width=25,
        )

        self.macro_var = tk.StringVar(self)
        self.macro_entry = ttk.Entry(
            master=self,
            textvariable=self.macro_var,
            justify=tk.LEFT,
            width=10,
        )
        self.macro_entry.insert(index=tk.END, string="10")

        self.add_button = ttk.Button(
            master=self,
            text="Add Problem-Solver Pair",
            width=15,
            command=self.add_experiment,
        )

        self.clear_queue_button = ttk.Button(
            master=self,
            text="Clear All Problem-Solver Pairs",
            width=15,
            command=self.clear_queue,
        )  # (self.experiment_added, self.problem_added, self.solver_added, self.macros_added, self.run_button_added))

        self.crossdesign_button = ttk.Button(
            master=self,
            text="Create Problem-Solver Group",
            width=50,
            command=self.crossdesign_function,
        )

        self.pickle_file_load_button = ttk.Button(
            master=self,
            text="Load Problem-Solver Pair",
            width=50,
            command=self.load_pickle_file_function,
        )

        self.attribute_description_label = tk.Label(
            master=self,
            text="Attribute Description Label for Problems:\n Objective (Single [S] or Multiple [M])\n Constraint (Unconstrained [U], Box[B], Determinisitic [D], Stochastic [S])\n Variable (Discrete [D], Continuous [C], Mixed [M])\n Gradient Available (True [G] or False [N])",
            font=nametofont("TkTooltipFont"),
        )
        self.attribute_description_label.place(x=450, rely=0.478)

        self.post_normal_all_button = ttk.Button(
            master=self,
            text="Post-Normalize Selected",
            width=20,
            state="normal",
            command=self.post_normal_all_function,
        )

        self.make_meta_experiment = ttk.Button(
            master=self,
            text="Create Problem-Solver Group from Selected",
            width=35,
            state="normal",
            command=self.make_meta_experiment_func,
        )

        self.pickle_file_pathname_label = tk.Label(
            master=self,
            text="File Selected:",
        )

        self.pickle_file_pathname_show = tk.Label(
            master=self,
            text="No File Selected!",
            foreground="red",
            wraplength="500",
        )

        self.style.configure("Bold.TLabel", font=nametofont("TkHeadingFont"))
        self.label_Workspace = ttk.Label(
            master=self, text="Workspace", style="Bold.TLabel"
        )
        self.queue_label_frame = ttk.LabelFrame(
            master=self, labelwidget=self.label_Workspace
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
            master=self, text="Problem Factors"
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
            master=self, text="Model Factors"
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
            self.problem_var = tk.StringVar(master=self)
            # sets the default OptionMenu value

            # creates drop down menu, for tkinter, it is called "OptionMenu"
            self.problem_menu = ttk.OptionMenu(
                self,
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
            parent=self,
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
        #     tk.messagebox.showwarning(master=self, title=" Warning", message=message)

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
                    master=self, title=" Warning", message=message
                )
        # else:
        #     message = "You are attempting to load a file, but haven't selected one yet.\nPlease select a file first."
        #     tk.messagebox.showwarning(master=self, title=" Warning", message=message)

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

        self.postrep_window = Toplevel(self)
        self.center_window(0.8)
        self.set_style()

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
        self.crossdesign_window = Toplevel(self)
        self.center_window(0.8)
        self.set_style()
        self.crossdesign_window.title("Cross-Design Problem-Solver Group")
        self.cross_app = CrossDesignWindow(self.crossdesign_window, self)

    # My code starts here
    # Open data farming window
    def datafarming_function(self) -> None:
        self.datafarming_window = Toplevel(self)
        self.center_window(0.8)
        self.set_style()
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

        self.postrep_window = Toplevel(self)
        self.center_window(0.8)
        self.set_style()
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
        self.postrep_window = Toplevel(self)
        self.center_window(0.8)
        self.set_style()
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
        self.postrep_window = Toplevel(self)
        self.center_window(0.8)
        self.set_style()
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
            master=self,
            text="Group Problem(s):*",
        )
        self.problem_var2 = tk.StringVar(master=self)

        self.problem_menu2 = ttk.OptionMenu(
            self,
            self.problem_var2,
            "Problem",
            *self.list_meta_experiment_problems,
            command=partial(self.show_problem_factors2, row_index),
        )

        self.problem_label2.place(relx=0.35, rely=0.1)
        self.problem_menu2.place(relx=0.45, rely=0.1)
        self.solver_label2 = tk.Label(
            master=self,
            text="Group Solver(s):*",
        )
        self.solver_var2 = tk.StringVar(master=self)
        self.solver_menu2 = ttk.OptionMenu(
            self,
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
            master=self,  # window label is used in
            text="Select Problem:",
        )
        self.problem_var = tk.StringVar(master=self)
        self.problem_menu = ttk.OptionMenu(
            self,
            self.problem_var,
            "Problem",
            *self.problem_list,
            command=self.show_problem_factors,
        )

        self.problem_label.place(relx=0.3, rely=0.1)
        self.problem_menu.place(relx=0.4, rely=0.1)
        self.solver_label = tk.Label(
            master=self,  # window label is used in
            text="Select Solver(s):*",
        )
        self.solver_var = tk.StringVar(master=self)
        self.solver_menu = ttk.OptionMenu(
            self,
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
            master=self, text="Solver Factors"
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
            master=self, text="Problem Factors"
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
            master=self, text="Model Factors"
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
            master=self, text="Solver Factors"
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


class PostNormalWindow(Toplevel):
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
        super().__init__(root, title="SimOpt GUI - Post-Normalization")
        self.center_window(0.8)  # 80% scaling

        self.post_norm_exp_list = experiment_list
        self.meta = meta
        self.main_window = main_window
        self.optimal_var = tk.StringVar(master=self)
        self.initial_var = tk.StringVar(master=self)
        self.check_var = tk.IntVar(master=self)
        self.init_var = tk.StringVar(self)
        self.proxy_var = tk.StringVar(self)
        self.proxy_sol = tk.StringVar(self)

        self.all_solvers = []
        for solvers in self.post_norm_exp_list:
            if solvers.solver.name not in self.all_solvers:
                self.all_solvers.append(solvers.solver.name)

        # ("my exp post pro ", experiment_list)
        self.selected = experiment_list

        self.frame = tk.Frame(self)
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
            master=self,
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
            master=self,
            text="The Initial Solution, x\u2080, is " + initsol + ".",
            wraplength="400",
        )

        self.n_opt_label = tk.Label(
            master=self,
            text="The Optimal Solution, x\u002a, is "
            + opt
            + " for this "
            + minmax
            + "imization Problem. \nIf the Proxy Optimal Value or the Proxy Optimal Solution is unspecified, SimOpt uses the best Solution found in the selected Problem-Solver Pair experiments as the Proxy Optimal Solution.",
            wraplength="600",
            justify="left",
        )

        self.n_optimal_label = tk.Label(
            master=self,
            text="Optimal Solution (optional):",
            wraplength="250",
        )
        self.n_proxy_val_label = tk.Label(
            master=self,
            text="Insert Proxy Optimal Value, f(x\u002a):",
            wraplength="250",
        )
        self.n_proxy_sol_label = tk.Label(
            master=self,
            text="Insert Proxy Optimal Solution, x\u002a:",
            wraplength="250",
        )

        # t = ["x","f(x)"]
        self.n_proxy_sol_entry = ttk.Entry(
            master=self,
            textvariable=self.proxy_sol,
            justify=tk.LEFT,
            width=8,
        )
        self.n_proxy_val_entry = ttk.Entry(
            master=self,
            textvariable=self.proxy_var,
            justify=tk.LEFT,
            width=8,
        )
        self.n_initial_entry = ttk.Entry(
            master=self,
            textvariable=self.init_var,
            justify=tk.LEFT,
            width=10,
        )

        self.n_crn_label = tk.Label(
            master=self,
            text="CRN for x\u2080 and Optimal x\u002a?",
            wraplength="310",
        )
        self.n_crn_checkbox = tk.Checkbutton(
            self, text="", variable=self.check_var
        )

        self.n_postreps_init_opt_label = tk.Label(
            master=self,
            text="Number of Post-Normalizations at x\u2080 and x\u002a:",
            wraplength="310",
        )

        self.n_postreps_init_opt_var = tk.StringVar(self)
        self.n_postreps_init_opt_entry = ttk.Entry(
            master=self,
            textvariable=self.n_postreps_init_opt_var,
            justify=tk.LEFT,
            width=15,
        )
        self.n_postreps_init_opt_entry.insert(index=tk.END, string="200")

        self.post_processing_run_label = tk.Label(
            master=self,  # window label is used for
            text="Click to Post-Normalize the Problem-Solver Pairs",
            wraplength="300",
        )

        self.post_processing_run_button = ttk.Button(
            master=self,  # window button is used in
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
            # self.destroy()
            self.post_processed_bool = True

            self.postrep_window = Toplevel(self)
            self.postrep_window.center_window(0.8)
            self.postrep_window.set_style()
            self.postrep_window.title("Plotting Page")
            self.destroy()
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


class PostProcessingWindow(Toplevel):
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
        super().__init__(root, title="SimOpt GUI - Post-Processing")
        self.center_window(0.8)  # 80% scaling

        self.meta = meta
        self.main_window = main_window
        self.my_experiment = myexperiment
        # ("my exp post pro ", experiment_list)
        self.selected = experiment_list

        self.frame = tk.Frame(self)

        self.title = tk.Label(
            master=self,
            text="Welcome to the Post-Processing Page",
            font=nametofont("TkHeadingFont"),
            justify="center",
        )
        if self.meta:
            self.title = tk.Label(
                master=self,
                text="Welcome to the Post-Processing \nand Post-Normalization Page",
                font=nametofont("TkHeadingFont"),
                justify="center",
            )

        self.n_postreps_label = tk.Label(
            master=self,
            text="Number of Postreplications at each Recommended Solution:",
            wraplength="250",
        )

        self.n_postreps_var = tk.StringVar(self)
        self.n_postreps_entry = ttk.Entry(
            master=self,
            textvariable=self.n_postreps_var,
            justify=tk.LEFT,
            width=15,
        )
        self.n_postreps_entry.insert(index=tk.END, string="100")

        self.crn_across_budget_label = tk.Label(
            master=self,
            text="Use CRN for Postreplications at Solutions Recommended at Different Times?",
            wraplength="250",
        )

        self.crn_across_budget_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_budget_var = tk.StringVar(self)
        # sets the default OptionMenu selection
        # self.crn_across_budget_var.set("True")
        # creates drop down menu, for tkinter, it is called "OptionMenu"
        self.crn_across_budget_menu = ttk.OptionMenu(
            self,
            self.crn_across_budget_var,
            "True",
            *self.crn_across_budget_list,
        )

        self.crn_across_macroreps_label = tk.Label(
            master=self,
            text="Use CRN for Postreplications at Solutions Recommended on Different Macroreplications?",
            wraplength="325",
        )

        self.crn_across_macroreps_list = ["True", "False"]
        # stays the same, has to change into a special type of variable via tkinter function
        self.crn_across_macroreps_var = tk.StringVar(self)

        self.crn_across_macroreps_menu = ttk.OptionMenu(
            self,
            self.crn_across_macroreps_var,
            "False",
            *self.crn_across_macroreps_list,
        )

        self.crn_norm_budget_label = tk.Label(
            master=self,
            text="Use CRN for Postreplications at x\u2080 and x\u002a?",
            wraplength="325",
        )
        self.crn_norm_across_macroreps_var = tk.StringVar(self)
        self.crn_norm_across_macroreps_menu = ttk.OptionMenu(
            self,
            self.crn_norm_across_macroreps_var,
            "True",
            *self.crn_across_macroreps_list,
        )

        self.n_norm_label = tk.Label(
            master=self,
            text="Post-Normalization Parameters",
            font=nametofont("TkHeadingFont"),
            wraplength="300",
        )

        self.n_proc_label = tk.Label(
            master=self,
            text="Post-Processing Parameters",
            font=nametofont("TkHeadingFont"),
            wraplength="300",
        )

        self.n_norm_ostreps_label = tk.Label(
            master=self,
            text="Number of Postreplications at x\u2080 and x\u002a:",
            wraplength="300",
        )

        self.n_norm_postreps_var = tk.StringVar(self)
        self.n_norm_postreps_entry = ttk.Entry(
            master=self,
            textvariable=self.n_norm_postreps_var,
            justify=tk.LEFT,
            width=15,
        )
        self.n_norm_postreps_entry.insert(index=tk.END, string="200")

        self.post_processing_run_label = tk.Label(
            master=self,  # window label is used for
            text="Complete Post-Processing of the Problem-Solver Pairs:",
            wraplength="250",
        )

        if self.meta:
            self.post_processing_run_label = tk.Label(
                master=self,  # window label is used for
                text="Complete Post-Processing and Post-Normalization of the Problem-Solver Pair(s)",
                wraplength="300",
            )

        self.post_processing_run_button = ttk.Button(
            master=self,  # window button is used in
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
            self.destroy()
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


class CrossDesignWindow(Toplevel):
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
            super().__init__(root, title="SimOpt GUI - Cross-Design")
            self.center_window(0.8)  # 80% scaling

            self.crossdesign_title_label = tk.Label(
                master=self,
                text="Create Cross-Design Problem-Solver Group",
                font=nametofont("TkHeadingFont"),
            )
            self.crossdesign_title_label.place(x=10, y=25)

            self.crossdesign_problem_label = tk.Label(
                master=self,
                text="Select Problems:",
            )
            self.crossdesign_problem_label.place(x=190, y=55)

            self.crossdesign_solver_label = tk.Label(
                master=self,
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
                    self, value=False
                )
                self.crossdesign_solver_checkbox = tk.Checkbutton(
                    master=self,
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
                    self, value=False
                )
                self.crossdesign_problem_checkbox = tk.Checkbutton(
                    master=self,
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
                    master=self,
                    text="Number of Macroreplications:",
                )
                self.crossdesign_macro_label.place(
                    x=15, y=80 + (25 * problem_cnt)
                )

                self.crossdesign_macro_var = tk.StringVar(self)
                self.crossdesign_macro_entry = ttk.Entry(
                    master=self,
                    textvariable=self.crossdesign_macro_var,
                    justify=tk.LEFT,
                    width=15,
                )
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")
                self.crossdesign_macro_entry.place(
                    x=15, y=105 + (25 * solver_cnt)
                )

                self.crossdesign_button = ttk.Button(
                    master=self,
                    text="Add Cross-Design Problem-Solver Group",
                    width=65,
                    command=self.confirm_cross_design_function,
                )
                self.crossdesign_button.place(x=15, y=135 + (25 * solver_cnt))

            if problem_cnt > solver_cnt:
                problem_cnt += 1

                self.crossdesign_macro_label = tk.Label(
                    master=self,
                    text="Number of Macroreplications:",
                )
                self.crossdesign_macro_label.place(
                    x=15, y=80 + (25 * problem_cnt)
                )

                self.crossdesign_macro_var = tk.StringVar(self)
                self.crossdesign_macro_entry = ttk.Entry(
                    master=self,
                    textvariable=self.crossdesign_macro_var,
                    justify=tk.LEFT,
                    width=15,
                )
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")

                self.crossdesign_macro_entry.place(
                    x=15, y=105 + (25 * problem_cnt)
                )

                self.crossdesign_button = ttk.Button(
                    master=self,
                    text="Add Cross-Design Problem-Solver Group",
                    width=45,
                    command=self.confirm_cross_design_function,
                )
                self.crossdesign_button.place(x=15, y=135 + (25 * problem_cnt))

            if problem_cnt == solver_cnt:
                problem_cnt += 1

                self.crossdesign_macro_label = tk.Label(
                    master=self,
                    text="Number of Macroreplications:",
                )
                self.crossdesign_macro_label.place(
                    x=15, y=80 + (25 * problem_cnt)
                )

                self.crossdesign_macro_var = tk.StringVar(self)
                self.crossdesign_macro_entry = ttk.Entry(
                    master=self,
                    textvariable=self.crossdesign_macro_var,
                    justify=tk.LEFT,
                    width=15,
                )
                self.crossdesign_macro_entry.insert(index=tk.END, string="10")
                self.crossdesign_macro_entry.place(
                    x=15, y=105 + (25 * problem_cnt)
                )

                self.crossdesign_button = ttk.Button(
                    master=self,
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
                master=self,
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
                master=self,
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
        self.destroy()
        ExperimentWindow.add_meta_exp_to_frame(
            self.main_window, self.crossdesign_macro_var
        )

        return self.crossdesign_MetaExperiment

        # (self.crossdesign_MetaExperiment)

    def get_crossdesign_meta_experiment(self) -> ProblemsSolvers:
        return self.crossdesign_MetaExperiment
