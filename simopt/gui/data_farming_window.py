"""GUI for data farming experiments."""

import ast
import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.font import nametofont

import pandas as pd

import simopt.directory as directory
from simopt.data_farming_base import DATA_FARMING_DIR, DataFarmingExperiment
from simopt.experiment_base import (
    create_design,
)
from simopt.gui.toplevel_custom import Toplevel

# Workaround for AutoAPI
model_directory = directory.model_directory
model_unabbreviated_directory = directory.model_unabbreviated_directory


class DataFarmingWindow(Toplevel):
    """Class to create the data farming window."""

    def __init__(self, root: tk.Tk, forced_creation: bool = False) -> None:
        """Initialize the data farming window.

        Args:
            root (tk.Tk): The root window of the application.
            forced_creation (bool, optional): Whether to create the window even if it
                already exists. Defaults to False.
        """
        if not forced_creation:
            super().__init__(
                root,
                title="SimOpt GUI - Model Data Farming",
                exit_on_close=True,
            )
            self.center_window(0.8)  # 80% scaling

            self.grid_rowconfigure(0, weight=0)
            self.grid_rowconfigure(1, weight=0)
            self.grid_rowconfigure(2, weight=0)
            self.grid_rowconfigure(3, weight=0)
            self.grid_rowconfigure(4, weight=0)
            self.grid_rowconfigure(5, weight=0)
            self.grid_rowconfigure(6, weight=0)
            self.grid_rowconfigure(7, weight=0)
            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(1, weight=1)
            self.grid_columnconfigure(2, weight=1)
            self.grid_columnconfigure(3, weight=1)
            self.grid_columnconfigure(4, weight=1)

            # Intitialize frames so prevous entries can be deleted
            self.design_frame = tk.Frame(master=self)
            self.design_frame.grid(row=5, column=0)

            self.create_design_frame = tk.Frame(master=self)
            self.run_frame = tk.Frame(master=self)
            self.factor_canvas = tk.Canvas(master=self)
            self.factors_frame = tk.Frame(master=self.factor_canvas)

            # Initial variable values
            self.factor_que_length = 1
            self.default_values_list = []
            self.checkstate_list = []
            self.min_list = []
            self.max_list = []
            self.dec_list = []

            # Create main window title
            self.title_frame = tk.Frame(master=self)
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
            self.modelselect_frame = tk.Frame(master=self)
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

            self.experiment_name = "experiment"

    def clear_frame(self, frame: tk.Frame) -> None:
        """Clear all widgets from a frame.

        Args:
            frame (tk.Frame): The frame to clear.
        """
        for widget in frame.winfo_children():
            widget.destroy()

    def load_design(self) -> None:
        """Load design from a CSV file."""
        # Clear previous selections
        self.clear_frame(frame=self.factors_frame)
        self.clear_frame(frame=self.create_design_frame)
        self.clear_frame(frame=self.run_frame)
        self.clear_frame(frame=self.design_frame)

        # Initialize frame canvas
        self.factor_canvas = tk.Canvas(master=self)
        self.factor_canvas.grid_rowconfigure(0, weight=1)
        self.factor_canvas.grid_columnconfigure(0, weight=1)
        self.factor_canvas.grid(row=4, column=0, sticky="nsew")

        self.factors_title_frame = tk.Frame(master=self)
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
        selected_file = filedialog.askopenfilename()
        self.csv_filename = Path(selected_file).resolve()

        # get experiment name
        filename = self.csv_filename.name
        name = filename.split(".")[0]
        # remove design from name if present
        self.experiment_name = name.replace("_design", "")

        # convert loaded design to data frame
        self.design_table = pd.read_csv(self.csv_filename, sep="\t", header=None, index_col=False)
        self.design_table.columns = self.design_table.iloc[0]

        # Get design information from table
        self.model_name = self.design_table.at[1, "name"]
        self.design_type = self.design_table.at[1, "design_type"]
        self.n_stacks = self.design_table.at[1, "num_stacks"]
        self.model_var.set(self.model_name)

        all_factor_names = list(self.design_table.columns[1:-3])
        # names of factors included in design
        self.factor_names = []
        # dictionary that contains true/false for whether factor is in design
        self.factor_status = {}
        # col correspond to factor names, exclude index and information cols
        for col in self.design_table.columns[1:-3]:
            factor_set = set(self.design_table[col])
            design_factor = len(factor_set) > 1
            self.factor_status[col] = design_factor

        # get default values for fixed factors
        # contains only factors not in design, factor default vals input as str
        self.default_factors = {}
        for factor in self.factor_status:
            if not self.factor_status[factor]:
                self.default_factors[factor] = self.design_table.at[1, factor]
            else:
                self.factor_names.append(factor)

        # TODO: consolidate with copy in data_farming_base.py
        # If for some reason the user provides the module name instead of the
        # abbreviated class name, set the proper name.
        if self.model_name not in model_directory:
            for name, model_class in directory.model_directory.items():
                if model_class.name == self.model_name:
                    self.model_name = name
                    break

        self.model_object = model_directory[self.model_name]()

        # Allow user to change default values
        for _, factor in enumerate(all_factor_names):
            self.factor_datatype = self.model_object.specifications[factor].get(
                "datatype"
            )
            self.factor_description = self.model_object.specifications[factor].get(
                "description"
            )

            if not self.factor_status[factor]:
                self.factor_default = self.default_factors[factor]

            else:
                self.factor_default = "Cannot Edit Design Factor"

            self.factors_frame.grid_rowconfigure(self.factor_que_length, weight=1)

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
        """Enable the run button."""
        self.run_button.configure(state="normal")

    def show_design_options(self) -> None:
        # Design type selection menu
        self.design_frame = tk.Frame(master=self)
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
            self.default_value.get() for self.default_value in self.default_values_list
        ]  # default value of each factor
        for factor_index, factor in enumerate(self.default_factors):
            # self.default_values = [
            #     self.default_value.get()
            #     for self.default_value in self.default_values_list
            # ]  # default value of each factor
            new_val = self.default_values[factor_index]
            self.design_table[factor] = new_val
            self.default_factors[factor] = new_val

        self.experiment_name = (
            self.design_filename_var.get()
        )  # name of design file specified by user

        design_csv_name = f"{self.experiment_name}_design.csv"

        self.csv_filename = DATA_FARMING_DIR / design_csv_name

        self.design_table.to_csv(self.csv_filename, index=False, sep="\t")

        # read new design csv and convert to df
        self.design_table = pd.read_csv(self.csv_filename, index_col=False)

        messagebox.showinfo(
            "Information",
            f"Design has been modified. "
            f"{design_csv_name} has been created in {DATA_FARMING_DIR}. ",
        )

        self.display_design_tree()
        self.con_design()

    def con_design(self) -> None:
        # Create design txt file
        # Load name specified by user
        self.experiment_name = self.design_filename_var.get()
        self.design_filename = f"{self.experiment_name}_design.csv"
        file_path = DATA_FARMING_DIR / self.design_filename
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        self.design_table.to_csv(
            file_path,
            sep="\t",
            index=False,
            header=False,
        )

        # get fixed factors in proper data type
        self.fixed_factors = self.convert_proper_datatype(self.default_factors)

        self.enable_run_button()

    def convert_proper_datatype(self, fixed_factors: dict) -> dict:
        """Convert fixed factor values from strings to their proper data types.

        Args:
            fixed_factors (dict): Dictionary containing fixed factor names and
                user-selected values as strings.

        Returns:
            dict: Dictionary with fixed factor names and values converted to their
                appropriate data types.
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
                    converted_fixed_factors[factor] = tuple(float(s) for s in tuple_str)
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
    def show_model_factors(self, _: tk.StringVar) -> object:
        """Show model factors in the GUI.

        Args:
            _ (tk.StringVar): The selected model from the drop-down menu.
        """
        self.factor_canvas.destroy()

        # Initialize frame canvas
        self.factor_canvas = tk.Canvas(master=self)
        self.factor_canvas.grid_rowconfigure(0, weight=1)
        self.factor_canvas.grid_columnconfigure(0, weight=1)
        self.factor_canvas.grid(row=4, column=0, sticky="nsew")
        self.factors_frame = tk.Frame(master=self.factor_canvas)
        self.factor_canvas.create_window((0, 0), window=self.factors_frame, anchor="nw")

        self.factors_frame.grid_rowconfigure(self.factor_que_length + 1, weight=1)

        self.factors_title_frame = tk.Frame(master=self)
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
            self.factor_description = self.model_object.specifications[factor].get(
                "description"
            )
            self.factor_default = self.model_object.specifications[factor].get(
                "default"
            )
            self.factor_isDatafarmable = self.model_object.specifications[factor].get(
                "isDatafarmable"
            )

            # Values to help with formatting
            entry_width = 10

            self.factors_frame.grid_rowconfigure(self.factor_que_length, weight=1)

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

            if (
                self.factor_datatype is float
                and self.factor_isDatafarmable is not False
            ):
                self.factors_frame.grid_rowconfigure(self.factor_que_length, weight=1)

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
                self.checkbox.grid(row=self.factor_que_length, column=3, sticky="nsew")
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

            elif (
                self.factor_datatype is int and self.factor_isDatafarmable is not False
            ):
                self.factors_frame.grid_rowconfigure(self.factor_que_length, weight=1)

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
                self.checkbox.grid(row=self.factor_que_length, column=3, sticky="nsew")
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

            elif self.factor_datatype is list or self.factor_isDatafarmable is False:
                self.factors_frame.grid_rowconfigure(self.factor_que_length, weight=1)

                if self.factor_datatype is list:
                    self.str_type = "list"
                elif self.factor_datatype is int:
                    self.str_type = "int"
                elif self.factor_datatype is float:
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
                self.factors_frame.grid_rowconfigure(self.factor_que_length, weight=1)

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
        self.create_design_frame = tk.Frame(master=self)
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
        design_table = pd.read_csv(self.csv_filename, index_col="design_num", sep="\t")
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

        for index, row in enumerate(design_table.itertuples(index=False)):
            self.design_tree.insert("", index, text=str(index), values=row[:-3])

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
        self.run_frame = tk.Frame(master=self)
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

    def create_design(self, *_: tuple) -> None:
        """Create a design `.txt` and `.csv` file based on user-specified options.

        Args:
            _ (tuple): Tuple containing the number of stacks and the design type
                (unused positional arguments).
        """
        self.create_design_frame = tk.Frame(master=self)
        self.create_design_frame.grid(row=6, column=0)
        self.create_design_frame.grid_rowconfigure(0, weight=0)
        self.create_design_frame.grid_rowconfigure(1, weight=1)
        self.create_design_frame.grid_columnconfigure(0, weight=1)

        # Dictionary used for tree view display of fixed factors
        self.fixed_str = {}

        # user specified design options
        n_stacks = int(self.stack_var.get())
        design_type = self.design_var.get()
        if design_type != "nolhs":
            error_msg = "Design type not supported."
            logging.error(error_msg)
            messagebox.showerror("Error", error_msg)
            return
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

        # values to index through factors
        maxmin_index = 0
        dec_index = 0

        # List to hold names of all factors part of model to be displayed in csv
        self.factor_names = []
        def_factor_str = {}

        # Write factor information to design txt file
        file_name = f"{self.experiment_name}.txt"
        file_path = DATA_FARMING_DIR / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as file:
            for factor_index, factor in enumerate(self.model_object.specifications):
                factor_datatype = self.model_object.specifications[factor].get("datatype")
                is_datafarmable_factor = self.model_object.specifications[factor].get(
                    "isDatafarmable", True
                )
                factor_include = check_values[factor_index]

                # get user inputs for design factors

                if factor_include:
                    self.factor_names.append(factor)

                    if not (factor_datatype in (float, int) and is_datafarmable_factor):
                        error_msg = "Factor datatype not supported."
                        logging.error(error_msg)

                    factor_min = str(min_values[maxmin_index])
                    factor_max = str(max_values[maxmin_index])
                    maxmin_index += 1

                    if factor_datatype is float:
                        # NOTE: this doesn't work with 1e-XX values
                        factor_dec = str(dec_values[dec_index])
                        dec_index += 1
                    else:  # factor is int
                        factor_dec = "0"

                    data_insert = f"{factor_min} {factor_max} {factor_dec}\n"
                    file_name = f"{self.experiment_name}.txt"
                    file_path = DATA_FARMING_DIR / file_name

                    file.write(data_insert)

                # add fixed factors to dictionary and increase index values
                else:
                    def_factor_str[factor] = default_values[factor_index]
                    if not is_datafarmable_factor:
                        continue
                    if factor_datatype is float:
                        dec_index += 1
                        maxmin_index += 1
                    elif factor_datatype is int:
                        maxmin_index += 1

        # convert fixed factors to proper data type
        self.fixed_factors = self.convert_proper_datatype(def_factor_str)
        self.design_filename = f"{self.experiment_name}_design"
        design_csv = f"{self.experiment_name}_design.csv"
        self.csv_filename = DATA_FARMING_DIR / design_csv

        # Use create_design to create a design txt file & design csv
        self.design_list = create_design(
            name=self.model_object.class_name_abbr,
            factor_headers=self.factor_names,
            factor_settings=Path(self.experiment_name),
            fixed_factors=self.fixed_factors,
            n_stacks=n_stacks,
            design_type=design_type,
        )
        # Pop up message that csv design file has been created
        messagebox.showinfo(
            "Information",
            f"Design file {design_csv} has been created in {DATA_FARMING_DIR}.",
        )

        # Display Design Values
        self.display_design_tree()

    def run_experiment(self, *_: tuple) -> None:
        """Run an experiment using the specified design and experiment options.

        Args:
            _ (tuple): Tuple containing the number of replications and whether to use
                common random numbers (CRN).
        """
        # Specify a common number of replications to run of the model at each
        # design point.
        n_reps = int(self.rep_var.get())

        # Specify whether to use common random numbers across different versions
        # of the model.
        crn_across_design_pts = self.crn_var.get() == "Yes"

        raw_results = f"{self.experiment_name}_raw_results.csv"
        output_filename = DATA_FARMING_DIR / raw_results

        # Create DataFarmingExperiment object.
        myexperiment = DataFarmingExperiment(
            model_name=self.model_object.name,
            factor_settings=None,
            factor_headers=self.factor_names,
            design_path=self.design_filename,
            model_fixed_factors=self.fixed_factors,
        )

        # Run replications and print results to file.
        myexperiment.run(n_reps=n_reps, crn_across_design_pts=crn_across_design_pts)
        myexperiment.print_to_csv(csv_file_name=output_filename, overwrite=True)

        # run confirmation message
        messagebox.showinfo(
            "Run Completed",
            f"Experiment Completed. Output file can be found at {output_filename}",
        )

    def include_factor(self, *_: tuple) -> None:
        """Include a factor in the experiment and enable related experiment options.

        Args:
            _ (tuple): Tuple containing the factor name and its checkstate value.
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
            self.factor_description = self.model_object.specifications[factor].get(
                "description"
            )
            self.factor_default = self.model_object.specifications[factor].get(
                "default"
            )
            self.factor_isDatafarmable = self.model_object.specifications[factor].get(
                "isDatafarmable"
            )

            # Disable / enable experiment option widgets depending on factor type
            if (
                self.factor_datatype in (int, float)
                and self.factor_isDatafarmable is not False
            ):
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
