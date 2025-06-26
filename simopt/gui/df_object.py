"""Datafarmable row classes.

This module contains the classes that represent the datafarmable rows in the GUI.
"""

from __future__ import annotations

import tkinter as tk
from abc import ABC, abstractmethod
from ast import literal_eval
from tkinter import ttk
from typing import Literal

from simopt.utils import classproperty, override


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

    @classproperty
    @abstractmethod
    def type(cls) -> tk.StringVar:
        """The type of the factor."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default(self) -> tk.Variable:
        """The default value of the factor."""
        raise NotImplementedError

    @default.setter
    @abstractmethod
    def default(self, default: tk.Variable) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_eval(self) -> object:
        """Evaluated default value of the factor."""
        raise NotImplementedError

    @property
    def include(self) -> tk.BooleanVar | None:
        """Whether to include the factor in the experiment."""
        return None

    @property
    def include_default_state(
        self,
    ) -> Literal["normal", "readonly", "disabled", None]:
        """Whether or not the default field is enabled."""
        if self.include is None:
            return None
        if self.include.get():
            return "disabled"
        return "normal"

    @property
    def include_datafarm_state(
        self,
    ) -> Literal["normal", "readonly", "disabled", None]:
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
        """Initialize a DFFactor instance.

        Args:
            name (str): The name of the factor.
            description (str): A description of the factor.
        """
        self.__name = tk.StringVar(value=name)
        self.__description = tk.StringVar(value=description)

    def get_name_label(self, frame: ttk.Frame) -> tk.Label:
        """Get the name label of the factor.

        Args:
            frame (ttk.Frame): The frame in which the label will be placed.

        Returns:
            tk.Label: The name label for the factor.
        """
        if not hasattr(self, "lbl_name"):
            self.lbl_name = tk.Label(
                master=frame,
                text=self.name.get(),
                justify=tk.LEFT,
            )
        return self.lbl_name

    def get_description_label(self, frame: ttk.Frame) -> tk.Label:
        """Get the description label of the factor.

        Args:
            frame (ttk.Frame): The frame in which the label will be placed.

        Returns:
            tk.Label: The description label for the factor.
        """
        if not hasattr(self, "lbl_description"):
            self.lbl_description = tk.Label(
                master=frame,
                text=self.description.get(),
                justify=tk.LEFT,
                anchor=tk.W,
                wraplength=200,
            )
        return self.lbl_description

    def get_type_label(self, frame: ttk.Frame) -> tk.Label:
        """Get the type label of the factor.

        Args:
            frame (ttk.Frame): The frame in which the label will be placed.

        Returns:
            tk.Label: The type label for the factor.
        """
        if not hasattr(self, "lbl_type"):
            self.lbl_type = tk.Label(
                master=frame,
                text=self.type.get(),
                justify=tk.CENTER,
                anchor=tk.W,
                width=1,
            )
        return self.lbl_type

    def get_default_entry(self, frame: ttk.Frame) -> ttk.Entry:
        """Get the default entry of the factor.

        Args:
            frame (ttk.Frame): The frame in which the entry will be placed.

        Returns:
            ttk.Entry: The default entry for the factor.
        """
        if not hasattr(self, "ent_default"):
            self.ent_default = ttk.Entry(
                master=frame,
                state=str(self.include_default_state),
                textvariable=self.default,
                justify=tk.RIGHT,
                width=1,
            )
        return self.ent_default

    def get_include_checkbutton(self, frame: ttk.Frame) -> tk.Checkbutton | None:
        """Get the include checkbutton of the factor.

        Args:
            frame (ttk.Frame): The frame in which the checkbutton will be placed.

        Returns:
            tk.Checkbutton | None: The include checkbutton for the factor,
                if applicable.
        """
        if self.include is None:
            return None
        if not hasattr(self, "chk_include"):
            self.chk_include = tk.Checkbutton(
                master=frame,
                variable=self.include,
                command=self._toggle_fields,
            )
        return self.chk_include

    def get_minimum_entry(self, frame: ttk.Frame) -> ttk.Entry | None:
        """Get the minimum entry of the factor.

        Args:
            frame (ttk.Frame): The frame in which the entry will be placed.

        Returns:
            ttk.Entry | None: The minimum entry for the factor, if applicable.
        """
        if self.minimum is None:
            return None
        if not hasattr(self, "ent_minimum"):
            self.ent_minimum = ttk.Entry(
                master=frame,
                state=str(self.include_datafarm_state),
                textvariable=self.minimum,
                justify=tk.RIGHT,
                width=1,
            )
        return self.ent_minimum

    def get_maximum_entry(self, frame: ttk.Frame) -> ttk.Entry | None:
        """Get the maximum entry of the factor.

        Args:
            frame (ttk.Frame): The frame in which the entry will be placed.

        Returns:
            ttk.Entry | None: The maximum entry for the factor, if applicable.
        """
        if self.maximum is None:
            return None
        if not hasattr(self, "ent_maximum"):
            self.ent_maximum = ttk.Entry(
                master=frame,
                state=str(self.include_datafarm_state),
                textvariable=self.maximum,
                justify=tk.RIGHT,
                width=1,
            )
        return self.ent_maximum

    def get_num_decimals_entry(self, frame: ttk.Frame) -> ttk.Entry | None:
        """Get the number of decimals entry of the factor.

        Args:
            frame (ttk.Frame): The frame in which the entry will be placed.

        Returns:
            ttk.Entry | None: The number of decimals entry for the factor,
                if applicable.
        """
        if self.num_decimals is None:
            return None
        if not hasattr(self, "ent_num_decimals"):
            self.ent_num_decimals = ttk.Entry(
                master=frame,
                state=str(self.include_datafarm_state),
                textvariable=self.num_decimals,
                justify=tk.RIGHT,
                width=1,
            )
        return self.ent_num_decimals

    def _toggle_fields(self) -> None:
        """Toggle the states of the datafarm fields."""
        if self.include is None:
            return
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

    @classproperty
    def type(cls) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="bool")

    @property
    def default(self) -> tk.BooleanVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.Variable) -> None:
        if not isinstance(default, tk.BooleanVar):
            error_msg = "Default value must be a BooleanVar."
            raise ValueError(error_msg)
        self.__default = default

    @property
    def default_eval(self) -> bool:
        """Evaluated default value of the factor."""
        return self.default.get()

    @property
    def include_default_state(self) -> Literal["readonly", "disabled", None]:
        """Whether or not the default field is enabled."""
        if self.include is None:
            return None
        if self.include.get():
            return "disabled"
        return "readonly"

    @property
    def include_datafarm_state(self) -> Literal["readonly", "disabled", None]:
        """Whether or not the datafarm fields are enabled."""
        if self.include is None:
            return None
        if self.include.get():
            return "readonly"
        return "disabled"

    def __init__(self, name: str, description: str, default: bool) -> None:
        """Initialize the boolean factor.

        Args:
            name (str): The name of the factor.
            description (str): A description of the factor.
            default (bool): The default value of the factor.
        """
        super().__init__(name, description)
        self.default = tk.BooleanVar(value=default)
        self.__include = tk.BooleanVar(value=False)

    @override
    def get_default_entry(self, frame: ttk.Frame) -> ttk.Entry:
        if not hasattr(self, "ent_default"):
            # Create a dropdown menu for boolean values
            self.ent_default = ttk.Combobox(
                master=frame,
                state=str(self.include_default_state),
                textvariable=self.default,
                values=["True", "False"],
                justify=tk.LEFT,
                width=1,
            )
            self.ent_default.current(0 if self.default.get() else 1)
        return self.ent_default

    def _toggle_fields(self) -> None:
        super()._toggle_fields()
        if self.ent_default.state() != ["disabled"]:
            self.ent_default.state(["readonly"])


class DFInteger(DFFactor):
    """Class to store integer factors for problems and solvers."""

    @classproperty
    def type(cls) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="int")

    @property
    def default(self) -> tk.IntVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.Variable) -> None:
        if not isinstance(default, tk.IntVar):
            error_msg = "Default value must be an IntVar."
            raise ValueError(error_msg)
        self.__default = default

    @property
    def default_eval(self) -> int:
        """Evaluated default value of the factor."""
        try:
            return int(self.default.get())
        except ValueError:
            raise ValueError(
                f"Default value for {self.name.get()} must be an integer."
            ) from None

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
        """Initialize the integer factor.

        Args:
            name (str): The name of the factor.
            description (str): A description of the factor.
            default (int): The default value of the factor.
        """
        super().__init__(name, description)
        self.__default = tk.IntVar(value=default)
        self.__include = tk.BooleanVar(value=False)
        self.__minimum = tk.IntVar(value=default)
        self.__maximum = tk.IntVar(value=default)


class DFIntegerNonDatafarmable(DFFactor):
    """Class to store non-datafarmable integer factors for problems and solvers."""

    @classproperty
    def type(cls) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="int")

    @property
    def default(self) -> tk.IntVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.Variable) -> None:
        if not isinstance(default, tk.IntVar):
            error_msg = "Default value must be an IntVar."
            raise ValueError(error_msg)
        self.__default = default

    @property
    def default_eval(self) -> int:
        """Evaluated default value of the factor."""
        try:
            return int(self.default.get())
        except ValueError:
            raise ValueError(
                f"Default value for {self.name.get()} must be an integer."
            ) from None

    def __init__(self, name: str, description: str, default: int) -> None:
        """Initialize the non-datafarmable integer factor.

        Args:
            name (str): The name of the factor.
            description (str): A description of the factor.
            default (int): The default value of the factor.
        """
        appended_description = "[Non-Datafarmable] " + description
        super().__init__(name, appended_description)
        self.__default = tk.IntVar(value=default)


class DFFloat(DFFactor):
    """Class to store float factors for problems and solvers."""

    @classproperty
    def type(cls) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="float")

    @property
    def default(self) -> tk.DoubleVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.Variable) -> None:
        if not isinstance(default, tk.DoubleVar):
            error_msg = "Default value must be a DoubleVar."
            raise ValueError(error_msg)
        self.__default = default

    @property
    def default_eval(self) -> float:
        """Evaluated default value of the factor."""
        try:
            return float(self.default.get())
        except ValueError:
            raise ValueError(
                f"Default value for {self.name.get()} must be a float."
            ) from None

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
        """Initialize the float factor.

        Args:
            name (str): The name of the factor.
            description (str): A description of the factor.
            default (float): The default value of the factor.
        """
        super().__init__(name, description)
        self.__default = tk.DoubleVar(value=default)
        self.__include = tk.BooleanVar(value=False)
        self.__minimum = tk.DoubleVar(value=default)
        self.__maximum = tk.DoubleVar(value=default)
        num_decimals = self.__find_num_decimals(default)
        self.__num_decimals = tk.IntVar(value=num_decimals)

    def __find_num_decimals(self, value: float) -> int:
        """Find the number of decimal places in a float value.

        Args:
            value (float): The float value to analyze.

        Returns:
            int: The number of decimal places in the given float.
        """
        # Case 1: Decimal point in value
        if "." in str(value):
            return len(str(value).split(".")[1])
        # Case 2: No decimal point in value, but xe-y format
        if "e-" in str(value):
            return int(str(value).split("e-")[1])
        # Case 3: No decimal point and not in xe-y format
        return 0


class DFTuple(DFFactor):
    """Class to store tuple factors for problems and solvers."""

    @classproperty
    def type(cls) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="tuple")

    @property
    def default(self) -> tk.StringVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.Variable) -> None:
        if not isinstance(default, tk.StringVar):
            error_msg = "Default value must be a StringVar."
            raise ValueError(error_msg)
        self.__default = default

    @property
    def default_eval(self) -> tuple:
        """Evaluated default value of the factor."""
        try:
            return tuple(literal_eval(self.default.get()))
        except ValueError:
            raise ValueError(
                f"Default value for {self.name.get()} must be a tuple."
            ) from None

    def __init__(self, name: str, description: str, default: tuple) -> None:
        """Initialize the tuple factor.

        Args:
            name (str): The name of the factor.
            description (str): A description of the factor.
            default (tuple): The default value of the factor.
        """
        super().__init__(name, description)
        self.__default = tk.StringVar(value=str(default))


class DFList(DFFactor):
    """Class to store list factors for problems and solvers."""

    @classproperty
    def type(cls) -> tk.StringVar:
        """The type of the factor."""
        return tk.StringVar(value="list")

    @property
    def default(self) -> tk.StringVar:
        """The default value of the factor."""
        return self.__default

    @default.setter
    def default(self, default: tk.Variable) -> None:
        if not isinstance(default, tk.StringVar):
            error_msg = "Default value must be a StringVar."
            raise ValueError(error_msg)
        self.__default = default

    @property
    def default_eval(self) -> list:
        """Evaluated default value of the factor."""
        try:
            return list(literal_eval(self.default.get()))
        except ValueError:
            raise ValueError(
                f"Default value for {self.name.get()} must be a list."
            ) from None

    def __init__(self, name: str, description: str, default: list) -> None:
        """Initialize the list factor.

        Args:
            name (str): The name of the factor.
            description (str): A description of the factor.
            default (list): The default value of the factor.
        """
        super().__init__(name, description)
        self.__default = tk.StringVar(value=str(default))


def spec_dict_to_df_dict(spec_dict: dict[str, dict]) -> dict[str, DFFactor]:
    """Convert a dictionary of specifications to a dictionary of data farming factors.

    Args:
        spec_dict (dict[str, dict]): A dictionary of factor specifications.

    Returns:
        dict[str, DFFactor]: A dictionary mapping factor names to `DFFactor` instances.
    """
    return {
        spec_name: spec_to_df(spec_name, spec) for spec_name, spec in spec_dict.items()
    }


def spec_to_df(spec_name: str, spec: dict) -> DFFactor:
    """Convert a specification to a data farming factor.

    Args:
        spec_name (str): The name of the factor.
        spec (dict): The specification dictionary for the factor.

    Returns:
        DFFactor: The corresponding data farming factor.
    """
    # Get the factor's datatype, description, and default value
    f_type = spec["datatype"]
    f_description = spec["description"]
    f_default = spec["default"]

    df_factor_map = {
        bool: DFBoolean,
        int: DFInteger,
        float: DFFloat,
        tuple: DFTuple,
        list: DFList,
    }

    # Check to see if we have a non-datafarmable integer
    if f_type is int and "isDatafarmable" in spec and not spec["isDatafarmable"]:
        return DFIntegerNonDatafarmable(spec_name, f_description, f_default)
    # Otherwise, just use the default mapping
    if f_type in df_factor_map:
        return df_factor_map[f_type](spec_name, f_description, f_default)
    raise NotImplementedError(f"Factor type [{f_type}] not yet implemented.")
