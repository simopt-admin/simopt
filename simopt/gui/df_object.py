import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import ttk
from typing import Literal

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
