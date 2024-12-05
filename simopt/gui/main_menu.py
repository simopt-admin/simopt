import tkinter as tk
from tkinter import ttk
from tkinter.font import nametofont
from typing import Final

from simopt.gui.data_farming_window import DataFarmingWindow
from simopt.gui.new_experiment_window import NewExperimentWindow
from simopt.gui.toplevel_custom import Toplevel

FONT_SCALE: Final[float] = 1.5


class MainMenuWindow(Toplevel):
    """Main menu window of the GUI."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the main menu window of the GUI.

        Parameters
        ----------
        root : tk.Tk
            The main window of the GUI

        """
        super().__init__(
            root, title="SimOpt GUI - Main Menu", exit_on_close=True
        )
        self.center_window(0.8)  # 80% scaling

        self.menu_frame = ttk.Frame(master=self)
        self.menu_frame.pack(anchor="center", expand=True)

        # Create new style for the labels and buttons
        header_font = nametofont("TkHeadingFont").copy()
        header_font_size = header_font.cget("size")
        scaled_header_font_size = int(header_font_size * FONT_SCALE)
        header_font.configure(size=scaled_header_font_size)

        option_font = nametofont("TkTextFont").copy()
        option_font_size = option_font.cget("size")
        scaled_option_font_size = int(option_font_size * FONT_SCALE)
        option_font.configure(size=scaled_option_font_size)
        self.style.configure("TButton", font=option_font)
        button_padding = scaled_option_font_size

        self.title_label = tk.Label(
            master=self.menu_frame,
            text="Welcome to SimOpt Library Graphic User Interface",
            justify="center",
            font=header_font,
        )
        self.title_label.grid(row=0, column=0, pady=10, sticky="nsew")

        self.separator = ttk.Separator(
            master=self.menu_frame, orient="horizontal"
        )
        self.separator.grid(row=1, column=0, pady=10, sticky="nsew")

        # Button to open model data farming window
        self.datafarm_model_button = ttk.Button(
            master=self.menu_frame,
            text="Data Farm Models",
            command=self.open_model_datafarming,
        )
        self.datafarm_model_button.grid(
            row=2,
            column=0,
            pady=10,
            sticky="nsew",
            ipadx=button_padding,
            ipady=button_padding,
        )

        # Button to open new experiment window
        self.new_experiment_button = ttk.Button(
            master=self.menu_frame,
            text="Data Farm Problems and/or Solvers",
            command=self.open_new_experiment,
        )
        self.new_experiment_button.grid(
            row=3,
            column=0,
            pady=10,
            sticky="nsew",
            ipadx=button_padding,
            ipady=button_padding,
        )

    def open_model_datafarming(self) -> None:
        """Open the model data farming window."""
        DataFarmingWindow(self.root)
        # Configure the exit button to close the window and close the menu
        self.destroy()

    def open_new_experiment(self) -> None:
        """Open the new experiment window."""
        NewExperimentWindow(self.root)
        # Configure the exit button to close the window and close the menu
        self.destroy()
