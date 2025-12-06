"""Main menu window of the GUI."""

import tkinter as tk
from tkinter import ttk
from tkinter.font import nametofont
from typing import Final

from simopt.gui.data_farming_window import DataFarmingWindow
from simopt.gui.new_experiment_window import NewExperimentWindow
from simopt.gui.toplevel_custom import Toplevel

FONT_SCALE: Final[float] = 1


class MainMenuWindow(Toplevel):
    """Main menu window of the GUI."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the main menu window of the GUI.

        Args:
            root (tk.Tk): The main window of the GUI.
        """
        super().__init__(root, title="SimOpt GUI - Main Menu", exit_on_close=True)
        # Set the size of the window to XX% of the screen size
        size_percent = 50
        self.center_window(size_percent / 100.0)

        self.menu_frame = ttk.Frame(master=self)
        self.menu_frame.pack(anchor="center", expand=True)

        # Create new style for the labels and buttons
        self.set_main_menu_style_changes()

        self.title_label = ttk.Label(
            master=self.menu_frame,
            text="Welcome to SimOpt Library Graphic User Interface",
            justify="center",
        )
        self.title_label.grid(row=0, column=0, pady=10, sticky="nsew")

        self.separator = ttk.Separator(master=self.menu_frame, orient="horizontal")
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
            ipadx=20,
            ipady=20,
        )

        # Button to open new experiment window
        self.new_experiment_button = ttk.Button(
            master=self.menu_frame,
            text="Simulation Optimization Experiments",
            command=self.open_new_experiment,
        )
        self.new_experiment_button.grid(
            row=3,
            column=0,
            pady=10,
            sticky="nsew",
            ipadx=20,
            ipady=20,
        )

        # Prevent window from getting launched in the background
        self.lift()

    def set_main_menu_style_changes(self) -> None:
        """Set the style of the main menu window."""
        self.header_font = nametofont("TkHeadingFont").copy()
        header_font_size = self.header_font.cget("size")
        scaled_header_font_size = int(header_font_size * FONT_SCALE)
        self.header_font.configure(size=scaled_header_font_size)
        self.style.configure("TLabel", font=self.header_font)

        self.option_font = nametofont("TkTextFont").copy()
        option_font_size = self.option_font.cget("size")
        scaled_option_font_size = int(option_font_size * FONT_SCALE)
        self.option_font.configure(size=scaled_option_font_size)
        self.style.configure("TButton", font=self.option_font)

    def reset_main_menu_style_changes(self) -> None:
        """Reset the style of the buttons."""
        self.style.configure("TLabel", font=nametofont("TkTextFont"))
        self.style.configure("TButton", font=nametofont("TkTextFont"))

    def __open_window(self, class_name: type) -> None:
        """Open a new window."""
        self.reset_main_menu_style_changes()
        new_window = class_name(self.root)
        self.destroy()
        # Bring new window to front
        new_window.lift()

    def open_model_datafarming(self) -> None:
        """Open the model data farming window."""
        self.__open_window(DataFarmingWindow)

    def open_new_experiment(self) -> None:
        """Open the new experiment window."""
        self.__open_window(NewExperimentWindow)
