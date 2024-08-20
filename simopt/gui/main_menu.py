import tkinter as tk
from tkinter.font import nametofont

from simopt.gui.experiment_window import ExperimentWindow
from simopt.gui.data_farming_window import DataFarmingWindow
from simopt.gui.new_experiment_window import NewExperimentWindow
from simopt.gui.toplevel_custom import Toplevel


class MainMenuWindow(Toplevel):
    """Main menu window of the GUI."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the main menu window of the GUI.

        Parameters
        ----------
        root : tk.Tk
            The main window of the GUI

        """
        super().__init__(root)
        self.configure_close()
        self.center_window(0.8)  # 80% scaling
        self.title("SimOpt Library GUI")

        self.menu_frame = tk.Frame(master=self)
        self.menu_frame.pack(anchor="center")

        font_scale = 1.5
        header_font = nametofont("TkHeadingFont").copy()
        header_font.configure(size=int(header_font.cget("size") * font_scale))
        option_font = nametofont("TkMenuFont").copy()
        option_font.configure(size=int(option_font.cget("size") * font_scale))

        self.title_label = tk.Label(
            master=self.menu_frame,
            text="Welcome to SimOpt Library Graphic User Interface",
            font=header_font,
            justify="center",
        )
        self.title_label.grid(row=0, column=0, pady=20)

        # Button to open original main window to run experiments across solvers & problems
        self.experiment_button = tk.Button(
            master=self.menu_frame,
            text="Run Single Problem-Solver Experiment",
            font=option_font,
            command=self.open_experiment_window,
        )
        self.experiment_button.grid(row=1, column=0, pady=10, sticky="nsew")
        self.experiment_button.configure(background="light gray")

        # Button to open model data farming window
        self.datafarm_model_button = tk.Button(
            master=self.menu_frame,
            text="Data Farm Models",
            font=option_font,
            command=self.open_model_datafarming,
        )
        self.datafarm_model_button.grid(row=2, column=0, pady=10, sticky="nsew")
        self.datafarm_model_button.configure(background="light gray")

        # # Button to open solver & problem data farming window
        # self.datafarm_prob_sol_button = tk.Button(
        #     master=self.menu_frame,
        #     text="Solver Data Farming",
        #     font=option_font,
        #     command=self.open_prob_sol_datafarming,
        # )
        # self.datafarm_prob_sol_button.grid(row=3, column=0, pady=10, sticky="nsew")
        # self.datafarm_prob_sol_button.configure(background="light gray")

        # Button to open new experiment window
        self.new_experiment_button = tk.Button(
            master=self.menu_frame,
            text="Data Farm Solvers, Problems, and Models",
            font=option_font,
            command=self.open_new_experiment,
        )
        self.new_experiment_button.grid(row=4, column=0, pady=10, sticky="nsew")
        self.new_experiment_button.configure(background="light gray")

        # Open the new experiment window and hide the main menu window
        # self.open_new_experiment()
        # self.withdraw()

    def open_experiment_window(self) -> None:
        """Open the experiment window."""
        experiment_app = ExperimentWindow(self.root)
        # Configure the exit button to close the window and close the menu
        experiment_app.configure_exit()
        self.destroy()

    def open_model_datafarming(self) -> None:
        """Open the model data farming window."""
        datafarming_app = DataFarmingWindow(self.root)
        # Configure the exit button to close the window and close the menu
        datafarming_app.configure_exit()
        self.destroy()

    def open_new_experiment(self) -> None:
        """Open the new experiment window."""
        new_experiment_app = NewExperimentWindow(self.root)
        # Configure the exit button to close the window and close the menu
        new_experiment_app.configure_exit()
        self.destroy()
