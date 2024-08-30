"""GUI for SimOpt Library."""  # noqa: N999

import tkinter as tk

from simopt.gui.main_menu import MainMenuWindow


class GUIMaster(tk.Tk):
    """The master class for the GUI."""

    def __init__(self) -> None:
        """Initialize the GUI."""
        super().__init__()
        # Minimize the GUI window
        self.withdraw()


def main() -> None:
    """Run the GUI."""
    root = GUIMaster()
    root.title("SimOpt Library Graphical User Interface")
    root.pack_propagate(False)

    # app = Experiment_Window(root)
    MainMenuWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
