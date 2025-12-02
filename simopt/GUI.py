"""GUI for SimOpt Library."""  # noqa: N999

import logging
import sys
import tkinter as tk

from numpy import seterr

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
    root.tk.call("tk", "scaling", 1.0)

    # Parse command line
    log_level = logging.INFO
    for arg in sys.argv:
        if arg == "--debug":
            log_level = logging.DEBUG
            seterr(all="raise")
            break
        if arg == "--silent":
            log_level = logging.CRITICAL
            break

    debug_format = "%(levelname)s: %(message)s"
    logging.basicConfig(level=log_level, format=debug_format)
    logging.debug("GUI started")

    # app = Experiment_Window(root)
    MainMenuWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
