import tkinter as tk
from tkinter import ttk
from tkinter.font import nametofont


class Toplevel(tk.Toplevel):
    """Custom Toplevel class for the GUI."""

    def __init__(
        self,
        root: tk.Tk,
        title: str = "SimOpt GUI",
        exit_on_close: bool = False,
    ) -> None:
        """Initialize the ToplevelCustom class.

        Parameters
        ----------
        root : tk.Tk
            The main window of the GUI
        exit_on_close : bool, optional
            If True, the program will exit when the window is closed.

        """
        super().__init__(root)
        self.root = root
        # Configure the close button
        if exit_on_close:
            self.protocol("WM_DELETE_WINDOW", self.root.quit)
        else:
            self.protocol("WM_DELETE_WINDOW", self.destroy)
        # Set title and theme
        self.title(title)
        self.set_theme()

    def set_theme(self) -> None:
        """Set the theme of the GUI."""
        # Configure the theme of the GUI
        self.style = ttk.Style()
        self.style.theme_use("alt")
        # Configure the default fonts based on screen size
        # https://tkinter-docs.readthedocs.io/en/latest/generic/fonts.html
        # Scale by width because it's easy to scroll vertically, but scrolling
        # horizontally is a pain. This way, the text will always fit on
        # the screen.
        width = self.winfo_screenwidth()
        font_medium = int(width / 200)
        font_large = int(font_medium * 1.2)
        font_small = int(font_medium * 0.8)

        # Adjust the default fonts
        nametofont("TkDefaultFont").configure(size=font_medium)
        nametofont("TkTextFont").configure(size=font_medium)
        nametofont("TkHeadingFont").configure(
            size=font_large, weight="bold"
        )  # Default header
        nametofont("TkCaptionFont").configure(size=font_large)
        nametofont("TkTooltipFont").configure(
            size=font_small, slant="italic"
        )  # Default small italics
        nametofont("TkFixedFont").configure(size=font_medium)
        nametofont("TkIconFont").configure(size=font_medium)
        nametofont("TkMenuFont").configure(size=font_medium)
        nametofont("TkSmallCaptionFont").configure(size=font_small)

    def center_window(self, scale: float) -> None:
        """Centers the window to the main display/monitor.

        Example Usage
        -------------
        position = center_window(self.root, 0.8)

        self.root.geometry(position)

        Parameters
        ----------
        scale : float
            The scale of the window

        """
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = int(screen_width * scale)
        height = int(screen_height * scale)
        x = int((screen_width / 2) - (width / 2))
        y = int(
            (screen_height / 2) - (height / 1.9)
        )  # Slight adjustment for taskbar
        position = f"{width}x{height}+{x}+{y}"
        self.geometry(position)
