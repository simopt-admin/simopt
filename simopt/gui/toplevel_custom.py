import sys
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
        """Initialize the Toplevel class.

        Args:
            root (tk.Tk): The main window of the GUI.
            title (str, optional): The title of the window. Defaults to "SimOpt GUI".
            exit_on_close (bool, optional): If True, the program will exit when the
                window is closed.
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
        self.set_style()

    def set_style(self) -> None:
        """Set the theme of the GUI."""
        # Configure the theme of the GUI
        self.style = ttk.Style()
        # Configure the default fonts based on screen size
        # https://tkinter-docs.readthedocs.io/en/latest/generic/fonts.html
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        # If it's wider than 16:9 (like 21:9 or 32:9), use the height to
        # calculate the equivalent width
        if (width / height) > (16 / 9):
            width = height * (16 / 9)
        # Otherwise, we're good with just using the width
        # Target a 1920x1080 screen
        scale = width / 1920

        font_medium = int(12 * scale)
        if sys.platform == "darwin":
            win_to_mac_scaling = 1.375
            font_medium = int(font_medium * win_to_mac_scaling)
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

        # Change the default button behavior to center text
        self.style.configure("TButton", justify="center")

        # Set Treeview style
        # TODO: see if this is the right scaling
        height = 30 * font_medium / 12
        self.style.configure("Treeview", rowheight=int(height))
        self.style.configure(
            "Treeview.Heading",
            font=nametofont("TkHeadingFont"),
        )

    def center_window(self, scale: float) -> None:
        """Center the window on the main display/monitor.

        Args:
            scale (float): The scale factor to apply to the window size
                (e.g., 0.8 for 80% of screen size).

        Example:
            ```python
            position = center_window(self.root, 0.8)
            self.root.geometry(position)
            ```
        """
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        width = int(screen_width * scale)
        height = int(screen_height * scale)
        x = int((screen_width / 2) - (width / 2))
        # Slight adjustment for taskbar
        y = int((screen_height / 2) - (height / 1.9))
        position = f"{width}x{height}+{x}+{y}"
        self.geometry(position)
