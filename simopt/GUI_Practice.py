# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:18:28 2023

@author: Owner
"""

import tkinter as tk

def button_click():
    print("Button clicked!")

# Create the main window
window = tk.Tk()

# Create a button
button = tk.Button(window, text="Click Me", command=button_click)
button.pack()

# Start the Tkinter event loop
window.mainloop()