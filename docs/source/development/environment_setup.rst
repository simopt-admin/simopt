Environment Setup
=================

Required Tools
--------------

- `Miniconda (recommended) or Anaconda <https://www.anaconda.com/download>`__

    - Miniconda is a minimal installer for Conda, which is a package manager for Python and other languages.
    - Anaconda is a larger distribution that includes many pre-installed packages, but since we are using a custom environment, Anaconda is not necessary.

- `Git <https://git-scm.com/downloads>`__ & `GitHub Desktop <https://desktop.github.com/>`__
    
    - Git is a version control system that allows you to track changes in your code and collaborate with others.
    - GitHub Desktop is a GUI for Git that simplifies the process of cloning repositories and managing branches.

- `Visual Studio Code <https://code.visualstudio.com/>`__
    
    - A lightweight code editor that supports many programming languages and has a rich ecosystem of extensions.
    - It is recommended to use VS Code for development due to its integrated terminal, debugging capabilities, and support for Python development.

Installation Steps
------------------

1. **Install Miniconda or Anaconda**

    - The default installer settings are generally fine.
    - *Tip:* Check `Add Anaconda/Miniconda to my PATH` during installation if Python isn't already on your system.

2. **Install Git**

    - During setup, avoid selecting `Vim` as the default editor unless you are familiar with it.
    - Choose `Visual Studio Code` as the default editor when prompted.

3. **Install GitHub Desktop**

    - Once installed, choose `Clone a repository from the Internet`, then paste the following URL: ``https://github.com/simopt-admin/simopt.git``
    - Remember the local directory you cloned the repository to. This is where you will find the `simopt` folder.

4. **Install Visual Studio Code**

    - Open the `simopt` folder in VS Code (`File > Open Folder...`).

5. **Setup the SimOpt Environment**

    - Open the terminal in VS Code (`View > Terminal`)
    - Run the following command(s):

        - Windows (cmd): ``setup_simopt.bat``
        - Windows (PowerShell): ``cmd /c setup_simopt.bat``
        - MacOS/Linux: ``chmod +x setup_simopt.sh && ./setup_simopt.sh``

    - The setup script should automatically activate the environment. If needed later, you can manually activate it with: ``conda activate simopt``
    - You can re-run the setup script at any time to update dependencies in the environment.

6. **Install recommended VS Code extensions**

    - Python (`ms-python.python`)

        - This will also automatically install the following extensions:

            - Pylance (`ms-python.vscode-pylance`)
            - Debugger (`ms-python.debugpy`)

    - Ruff (`charliermarsh.ruff`)
    - GitHub Copilot (`GitHub.copilot`, optional but helpful)

        - This extension requires a GitHub account with Copilot access.
        - Access to Copilot is free for students and educators.
        - You can request access to Copilot through the `GitHub Student Developer Pack <https://education.github.com/pack/>`__. 

7. **Set the Python interpreter**

    - Open any `.py` file and look at the bottom-left corner of VS Code.

        .. image:: ./_static/version_preview.png
            :alt: A preview of the Python version in the lower right corner of VS Code
            :align: center

    - If the Python environment doesn’t show `(simopt)`, click the interpreter name and select one that includes `simopt`.

        .. image:: ./_static/version_dropdown.png
            :alt: A dropdown of Python interpreters visible in VS Code
            :align: center

Running the GUI
---------------

To run the GUI with debugging, use one of the following methods:

- From the menu bar, choosing `Run > Start Debugging`
- Pressing `F5`

Alternatively, you can run the application without debugging using one of the following methods:

- From the menu bar, choosing `Run > Run Without Debugging`
- Pressing `Ctrl + F5`
- Running ``python -m simopt.GUI`` in the terminal

NOTE: If launching via VS Code, you may be prompted to configure a launch environment. Choose `Python Module` and input ``simopt.GUI`` as the module name.
