when working on documentation of functions and methods, make sure not to add any docstring to a private function unless you are explicitly told to do that.

when you create a funciton and decide it's private do not add every detailed doc. just a sentence summarize.

after code change, do try to review the code, but do not automatically run pytest for the entire test suite which will take a long time. you can identify possible test impacted and run targeted tests.
when running tests, you should always use `python -m pytest` as the start of the command. DO NOT use `pytest`. you should also use multiple process by attaching `-n 8`.

when you run any python command, add the env var `MRG32K3A_BACKEND=rust` 

when you use maturin to install a rust library, make sure to call `maturin develop -r` to install the fast compilation

when writing code, unless absolutely necessary, do not add type hints for variables inside functions and methods.

when doing refactor, or extracting existing logic into a separate function, make sure to keep the names of variables, functio3ns, etc. and keep original comments

All the code that is coming from simopt/plots I should be considered deprecated. You should use a simopt/analysis and simopt/experiment. benchmark.py gives a good example how to write code.
