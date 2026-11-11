## Candidate solution simulated, but attributes not used immediately

### Clear match

- `simopt/solvers/astrodf.py:828`
  - `candidate_solution` is created and then simulated at `simopt/solvers/astrodf.py:836` or adaptively sampled at `simopt/solvers/astrodf.py:838`.
  - Its fields are not read until `simopt/solvers/astrodf.py:847` (`objectives_mean`), and later `objectives_var` / `n_reps` are used at `simopt/solvers/astrodf.py:873`.

### Near-matches

- `simopt/solvers/neldmd.py:180`
  - `p_new` is created and simulated at `simopt/solvers/neldmd.py:182`.
  - It is then inserted into `sort_sol` at `simopt/solvers/neldmd.py:185`, and its statistics are only consumed later through `_sort_and_end_update`.

- `simopt/solvers/neldmd.py:143`
  - Initial `solution`s are simulated in a loop at `simopt/solvers/neldmd.py:145`.
  - They are only consumed later when sorting at `simopt/solvers/neldmd.py:152`.

### Not matches

- `simopt/solvers/strong.py:167`
  - `candidate_solution.objectives_mean` is read immediately after `simulate`.

- `simopt/solvers/aloe.py:129`
  - `candidate_solution.objectives_mean` is read immediately after `simulate`.

- `simopt/solvers/aloe.py:161`
  - `candidate_solution.objectives_mean` is read right away in the Armijo check.

- `simopt/solvers/adam.py:125`
  - `candidate_solution.objectives_mean` is read immediately after `simulate`.

- `simopt/solvers/utils.py:41`
  - `candidate_solution.objectives_mean` is read immediately after `simulate`.
