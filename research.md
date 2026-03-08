# `Solution.rng_list` Lifecycle (SimOpt)

This note traces how `Solution.rng_list` is created, attached, consumed, and advanced through the full run lifecycle.

## Key objects and roles

- `Solver.rng_list`: solver-internal randomness (sampling directions, random restart picks, etc.).
- `Solver.solution_progenitor_rngs`: template RNGs used to seed every new `Solution`.
- `Solution.rng_list`: RNGs actually used when simulating that specific solution.
- `Problem.simulate`: consumes `solution.rng_list` and advances each RNG by one subsubstream per replication.

## Main lifecycle during `run_solver`

### 1) RNG setup per macrorep

In `simopt/experiment/run_solver.py::_set_up_rngs`:

- Temporary solver RNGs are attached first (`[2, i+1, 0]` for `i=0..2`), mainly to establish expected solver RNG count.
- Simulation RNGs are created and stored in `solver.solution_progenitor_rngs`:
  - `MRG32k3a(s_ss_sss_index=[mrep + 3, i, 0])` for each model RNG index `i`.
- Solver RNGs are then replaced by macrorep-specific solver RNGs:
  - `MRG32k3a(s_ss_sss_index=[mrep + 3, problem.model.n_rngs + i, 0])`.

So each macrorep gets its own stream namespace.

### 2) Creating a new solution attaches RNG copies

In `Solver.create_new_solution`:

- Creates `Solution(x)`.
- Attaches RNGs via `new_solution.attach_rngs(self.solution_progenitor_rngs, copy=True)`.
- `copy=True` deep-copies each RNG, so each solution has an independent RNG state trajectory.

If `crn_across_solns=False`, solver advances each progenitor RNG by `problem.model.n_rngs` substreams after each new solution creation. That forces different random-number regimes across solutions.

### 3) Simulation consumes and advances `solution.rng_list`

In `Problem.simulate(solution, num_macroreps)` loop:

- Calls:
  - `self.model.before_replicate(solution.rng_list)`
  - `self.before_replicate(solution.rng_list)`
  - optional experiment override `before_replicate_override(self.model, solution.rng_list)`
- Runs `self.replicate(solution.x)` and stores result in solution arrays.
- Advances every RNG in `solution.rng_list` with `advance_subsubstream()`.

Therefore: one replication = one subsubstream step per RNG in that solution.

### 4) Re-simulating same `Solution`

If the same `Solution` instance is simulated again later, it continues from the already advanced RNG state. It does not reset unless caller reattaches fresh RNGs.

## How models use `solution.rng_list`

`Model.before_replicate(rng_list)` maps list positions to input models. Typical pattern:

- `rng_list[0]` -> arrival noise
- `rng_list[1]` -> service noise
- etc.

Some models intentionally share an RNG across components (for coupling/CRN effects), e.g. table allocation uses `rng_list[0]` for both arrival time and arrival count.

Special case: `RMITD` passes the full RNG list into a custom input model (`DemandInputModel.set_rng`) which splits into two internal RNG handles.

## Post-processing lifecycle (separate from solver solve loop)

### Post-replication (`simopt/experiment/post_replicate.py`)

For each solution `x` in solver history:

- Creates fresh `Solution(x)`.
- Attaches RNGs either:
  - `copy=True` if `crn_across_budget=True` (reuse same starting RNG state across budget steps), or
  - `copy=False` if `crn_across_budget=False` (shared state advances across steps).
- Calls `problem.simulate(solution, n_postreps)`.

Macrorep-level RNG seed policy is controlled by `crn_across_macroreps`.

### Post-normalization (`simopt/experiment/post_normalize.py`)

- Creates `Solution(x)`.
- Attaches with `copy=False`.
- Simulates to get objective sample for normalization.

## Important invariants

- `Solution.__init__` does not initialize `rng_list`; caller must call `attach_rngs` before `simulate`.
- `len(solution.rng_list)` must satisfy model expectations (`model.n_rngs` and indexing in `before_replicate`).
- `Problem.simulate` mutates RNG state in-place via subsubstream advancement.

## Notable edge behavior

- `NelderMead` sometimes constructs `Solution(...)` directly and calls `attach_rngs(..., copy=True)` instead of using `create_new_solution`. This bypasses the `crn_across_solns=False` progenitor-advance logic in `create_new_solution`.
- There is an inline FIXME in `run_solver._set_up_rngs` noting the initial temporary solver RNGs appear to be overridden.

## Practical mental model

- `solution_progenitor_rngs` = template seed state for new solutions.
- `Solution.rng_list` = private simulation RNG state for that solution (unless intentionally shared with `copy=False`).
- `Problem.simulate` = deterministic consumer of solution RNGs that advances subsubstreams each replication.
- CRN settings decide whether different solutions start from aligned RNG states or offset ones.
