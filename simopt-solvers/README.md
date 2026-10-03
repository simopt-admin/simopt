# simopt-solvers

Simulation-optimization solvers for models built with
[simopt-dsl](../simopt-dsl).

```python
import simopt_dsl as dsl
from simopt_solvers import ADAM, solve

model = dsl.Model()
x = model.add_continuous_vector(-10, 10, name="x", shape=2, initial=[5, -5])


def run(decisions, rngs):
    x1, x2 = decisions["x"]
    return {"cost": (x1 - 3) ** 2 + (x2 + 1) ** 2 + rngs[0].normalvariate(0, 1)}, {}


simulation = model.add_simulation(run=run, decisions={"x": x})
model.minimize(dsl.mean(simulation.metric("cost")))

result = solve(model, ADAM(), budget=2000, seed=0)
print(result.x)
```

`seed` is a run index: the same seed reproduces the same run, and different
seeds give independent runs.

FCSA needs `cvxpy` and DASSO needs `pydoe`; install them with
`pip install 'simopt-solvers[fcsa]'`, `[dasso]`, or `[all]`.
