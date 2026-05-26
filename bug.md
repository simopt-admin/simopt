# FCSA objective-direction bug

`TestFacsize1Fcsa` exposed an old objective-direction bug in FCSA.

`FACSIZE-1` is a minimization problem with `problem.minmax[0] == -1`. Before
`Solver.evaluate()` standardized evaluated objectives to minimization, FCSA used this
comparison when checking whether a feasible solution improved over the incumbent:

```python
problem.minmax[0] * solution.objectives_mean
< problem.minmax[0] * self._best_solution.objectives_mean
```

For a minimization problem, that becomes:

```python
-new_objective < -best_objective
```

which is equivalent to:

```python
new_objective > best_objective
```

So the old FCSA path treated a higher objective value as an improvement on
minimization problems. The expected artifact for `TestFacsize1Fcsa` appears to encode
that old behavior: it expected only 2 recommended solutions, while the corrected
minimization convention reports many feasible improving incumbents.

Under the new convention, `Solver.evaluate()` returns objective values in
minimization form for both min and max problems, so FCSA should compare incumbents
directly with:

```python
solution.objectives_mean < self._best_solution.objectives_mean
```
