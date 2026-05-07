# Review: simopt/models/ambulance.py

## Findings

### High: queued dispatch leaves the ambulance marked available

In the service-completion path, the ambulance is set available at
`simopt/models/ambulance.py:299`, then a queued call is assigned and a new
completion is scheduled at `simopt/models/ambulance.py:302` and
`simopt/models/ambulance.py:328`, but `ambs[i, 2]` is never set back to
`BUSY`.

Later arrivals can dispatch that same ambulance while it is still serving the
queued call, underestimating response times and corrupting gradients.

### Medium: `check_deterministic_constraints` accepts wrong-length vectors

`simopt/models/ambulance.py:418` only checks bounds, so `()` or `(1.0,)` pass.
Those vectors become malformed `variable_locs` at
`simopt/models/ambulance.py:391`, then `replicate()` indexes them using
`variable_base_count` at `simopt/models/ambulance.py:164`.

Add a dimension check such as `len(_x) == self.dim`.

### Low: beta shape parameters are not validated positive

`call_loc_beta_x` and `call_loc_beta_y` are plain `tuple[float, float]` fields
at `simopt/models/ambulance.py:49`, but they are passed to `betavariate` at
`simopt/models/ambulance.py:199`.

Invalid values are accepted at config time and fail later during replication.

## Checks

- `ruff check simopt/models/ambulance.py` passed.
- Smoke simulations ran successfully.
- A deterministic sequence reproduced the queued-dispatch availability bug.
