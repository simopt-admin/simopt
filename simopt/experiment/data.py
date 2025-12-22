# ruff: noqa: D100, D101

import pandera.pandas as pa


class SolverHistorySchema(pa.DataFrameModel):
    mrep: int
    step: int
    budget: int
    solution: tuple[float, ...]


class ManySolverHistorySchema(pa.DataFrameModel):
    experiment: int
    mrep: int
    step: int
    budget: int
    solution: tuple[float, ...]


class PostReplicateSchema(pa.DataFrameModel):
    mrep: int
    step: int
    rep: int
    objective: float
    stochastic_constraints: object  # np.ndarray


class ManyPostReplicateSchema(pa.DataFrameModel):
    experiment: int
    mrep: int
    step: int
    rep: int
    objective: float
    stochastic_constraints: object  # np.ndarray
