# ruff: noqa: D100, D101

import pandera.pandas as pa


class SolverHistorySchema(pa.DataFrameModel):
    mrep: int
    step: int
    budget: int
    solution: tuple[float, ...]
