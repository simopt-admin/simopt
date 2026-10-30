"""Tests for deriving solver specifications from configuration fields."""

import pytest
from pydantic import BaseModel, Field

from simopt_solvers._utils import get_specifications


def test_specifications_preserve_defaults_aliases_and_metadata():
    class Config(BaseModel):
        count: int = Field(3, alias="n", description="Sample count")
        weights: tuple[float, ...] = Field(
            default_factory=lambda: (1.0, 2.0),
            json_schema_extra={"isDatafarmable": False},
        )

    specifications = get_specifications(Config)

    assert specifications["n"] == {
        "description": "Sample count",
        "datatype": int,
        "default": 3,
    }
    assert specifications["weights"] == {
        "description": None,
        "datatype": list,
        "default": [1.0, 2.0],
        "isDatafarmable": False,
    }


def test_specifications_resolve_factories_using_prior_field_defaults():
    class Config(BaseModel):
        bounds: tuple[int, int] = (1, 5)
        upper: int = Field(default_factory=lambda data: data["bounds"][1])

    assert get_specifications(Config)["upper"]["default"] == 5


def test_specifications_reject_required_fields():
    class Config(BaseModel):
        count: int

    with pytest.raises(ValueError, match="Configuration field 'count' must have a default"):
        get_specifications(Config)


def test_specifications_allow_callable_schema_metadata():
    def add_title(schema: dict) -> None:
        schema["title"] = "Sample count"

    class Config(BaseModel):
        count: int = Field(3, json_schema_extra=add_title)

    assert get_specifications(Config)["count"]["default"] == 3
