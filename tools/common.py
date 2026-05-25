"""Shared filename rules for generated experiment test results."""

import platform
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "simopt.toml"
MACHINE_TAG_SEPARATOR = "_"
RESULTS_SUFFIX = ".pickle.zst"


def sanitize_name(name: str) -> str:
    """Return the result-file-safe form of a problem or solver name."""
    return "".join(e for e in name if e.isalnum())


def machine_info_tag() -> str:
    """Return a stable machine tag for platform-specific expected results."""
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    machine_aliases = {
        "amd64": "x64",
        "x86_64": "x64",
        "aarch64": "arm64",
    }
    return f"{system}_{machine_aliases.get(machine, machine)}"


def expected_results_filename(problem_name: str, solver_name: str) -> str:
    """Return the expected-results filename for a problem-solver pair."""
    filename = f"{sanitize_name(problem_name)}_{sanitize_name(solver_name)}"
    if uses_machine_specific_results(solver_name):
        filename = f"{filename}{MACHINE_TAG_SEPARATOR}{machine_info_tag()}"
    return f"{filename}{RESULTS_SUFFIX}"


def uses_machine_specific_results(solver_name: str) -> bool:
    """Return whether this solver has machine-specific results."""
    return solver_name in _machine_specific_experiments()


def is_result_file_for_current_machine(path: Path) -> bool:
    """Return whether a result file should be tested on this machine."""
    stem = _result_stem(path.name)
    if stem is None:
        return False

    machine_tag_suffix = f"{MACHINE_TAG_SEPARATOR}{machine_info_tag()}"
    if stem.endswith(machine_tag_suffix):
        base_stem = stem[: -len(machine_tag_suffix)]
        return _uses_machine_specific_stem(base_stem)

    if _is_machine_specific_stem_with_tag(stem):
        return False

    return not _uses_machine_specific_stem(stem)


def current_machine_result_file(path: Path) -> Path | None:
    """Return the current machine's result file for a machine-specific result."""
    stem = _result_stem(path.name)
    if stem is None:
        return None

    machine_tag_suffix = f"{MACHINE_TAG_SEPARATOR}{machine_info_tag()}"
    base_stem = _machine_specific_base_stem(stem)

    if base_stem is None:
        return None
    return path.with_name(f"{base_stem}{machine_tag_suffix}{RESULTS_SUFFIX}")


def require_current_machine_result_file(path: Path) -> None:
    """Raise if a machine-specific result is missing for the current machine."""
    current_machine_path = current_machine_result_file(path)
    if current_machine_path is not None and not current_machine_path.exists():
        raise FileNotFoundError(
            "Missing platform-specific expected results file for "
            f"{path.name}: expected {current_machine_path.name}"
        )


def _result_stem(filename: str) -> str | None:
    if not filename.endswith(RESULTS_SUFFIX):
        return None
    return filename[: -len(RESULTS_SUFFIX)]


def _machine_specific_experiments() -> list[str]:
    with CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)
    return config.get("machine_specific_experiments", [])


def _uses_machine_specific_stem(stem: str) -> bool:
    for solver_name in _machine_specific_experiments():
        if stem.endswith(f"_{sanitize_name(solver_name)}"):
            return True
    return False


def _is_machine_specific_stem_with_tag(stem: str) -> bool:
    return _machine_specific_base_stem(stem) is not None and not _uses_machine_specific_stem(stem)


def _machine_specific_base_stem(stem: str) -> str | None:
    for solver_name in _machine_specific_experiments():
        solver_stem = f"_{sanitize_name(solver_name)}"
        if stem.endswith(solver_stem):
            return stem

        tagged_solver_stem = f"{solver_stem}{MACHINE_TAG_SEPARATOR}"
        if tagged_solver_stem in stem:
            return stem.split(tagged_solver_stem, maxsplit=1)[0] + solver_stem

    return None
