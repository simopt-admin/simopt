import importlib
import os
from collections.abc import Callable
from types import ModuleType

from simopt.model import Model


def find_model_patches(module: ModuleType) -> list[Callable]:
    patches: list[Callable] = []
    for attr in dir(module):
        if not attr.startswith("patch_model"):
            continue
        candidate = getattr(module, attr)
        if callable(candidate):
            patches.append(candidate)
    return patches


def patch(model_class: type[Model], patch_function: Callable) -> None:
    # Ask the extension what class to patch and with what method
    class_name, method = patch_function()
    # Patch the method into the model_class if it matches
    full_name = model_class.__module__ + "." + model_class.__qualname__
    if full_name == class_name:
        model_class.replicate = method


def load_module(module_name: str, model_class: type[Model]) -> None:
    # Import the specified library
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"SimOpt failed to load extension '{module_name}'") from e

    # Find all patch_model* functions in the library
    patches = find_model_patches(module)
    if not patches:
        raise ImportError(f"'{module_name}' does not have any 'patch_model*' functions")

    # Apply each patch to the model_class
    for p in patches:
        patch(model_class, p)


def patch_model(model_class: type[Model]) -> None:
    env_var = os.environ.get("SIMOPT_EXT")
    if not env_var:
        return

    # Assume that the user has specified a comma-separated list of libraries to import.
    # For example, SIMOPT_EXT="simopt_extension_a,simopt_extension_b"
    for part in env_var.split(","):
        module_name = part.strip()
        if module_name:
            load_module(module_name, model_class)
