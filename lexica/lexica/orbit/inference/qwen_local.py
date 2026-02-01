"""
Local Qwen inference wrapper for Orbit.

This module provides a thin convenience layer over the Orbit model.
All model lifecycle management is handled by load_model.py.
"""

from typing import Any
from lexica.orbit.inference.load_model import Orbit

# Singleton Orbit instance (lazy-loaded)
_orbit_instance: Orbit | None = None


def _get_orbit() -> Orbit:
    global _orbit_instance
    if _orbit_instance is None:
        _orbit_instance = Orbit()
    return _orbit_instance


def generate(prompt: str, **kwargs: Any) -> str:
    """
    Generate raw model output from a natural language prompt.

    This function does NOT perform validation or IR construction.
    It is intended for internal use by the Orbit translator.
    """
    orbit = _get_orbit()
    return orbit.generate(prompt, **kwargs)
