"""Protocol-aware wind-power event measurement."""
from .protocol import Protocol, chronological_split, fit_scale, regularize
from .catalog import detect, describe, composite_intervals, build_catalog

__version__ = "0.18.0"
__all__ = ["Protocol", "chronological_split", "fit_scale", "regularize", "detect", "describe", "composite_intervals", "build_catalog"]
