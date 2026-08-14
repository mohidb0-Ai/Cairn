"""Cairn — a small, dependency-aware orchestrator for machine learning work:
training pipelines, experiment tracking, and a model registry, all as one
lightweight Python library.

    from cairn import step, Pipeline, track, registry

Author: Mohid Bin Farooq
"""

from cairn.pipeline import Pipeline, step
from cairn.registry import registry
from cairn.tracking import track

__version__ = "0.1.0"
__author__ = "Mohid Bin Farooq"
__all__ = ["Pipeline", "step", "track", "registry", "__version__"]
