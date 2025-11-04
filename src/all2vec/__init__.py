"""
ALL2Vec: Continuous Predictive Representations for Dynamic Visual Streams

Author: Ken I.
Email: ken.i.research@gmail.com
Paper: https://doi.org/10.5281/zenodo.17513405

__version__ = "0.1.0-dev"
__author__ = "Ken I."


This package provides a minimal, modular implementation of:

- `model`: core neural components and feature extractor
- `train`: live training / adaptation loop with logging
- `visualize`: interactive matplotlib dashboard

Importing from here re-exports the most commonly-used entry points.
"""

from .model import ModelConfig, FeatureExtractor, ALL2VecModel, init_state
from .train import run_live
from .visualize import Visualizer

__all__ = [
    "ModelConfig",
    "FeatureExtractor",
    "ALL2VecModel",
    "init_state",
    "run_live",
    "Visualizer",
]
