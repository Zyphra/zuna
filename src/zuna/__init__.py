"""
Zuna: EEG Foundation Model and Preprocessing Pipeline
"""

__version__ = "0.1.0"

from .pipeline import run_zuna_pipeline, zuna_preprocessing, zuna_inference, zuna_pt_to_fif, zuna_plot

__all__ = [
    'run_zuna_pipeline',
    'zuna_preprocessing',
    'zuna_inference',
    'zuna_pt_to_fif',
    'zuna_plot',
]
