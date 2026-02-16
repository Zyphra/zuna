"""
Zuna: EEG Foundation Model and Preprocessing Pipeline
"""

__version__ = "0.1.0"

from .pipeline import zuna_preprocessing, zuna_inference, zuna_pt_to_fif #zuna_plot
from .visualization.compare import compare_plot_pipeline
__all__ = [
    'zuna_preprocessing',
    'zuna_inference',
    'zuna_pt_to_fif',
    'compare_plot_pipeline', #'zuna_plot',
]
