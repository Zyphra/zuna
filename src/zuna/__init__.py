"""
Zuna: EEG Foundation Model

Main functions:
    zuna.preprocessing()          - .fif → .pt (resample, filter, epoch, normalize)
    zuna.inference()              - .pt → .pt (model reconstruction)
    zuna.pt_to_fif()              - .pt → .fif (denormalize, concatenate)
    zuna.compare_plot_pipeline()  - Generate comparison plots

See tutorials/run_zuna_pipeline.py for a complete working example.
Use help(zuna.preprocessing) etc. for detailed documentation.
"""

__version__ = "0.1.0"

from .preprocessing.batch import process_directory as preprocessing
from .pipeline import zuna_inference as inference, zuna_pt_to_fif as pt_to_fif
from .visualization.compare import compare_plot_pipeline

__all__ = [
    'preprocessing',
    'inference',
    'pt_to_fif',
    'compare_plot_pipeline',
]
