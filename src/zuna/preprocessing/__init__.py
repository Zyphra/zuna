"""
EEG Preprocessing module.
"""
from .config import ProcessingConfig
from .processor import EEGProcessor
from .io import save_pt, load_pt, pt_to_raw
from .batch import preprocessing
from .interpolation import upsample_channels

__all__ = [
    'ProcessingConfig',
    'EEGProcessor',
    'save_pt',
    'load_pt',
    'pt_to_raw',
    'preprocessing',
    'upsample_channels',
]
