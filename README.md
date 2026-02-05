# Zuna: EEG Foundation Model & Preprocessing

Zuna provides both EEG preprocessing and foundation model inference in one package.

## Features

- 🧹 **Clean preprocessing pipeline** with automatic artifact removal
- 🔄 **Full reversibility** - convert PT → Raw with original scaling
- ⚙️ **Configurable** - toggle each processing step on/off
- 🎯 **Simple API** - just 2 main functions: `raw_to_pt()` and `pt_to_raw()`
- 📦 **Flexible batching** - choose to save/discard incomplete epoch batches
- 🤖 **Foundation model** - pretrained model weights via Hugging Face

## Installation

```bash
# Install from PyPI
pip install zuna

# Or install in development mode
cd zuna
pip install -e .
```

## Quick Start - Preprocessing

```python
import zuna
import mne

# 1. Load your EEG data
raw = mne.io.read_raw_fif('your_data.fif', preload=True)

# 2. Set montage (REQUIRED - provides 3D coordinates)
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# 3. Process to PT format
metadata = zuna.raw_to_pt(raw, 'output.pt')

# 4. Later, reconstruct back to Raw
raw_reconstructed = zuna.pt_to_raw('output.pt')
```

## Configuration Options

```python
import zuna

# Configure preprocessing
metadata = zuna.raw_to_pt(
    raw,
    'output.pt',
    drop_bad_channels=True,
    drop_bad_epochs=True,
    apply_notch_filter=True,
    save_incomplete_batches=False,  # Discard if < 64 epochs
    target_sfreq=256.0,
    epoch_duration=5.0,
)
```

See full documentation for all configuration options