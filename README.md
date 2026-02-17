# Zuna: EEG Foundation Model

[![HuggingFace ZUNA](https://img.shields.io/badge/HuggingFace-ZUNA-FFD21E?logo=huggingface&logoColor=black&labelColor=555555)](https://huggingface.co/Zyphra/ZUNA)

Zuna is a pretrained EEG foundation model that reconstructs and denoises EEG signals. It takes raw EEG recordings, processes them through a transformer-based model, and outputs cleaned reconstructions.

## Installation

```bash
pip install zuna
```

Or install in development mode:

```bash
git clone https://github.com/Zyphra/zuna.git
cd zuna
pip install -e .
```

## Quick Start

See `tutorials/run_zuna_pipeline.py` for a complete working example. Edit the paths and options, then run:

```bash
python tutorials/run_zuna_pipeline.py
```

The pipeline takes your `.fif` files (with a channel montage set) and runs 4 steps:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `zuna.preprocessing` | .fif → .pt (resample, filter, epoch, normalize) |
| 2 | `zuna.inference` | .pt → .pt (model reconstruction) |
| 3 | `zuna.pt_to_fif` | .pt → .fif (denormalize, concatenate) |
| 4 | `zuna.compare_plot_pipeline` | Generate comparison plots |

It creates this directory structure under your working directory:

```
working_dir/
    1_fif_filter/     - Preprocessed .fif files (for comparison)
    2_pt_input/       - Preprocessed .pt files (model input)
    3_pt_output/      - Model output .pt files
    4_fif_output/     - Final reconstructed .fif files
    FIGURES/          - Comparison plots (if enabled)
```

Model weights are automatically downloaded from HuggingFace on first run.

## API Reference

For detailed documentation on any function, use Python's `help()`:

```python
import zuna
help(zuna.preprocessing)
help(zuna.inference)
help(zuna.pt_to_fif)
help(zuna.compare_plot_pipeline)
```

## Options

### Channel upsampling

Add channels from the standard 10-05 montage (the model will interpolate them):

```python
preprocessing(
    input_dir="...",
    output_dir="...",
    # Add specific channels by name
    target_channel_count=['AF3', 'AF4', 'F1', 'F2', 'FC1', 'FC2'],
    # Or upsample to N channels (greedy spatial selection)
    # target_channel_count=40,
)
```

### Bad channel zeroing

Zero out known bad channels so the model interpolates them:

```python
preprocessing(
    input_dir="...",
    output_dir="...",
    bad_channels=['Cz', 'Fz'],
)
```

### Visualization

Generate comparison plots between input and output:

```python
compare_plot_pipeline(
    input_dir="...",
    fif_input_dir=".../1_fif_filter",
    fif_output_dir=".../4_fif_output",
    pt_input_dir=".../2_pt_input",
    pt_output_dir=".../3_pt_output",
    output_dir=".../FIGURES",
    plot_pt=True,   # Compare .pt files (epoch-level)
    plot_fif=True,  # Compare .fif files (full recording)
)
```

### GPU selection

```python
inference(
    input_dir="...",
    output_dir="...",
    gpu_device=0,  # GPU ID (default: 0), or "" for CPU
)
```

## Setting Montages

Your input `.fif` files must have a channel montage with 3D positions. If your files don't have one:

```python
import mne

raw = mne.io.read_raw_fif('data.fif', preload=True)
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)
raw.save('data_with_montage.fif', overwrite=True)
```

## Disclaimer

This software and related services ("Services") are provided for research use only and are not intended for use in the diagnosis, cure, mitigation, treatment, or prevention of any disease or health condition. The Services have not been validated for any medical or clinical use. The information provided through the Services is for general informational purposes only and is not a substitute for any professional medical or healthcare advice. We do not warrant that any information provided through the Services is accurate, complete, or useful to you. Any reliance you place on such information is strictly at your own risk.
