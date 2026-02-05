#!/usr/bin/env python3
"""
Tutorial: Zuna Inference Pipeline

This tutorial shows how inference can work with automatic preprocessing.
The preprocessing step uses a temporary directory, so users don't need to
manage intermediate PT files.
"""
import tempfile
from pathlib import Path
import zuna
import mne


def inference_pipeline(raw: mne.io.Raw, model_name: str = "zuna-base"):
    """
    Run inference on raw EEG data with automatic preprocessing.

    This function automatically:
    1. Preprocesses raw → PT in a temp directory
    2. Runs inference on PT file
    3. Cleans up temp files
    4. Returns predictions

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with montage set
    model_name : str
        Name of Zuna model to use

    Returns
    -------
    predictions : dict
        Model predictions
    """
    # Create temporary directory for intermediate PT file
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Using temporary directory: {tmp_dir}")

        # Preprocess to PT
        pt_path = Path(tmp_dir) / "temp_processed.pt"
        print("Preprocessing...")
        metadata = zuna.raw_to_pt(
            raw,
            str(pt_path),
            save_incomplete_batches=True,  # Always save for inference
            min_epochs_to_save=1,
        )

        print(f"  Preprocessed {metadata['n_epochs_saved']} epochs")

        # TODO: Run inference
        # predictions = zuna.inference.predict(pt_path, model_name=model_name)

        print("\n⚠️  Inference code not yet implemented!")
        print("   This is a placeholder showing how it would work.")

        # For now, just return metadata
        return {
            'status': 'preprocessed',
            'epochs': metadata['n_epochs_saved'],
            'channels': metadata['final_n_channels'],
            'temp_file_used': str(pt_path),
        }

    # Temp directory and files are automatically cleaned up here


def inference_from_file(raw_file_path: str, model_name: str = "zuna-base"):
    """
    Run inference directly from a raw EEG file path.

    This is the simplest API - users just provide a path!

    Parameters
    ----------
    raw_file_path : str
        Path to raw EEG file (.fif, .edf, etc.)
    model_name : str
        Name of Zuna model to use

    Returns
    -------
    predictions : dict
        Model predictions
    """
    # Load raw data
    if raw_file_path.endswith('.fif'):
        raw = mne.io.read_raw_fif(raw_file_path, preload=True, verbose=False)
    elif raw_file_path.endswith('.edf'):
        raw = mne.io.read_raw_edf(raw_file_path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {raw_file_path}")

    # Set standard montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    # Run inference
    return inference_pipeline(raw, model_name)


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Zuna Inference Pipeline Example")
    print("="*80)

    # Option 1: User has already loaded raw data
    # raw = mne.io.read_raw_fif('my_data.fif', preload=True)
    # montage = mne.channels.make_standard_montage('standard_1005')
    # raw.set_montage(montage)
    # predictions = inference_pipeline(raw)

    # Option 2: User just provides a file path (simplest!)
    # predictions = inference_from_file('my_data.fif')

    print("\nHow it would work:")
    print("  1. User provides raw file path or mne.Raw object")
    print("  2. Zuna preprocesses to PT in temp directory")
    print("  3. Zuna loads model and runs inference")
    print("  4. Temp files are automatically cleaned up")
    print("  5. User gets predictions!")
    print("\nBenefit: Users never need to manage PT files explicitly.")
    print("         (But they CAN if they want, using tutorial_preprocessing.py)")
