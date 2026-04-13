from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from zuna.pipeline import pt_to_fif
from zuna.preprocessing.batch import (
    _add_epochs_to_cache,
    _flush_remaining_cache,
    _reset_epoch_cache,
)
from zuna.preprocessing.config import ProcessingConfig
from zuna.preprocessing.io import load_pt


CHANNEL_NAMES = ["Fz", "Cz"]
POSITIONS = np.array([[0.0, 0.0, 0.04], [0.01, 0.0, 0.04]], dtype=np.float32)


@pytest.fixture(autouse=True)
def reset_epoch_cache():
    _reset_epoch_cache()
    yield
    _reset_epoch_cache()


def _make_epochs(n_epochs: int, value: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    epochs = [
        np.full((len(CHANNEL_NAMES), 4), value + epoch_idx, dtype=np.float32)
        for epoch_idx in range(n_epochs)
    ]
    positions = [POSITIONS.copy() for _ in range(n_epochs)]
    return epochs, positions


def _make_metadata(original_filename: str) -> Dict[str, object]:
    return {
        "original_filename": original_filename,
        "channel_names": CHANNEL_NAMES.copy(),
        "sampling_rate": 256.0,
    }


def _save_file_epochs(
    output_path: Path,
    config: ProcessingConfig,
    *,
    file_counter: int,
    original_filename: str,
    n_epochs: int,
    value: float,
) -> List[str]:
    epochs, positions = _make_epochs(n_epochs, value)
    saved_files = _add_epochs_to_cache(
        epochs,
        positions,
        _make_metadata(original_filename),
        file_counter,
        output_path,
        config,
    )
    remaining = _flush_remaining_cache(output_path)
    if remaining:
        saved_files.append(remaining)
    return saved_files


def test_exact_multiple_boundary_uses_next_file_metadata(tmp_path: Path):
    config = ProcessingConfig(epochs_per_file=64)

    first_files = _save_file_epochs(
        tmp_path,
        config,
        file_counter=0,
        original_filename="subject_a.fif",
        n_epochs=64,
        value=1.0,
    )
    second_files = _save_file_epochs(
        tmp_path,
        config,
        file_counter=1,
        original_filename="subject_b.fif",
        n_epochs=5,
        value=2.0,
    )

    assert len(first_files) == 1
    assert len(second_files) == 1

    second_pt = load_pt(second_files[0])
    assert second_pt["metadata"]["original_filename"] == "subject_b.fif"
    assert second_pt["metadata"]["channel_names"] == CHANNEL_NAMES


def test_multiple_exact_multiple_files_do_not_cascade_stale_metadata(tmp_path: Path):
    config = ProcessingConfig(epochs_per_file=64)

    file_specs = [
        (0, "subject_a.fif", 64, 1.0),
        (1, "subject_b.fif", 64, 2.0),
        (2, "subject_c.fif", 64, 3.0),
        (3, "subject_d.fif", 7, 4.0),
    ]

    saved_by_file = {}
    for file_counter, original_filename, n_epochs, value in file_specs:
        saved_by_file[original_filename] = _save_file_epochs(
            tmp_path,
            config,
            file_counter=file_counter,
            original_filename=original_filename,
            n_epochs=n_epochs,
            value=value,
        )

    for original_filename, saved_files in saved_by_file.items():
        assert saved_files
        for saved_file in saved_files:
            pt_data = load_pt(saved_file)
            assert pt_data["metadata"]["original_filename"] == original_filename


def test_switching_files_with_buffered_epochs_raises(tmp_path: Path):
    config = ProcessingConfig(epochs_per_file=64)

    epochs, positions = _make_epochs(10, 1.0)
    _add_epochs_to_cache(
        epochs,
        positions,
        _make_metadata("subject_a.fif"),
        0,
        tmp_path,
        config,
    )

    next_epochs, next_positions = _make_epochs(1, 2.0)
    with pytest.raises(RuntimeError, match="unsaved epochs"):
        _add_epochs_to_cache(
            next_epochs,
            next_positions,
            _make_metadata("subject_b.fif"),
            1,
            tmp_path,
            config,
        )


def test_pt_to_fif_groups_outputs_by_original_filename_after_exact_boundary(tmp_path: Path):
    config = ProcessingConfig(epochs_per_file=64)

    _save_file_epochs(
        tmp_path,
        config,
        file_counter=0,
        original_filename="subject_a.fif",
        n_epochs=128,
        value=1.0,
    )
    _save_file_epochs(
        tmp_path,
        config,
        file_counter=1,
        original_filename="subject_b.fif",
        n_epochs=32,
        value=2.0,
    )

    output_dir = tmp_path / "reconstructed"
    pt_to_fif(str(tmp_path), str(output_dir))

    reconstructed = sorted(path.name for path in output_dir.glob("*.fif"))
    assert reconstructed == ["subject_a.fif", "subject_b.fif"]
