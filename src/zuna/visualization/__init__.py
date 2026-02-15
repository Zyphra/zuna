"""
Visualization utilities for Zuna pipeline.
"""

from pathlib import Path


def compare_pipeline_outputs(
    input_dir: str,
    working_dir: str,
    plot_pt: bool = False,
    plot_fif: bool = True,
    num_samples: int = 2,
    sample_from_ends: bool = True,
):
    """
    Compare input vs output files (both .pt and .fif) from the pipeline.

    Args:
        working_dir: Working directory containing subdirectories:
                    - 1_fif_input/ (preprocessed FIF files)
                    - 2_pt_input/ (preprocessed PT files)
                    - 3_pt_output/ (model output PT files)
                    - 4_fif_output/ (reconstructed FIF files)
        plot_pt: Whether to plot .pt file comparisons (default: False)
        plot_fif: Whether to plot .fif file comparisons (default: True)
        num_samples: Number of files to compare (default: 2)
        sample_from_ends: If True, pick first and last files; if False, random (default: True)
    """
    from .compare import compare_pipeline

    working_path = Path(working_dir)

    # Set up paths
    preprocessed_fif_dir = working_path / "1_fif_input"
    pt_input_dir = working_path / "2_pt_input"
    pt_output_dir = working_path / "3_pt_output"
    fif_output_dir = working_path / "4_fif_output"
    figures_dir = working_path / "FIGURES"

    # Create figures directory
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    compare_pipeline(
        input_dir=str(input_dir),
        fif_input_dir=str(preprocessed_fif_dir),
        fif_output_dir=str(fif_output_dir),
        pt_input_dir=str(pt_input_dir),
        pt_output_dir=str(pt_output_dir),
        output_dir=str(figures_dir),
        plot_pt=plot_pt,
        plot_fif=plot_fif,
        num_samples=num_samples,
        sample_from_ends=sample_from_ends,
        include_original_fif=include_original_fif,
        normalize_for_comparison=normalize_for_comparison,
    )
