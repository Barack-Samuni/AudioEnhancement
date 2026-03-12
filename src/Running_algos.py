from pathlib import Path

import numpy as np

from ANC.anc_helpers import alignment_process, analyze_nkf, analyze_nlms, analyze_rls, show_spectrogram
from src import files_handler as io_loader
from src import utils as ut

# Constants
DEFAULT_SAMPLE_RATE = 16000
NLMS_FILTER_WINDOW = 1024


def main() -> None:
    # 1. Select the files and initialize parameters
    alignment = False  # Flag to skip manual alignment if data is already correlated
    noise_files, project_root, signal_files = io_loader.load_data()
    iteration = 0
    fs_resample = DEFAULT_SAMPLE_RATE
    # Iterate through pairs of signal and noise reference files
    for s_path, n_path in zip(signal_files, noise_files):
        # Load audio data from selected paths
        sig, fs_sig = io_loader.load_sound(s_path)
        noise, fs_noise = io_loader.load_sound(n_path)
        # Ensure signals are mono for processing
        sig = io_loader.stereo_to_mono(sig)
        noise = io_loader.stereo_to_mono(noise)
        # Resample both signals to the target processing frequency (usually 16kHz)
        resampled_sig, _ = ut.resample_fs(sig, fs_sig, fs_resample)
        resampled_noise, _ = ut.resample_fs(noise, fs_noise, fs_resample)

        # Perform Time Alignment if the alignment flag is False
        if not alignment:  # TO DO correlation issue - fixing required
            # Estimate delay using GCC-PHAT cross-correlation
            resampled_noise, resampled_sig = alignment_process(fs_resample, resampled_noise, resampled_sig)
            io_loader.save_sound(rf"{project_root}\corr_noise1.wav", resampled_noise, fs_resample)
            io_loader.save_sound(rf"{project_root}\corr_sig1.wav", resampled_sig, fs_resample)

        # NO MANDOTRY- APPLYS EXPONENTIAL TF TO THE NOISE AND NORMALIZE
        resampled_noise = ut.distortion_ir(resampled_noise)
        # Analyze initial coherence between noise and signal before filtering
        ut.coherence_of_sigs(resampled_sig, resampled_noise, fs_resample, True)
        # Run various ANC algorithms (NLMS, NKF, RLS)
        process_ancs(fs_resample, iteration, project_root, resampled_noise, resampled_sig)
        iteration += 1


def process_ancs(
    fs_resample: int, iteration: int, project_root: Path, resampled_noise: np.ndarray, resampled_sig: np.ndarray
):
    """
    Runs the noise cancellation pipeline using different algorithms and saves results.

    Args:
        fs_resample (int): The sampling frequency (Hz) of the audio signals after resampling.
        iteration (int): The current test or epoch number, used for versioning saved outputs.
        project_root (Path): Pathlib object pointing to the root directory of the project.
        resampled_noise (np.ndarray): The reference noise signal captured by the less sensitive mic.
        resampled_sig (np.ndarray): The primary signal (voice + noise) captured by the sensitive mic.
    """
    # Show input stfts
    show_spectrogram(resampled_noise, fs_resample, "Noise microphone (less sensitive)")
    show_spectrogram(resampled_sig, fs_resample, "Signal + noise before enhancement (sensitive mic)")

    results_dir = io_loader.get_results_dir(project_root)

    analyze_nlms(
        fs_resample=fs_resample,
        iteration=iteration,
        resampled_noise=resampled_noise,
        resampled_sig=resampled_sig,
        results_dir=results_dir,
    )

    analyze_nkf(
        fs_resample=fs_resample,
        iteration=iteration,
        resampled_noise=resampled_noise,
        resampled_sig=resampled_sig,
        results_dir=results_dir,
    )

    analyze_rls(
        fs_resample=fs_resample,
        iteration=iteration,
        resampled_noise=resampled_noise,
        resampled_sig=resampled_sig,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
