from pathlib import Path

import numpy as np

from ANC.anc_helpers import alignment_process, analyze_results, show_spectrogram
from ANC.nkf import process_nkf
from ANC.nlms_filter import nlms_calculation
from ANC.rls_filter import RLSFilter
from src import files_handler as io_loader
from src import utils as ut

# Constants
DEFAULT_SAMPLE_RATE = 16000
NLMS_FILTER_WINDOW = 1024
NLMS_MU = 0.1
RLS_N_TAPS = 64


def main() -> None:
    # 1. Select the files and initialize parameters
    alignment = True  # Flag to skip manual alignment if data is already correlated
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

        resampled_noise = ut.distortion_ir(resampled_noise)
        # NO MANDOTRY- APPLYS EXPONENTIAL TF TO THE NOISE AND NORMALIZE
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
    """
    # Show input signals
    show_spectrogram(resampled_noise, fs_resample, "Noise microphone (less sensitive)")
    show_spectrogram(resampled_sig, fs_resample, "Signal + noise before enhancement (sensitive mic)")

    results_dir = io_loader.get_results_dir(project_root)

    # --- NLMS Algorithm ---
    nlms_result = nlms_calculation(
        total_sig=resampled_sig,
        noise=resampled_noise,
        fs1=fs_resample,
        fs2=fs_resample,
        fs_resample=DEFAULT_SAMPLE_RATE,
        filter_window=NLMS_FILTER_WINDOW,
        mu=NLMS_MU,
    )
    analyze_results(
        nlms_result,
        resampled_noise,
        fs_resample,
        results_dir,
        f"NLMS{iteration}.wav",
        "Signal after NLMS only",
    )

    # --- NKF (Neural Kalman Filter) ---
    nkf_result = process_nkf(sig=resampled_sig, noise=resampled_noise, fs_sig=fs_resample, fs_noise=fs_resample)
    analyze_results(
        nkf_result,
        resampled_noise,
        fs_resample,
        results_dir,
        f"nkf {iteration}.wav",
        "Signal after NKF only",
    )

    # --- RLS (Recursive Least Squares) ---
    rls_flit = RLSFilter(n_taps=RLS_N_TAPS)
    noise_estimation, rls_error = rls_flit.process(noisy_signal=resampled_sig, noise=resampled_noise)
    analyze_results(
        rls_error,
        resampled_noise,
        fs_resample,
        results_dir,
        f"error_RLS{iteration}.wav",
        "error after RLS only",
    )
    analyze_results(
        noise_estimation,
        resampled_noise,
        fs_resample,
        results_dir,
        f"sig_RLS{iteration}.wav",
        "noise estimation after RLS only",
    )


if __name__ == "__main__":
    main()
