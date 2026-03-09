from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

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
GCC_PHAT_ALIGNMENT_SECONDS = 10
ALIGNMENT_SAFETY_BUFFER = 0.001  # seconds


def show_spectrogram(sig: np.ndarray, fs: int, title: str = "Spectrogram") -> None:
    """
    Computes and displays the STFT of a signal in dB scale.
    Uses the utility functions from utils.py.
    """
    # 1. Calculate STFT in dB mode using your utils function
    # f: frequency bins, t: time bins, stft_db: magnitude in dB
    f, t, stft_db = ut.calc_stft(sig, fs, mode="linear")

    # 3. Use your utility plot function
    # We pass the dB data and specify mode="linear" to plot_stft
    # because we already converted it to dB in the calculation step.
    ut.plot_stft(stft_db, t=t, f=f, mode="dB", title=title)


def main() -> None:
    # 1. Select the files and initialize parameters
    alignment = True  # Flag to skip manual alignment if data is already correlated
    fs_resample, iteration, noise_files, project_root, signal_files = load_data()

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
            alignment_window = fs_resample * GCC_PHAT_ALIGNMENT_SECONDS
            tau = ut.gcc_phat(
                resampled_sig[:alignment_window], resampled_noise[:alignment_window], fs=fs_resample, interp=1
            )
            sign_tau = np.sign(tau)  # sign of this delay
            tau = max(0, int((np.abs(tau) - ALIGNMENT_SAFETY_BUFFER) * fs_resample))  # Checks alignment buffer
            tau = int(sign_tau * tau)  # applys back the delay
            # Use torch to shift the noise signal by padding with zeros
            resampled_noise, resampled_sig = ut.adjusting_delays(
                sig_to_adjust=resampled_noise, sig_source=resampled_sig, tau=tau
            )
            # ensure lengths match exactly
            resampled_noise, resampled_sig = ut.match_sigs(ref=resampled_noise, sig=resampled_sig)
            # Save the newly aligned/correlated signals
            io_loader.save_sound(rf"{project_root}\corr_noise1.wav", resampled_noise, fs_resample)
            io_loader.save_sound(rf"{project_root}\corr_sig1.wav", resampled_sig, fs_resample)

        resampled_noise = ut.distortion_ir(resampled_noise)
        # NO MANDOTRY- APPLYS EXPONENTIAL TF TO THE NOISE AND NORMALIZE

        # Analyze initial coherence between noise and signal before filtering
        ut.coherence_of_sigs(resampled_sig, resampled_noise, fs_resample, True)

        # Run various ANC algorithms (NLMS, NKF, RLS)
        process_ancs(fs_resample, iteration, project_root, resampled_noise, resampled_sig)
        iteration += 1


def load_data() -> Tuple[int, int, list[str], Path, list[str]]:
    """
    Handles file selection via UI and prepares project variables.
    """
    print("Please select the TOTAL signals (signal + noise):")
    signal_files = io_loader.select_audio_files()

    print("Please select the NOISE reference signals:")
    noise_files = io_loader.select_audio_files()

    project_root = Path(signal_files[0]).parent
    iteration = 0
    fs_resample = DEFAULT_SAMPLE_RATE

    # Validation: Ensure we have a reference for every signal
    if len(signal_files) != len(noise_files):
        raise IndexError(f"Mismatch: Found {len(signal_files)} signals but {len(noise_files)} noise files.")
    return fs_resample, iteration, noise_files, project_root, signal_files


def save_and_analyze_result(
    result: Union[np.ndarray, torch.Tensor],
    noise: np.ndarray,
    fs: int,
    results_dir: Path,
    filename: str,
    title: str,
) -> None:
    """
    Helper function to save, visualize, and analyze ANC results.

    Args:
        result: The processed signal
        noise: Reference noise signal for coherence analysis
        fs: Sample rate
        results_dir: Directory to save results
        filename: Output filename
        title: Title for spectrogram
    """
    if torch.is_tensor(result):
        result = result.detach().cpu().numpy()

    show_spectrogram(result, fs, title)
    output_path = results_dir / filename
    io_loader.save_sound(str(output_path), result, fs)
    ut.coherence_of_sigs(result, noise, fs, True)


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
    save_and_analyze_result(
        nlms_result,
        resampled_noise,
        fs_resample,
        results_dir,
        f"NLMS{iteration}.wav",
        "Signal after NLMS only",
    )

    # --- NKF (Neural Kalman Filter) ---
    nkf_result = process_nkf(sig=resampled_sig, noise=resampled_noise, fs_sig=fs_resample, fs_noise=fs_resample)
    save_and_analyze_result(
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
    save_and_analyze_result(
        rls_error,
        resampled_noise,
        fs_resample,
        results_dir,
        f"error_RLS{iteration}.wav",
        "error after RLS only",
    )
    save_and_analyze_result(
        noise_estimation,
        resampled_noise,
        fs_resample,
        results_dir,
        f"sig_RLS{iteration}.wav",
        "noise estimation after RLS only",
    )


if __name__ == "__main__":
    main()
