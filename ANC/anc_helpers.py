from pathlib import Path
from typing import Union

import numpy as np
import torch

from ANC.nkf import process_nkf
from ANC.nlms_filter import nlms_calculation
from ANC.rls_filter import RLSFilter
from src import files_handler as io_loader
from src import utils as ut

NLMS_FILTER_WINDOW = 1024
NLMS_MU = 0.1
RLS_N_TAPS = 64
DEFAULT_SAMPLE_RATE = 16000


GCC_PHAT_ALIGNMENT_SECONDS = 10
ALIGNMENT_SAFETY_BUFFER = 0.001  # seconds


def analyze_results(
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


def alignment_process(
    fs_resample: int, resampled_noise: np.ndarray, resampled_sig: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns a noise signal to a reference signal using GCC-PHAT cross-correlation.

    This function calculates the time delay (tau) between two signals within a
    specific window, applies a safety buffer to the delay, and shifts the noise
    signal to synchronize it with the reference signal.

    Args:
        fs_resample: The sampling frequency of the input signals (Hz).
        resampled_noise: The noise signal array to be shifted/aligned.
        resampled_sig: The reference signal array.

    Returns:
        A tuple containing (aligned_noise, aligned_sig), both truncated to
        the same length.
    """
    alignment_window = fs_resample * GCC_PHAT_ALIGNMENT_SECONDS
    tau = ut.gcc_phat(resampled_sig[:alignment_window], resampled_noise[:alignment_window], fs=fs_resample, interp=1)
    sign_tau = np.sign(tau)  # sign of this delay
    tau = max(0, int((np.abs(tau) - ALIGNMENT_SAFETY_BUFFER) * fs_resample))  # Checks alignment buffer
    tau = int(sign_tau * tau)  # applys back the delay
    # Use torch to shift the noise signal by padding with zeros
    resampled_noise, resampled_sig = ut.adjusting_delays(
        sig_to_adjust=resampled_noise, sig_source=resampled_sig, tau=tau
    )
    # ensure lengths match exactly
    resampled_noise, resampled_sig = ut.match_sigs(ref=resampled_noise, sig=resampled_sig)  # check?
    # Save the newly aligned/correlated signals
    return resampled_noise, resampled_sig


def analyze_rls(
    fs_resample: int, iteration: int, resampled_noise: np.ndarray, resampled_sig: np.ndarray, results_dir: Path
):
    """
    Executes the Recursive Least Squares (RLS) filter test.

    Args:
        fs_resample (int): The audio sampling rate (Hz).
        iteration (int): The current test index for file naming.
        resampled_noise (np.ndarray): The reference noise signal.
        resampled_sig (np.ndarray): The primary signal containing noise and target.
        results_dir (Path): The directory where output files will be saved.
    """
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


def analyze_nkf(
    fs_resample: int, iteration: int, resampled_noise: np.ndarray, resampled_sig: np.ndarray, results_dir: Path
):
    """
    Executes the Neural Kalman Filter (NKF) test.

    Args:
        fs_resample (int): The audio sampling rate (Hz).
        iteration (int): The current test index for file naming.
        resampled_noise (np.ndarray): The reference noise signal.
        resampled_sig (np.ndarray): The primary signal containing noise and target.
        results_dir (Path): The directory where output files will be saved.
    """
    nkf_result = process_nkf(sig=resampled_sig, noise=resampled_noise, fs_sig=fs_resample, fs_noise=fs_resample)
    analyze_results(
        nkf_result,
        resampled_noise,
        fs_resample,
        results_dir,
        f"nkf {iteration}.wav",
        "Signal after NKF only",
    )


def analyze_nlms(
    fs_resample: int, iteration: int, resampled_noise: np.ndarray, resampled_sig: np.ndarray, results_dir: Path
):
    """
    Executes the Normalized LEAST Mean Squares (NLMS) filter test.

    Args:
        fs_resample (int): The audio sampling rate (Hz).
        iteration (int): The current test index for file naming.
        resampled_noise (np.ndarray): The reference noise signal.
        resampled_sig (np.ndarray): The primary signal containing noise and target.
        results_dir (Path): The directory where output files will be saved.
    """
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
