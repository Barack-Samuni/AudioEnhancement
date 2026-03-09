from pathlib import Path
from typing import Union

import numpy as np
import torch

from src import files_handler as io_loader
from src import utils as ut

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
