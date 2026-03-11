import numpy as np
from pyroomacoustics.adaptive import NLMS

import src.utils as ut
from src import files_handler as io_loader


def nlms_calculation(
    total_sig: np.ndarray,
    noise: np.ndarray,
    fs1: int,
    fs2: int,
    fs_resample: int = 16000,
    filter_window: int = 1024,
    mu: float = 0.1,
) -> np.ndarray:
    """
    Applies NLMS and filtering using pyroomacoustics NLMS.

    Args:
        total_sig: Primary signal (signal + noise)
        noise: Reference noise signal
        fs1: Sample rate of total_sig
        fs2: Sample rate of noise
        fs_resample: Target sample rate (default: 16000 Hz)
        filter_window: Number of filter taps (default: 1024)
        mu: Step size parameter (default: 0.1)

    Returns:
        Filtered signal (error signal)

    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if not isinstance(total_sig, np.ndarray) or not isinstance(noise, np.ndarray):
        raise ValueError("Both total_sig and noise must be numpy arrays")

    if total_sig.size == 0 or noise.size == 0:
        raise ValueError("Input signals cannot be empty")

    if filter_window <= 0 or filter_window > len(total_sig):
        raise ValueError(f"filter_window must be positive and <= signal length ({len(total_sig)})")

    # 1. Resampling logic
    if fs1 != fs_resample:
        total_sig, _ = ut.resample_fs(total_sig, fs_old=fs1, fs_new=fs_resample)

    if fs2 != fs_resample:
        noise, _ = ut.resample_fs(noise, fs_old=fs2, fs_new=fs_resample)

    # 2. Use only first channel (Mono conversion from multichannels audio)
    total_sig = io_loader.stereo_to_mono(total_sig)
    noise = io_loader.stereo_to_mono(noise)

    # 3. Synchronize signal lengths
    noise, total_sig, _ = ut.adjust_min_length(noise, total_sig)

    nlms_filter = NLMS(length=filter_window, mu=mu)
    e = np.zeros(len(total_sig))

    # 5. Adaptive filtering loop
    # Using the pyroomacoustics built-in update logic
    for n in range(len(total_sig)):
        nlms_filter.update(noise[n], total_sig[n])
        e[n] = total_sig[n] - np.dot(nlms_filter.w, nlms_filter.x)

    return e
