import utils
from utils import resample_fs
import numpy as np
from pyroomacoustics.adaptive import NLMS


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
        total_sig, _ = resample_fs(total_sig, fs_old=fs1, fs_new=fs_resample)

    if fs2 != fs_resample:
        noise, _ = resample_fs(noise, fs_old=fs2, fs_new=fs_resample)

    # 2. Use only first channel (Mono conversion)
    if total_sig.ndim > 1:
        total_sig = total_sig.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    # 3. Synchronize signal lengths
    total_sig, noise = utils.match_sigs(ref=total_sig, sig=noise)

    l = NLMS(length=filter_window, mu=mu)
    e = np.zeros(len(total_sig))

    # 5. Adaptive filtering loop
    # Using the pyroomacoustics built-in update logic
    for n in range(len(total_sig)):
        l.update(noise[n], total_sig[n])
        e[n] = total_sig[n] - np.dot(l.w, l.x)

    return e


# Backward-compatible alias for existing callers.
NLMS_calculation = nlms_calculation
