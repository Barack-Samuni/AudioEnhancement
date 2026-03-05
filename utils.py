import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
import torch
from typing import Tuple, Optional, Literal

# 1. Spectral Analysis Parameters
# WIN_DUR: The length of the analysis window in seconds (64ms)
# HOP_FRAC: The step size between windows as a fraction of window length
# EPSILON: constant to prevent division by 0
WIN_DUR = 0.064
HOP_FRAC = 0.2
EPSILON = 1e-15

# Utility functions

FilterType = Literal[
    "bandpass", "lowpass", "highpass", "bandstop",
    "band", "pass", "bp", "bands", "stop", "bs",
    "low", "lp", "l", "high", "hp", "h",
]


def butter_filter(
    data: np.ndarray,
    cutoff: float,
    fs: int,
    btype: FilterType,
    order: int = 5,
) -> np.ndarray:
    """
    Applies a Butterworth filter using zero-phase filtering to preserve timing.

    Args:
        data: The input audio signal array
        cutoff: The filter frequency in Hz
        fs: Sampling rate of the input data
        btype: 'low' for low-pass or 'high' for high-pass
        order: The steepness of the filter roll-off (default: 5)

    Returns:
        Filtered signal array
    """
    nyq = 0.5 * fs  # Nyquist Frequency (highest representable frequency)
    normal_cutoff = cutoff / nyq

    # Generate the filter coefficients (numerator 'b' and denominator 'a')
    b, a = sg.butter(order, normal_cutoff, btype=btype, analog=False)

    # Use filtfilt to apply the filter forward and backward, resulting in zero phase distortion
    y = sg.filtfilt(b, a, data)
    return y


def resample_fs(sig: np.ndarray, fs_old: int, fs_new: int) -> Tuple[np.ndarray, int]:
    """
    Changes the sampling rate of a signal.

    If downsampling, it applies a low-pass filter first to prevent aliasing.

    Args:
        sig: Input signal array
        fs_old: Current sampling rate
        fs_new: Target sampling rate

    Returns:
        tuple: (resampled signal, new sample rate)
    """
    if fs_old == fs_new:
        return sig, fs_old

    # If the new rate is lower, we must remove frequencies above the new Nyquist limit
    if fs_new < fs_old:
        fn = fs_new / 2
        sig = butter_filter(data=sig, cutoff=fn, fs=fs_old, btype='low')

    num_samples = int(len(sig) * fs_new / fs_old)
    # Perform resampling using Fourier method (high quality)
    resampled_sig = sg.resample(sig, num_samples)
    return resampled_sig, fs_new


def calc_stft(sig: np.ndarray, fs: int, mode: str = "linear") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Short-Time Fourier Transform to analyze signal frequency over time.

    Args:
        sig: Input signal array
        fs: Sampling rate
        mode: Output mode - 'linear' for magnitude, 'dB' for decibels, 'complex' for complex values

    Returns:
        tuple: (frequency bins, time bins, STFT result)
    """
    n_overlap, window_size = stft_params_calc(fs)
    f1, t1, sig_stft = sg.stft(x=sig, fs=fs, noverlap=n_overlap, nperseg=window_size)
    stft_abs = np.abs(sig_stft)

    if mode == "linear":
        return f1, t1, stft_abs
    if mode == "dB":
        # Convert to logarithmic scale (Decibels) for better visualization of quiet sounds
        return f1, t1, 20 * np.log10(stft_abs + EPSILON)

    # Return complex numbers for processing/reconstruction
    return f1, t1, sig_stft


def stft_params_calc(fs: int) -> Tuple[int, int]:
    """
    Helper to convert time durations (seconds) into discrete samples based on fs.

    Args:
        fs: Sampling rate

    Returns:
        tuple: (number of overlapping samples, window size in samples)
    """
    window_size = int(fs * WIN_DUR)
    n_overlap = int(((WIN_DUR * fs) * (1 - HOP_FRAC)))
    return n_overlap, window_size


def plot_stft(stft: np.ndarray, t: Optional[np.ndarray] = None, f: Optional[np.ndarray] = None,
              mode: str = "dB", vmin: float = -90, vmax: float = -20, title: str = "") -> None:
    """
    Plots a spectrogram heatmap.

    Args:
        stft: STFT magnitude data
        t: Time bins (optional)
        f: Frequency bins (optional)
        mode: Display mode - 'dB' for decibels (default), '' for linear
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        title: Plot title
    """
    _, ax = plt.subplots(figsize=(10, 4))
    if mode == "dB":
        plot_data = 20 * np.log10(stft + EPSILON)
    else:
        plot_data = stft

    # Define the coordinates for the x (time) and y (frequency) axes
    if t is not None and f is not None:
        extent = [t[0], t[-1], f[0], f[-1]]
    else:
        extent = None

    ax.imshow(
        plot_data,
        origin='lower',
        aspect='auto',
        cmap='inferno',
        vmin=vmin if mode == "log" else None,
        vmax=vmax if mode == "log" else None,
        extent=extent
    )

    ax.set_ylabel('Frequency' + (' (Hz)' if f is not None else ' (Index)'))
    ax.set_xlabel('Time' + (' (sec)' if t is not None else ' (Index)'))
    ax.set_title(f"STFT {title} , ({mode.capitalize()})")
    plt.show()


def coherence_of_sigs(sig: np.ndarray, noise: np.ndarray, fs: int) -> None:
    """
    Calculates and plots the coherence between two signals across frequencies.

    Coherence measures the linear relationship between two signals at each frequency.
    Values near 1.0 indicate high correlation, which is crucial for effective ANC.

    Args:
        sig: First signal array
        noise: Second signal array (reference noise)
        fs: Sampling rate
    """
    n_overlap, window_size = stft_params_calc(fs)
    f, cxy = sg.coherence(sig, noise, fs=fs, noverlap=n_overlap, nperseg=window_size)
    plt.plot(f, cxy)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.show(block=True)


def match_sigs(ref: np.ndarray, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensures 'sig' is exactly the same length as 'ref' by padding or cropping.

    Args:
        ref: Reference signal array
        sig: Signal to be matched to reference length

    Returns:
        tuple: (ref signal unchanged, sig signal adjusted to match length)
    """
    diff = len(ref) - len(sig)

    if diff > 0:
        # If 'sig' is shorter, add zeros at the end
        sig = np.pad(sig, (0, diff), mode='constant')
    elif diff < 0:
        # If 'sig' is longer, cut the extra samples
        sig = sig[:len(ref)]

    return ref, sig


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    """
    Generalized Cross Correlation with Phase Transform (GCC-PHAT).

    Calculates the time delay required to align two signals by finding the peak
    in their cross-correlation. This is crucial for synchronizing reference and
    primary signals before adaptive noise cancellation.

    The Phase Transform (PHAT) normalization removes amplitude effects and focuses
    solely on phase differences, providing a sharp correlation peak even in noisy
    or reverberant conditions.

    Args:
        sig: Primary/trusted signal (stationary reference)
        refsig: Secondary/reference signal to be aligned (the signal to "move")
        fs: Sampling rate in Hz (default: 1)
        max_tau: Maximum time delay to search in seconds (optional)
        interp: Interpolation factor for sub-sample precision (default: 16)

    Returns:
        float: Time delay 'tau' in seconds required to align refsig to sig

    Note:
        The returned tau should be applied to refsig to ensure both signals
        are perfectly in-phase before adaptive filtering.
    """

    # Ensure data is in NumPy format for FFT processing
    if torch.is_tensor(sig):
        sig = sig.numpy()
    if torch.is_tensor(refsig):
        refsig = refsig.numpy()

    # Define FFT length to avoid circular overlap artifacts
    n = sig.shape[0] + refsig.shape[0]

    # Convert signals to frequency domain
    sig_fft = np.fft.rfft(sig, n=n)
    refsig_fft = np.fft.rfft(refsig, n=n)

    # Calculate the cross-power spectrum (correlation in frequency)
    cross_power = sig_fft * np.conj(refsig_fft)

    # Normalize by magnitude (PHAT): This removes amplitude influence and
    # focuses solely on the phase difference to get a sharp correlation peak.
    cc = np.fft.irfft(cross_power / (np.abs(cross_power) + EPSILON), n=(interp * n))

    # Determine the search range for the delay
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    # Center the correlation result so zero-lag is at the middle
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # Find the peak index: This tells us how much 'refsig' is offset from 'sig'
    shift = np.argmax(np.abs(cc)) - max_shift

    # Convert index shift into time delay in seconds
    tau = shift / float(interp * fs)

    return tau
