import sys
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
import torch

# 1. Spectral Analysis Parameters
# WIN_DUR: The length of the analysis window in seconds (64ms)
# HOP_FRAC: The step size between windows as a fraction of window length
# EPSILON: constant to prevent division by 0
WIN_DUR = 0.064
HOP_FRAC = 0.2
EPSILON = sys.float_info.epsilon
IMPULSE_RESPONSE_LENGTH = 20

# Utility functions

FilterType = Literal[
    "bandpass",
    "lowpass",
    "highpass",
    "bandstop",
    "band",
    "pass",
    "bp",
    "bands",
    "stop",
    "bs",
    "low",
    "lp",
    "l",
    "high",
    "hp",
    "h",
]


def lin2dB(sig: np.ndarray, Power: bool = False) -> np.ndarray:
    """
    Converts a linear signal to the Decibel (dB) scale.

    Parameters:
    -----------
    sig:   The input signal (Amplitude or Power).
    Power: If True, uses 10*log10 (for Power/Energy).
           If False, uses 20*log10 (for Amplitude/Voltage).
    """
    # 1. Take the absolute value to ensure we don't log a negative number
    # 2. Add EPSILON to prevent log(0) which returns -inf
    if Power:
        # P_dB = 10 * log10(P_linear)
        return 10 * np.log10(np.abs(sig) + EPSILON)
    else:
        # A_dB = 20 * log10(A_linear)
        return 20 * np.log10(np.abs(sig) + EPSILON)


def dB2lin(db_sig: np.ndarray, Power: bool = False) -> np.ndarray:
    """
    Converts a Decibel (dB) value back to the linear scale.

    Parameters:
    -----------
    db_sig: The input signal in dB.
    Power:  If True, uses 10^(dB/10).
            If False, uses 10^(dB/20).
    """
    if Power:
        # P_linear = 10^(P_dB / 10)
        return 10 ** (db_sig / 10.0)
    else:
        # A_linear = 10^(A_dB / 20)
        return 10 ** (db_sig / 20.0)


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
        sig = butter_filter(data=sig, cutoff=fn, fs=fs_old, btype="low")

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
        return f1, t1, lin2dB(sig, False)

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


def plot_stft(
    stft: np.ndarray,
    t: Optional[np.ndarray] = None,
    f: Optional[np.ndarray] = None,
    mode: str = "dB",
    vmin: float = -90,
    vmax: float = -20,
    title: str = "",
) -> None:
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
        plot_data = lin2dB(stft, False)
    else:
        plot_data = stft

    # Define the coordinates for the x (time) and y (frequency) axes
    if t is not None and f is not None:
        extent = [t[0], t[-1], f[0], f[-1]]
    else:
        extent = None

    ax.imshow(
        plot_data,
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=vmin if mode == "dB" else None,
        vmax=vmax if mode == "dB" else None,
        extent=extent,
    )

    ax.set_ylabel("Frequency" + (" (Hz)" if f is not None else " (Index)"))
    ax.set_xlabel("Time" + (" (sec)" if t is not None else " (Index)"))
    ax.set_title(f"STFT {title} , ({mode.capitalize()})")
    plt.show()


def coherence_of_sigs(
    sig: np.ndarray, noise: np.ndarray, fs: int, plot_coher: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates and optionally plots the Magnitude Squared Coherence between two signals.

    Coherence (Cxy) is a function of frequency with values between 0 and 1.
    - 1.0: Perfect linear relationship (The filter can perfectly cancel this).
    - 0.0: No relationship (The filter will fail to remove this noise).

    Parameters:
    -----------
    sig:         The primary signal (Speech + Noise).
    noise:       The reference noise signal.
    fs:          Sampling frequency (usually 16000).
    plot_coher:  If True, displays the coherence plot.

    Returns:
    --------
    f (ndarray):   Frequency vector.
    Cxy (ndarray): Coherence values for each frequency.
    """
    n_overlap, window_size = stft_params_calc(fs)
    f, cxy = sg.coherence(sig, noise, fs=fs, noverlap=n_overlap, nperseg=window_size)
    if plot_coher:
        plt.plot(f, cxy)
        plt.xlabel("frequency [Hz]")
        plt.ylabel("Coherence")
        plt.show(block=True)
    return f, cxy


# 2. Matching and Correlation


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
        sig = np.pad(sig, (0, diff), mode="constant")
    elif diff < 0:
        # If 'sig' is longer, cut the extra samples
        sig = sig[: len(ref)]

    return ref, sig


def adjust_min_length(noise: np.ndarray, total_sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Sincronizes two signals by cropping both to the length of the shorter one.
    This is essential for element-wise operations like subtraction in ANC.

    Parameters:
    -----------
    noise (np.ndarray):     The noise reference signal array.
    total_sig (np.ndarray): The primary signal array (speech + noise).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, int]:
        - The cropped noise signal.
        - The cropped primary signal.
        - The new common length (min_len).
    """
    min_len = min(len(total_sig), len(noise))
    noise = noise[:min_len]
    total_sig = total_sig[:min_len]
    return noise, total_sig, min_len


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
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    # Find the peak index: This tells us how much 'refsig' is offset from 'sig'
    shift = np.argmax(np.abs(cc)) - max_shift

    # Convert index shift into time delay in seconds
    tau = shift / float(interp * fs)

    return tau


def adjusting_delays(sig_to_adjust: np.ndarray, sig_source: np.ndarray, tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns two signals in time by shifting one relative to the other based on a calculated delay.

    This function takes a signal that needs shifting (sig_to_adjust) and pads it with zeros
    at the beginning to match the timing of a reference source (sig_source).

    Args:
        sig_to_adjust (np.ndarray): The signal to be delayed (usually the reference noise).
        sig_source (np.ndarray):    The anchor signal (usually the primary noisy microphone).
        tau (int):                 The delay amount in samples (calculated via GCC-PHAT).

    Returns:
        Tuple[np.ndarray, np.ndarray]: The synchronized (adjusted) signal and the source signal.
    """
    sig_source = torch.from_numpy(sig_source).float()
    sig_to_adjust = torch.from_numpy(sig_to_adjust).float()
    tau_samples = int(tau)
    sig_to_adjust = torch.cat([torch.zeros(int(tau_samples)), sig_to_adjust])[: sig_source.shape[-1]]
    sig_to_adjust = sig_to_adjust.numpy()
    sig_source = sig_source.numpy()
    return sig_to_adjust, sig_source


# 3. Transfer functions processes


def distortion_ir(noise: np.ndarray) -> np.ndarray:
    """
    Simulates room acoustic distortion by applying an Exponential Decay Impulse Response.

    This function models how sound reverberates in a room by applying a simple
    exponentially decaying impulse response filter. This simulates the effect of
    reflections and acoustic distortion that occurs when noise travels through a
    physical space before being captured by a microphone.

    The impulse response uses:
    - Exponential decay over 20 samples to simulate reflection damping
    - Random phase shifts (+1 or -1) to model complex reflection patterns
    - Normalization to prevent clipping

    Args:
        noise: Input noise signal as numpy array

    Returns:
        Distorted noise signal with simulated room acoustics

    Note:
        This is used to test ANC algorithm robustness against acoustic path mismatch
        between reference and primary microphones in real-world scenarios.
    """
    # Generate exponentially decaying impulse response with random phase
    room_ir = np.exp(-np.linspace(0, 1, IMPULSE_RESPONSE_LENGTH)) * np.random.choice([1, -1], IMPULSE_RESPONSE_LENGTH)

    # Apply room impulse response filter
    noise = sg.lfilter(room_ir, [1], noise)

    # Normalize to prevent clipping
    noise = normalize_sig(noise)

    return noise


def normalize_sig(sig):
    """
    Performs Peak Normalization on a digital audio signal.

    This process finds the highest absolute value in the signal and scales
    every sample so that the peak reaches exactly 1.0 (or -1.0).

    Parameters:
    -----------
    sig (np.ndarray): The input audio signal array. Can be a raw signal
                      from a WAV file or the output of an ANC filter
                      (like NLMS, RLS, or NKF).

    Returns:
    --------
    np.ndarray: The normalized signal where all values are within the
                range [-1.0, 1.0].
    """
    return sig / np.max(np.abs(sig))
