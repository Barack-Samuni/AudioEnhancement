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
EPSILON = 1e-15


# 2. Utility Functions

def butter_filter(data, cutoff, fs, btype, order=5):
    """
    Applies a Butterworth filter using zero-phase filtering to preserve timing.

    Args:
        data: The input audio signal array.
        cutoff: The filter frequency in Hz.
        fs: Sampling rate of the input data.
        btype: 'low' for low-pass or 'high' for high-pass.
        order: The steepness of the filter roll-off.
    """
    nyq = 0.5 * fs  # Nyquist Frequency (highest representable frequency)
    normal_cutoff = cutoff / nyq

    # Generate the filter coefficients (numerator 'b' and denominator 'a')
    b, a = sg.butter(order, normal_cutoff, btype=btype, analog=False)

    # Use filtfilt to apply the filter forward and backward, resulting in zero phase distortion
    y = sg.filtfilt(b, a, data)
    return y


def resample_fs(sig, fs_old, fs_new):
    """
    Changes the sampling rate of a signal.
    If downsampling, it applies a low-pass filter first to prevent aliasing.
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


def calc_stft(sig, fs, mode="linear"):
    """
    Computes the Short-Time Fourier Transform to analyze signal frequency over time.
    """
    n_overlap, window_size = stft_params_calc(fs)
    f1, t1, sig_stft = sg.stft(x=sig, fs=fs, noverlap=n_overlap, nperseg=window_size)
    stft_abs = np.abs(sig_stft)

    if mode == "linear":
        return f1, t1, stft_abs
    elif mode == "dB":
        # Convert to logarithmic scale (Decibels) for better visualization of quiet sounds
        return f1, t1, 20 * np.log10(stft_abs + EPSILON)
    else:
        # Return complex numbers for processing/reconstruction
        return f1, t1, sig_stft


def stft_params_calc(fs) -> tuple[int, int]:
    """
    Helper to convert time durations (seconds) into discrete samples based on fs.
    """
    window_size = int(fs * WIN_DUR)
    n_overlap = int(((WIN_DUR * fs) * (1 - HOP_FRAC)))
    return n_overlap, window_size


def plot_stft(stft,t=None, f=None, mode="dB", vmin=-90, vmax=-20,title=""):
    """
    Plots a spectrogram heatmap on the provided Matplotlib axis.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    if mode == "dB":
        plot_data = 20 * np.log10(stft + EPSILON)
    else:
        plot_data = stft

    # Define the coordinates for the x (time) and y (frequency) axes
    if t is not None and f is not None:
        extent = [t[0], t[-1], f[0], f[-1]]
    else:
        extent = None

    img = ax.imshow(
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


def coherence_of_sigs(sig, noise, fs):
    """
    Calculates the Coherence (linear relationship) between two signals across frequencies.
    Values near 1.0 indicate high correlation, crucial for effective ANC.
    """
    n_overlap, window_size = stft_params_calc(fs)
    f, Cxy = sg.coherence(sig, noise, fs=fs, noverlap=n_overlap, nperseg=window_size)
    plt.plot(f, Cxy)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.show(block=True)


def match_sigs(ref: np.ndarray, sig: np.ndarray):
    """
    Ensures 'sig' is exactly the same length as 'ref' by padding or cropping.
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

    LOGIC:
    - sig: The primary/trusted signal (Stationary reference).
    - refsig: The secondary/reference noise signal (The one we "move").

    This function calculates the time delay 'tau' required to align 'refsig' to 'sig'.
    By applying 'tau' to 'refsig', we ensure that the noise in both channels is
    perfectly in-phase before adaptive filtering.
    """
    # Ensure data is in NumPy format for FFT processing
    if torch.is_tensor(sig):
        sig = sig.numpy()
    if torch.is_tensor(refsig):
        refsig = refsig.numpy()

    # Define FFT length to avoid circular overlap artifacts
    n = sig.shape[0] + refsig.shape[0]

    # Convert signals to frequency domain
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    # Calculate the cross-power spectrum (correlation in frequency)
    R = SIG * np.conj(REFSIG)

    # Normalize by magnitude (PHAT): This removes amplitude influence and
    # focuses solely on the phase difference to get a sharp correlation peak.
    cc = np.fft.irfft(R / (np.abs(R) + EPSILON), n=(interp * n))

    # Determine the search range for the delay
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    # Center the correlation result so zero-lag is at the middle
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # Find the peak index: This tells us how much 'refsig' is offset from 'sig'
    shift = np.argmax(np.abs(cc)) - max_shift

    # Convert index shift into time delay in seconds
    tau = shift / float(interp * fs)

    return tau