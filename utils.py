import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg

# 1. Spectral Params
WIN_DUR = 0.064
HOP_FRAC = 0.2
EPSILON = 1e-10

# 2. Utils functions

def butter_filter(data, cutoff, fs, btype, order=5):
    """
    data:   The signal you want to filter
    cutoff: The critical frequency (Hz)
    fs:     The sampling rate of your data (Hz)
    btype:  'low' or 'high'
    order:  The polynomial order (higher = steeper roll-off)
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq

    # Get filter coefficients
    b, a = sg.butter(order, normal_cutoff, btype=btype, analog=False)

    # Apply filter
    y = sg.filtfilt(b, a, data)
    return y

def resample_fs(sig,fs_old,fs_new):
    if fs_old==fs_new:
        return sig,fs_old
    if fs_new<fs_old:
        fn=fs_new/2
        sig=butter_filter(data=sig,cutoff=fn,fs=fs_old,btype='low')
    num_samples = int(len(sig) * fs_new / fs_old)
    resampled_sig = sg.resample(sig, num_samples)
    return resampled_sig,fs_new

def calc_stft(sig, fs, mode="linear"):
    n_overlap, window_size = stft_params_calc(fs)
    f1, t1, sig_stft = sg.stft(x=sig, fs=fs, noverlap=n_overlap, nperseg=window_size)
    stft_abs = np.abs(sig_stft)

    if mode == "linear":
        return f1, t1, stft_abs
    elif mode == "dB":  # Added the missing condition here
        # Adding a small epsilon (1e-10) prevents log(0) errors
        return f1, t1, 20 * np.log10(stft_abs + EPSILON)
    else:  # Added the missing colon here
        return f1, t1, sig_stft


def stft_params_calc(fs) -> tuple[int, int]:
    window_size = int(fs * WIN_DUR)
    n_overlap = int(((WIN_DUR * fs) * (1 - HOP_FRAC)))
    return n_overlap, window_size


def plot_stft(stft, ax, t=None, f=None, mode="dB", vmin=-90, vmax=-20):
    """
    psd:  The STFT magnitude or power array
    ax:   The matplotlib axis to plot on
    t, f: Optional time and frequency arrays
    mode: "log" (dB) or "linear"
    """
    # 1. Determine the Plotting Data
    if mode == "dB":
        # Add epsilon to avoid log(0)
        plot_data = 20 * np.log10(stft + EPSILON)
        label = "Magnitude (dB)"
    else:
        plot_data = stft
        label = "Magnitude (Linear)"

    # 2. Handle optional t and f
    if t is not None and f is not None:
        extent = [t[0], t[-1], f[0], f[-1]]
    else:
        extent = None # Defaults to pixel indices

    # 3. Create the image
    img = ax.imshow(
        plot_data,
        origin='lower',
        aspect='auto',
        cmap='inferno',
        vmin=vmin if mode == "log" else None,
        vmax=vmax if mode == "log" else None,
        extent=extent
    )

    # 4. Labeling logic
    ax.set_ylabel('Frequency' + (' (Hz)' if f is not None else ' (Index)'))
    ax.set_xlabel('Time' + (' (sec)' if t is not None else ' (Index)'))
    ax.set_title(f"STFT Spectrogram ({mode.capitalize()})")
    plt.show()


def coherence_of_sigs(sig,noise,fs):
    n_overlap, window_size = stft_params_calc(fs)
    f, Cxy = sg.coherence(sig, noise, fs=fs,noverlap=n_overlap, nperseg=window_size)
    plt.plot(f, Cxy)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.show(block=True)

def match_sigs(ref: np.ndarray, sig: np.ndarray):
    """
    Adjusts 'sig' to match the length of 'ref' by padding with zeros
    or cropping from the end.
    """
    diff = len(ref) - len(sig)

    if diff > 0:
        # Case: sig is shorter than ref -> Pad with zeros at the end
        # (0, diff) means 0 padding at start, 'diff' padding at end
        sig = np.pad(sig, (0, diff), mode='constant')
    elif diff < 0:
        # Case: sig is longer than ref -> Crop the end
        sig = sig[:len(ref)]

    return ref, sig

import torch
import numpy as np


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    Code src: https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py

    ref-the one signal we want to corr with sig
    sig - the trusted signal that we want to corr on
    '''

    if torch.is_tensor(sig):
        sig = sig.numpy()
    if torch.is_tensor(refsig):
        refsig = refsig.numpy()

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / (np.abs(R)+1e-15), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau

