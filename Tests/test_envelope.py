import random
from typing import Tuple

import files_handler as io_loader
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy import ndarray

from src import utils as ut


def test_main_checks():
    maximum_flit_window = np.arange(start=50, stop=500, step=50)
    cut_lpf = np.arange(start=10, stop=50, step=5)
    maximum_flit_check(cut_lpf[1], maximum_flit_window)
    # lpf_checks(cut_lpf,maximum_flit_window=)


def maximum_flit_check(cut_lpf: int, maximum_flit_window: np.ndarray) -> None:
    for i in range(len(maximum_flit_window)):
        envelope, fs_noise, noise, normalized_envelope, normalized_noise, t = get_envelope(
            maximum_flit_window=maximum_flit_window[i], cut_lpf=cut_lpf
        )
        tests_envs(
            envelope=envelope,
            fs_noise=fs_noise,
            maximum_flit_window=maximum_flit_window[i],
            noise=noise,
            normalized_envelope=normalized_envelope,
            normalized_noise=normalized_noise,
            t=t,
        )


def lpf_checks(cut_lpf: np.ndarray, maximum_flit_window: int) -> None:
    for i in range(len(cut_lpf)):
        envelope, fs_noise, noise, normalized_envelope, normalized_noise, t = get_envelope(
            maximum_flit_window=maximum_flit_window, cut_lpf=cut_lpf[i]
        )
        tests_envs(
            envelope=envelope,
            fs_noise=fs_noise,
            maximum_flit_window=maximum_flit_window,
            noise=noise,
            normalized_envelope=normalized_envelope,
            normalized_noise=normalized_noise,
            t=t,
        )


def tests_envs(
    envelope: ndarray,
    fs_noise: int,
    maximum_flit_window: int,
    noise: ndarray,
    normalized_envelope: ndarray,
    normalized_noise: ndarray,
    t: ndarray,
):
    test_positivity(envelope=envelope)
    test_envelope_bounds(noise=noise, envelope=envelope)
    test_envelope_smoothness(noise=noise, envelope=envelope)
    test_max_filter_logic(noise=noise, envelope=envelope, maximum_flit_window=maximum_flit_window)
    test_visual_env(envelope, fs_noise, noise, normalized_envelope, normalized_noise, t)


def test_visual_env(
    envelope: ndarray, fs_noise: int, noise: ndarray, normalized_envelope: ndarray, normalized_noise, t
) -> None:
    # Create a figure with a consistent 1x3 grid
    plt.figure(figsize=(15, 5))

    # Plot 1: Raw Noise and the Envelope
    plt.subplot(1, 3, 1)
    plt.plot(t[: 10 * fs_noise], noise[: 10 * fs_noise], label="Noise", alpha=0.5)
    plt.plot(t[: 10 * fs_noise], envelope[: 10 * fs_noise], label="Envelope", linewidth=2, color="red")
    plt.title("Original Noise & Envelope")
    plt.legend()

    # Plot 2: The Normalized Envelope (Zoomed)
    plt.subplot(1, 3, 2)
    plt.plot(t[: 10 * fs_noise], normalized_envelope[: 10 * fs_noise], color="orange")
    plt.title("Normalized Envelope (10s)")

    # Plot 3: The Full Normalized Envelope
    plt.subplot(1, 3, 3)
    plt.plot(t, normalized_noise, label="Noise", alpha=0.5)
    plt.plot(t, normalized_envelope, color="green")
    plt.title("Full Normalized Envelope")

    plt.tight_layout()  # Prevents labels from overlapping
    plt.show()


def get_envelope(maximum_flit_window: int, cut_lpf: int) -> Tuple[ndarray, int, ndarray, ndarray, ndarray, ndarray]:
    noise_file, project_root, signal_file = io_loader.load_data()
    noise, fs_noise = io_loader.load_sound(noise_file[0])
    t = np.linspace(0, len(noise) / fs_noise, len(noise))
    envelope = ut.maximum_filter_env(noise, maximum_flit_window)
    envelope = ut.butter_filter(data=envelope, cutoff=cut_lpf, fs=fs_noise, btype="lowpass")
    normalized_envelope = ut.normalize_sig(envelope)
    normalized_noise = ut.normalize_sig(noise)
    return envelope, fs_noise, noise, normalized_envelope, normalized_noise, t


def test_positivity(envelope):
    assert np.all(envelope >= 0), "Envelope should be positive."


def test_envelope_bounds(noise, envelope):
    """
    Validation: Physical Boundary Constraint.
    The envelope extracted by a maximum filter must always be greater than
    or equal to the absolute value of the source signal.
    """
    # We subtract a tiny EPSILON to prevent test failure due to
    # microscopic floating-point rounding errors.
    assert np.all(envelope >= np.abs(noise) - ut.EPSILON), "Envelope dropped below the absolute signal magnitude."


def test_envelope_smoothness(noise, envelope):
    """
    Validation: Signal Complexity Reduction.
    The 'Total Variation' (sum of absolute differences) of the envelope
    should be lower than the raw signal because the envelope ignores
    high-frequency oscillations.
    """
    sig_diff = np.sum(np.abs(np.diff(noise)))
    env_diff = np.sum(np.abs(np.diff(envelope)))

    # A valid envelope 'strips away' the carrier wave,
    # resulting in a much shorter total path length.
    assert env_diff < sig_diff, f"Envelope ({env_diff}) is noisier than the signal ({sig_diff})."


def test_max_filter_logic(noise, envelope, maximum_flit_window):
    """
    Validation: Mathematical Definition Check.
    Picks a random point and manually verifies that the envelope value
    matches the maximum absolute value within the sliding window.
    """
    # Select a random index, avoiding edges where the window might
    # behave differently (padding/truncation).
    idx = random.randint(maximum_flit_window, len(noise) - maximum_flit_window)

    # Replicate the max-filter logic: Look at the raw signal values
    # within the window centered at our random index.
    half_w = maximum_flit_window // 2
    window_slice = np.abs(noise[idx - half_w : idx + half_w])
    expected_max = np.max(window_slice)

    # Ensure the calculated envelope matches our manual 'truth'
    # for this specific point.
    assert envelope[idx] == pytest.approx(expected_max), f"Envelope at index {idx} does not match the local maximum."
