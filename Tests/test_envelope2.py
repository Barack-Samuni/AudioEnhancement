import random
from itertools import product
from typing import Any, Tuple

import files_handler as io_loader
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import utils as ut

# Define the grid of parameters to be tested:
# WINDOW_SIZES: The number of samples for the sliding max filter (50 to 450)
# LPF_CUTOFFS: The frequency threshold in Hz for the Butterworth low-pass filter (10 to 45)
WINDOW_SIZES = np.arange(start=50, stop=500, step=50)
LPF_CUTOFFS = np.arange(start=10, stop=50, step=5)


@pytest.fixture(scope="session")
def load_data():
    """
    Session-scoped fixture: Loads the audio file once per test session.
    - select_audio_files: Opens a file dialog for the user.
    - load_sound: Reads the audio data and returns the signal (noise) and sample rate (fs).
    """
    print("\n[Disk IO] Loading master audio file...")
    noise_file = io_loader.select_audio_files()
    noise, fs_noise = io_loader.load_sound(noise_file[0])
    return {"noise": noise, "fs": fs_noise}


@pytest.mark.parametrize(
    "window_size, cutoff",
    [(w, c) for w, c in product(WINDOW_SIZES, LPF_CUTOFFS)],
    # Generates descriptive names for each test case in the pytest runner
    ids=[f"Window_{w}samples,Cutoff_{c}Hz" for w, c in product(WINDOW_SIZES, LPF_CUTOFFS)],
)
def test_env(window_size, cutoff, load_data):
    """
    Core Test Suite: Validates the envelope extraction algorithm.
    Steps:
    1. Extract data from the fixture.
    2. Process the noise to get raw and normalized envelopes via get_env.
    3. Run a battery of validation functions (positivity, bounds, logic, etc.).
    4. Generate visual plots for debugging.
    """
    noise = load_data["noise"]
    fs_noise = load_data["fs"]

    # Process the signal and handle time offsets
    envelope, noise, normalized_envelope, normalized_noise, t = get_env(cutoff, fs_noise, noise, window_size)

    # Validate that all envelope values are >= 0
    check_positivity(envelope=envelope, window_size=window_size, cutoff=cutoff)

    # Validate that normalized signals stay within the [0, 1] range
    check_normalization(sig=normalized_envelope, label="Envelope")

    # Validate that the envelope physically 'contains' the signal (with tolerance)
    check_envelope_bounds(envelope=envelope, window_size=window_size, cutoff=cutoff, noise=noise)

    # Validate that the envelope is smoother (lower variation) than the raw noise
    check_envelope_smoothness(envelope=envelope, window_size=window_size, cutoff=cutoff, noise=noise)

    # Mathematically verify the max-filter logic at a random index
    check_max_filter_logic(envelope=envelope, window_size=window_size, cutoff=cutoff, noise=noise)

    # Render plots for visual verification
    check_visual_env(
        t=t,
        noise=noise,
        envelope=envelope,
        normalized_noise=normalized_noise,
        normalized_envelope=normalized_envelope,
        window_size=window_size,
        lpf_cutoff=cutoff,
        fs_noise=fs_noise,
    )


def get_env(cutoff: int, fs_noise: int, noise: np.ndarray, window_size: int) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Signal Processing Pipeline:
    - Generates time vector 't'.
    - Applies sliding maximum filter for initial envelope detection.
    - Applies Butterworth Low-Pass Filter (LPF) to smooth the envelope.
    - Normalizes signals for comparison.
    - Validates data integrity (None checks, shape checks).
    - Synchronizes signal offsets to account for filter group delays.
    """
    # Create the time axis based on sample rate and signal length
    t = np.linspace(0, len(noise) / fs_noise, len(noise))

    # Step 1: Maximum Filter - extracts peaks over a window
    envelope = ut.maximum_filter_env(noise, window_size)

    # Step 2: Low-Pass Filter - removes high frequency noise from the envelope
    envelope = ut.butter_filter(data=envelope, cutoff=cutoff, fs=fs_noise, btype="lowpass")

    # Step 3: Normalization - scales signals between 0 and 1
    normalized_envelope = ut.normalize_sig(envelope)
    normalized_noise = ut.normalize_sig(noise)

    # Internal Integrity Check: Ensure objects are not None
    check_none_vals(envelope=envelope, window_size=window_size, cutoff=cutoff)

    # Internal Integrity Check: Ensure input and output array lengths match
    check_array_shapes(envelope=envelope, noise=noise)

    # Offset Alignment: Compensates for processing delays to ensure t and signals align
    envelope = ut.optimal_offsets(envelope, fs_noise)
    normalized_envelope = ut.optimal_offsets(normalized_envelope, fs_noise)
    normalized_noise = ut.optimal_offsets(normalized_noise, fs_noise)
    noise = ut.optimal_offsets(noise, fs_noise)
    t = ut.optimal_offsets(t, fs_noise)

    return envelope, noise, normalized_envelope, normalized_noise, t


# --- Validation Functions (English Documentation) ---


def check_none_vals(envelope, window_size, cutoff):
    """Ensure the generated envelope is not None."""
    assert envelope is not None, f"Envelope is None for params: {(window_size, cutoff)}"


def check_array_shapes(envelope, noise):
    """Check: Output consistency. Envelope length must match source signal length."""
    assert len(envelope) == len(noise)


def check_positivity(envelope, window_size, cutoff):
    """Verify that the envelope values remain non-negative."""
    assert np.all(envelope >= 0 - ut.EPSILON), f"Envelope should be positive for params: {(window_size, cutoff)}"


def check_normalization(sig, label):
    """Ensure the signal values stay within the [0, 1] range (plus epsilon)."""
    assert np.max(sig) <= 1.0 + ut.EPSILON, f"{label} max exceeds 1.0"
    assert np.min(sig) >= 0.0 - ut.EPSILON, f"{label} min below 0.0"


def check_envelope_bounds(envelope, window_size, cutoff, noise):
    """
    Validation: Physical Boundary Constraint.
    The envelope extracted by a maximum filter at least will be greater than
     or equal to 80% absolute value of the source signal.
    """
    # Using 0.80 multiplier as per user logic to allow for smoothing tolerance
    assert np.all(
        envelope >= (np.abs(noise) * 0.80)
    ), f"Envelope dropped below the absolute signal magnitude for params: {(window_size, cutoff)}"


def check_envelope_smoothness(envelope, window_size, cutoff, noise):
    """
    Validation: Signal Complexity Reduction.
    The 'Total Variation' of the envelope should be lower than the raw signal.
    """
    sig_variation = np.sum(np.abs(np.diff(noise)))
    env_variation = np.sum(np.abs(np.diff(envelope)))
    assert (
        env_variation < sig_variation
    ), f"Envelope ({env_variation}) is noisier than the signal ({sig_variation}) for params: {(window_size, cutoff)}."


def check_max_filter_logic(envelope, window_size, cutoff, noise):
    """
    Validation: Mathematical Definition Check.
    Verifies a random point to ensure envelope matches local maximum absolute value.
    """
    idx = random.randint(window_size, len(envelope) - window_size)
    half_w = window_size // 2
    local_segment = np.abs(noise[idx - half_w : idx + half_w])
    expected_max = np.max(local_segment)

    # Comparing envelope at idx with expected max using relative tolerance
    assert envelope[idx] == pytest.approx(
        expected_max, rel=0.15
    ), f"Envelope for params: {(window_size, cutoff)}  at index {idx} does not match the local maximum."


def check_visual_env(t, noise, envelope, normalized_noise, normalized_envelope, window_size, lpf_cutoff, fs_noise):
    """Generate diagnostic plots for visual inspection of the envelope results."""
    plt.figure(figsize=(15, 5))
    t_prev = np.minimum(10, int(np.max(t)))
    samps = ut.time_to_indices(fs_noise, t_prev)

    # Plot 1: Raw Noise and the Envelope
    plt.subplot(1, 3, 1)
    plt.plot(t[:samps], noise[:samps], label="Noise", alpha=0.5)
    plt.plot(t[:samps], envelope[:samps], label="Envelope", linewidth=2, color="red")
    plt.title(f"Original Noise & Envelope {window_size}samples, {lpf_cutoff}hz")
    plt.legend()

    # Plot 2: The Normalized Envelope (Zoomed)
    plt.subplot(1, 3, 2)
    plt.plot(t[:samps], normalized_envelope[:samps], color="orange")
    plt.title(f"Normalized Envelope ({t_prev})")

    # Plot 3: The Full Normalized Envelope
    plt.subplot(1, 3, 3)
    plt.plot(t, normalized_noise, label="Noise", alpha=0.5)
    plt.plot(t, normalized_envelope, color="green")
    plt.title("Full Normalized Envelope")

    plt.tight_layout()
    plt.show()
