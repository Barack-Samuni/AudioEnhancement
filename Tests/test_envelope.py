import random
from typing import Any, Dict

import files_handler as io_loader
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import utils as ut

WINDOW_SIZES = np.arange(start=50, stop=500, step=50)
LPF_CUTOFFS = np.arange(start=10, stop=50, step=5)


@pytest.fixture(scope="session")
def raw_audio():
    """Loads the sound file from disk ONLY ONCE for the entire run."""
    print("\n[Disk IO] Loading master audio file...")
    noise_file = io_loader.select_audio_files()
    noise, fs = io_loader.load_sound(noise_file[0])
    return {"noise": noise, "fs": fs}


# --- 3. Module Level: Parameters ---
# We set scope="module" here to allow env_data (also module-scoped) to access them.
@pytest.fixture(scope="module", params=WINDOW_SIZES, ids=[f"Win_{w}" for w in WINDOW_SIZES])
def window_size(request):
    return request.param


@pytest.fixture(scope="module", params=LPF_CUTOFFS, ids=[f"LPF_{c}Hz" for c in LPF_CUTOFFS])
def lpf_cutoff(request):
    return request.param


@pytest.fixture(scope="module")
def env_data(raw_audio, window_size, lpf_cutoff) -> Dict[str, Any]:
    # We pass the ALREADY LOADED raw_audio dictionary into the logic function
    envelope, fs, noise, norm_env, norm_noise, t = get_envelope_logic(
        raw_audio=raw_audio,  # Pass it here!
        maximum_flit_window=window_size,
        cut_lpf=lpf_cutoff,
    )

    return {
        "envelope": envelope,
        "noise": noise,
        "fs": fs,
        "norm_env": norm_env,
        "norm_noise": norm_noise,
        "t": t,
        "window": window_size,
        "cutoff": lpf_cutoff,
    }


def get_envelope_logic(raw_audio: dict, maximum_flit_window: int, cut_lpf: int):
    """
    Accepts the preloaded raw_audio dictionary.
    """
    # Now raw_audio is a dictionary, so this works:
    noise = raw_audio["noise"]
    fs_noise = raw_audio["fs"]

    t = np.linspace(0, len(noise) / fs_noise, len(noise))

    # Apply Maximum Filter
    envelope = ut.maximum_filter_env(noise, maximum_flit_window)

    # Apply Butterworth Low-Pass Filter
    envelope = ut.butter_filter(data=envelope, cutoff=cut_lpf, fs=fs_noise, btype="lowpass")

    # Normalize signals
    normalized_envelope = ut.normalize_sig(envelope)
    normalized_noise = ut.normalize_sig(noise)

    return envelope, fs_noise, noise, normalized_envelope, normalized_noise, t


def test_positivity(env_data):
    assert np.all(env_data["envelope"] >= 0), "Envelope should be positive."


def test_envelope_bounds(env_data):
    """
    Validation: Physical Boundary Constraint.
    The envelope extracted by a maximum filter must always be greater than
    or equal to the absolute value of the source signal.
    """
    # We subtract a tiny EPSILON to prevent test failure due to
    # microscopic floating-point rounding errors.
    assert np.all(
        env_data["envelope"] >= np.abs(env_data["noise"]) - ut.EPSILON
    ), "Envelope dropped below the absolute signal magnitude."


def test_envelope_smoothness(env_data):
    """
    Validation: Signal Complexity Reduction.
    The 'Total Variation' (sum of absolute differences) of the envelope
    should be lower than the raw signal because the envelope ignores
    high-frequency oscillations.
    """
    sig_variation = np.sum(np.abs(np.diff(env_data["noise"])))
    env_variation = np.sum(np.abs(np.diff(env_data["envelope"])))
    # A valid envelope 'strips away' the carrier wave,
    # resulting in a much shorter total path length.
    assert env_variation < sig_variation, f"Envelope ({env_variation}) is noisier than the signal ({sig_variation})."


def test_max_filter_logic(env_data):
    """
    Validation: Mathematical Definition Check.
    Picks a random point and manually verifies that the envelope value
    matches the maximum absolute value within the sliding window.
    """
    # Select a random index, avoiding edges where the window might
    # behave differently (padding/truncation).
    noise = env_data["noise"]
    envelope = env_data["envelope"]
    maximum_flit_window = env_data["window"]
    idx = random.randint(maximum_flit_window, len(noise) - maximum_flit_window)

    # Replicate the max-filter logic: Look at the raw signal values
    # within the window centered at our random index.
    half_w = maximum_flit_window // 2
    local_segment = np.abs(noise[idx - half_w : idx + half_w])
    expected_max = np.max(local_segment)

    # Ensure the calculated envelope matches our manual 'truth'
    # for this specific point.
    assert envelope[idx] == pytest.approx(
        expected_max, rel=0.5
    ), f"Envelope at index {idx} does not match the local maximum."


def test_normalization_integrity(env_data):
    """Check: Signal normalization stays within [0, 1] range."""
    norm_env = env_data["norm_env"]
    assert np.max(norm_env) <= 1.0 + ut.EPSILON
    assert np.min(norm_env) >= 0.0 - ut.EPSILON


def test_array_shapes(env_data):
    """Check: Output consistency. Envelope length must match source signal length."""
    assert len(env_data["envelope"]) == len(env_data["noise"])


def test_visual_env(env_data) -> None:
    # Create a figure with a consistent 1x3 grid
    noise = env_data["noise"]
    envelope = env_data["envelope"]
    fs_noise = env_data["fs"]
    normalized_envelope = env_data["norm_env"]
    normalized_noise = env_data["norm_noise"]
    t = env_data["t"]

    plt.figure(figsize=(15, 5))

    # Plot 1: Raw Noise and the Envelope
    plt.subplot(1, 3, 1)
    plt.plot(t[: 10 * fs_noise], noise[: 10 * fs_noise], label="Noise", alpha=0.5)
    plt.plot(t[: 10 * fs_noise], envelope[: 10 * fs_noise], label="Envelope", linewidth=2, color="red")
    plt.title(f"Original Noise & Envelope {env_data['window']}smps, {env_data['cutoff']}hz")
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
