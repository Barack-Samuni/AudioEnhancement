import files_handler as io_loader
import matplotlib.pyplot as plt
import numpy as np

from src import utils as ut


def test_envelope():
    noise_file, project_root, signal_file = io_loader.load_data()
    noise, fs_noise = io_loader.load_sound(noise_file[0])
    t = np.linspace(0, len(noise) / fs_noise, len(noise))
    # envelope = ut.get_sig_envelope(noise, fs_noise, 5)
    # envelope = ut.get_peak_envelope(noise, fs_noise,attack_ms=2,release_ms=100)
    envelope = ut.maximum_filter_env(noise, 200)
    envelope = ut.butter_filter(data=envelope, cutoff=15, fs=fs_noise, btype="lowpass")
    normalized_envelope = ut.normalize_sig(envelope)
    normalized_noise = ut.normalize_sig(noise)

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
