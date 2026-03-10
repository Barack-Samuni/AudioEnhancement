import files_handler as io_loader
import matplotlib.pyplot as plt
import numpy as np

from src import utils as ut


def test_envelope():
    noise_file, project_root, signal_file = io_loader.load_data()
    noise, fs_noise = io_loader.load_sound(noise_file[0])
    t = np.linspace(0, len(noise) / fs_noise, len(noise))
    # A 50Hz sine wave with amplitude 1.0

    envelope = ut.get_sig_envelope(noise, fs_noise, 15)
    normalized_envelope = ut.normalize_sig(envelope)
    plt.plot(t[: 10 * fs_noise], noise[: 10 * fs_noise])
    plt.plot(t[: 10 * fs_noise], envelope[: 10 * fs_noise])
    plt.figure()
    plt.plot(t[: 10 * fs_noise], normalized_envelope[: 10 * fs_noise])
    plt.show()
