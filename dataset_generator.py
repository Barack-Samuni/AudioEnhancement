import os
import random
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def convolve_rir(signal, rir):
    """Fast convolution with truncation."""
    out = fftconvolve(signal, rir)
    return out[:len(signal)]


def match_length(noise, target_len):
    """Loop noise if shorter than speech."""
    if len(noise) < target_len:
        repeats = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeats)

    return noise[:target_len]


def compute_scaling(clean, noise, target_snr_db):
    """
    Compute exact scaling factor to achieve target SNR.
    """

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    target_noise_power = clean_power / (10 ** (target_snr_db / 10))

    scale = np.sqrt(target_noise_power / noise_power)

    return scale


# -------------------------------------------------
# Main dataset generator
# -------------------------------------------------

def generate_noisy_dataset(
    clean_dir,
    noise_dir,
    rir_dir,
    output_dir,
    snr_db=None,
    snr_range=(0, 20),
):

    os.makedirs(output_dir, exist_ok=True)

    clean_files = os.listdir(clean_dir)
    noise_files = os.listdir(noise_dir)
    rir_files = os.listdir(rir_dir)

    for f in tqdm(clean_files, desc="Generating mixtures"):

        clean_path = os.path.join(clean_dir, f)
        clean, sr = librosa.load(clean_path, sr=None)

        # random noise selection
        noise_file = random.choice(noise_files)
        noise, _ = librosa.load(os.path.join(noise_dir, noise_file), sr=sr)

        # random RIR selection
        rir_file = random.choice(rir_files)
        rir, _ = librosa.load(os.path.join(rir_dir, rir_file), sr=sr)

        # apply reverberation
        clean_rev = convolve_rir(clean, rir)
        noise_rev = convolve_rir(noise, rir)

        # match noise length
        noise_rev = match_length(noise_rev, len(clean_rev))

        # choose SNR
        if snr_db is not None:
            target_snr = snr_db
        else:
            target_snr = random.uniform(*snr_range)

        # scale noise
        scale = compute_scaling(clean_rev, noise_rev, target_snr)

        noise_scaled = noise_rev * scale

        mixture = clean_rev + noise_scaled

        out_name = os.path.splitext(f)[0] + "_noisy.wav"

        sf.write(
            os.path.join(output_dir, out_name),
            mixture,
            sr,
        )