from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import scipy.signal as sg
import torch

import IO as IO_LOADER
import utils as ut
from ANC.nkf import process_nkf
from ANC.NLMS import NLMS_calculation
from ANC.rls_filter import RLSFilter

# Constants
DEFAULT_SAMPLE_RATE = 16000
NLMS_FILTER_WINDOW = 1024
NLMS_MU = 0.1
RLS_N_TAPS = 64
IMPULSE_RESPONSE_LENGTH = 20
GCC_PHAT_ALIGNMENT_SECONDS = 10
ALIGNMENT_SAFETY_BUFFER = 0.001  # seconds


def show_spectrogram(sig: np.ndarray, fs: int, title: str = "Spectrogram") -> None:
    """
    Computes and displays the STFT of a signal in dB scale.
    Uses the utility functions from utils.py.
    """
    # 1. Calculate STFT in dB mode using your utils function
    # f: frequency bins, t: time bins, stft_db: magnitude in dB
    f, t, stft_db = ut.calc_stft(sig, fs, mode="dB")

    # 3. Use your utility plot function
    # We pass the dB data and specify mode="linear" to plot_stft
    # because we already converted it to dB in the calculation step.
    ut.plot_stft(stft_db, t=t, f=f, mode="", title=title)


def main() -> None:
    # 1. Select the files and initialize parameters
    alignment = False  # Flag to skip manual alignment if data is already correlated
    fs_resample, iteration, noise_files, project_root, signal_files = load_data()

    # Iterate through pairs of signal and noise reference files
    for s_path, n_path in zip(signal_files, noise_files):
        # Load audio data from selected paths
        sig, _fs_sig = IO_LOADER.load_sound(s_path)
        noise, fs_noise = IO_LOADER.load_sound(n_path)

        # Ensure signals are mono for processing
        sig = IO_LOADER.stereo_to_mono(sig)
        noise = IO_LOADER.stereo_to_mono(noise)

        # Resample both signals to the target processing frequency (usually 16kHz)
        resampled_sig, _ = ut.resample_fs(sig, fs_noise, fs_resample)
        resampled_noise, _ = ut.resample_fs(noise, fs_noise, fs_resample)

        # Perform Time Alignment if the alignment flag is False
        if not alignment:
            # Estimate delay using GCC-PHAT cross-correlation
            alignment_window = fs_resample * GCC_PHAT_ALIGNMENT_SECONDS
            tau = ut.gcc_phat(
                resampled_sig[:alignment_window], resampled_noise[:alignment_window], fs=fs_resample, interp=1
            )
            tau = max(0, int((tau - ALIGNMENT_SAFETY_BUFFER) * fs_resample))  # Convert to sample count

            # Use torch to shift the noise signal by padding with zeros
            resampled_sig = torch.from_numpy(resampled_sig).float()
            resampled_noise = torch.from_numpy(resampled_noise).float()
            tau_samples = int(tau)
            resampled_noise = torch.cat([torch.zeros(int(tau_samples)), resampled_noise])[: resampled_sig.shape[-1]]

            # Convert back to numpy and ensure lengths match exactly
            resampled_sig = resampled_sig.numpy()
            resampled_noise = resampled_noise.numpy()
            resampled_noise, resampled_sig = ut.match_sigs(resampled_noise, resampled_sig)

            # Save the newly aligned/correlated signals
            IO_LOADER.save_sound(rf"{project_root}\corr_noise.wav", resampled_noise, fs_resample)
            IO_LOADER.save_sound(rf"{project_root}\corr_sig.wav", resampled_sig, fs_resample)

        resampled_noise = distortion_ir(
            resampled_noise
        )  # NO MANDOTRY- APPLYS EXPONENTIAL TF TO THE NOISE AND NORMALIZE
        # Analyze initial coherence between noise and signal before filtering
        ut.coherence_of_sigs(resampled_sig, resampled_noise, fs_resample)

        # Run various ANC algorithms (NLMS, NKF, RLS)
        process_ancs(fs_resample, iteration, project_root, resampled_noise, resampled_sig)
        iteration += 1


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
    noise = noise / np.max(np.abs(noise))

    return noise


def get_results_dir(root_path: Path) -> Path:
    """
    Creates and returns a directory path for results based on the current date.
    """
    # %H: Hour (24-hour clock), %M: Minute
    today = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # 2. Define the path: ProjectRoot / results / 2026-03-04
    results_base = Path(root_path) / "results"
    daily_dir = results_base / today

    # 3. Create the directories (parents=True creates 'results' if it's missing)
    daily_dir.mkdir(parents=True, exist_ok=True)

    return daily_dir


def load_data() -> tuple[int, int, list[str], Path, list[str]]:
    """
    Handles file selection via UI and prepares project variables.
    """
    print("Please select the TOTAL signals (signal + noise):")
    signal_files = IO_LOADER.select_audio_files()

    print("Please select the NOISE reference signals:")
    noise_files = IO_LOADER.select_audio_files()

    project_root = Path(signal_files[0]).parent
    iteration = 0
    fs_resample = DEFAULT_SAMPLE_RATE

    # Validation: Ensure we have a reference for every signal
    if len(signal_files) != len(noise_files):
        raise IndexError(f"Mismatch: Found {len(signal_files)} signals but {len(noise_files)} noise files.")
    return fs_resample, iteration, noise_files, project_root, signal_files


def save_and_analyze_result(
    result: Union[np.ndarray, torch.Tensor],
    noise: np.ndarray,
    fs: int,
    results_dir: Path,
    filename: str,
    title: str,
) -> None:
    """
    Helper function to save, visualize, and analyze ANC results.

    Args:
        result: The processed signal
        noise: Reference noise signal for coherence analysis
        fs: Sample rate
        results_dir: Directory to save results
        filename: Output filename
        title: Title for spectrogram
    """
    if torch.is_tensor(result):
        result = result.detach().cpu().numpy()

    show_spectrogram(result, fs, title)
    output_path = results_dir / filename
    IO_LOADER.save_sound(str(output_path), result, fs)
    ut.coherence_of_sigs(result, noise, fs)


def process_ancs(
    fs_resample: int, iteration: int, project_root: Path, resampled_noise: np.ndarray, resampled_sig: np.ndarray
):
    """
    Runs the noise cancellation pipeline using different algorithms and saves results.
    """
    # Show input signals
    show_spectrogram(resampled_noise, fs_resample, "Noise microphone (less sensitive)")
    show_spectrogram(resampled_sig, fs_resample, "Signal + noise before enhancement (sensitive mic)")

    results_dir = get_results_dir(project_root)

    # --- NLMS Algorithm ---
    nlms_result = NLMS_calculation(
        total_sig=resampled_sig,
        noise=resampled_noise,
        fs1=fs_resample,
        fs2=fs_resample,
        fs_resample=DEFAULT_SAMPLE_RATE,
        filter_window=NLMS_FILTER_WINDOW,
        mu=NLMS_MU,
    )
    save_and_analyze_result(
        nlms_result,
        resampled_noise,
        fs_resample,
        results_dir,
        f"NLMS{iteration}.wav",
        "Signal after NLMS only",
    )

    # --- NKF (Neural Kalman Filter) ---
    nkf_result = process_nkf(sig=resampled_sig, noise=resampled_noise)
    save_and_analyze_result(
        nkf_result,
        resampled_noise,
        fs_resample,
        results_dir,
        f"nkf{iteration}.wav",
        "Signal after NKF only",
    )

    # --- RLS (Recursive Least Squares) ---
    rls_flit = RLSFilter(n_taps=RLS_N_TAPS)
    _, rls_res = rls_flit.process(noisy_signal=resampled_sig, noise=resampled_noise)
    save_and_analyze_result(
        rls_res,
        resampled_noise,
        fs_resample,
        results_dir,
        f"RLS{iteration}.wav",
        "Signal after RLS only",
    )


if __name__ == "__main__":
    main()
