import os
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import List, Tuple

import numpy as np
import soundfile as sf


def select_audio_files() -> List[str]:
    """
    Opens a dialog to select multiple .wav audio files.

    Returns:
        List of selected file paths
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_paths = filedialog.askopenfilenames(
        title="Select Audio Files", filetypes=[("Audio Files", "*.wav"), ("All Files", "*.*")]
    )
    root.destroy()
    return list(file_paths)


def load_sound(filename: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.

    Args:
        filename: Path to audio file

    Returns:
        Tuple: (signal data, sample rate)

    Raises:
        FileNotFoundError: If the file does not exist
        RuntimeError: If file cannot be read
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Audio file not found: {filename}")

    try:
        sig, fs = sf.read(filename)
        return sig, fs
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {filename}: {e}") from e


def save_sound(filename: str, data: np.ndarray, fs: int) -> None:
    """
    Save audio data to file.

    Args:
        filename: Output file path
        data: Audio signal data
        fs: Sample rate

    Raises:
        RuntimeError: If file cannot be written
    """
    try:
        sf.write(filename, data, fs)
        print(f"Output saved to: {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to save audio file {filename}: {e}") from e


def stereo_to_mono(data: np.ndarray) -> np.ndarray:
    """
    Convert stereo (or multi-channel) audio to mono by averaging channels.

    Args:
        data: Audio signal array (mono or multi-channel)

    Returns:
        Mono audio signal array
    """
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data


def load_data() -> tuple[list[str], Path, list[str]]:
    """
    Handles file selection via UI and prepares project variables.
    """
    print("Please select the TOTAL signals (signal + noise):")
    signal_files = select_audio_files()

    print("Please select the NOISE reference signals:")
    noise_files = select_audio_files()

    project_root = Path(signal_files[0]).parent

    # Validation: Ensure we have a reference for every signal
    if len(signal_files) != len(noise_files):
        raise IndexError(f"Mismatch: Found {len(signal_files)} signals but {len(noise_files)} noise files.")
    return noise_files, project_root, signal_files


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
