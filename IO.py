import soundfile as sf
import glob
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple

def select_audio_files() -> List[str]:
    """
    Opens a dialog to select multiple .wav audio files.

    Returns:
        List of selected file paths
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_paths = filedialog.askopenfilenames(
        title="Select Audio Files",
        filetypes=[("Audio Files", "*.wav"), ("All Files", "*.*")]
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
        FileNotFoundError: If file doesn't exist
        RuntimeError: If file cannot be read
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Audio file not found: {filename}")

    try:
        sig, fs = sf.read(filename)
        return sig, fs
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {filename}: {e}")

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
        raise RuntimeError(f"Failed to save audio file {filename}: {e}")

def stereo_to_mono(data: np.ndarray) -> np.ndarray:
    """
    Convert stereo (or multi-channel) audio to mono by averaging channels.

    Args:
        data: Audio signal array (mono or multi-channel)

    Returns:
        Mono audio signal array
    """
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return data