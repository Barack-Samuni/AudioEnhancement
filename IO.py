import soundfile as sf
import glob
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

def select_audio_files():
    """Opens a dialog to select specific multiple .wav files."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_paths = filedialog.askopenfilenames(
        title="Select Audio Files",
        filetypes=[("Audio Files", "*.wav"), ("All Files", "*.*")]
    )
    root.destroy()
    return list(file_paths)

def load_sound(filename):
    sig, fs = sf.read(filename)
    return sig, fs

def save_sound(filename, data, fs):
    sf.write(filename, data, fs)
    print(f"Output saved to: {filename}")

def stereo_to_mono(data):
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return data