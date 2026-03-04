from typing import Any

import numpy as np
from datetime import datetime
import IO as ioloader
import torch
from ANC.NLMS import NLMS_calculation
from ANC.nkf import process_nkf
from ANC.rls_filter import RLSFilter
import utils as ut
from pathlib import Path
import scipy.signal as sg

def main():
    # 1. Select the files
    alignment = True#Means all data is correlated
    fs_resample, iteration, noise_files, project_root, signal_files = load_data()

    # 3. If you have a list of "Signals" and a list of "Noise References," zip creates a single iterator of pairs:
    for s_path, n_path in zip(signal_files, noise_files):
        # Load data
        sig, fs_sig = ioloader.load_sound(s_path)
        noise, fs_noise = ioloader.load_sound(n_path)
        sig=ioloader.stereo_to_mono(sig)
        noise=ioloader.stereo_to_mono(noise)
        resampled_sig, _=ut.resample_fs(sig, fs_noise, fs_resample)
        resampled_noise,_=ut.resample_fs(noise, fs_noise, fs_resample)

        if not alignment:
            tau = ut.gcc_phat(resampled_sig[:fs_resample * 10], resampled_noise[:fs_resample * 10], fs=fs_resample, interp=1)
            tau = max(0, int((tau - 0.001) * fs_resample))
            resampled_sig = torch.from_numpy(resampled_sig).float()
            resampled_noise = torch.from_numpy(resampled_noise).float()
            resampled_noise = torch.cat([torch.zeros(tau), resampled_noise])[:resampled_sig.shape[-1]]
            resampled_sig = resampled_sig.numpy()
            resampled_noise = resampled_noise.numpy()
            resampled_noise, resampled_sig=ut.match_sigs(resampled_noise,resampled_sig)
            ioloader.save_sound(rf"{project_root}\corr_noise.wav",resampled_noise,fs_resample)
            ioloader.save_sound(rf"{project_root}\corr_sig.wav",resampled_sig,fs_resample)

        # resampled_noise = distortion_ir(resampled_noise)
        ut.coherence_of_sigs(resampled_sig,resampled_noise,fs_resample)
        process_ancs(fs_resample, iteration, project_root, resampled_noise, resampled_sig)
        iteration += 1


def distortion_ir(noise) -> Any:
    room_ir = np.exp(-np.linspace(0, 1, 20)) * np.random.choice([1, -1], 20)
    noise = sg.lfilter(room_ir, [1], noise)
    noise = noise / np.max(np.abs(noise))
    return noise


def get_results_dir(root_path):
    # 1. Get today's date in YYYY-MM-DD format
    today = datetime.now().strftime('%Y-%m-%d')

    # 2. Define the path: ProjectRoot / results / 2026-03-04
    results_base = Path(root_path) / "results"
    daily_dir = results_base / today

    # 3. Create the directories (parents=True creates 'results' if it's missing)
    daily_dir.mkdir(parents=True, exist_ok=True)

    return daily_dir

def load_data():
    print("Please select the TOTAL signals (Signal + Noise):")
    signal_files = ioloader.select_audio_files()

    print("Please select the NOISE reference signals:")
    noise_files = ioloader.select_audio_files()

    project_root = Path(signal_files[0]).parent
    iteration = 0
    fs_resample = 16000
    # 2. Validation
    if len(signal_files) != len(noise_files):
        raise IndexError(f"Mismatch: Found {len(signal_files)} signals but {len(noise_files)} noise files.")
    return fs_resample, iteration, noise_files, project_root, signal_files


def process_ancs(fs_resample: int, iteration: int, project_root: Path, resampled_noise: np.ndarray,
                 resampled_sig: np.ndarray):
    nlms_result = NLMS_calculation(total_sig=resampled_sig, noise=resampled_noise, fs1=fs_resample, fs2=fs_resample,
                                   fs_resample=16000, filter_window=1024)
    dir=get_results_dir(project_root)
    path_nlms = rf"{dir}\NLMS{iteration}.wav"
    ioloader.save_sound(path_nlms,nlms_result,fs_resample)
    ut.coherence_of_sigs(nlms_result, resampled_noise, fs_resample)
    nkf_result = process_nkf(sig=resampled_sig, noise=resampled_noise)
    path_nkf = rf"{dir}\nkf{iteration}.wav"
    ut.coherence_of_sigs(nkf_result, resampled_noise, fs_resample)
    ioloader.save_sound(path_nkf,nkf_result,fs_resample)
    rls_flit=RLSFilter(n_taps=64)
    _, rls_res = rls_flit.process(noisy_signal=resampled_sig, noise=resampled_noise)
    path_rls = rf"{dir}\RLS{iteration}.wav"
    ut.coherence_of_sigs(rls_res, resampled_noise, fs_resample)
    ioloader.save_sound(path_rls,rls_res,fs_resample)


if __name__ == "__main__":
    main()
