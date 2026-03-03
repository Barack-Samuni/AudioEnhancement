import utils
from utils import resample_fs
import numpy as np
import IO as ioloader
from pyroomacoustics.adaptive import NLMS



def NLMS_calculation(total_sig, noise, fs1, fs2, fs_resample=16000,filter_window=1024,mu=0.1,output_file=""):
    """
    Applies NLMS and filtering using pyroomacoustics NLMS.
    """  # <--- Fixed closing quotes here

    # 1. Resampling logic
    if fs1 != fs_resample:
        total_sig, _ = resample_fs(total_sig, fs_old=fs1, fs_new=fs_resample)

    if fs2 != fs_resample:
        noise, _ = resample_fs(noise, fs_old=fs2, fs_new=fs_resample)

    # 2. Use only first channel (Mono conversion)
    if total_sig.ndim > 1:
        total_sig = total_sig.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    # 3. Synchronize signal lengths
    total_sig, noise = utils.match_sigs(ref=total_sig, sig=noise)

    l = NLMS(length=filter_window,mu=mu)
    e = np.zeros(min_len)

    # 5. Adaptive filtering loop
    # Using the pyroomacoustics built-in update logic
    for n in range(min_len):
        l.update(noise[n], total_sig[n])
        e[n] = total_sig[n] - np.dot(l.w, l.x)

    ioloader.save_sound(filename=output_file,data=e,fs=fs_resample)
    return e