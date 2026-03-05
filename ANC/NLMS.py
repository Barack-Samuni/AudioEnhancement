import utils
from utils import resample_fs
import numpy as np
from pyroomacoustics.adaptive import NLMS



def NLMS_calculation(total_sig, noise, fs1, fs2, fs_resample=16000,filter_window=1024,mu=0.1):
    """
        Applies NLMS adaptive filtering to cancel noise from a primary signal.

        Mathematical Logic:
        The filter tries to find a weight vector 'w' such that:
        total_sig[n] ≈ Filter_Output(noise[n])
        The result returned (e) is: total_sig[n] - Filter_Output(noise[n])

        Parameters:
        -----------
        total_sig (ndarray): The 'Primary' input. Contains the desired speech + noise.
        noise (ndarray):     The 'Reference' input. Contains ONLY the noise to be removed.
        fs1 (int):           Original sampling rate of the total_sig.
        fs2 (int):           Original sampling rate of the noise reference.
        fs_resample (int):   Target sampling rate. Default 16kHz is standard for speech processing.
        filter_window (int): The number of 'taps' (coefficients).
                             Longer windows handle echoes/reverb better but take longer to converge.
        mu (float):          The 'Step Size' or Learning Rate.
                             Higher = Faster learning, but more instability/distortion.
                             Lower  = Better steady-state quality, but slower to adapt to changes.
        """

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
    e = np.zeros(len(total_sig))

    # 5. Adaptive filtering loop
    # Using the pyroomacoustics built-in update logic
    for n in range(len(total_sig)):
        l.update(noise[n], total_sig[n])
        e[n] = total_sig[n] - np.dot(l.w, l.x)

    return e