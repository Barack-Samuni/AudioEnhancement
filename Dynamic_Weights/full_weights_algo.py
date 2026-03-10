import numpy as np

from src import utils as ut

LPF_SUGGESTED_WINDOW = 15


def full_weighting_process(noise: np.ndarray, anc_sig: np.ndarray, single_channel: np.ndarray, fs_resample: int):
    envelope_noise = ut.get_sig_envelope(noise, fs_resample, LPF_SUGGESTED_WINDOW)
    normalized_envelope = ut.normalize_sig(envelope_noise)
    single_channel_weighted = normalized_envelope * single_channel
    return single_channel_weighted + anc_sig
