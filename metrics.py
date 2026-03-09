import numpy as np
from scipy.signal import resample
from pystoi import stoi
from pesq import pesq
from mir_eval.separation import bss_eval_sources


def align_length(ref, deg):
    L = min(len(ref), len(deg))
    return ref[:L], deg[:L]


def resample_if_needed(sig, orig_sr, target_sr):
    if orig_sr == target_sr:
        return sig

    duration = len(sig) / orig_sr
    target_len = int(duration * target_sr)

    return resample(sig, target_len)


# -------------------------------------------------
# SI-SNR
# -------------------------------------------------

def si_snr(reference, estimation):

    reference, estimation = align_length(reference, estimation)

    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)

    ref_energy = np.sum(reference ** 2)

    projection = np.sum(reference * estimation) * reference / ref_energy
    noise = estimation - projection

    ratio = np.sum(projection ** 2) / np.sum(noise ** 2)

    return 10 * np.log10(ratio)


# -------------------------------------------------
# STOI
# -------------------------------------------------

def compute_stoi(reference, estimation, sr):

    reference, estimation = align_length(reference, estimation)

    return stoi(reference, estimation, sr, extended=False)


# -------------------------------------------------
# PESQ
# -------------------------------------------------

def compute_pesq(reference, estimation, sr):

    reference, estimation = align_length(reference, estimation)

    if sr not in [8000, 16000]:
        reference = resample_if_needed(reference, sr, 16000)
        estimation = resample_if_needed(estimation, sr, 16000)
        sr = 16000

    mode = "wb" if sr == 16000 else "nb"

    try:
        return pesq(sr, reference, estimation, mode)
    except:
        return np.nan


# -------------------------------------------------
# SDR
# -------------------------------------------------

def compute_sdr(reference, estimation):

    reference, estimation = align_length(reference, estimation)

    sdr, _, _, _ = bss_eval_sources(
        reference[np.newaxis, :],
        estimation[np.newaxis, :]
    )

    return float(sdr[0])


# -------------------------------------------------
# GCC-PHAT
# -------------------------------------------------

def gcc_phat(sig, refsig, fs=1, interp=16):

    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)

    R = SIG * np.conj(REF)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)

    return tau


def compute_gcc_segment(reference, estimation, sr, segment_sec=8):

    reference, estimation = align_length(reference, estimation)

    seg_len = int(segment_sec * sr)

    if len(reference) < seg_len:
        seg_len = len(reference)

    start = len(reference) // 2 - seg_len // 2
    end = start + seg_len

    ref_seg = reference[start:end]
    est_seg = estimation[start:end]

    return gcc_phat(est_seg, ref_seg, fs=sr)