import soundfile as sf
import numpy as np
from scipy.signal import resample
from pesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz
import config

def preprocess_audio(clean_path, processed_path, target_sr=16000):
    """
    Loads audio files, converts to mono, resamples, and aligns lengths.
    Internal helper function to prevent code duplication.
    """
    # Load audio files
    clean_audio, sr_clean = sf.read(clean_path)
    deg_audio, sr_deg = sf.read(processed_path)

    # Convert to mono if stereo
    if len(clean_audio.shape) > 1: 
        clean_audio = np.mean(clean_audio, axis=1)
    if len(deg_audio.shape) > 1: 
        deg_audio = np.mean(deg_audio, axis=1)

    # Resample to 16kHz (Critical requirement for PESQ)
    if sr_clean != target_sr:
        clean_audio = resample(clean_audio, int(len(clean_audio) * target_sr / sr_clean))
    if sr_deg != target_sr:
        deg_audio = resample(deg_audio, int(len(deg_audio) * target_sr / sr_deg))

    # Align lengths by cropping to the shortest signal
    min_len = min(len(clean_audio), len(deg_audio))
    return clean_audio[:min_len], deg_audio[:min_len], target_sr

# --- Metric Functions ---

def get_stoi_score(clean_path, processed_path):
    """Calculates the Short-Time Objective Intelligibility (STOI) score."""
    ref, deg, sr = preprocess_audio(clean_path, processed_path)
    score = stoi(ref, deg, sr, extended=False)
    return float(score)

def get_pesq_score(clean_path, processed_path):
    """Calculates the Perceptual Evaluation of Speech Quality (PESQ) score."""
    ref, deg, sr = preprocess_audio(clean_path, processed_path)
    try:
        # Using 'wb' for Wide-band; can be changed to 'nb' for Narrow-band
        score = pesq(sr, ref, deg, 'wb')
        return float(score)
    except Exception as e:
        return f"Error: {e}"

def get_snr_score(clean_path, processed_path):
    """Calculates the Signal-to-Noise Ratio (SNR)."""
    ref, deg, _ = preprocess_audio(clean_path, processed_path)
    noise = ref - deg
    snr = 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-10))
    return float(snr)

def get_llr_score(clean_path, processed_path, frame_len=0.03, order=12):
    """Calculates the Log Likelihood Ratio (LLR) - lower values indicate better quality."""
    ref, deg, sr = preprocess_audio(clean_path, processed_path)
    
    def lpc(signal, order):
        """Internal helper to calculate Linear Predictive Coding coefficients."""
        r = np.correlate(signal, signal, mode='full')[len(signal)-1:]
        if r[0] == 0: 
            return np.zeros(order + 1)
        # Using scipy.linalg.toeplitz for efficient calculation
        a = np.linalg.solve(toeplitz(r[:order]), r[1:order+1])
        return np.concatenate(([1], -a))

    hop = int(frame_len * sr)
    llr_vals = []
    for i in range(0, len(ref) - hop, hop):
        ref_f, deg_f = ref[i:i+hop], deg[i:i+hop]
        a_ref, a_deg = lpc(ref_f, order), lpc(deg_f, order)
        
        # Calculate spectral distance using the autocorrelation matrix
        R_ref = toeplitz(np.correlate(ref_f, ref_f, mode='full')[len(ref_f)-1:][:order+1])
        num = np.dot(np.dot(a_deg, R_ref), a_deg)
        den = np.dot(np.dot(a_ref, R_ref), a_ref)
        if num > 0 and den > 0: 
            llr_vals.append(np.log10(num / den))
            
    return float(np.mean(llr_vals)) if llr_vals else 0.0

def get_cbak_score(clean_path, processed_path):
    """
    Calculates the CBAK (Composite Background Noise Quality) score.
    Predicts background noise intrusiveness (Range: 1-5, higher is better).
    Based on Loizou's regression formulas.
    """
    p = get_pesq_score(clean_path, processed_path)
    l = get_llr_score(clean_path, processed_path)
    
    # Loizou regression formula for CBAK prediction
    cbak = 1.63 + 0.40 * p - 0.05 * l
    
    return float(np.clip(cbak, 1.0, 5.0))

def get_covl_score(clean_path, processed_path):
    """
    Calculates the COVL (Composite Overall Quality) score.
    Combines PESQ and LLR to predict overall quality (Range: 1-5, higher is better).
    """
    p = get_pesq_score(clean_path, processed_path)
    l = get_llr_score(clean_path, processed_path)
    
    # Loizou regression formula for COVL prediction
    covl = 1.59 + 1.09 * p - 0.62 * l
    return float(np.clip(covl, 1.0, 5.0))

# --- Execution ---

path_a = config.PATH_A
path_b = config.PATH_B

print(f"STOI Score: {get_stoi_score(path_a, path_b):.4f}")
print(f"PESQ Score: {get_pesq_score(path_a, path_b):.4f}")
print(f"SNR Score:  {get_snr_score(path_a, path_b):.2f} dB")
print(f"LLR Score:  {get_llr_score(path_a, path_b):.4f}")
print(f"CBAK Score: {get_cbak_score(path_a, path_b):.4f}")
print(f"COVL Score: {get_covl_score(path_a, path_b):.4f}")