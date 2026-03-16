import soundfile as sf
import numpy as np
from scipy.signal import resample
from pesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz

def preprocess_audio(clean_path, processed_path, target_sr=16000):
    """
    טעינה, המרה למונו, שינוי קצב דגימה והשוואת אורכים.
    פונקציית עזר פנימית כדי למנוע כפל קוד.
    """
    # טעינה
    clean_audio, sr_clean = sf.read(clean_path)
    deg_audio, sr_deg = sf.read(processed_path)

    # המרה למונו (אם סטריאו)
    if len(clean_audio.shape) > 1: clean_audio = np.mean(clean_audio, axis=1)
    if len(deg_audio.shape) > 1: deg_audio = np.mean(deg_audio, axis=1)

    # Resampling ל-16kHz (קריטי ל-PESQ)
    if sr_clean != target_sr:
        clean_audio = resample(clean_audio, int(len(clean_audio) * target_sr / sr_clean))
    if sr_deg != target_sr:
        deg_audio = resample(deg_audio, int(len(deg_audio) * target_sr / sr_deg))

    # השוואת אורכים
    min_len = min(len(clean_audio), len(deg_audio))
    return clean_audio[:min_len], deg_audio[:min_len], target_sr

# --- פונקציה נפרדת לכל מדד ---

def get_stoi_score(clean_path, processed_path):
    """מחשב את מדד מובנות הדיבור (Short-Time Objective Intelligibility)"""
    ref, deg, sr = preprocess_audio(clean_path, processed_path)
    score = stoi(ref, deg, sr, extended=False)
    return float(score)

def get_pesq_score(clean_path, processed_path):
    """מחשב את מדד איכות הדיבור (Perceptual Evaluation of Speech Quality)"""
    ref, deg, sr = preprocess_audio(clean_path, processed_path)
    try:
        # 'wb' עבור Wide-band, ניתן להחליף ל-'nb' עבור Narrow-band
        score = pesq(sr, ref, deg, 'wb')
        return float(score)
    except Exception as e:
        return f"Error: {e}"

def get_snr_score(clean_path, processed_path):
    """מחשב את יחס האות לרעש (Signal-to-Noise Ratio) בצורה פשוטה"""
    ref, deg, _ = preprocess_audio(clean_path, processed_path)
    noise = ref - deg
    snr = 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-10))
    return float(snr)

def get_llr_score(clean_path, processed_path, frame_len=0.03, order=12):
    """מחשב את מדד ה-Log Likelihood Ratio (נמוך זה טוב)"""
    ref, deg, sr = preprocess_audio(clean_path, processed_path)
    
    def lpc(signal, order):
        r = np.correlate(signal, signal, mode='full')[len(signal)-1:]
        if r[0] == 0: return np.zeros(order + 1)
        # שימוש ב-toeplitz שיובא מ-scipy
        a = np.linalg.solve(toeplitz(r[:order]), r[1:order+1])
        return np.concatenate(([1], -a))

    hop = int(frame_len * sr)
    llr_vals = []
    for i in range(0, len(ref) - hop, hop):
        ref_f, deg_f = ref[i:i+hop], deg[i:i+hop]
        a_ref, a_deg = lpc(ref_f, order), lpc(deg_f, order)
        
        # עדכון כאן: שימוש ב-toeplitz של scipy
        R_ref = toeplitz(np.correlate(ref_f, ref_f, mode='full')[len(ref_f)-1:][:order+1])
        num = np.dot(np.dot(a_deg, R_ref), a_deg)
        den = np.dot(np.dot(a_ref, R_ref), a_ref)
        if num > 0 and den > 0: llr_vals.append(np.log10(num / den))
            
    return float(np.mean(llr_vals)) if llr_vals else 0.0

def get_cbak_score(clean_path, processed_path):
    """
    מחשב את מדד ה-CBAK (Composite Background Noise Quality).
    מנבא עד כמה רעש הרקע חודרני/מפריע (1-5, גבוה זה טוב).
    """
    # CBAK מתבסס על שילוב של PESQ ו-LLR (לעיתים גם WSS, אך פה נשתמש בגרסה הנפוצה)
    p = get_pesq_score(clean_path, processed_path)
    l = get_llr_score(clean_path, processed_path)
    
    # נוסחת הרגרסיה של Loizou לחיזוי CBAK
    cbak = 1.63 + 0.40 * p - 0.05 * l
    
    return float(np.clip(cbak, 1.0, 5.0))

def get_covl_score(clean_path, processed_path):
    """מחשב את מדד ה-Overall Quality (COVL) המבוסס על PESQ ו-LLR"""
    p = get_pesq_score(clean_path, processed_path)
    l = get_llr_score(clean_path, processed_path)
    
    # נוסחת Loizou לחיזוי איכות כללית
    covl = 1.59 + 1.09 * p - 0.62 * l
    return float(np.clip(covl, 1.0, 5.0))

path_a = rf"recordings\recording_ori_gt.wav"
path_b = rf"recordings\output_with_noise.wav"

print(f"STOI Score: {get_stoi_score(path_a, path_b):.4f}")
print(f"PESQ Score: {get_pesq_score(path_a, path_b):.4f}")
print(f"SNR Score:  {get_snr_score(path_a, path_b):.2f} dB")
print(f"LLR:  {get_llr_score(path_a, path_b):.4f}")
print(f"CBAK: {get_cbak_score(path_a, path_b):.4f}")
print(f"COVL: {get_covl_score(path_a, path_b):.4f}")