import numpy as np
from scipy.io import wavfile

def add_white_noise(input_path, output_path, snr_db):
    # טעינת הקובץ
    fs, data = wavfile.read(input_path)
    data = data.astype(np.float32)
    
    # חישוב הספק האות
    sig_power = np.mean(data**2)
    sig_db = 10 * np.log10(sig_power)
    
    # חישוב הספק הרעש הנדרש לפי ה-SNR
    noise_db = sig_db - snr_db
    noise_power = 10 ** (noise_db / 10)
    
    # יצירת רעש לבן
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    
    # הוספת הרעש לאות המקורי
    noisy_signal = data + noise
    
    # שמירה (המרה חזרה לפורמט המתאים)
    noisy_signal = np.clip(noisy_signal, -32768, 32767) # למניעת עיוות (Clipping)
    wavfile.write(output_path, fs, noisy_signal.astype(np.int16))

def main():
    input_audio = rf"recordings\recording_ori_gt.wav"
    output_audio = rf"recordings\output_with_noise.wav"
    snr_db = 10
    add_white_noise(input_path=input_audio,  output_path=output_audio, snr_db=10)