import numpy as np
import soundfile as sf

from pns.noise_suppressor import NoiseSuppressor

def denoise_single_file(input_path, output_path):
    # קריאת הקובץ
    noisy_wav, fs = sf.read(input_path)
    
    # בדיקה אם הקובץ הוא סטריאו (יותר מערוץ אחד) או מונו
    channels = noisy_wav.shape[1] if noisy_wav.ndim > 1 else 1
    
    print(f"מעבד קובץ: {input_path}")
    print(f"קצב דגימה: {fs} Hz")
    
    if channels > 1:
        xfinal = np.zeros(noisy_wav.shape)
        for ch in range(channels):
            noise_suppressor = NoiseSuppressor(fs)
            frame_size = noise_suppressor.get_frame_size()
            
            k = 0
            while k + frame_size < len(noisy_wav):
                frame = noisy_wav[k : k + frame_size, ch]
                xfinal[k : k + frame_size, ch] = noise_suppressor.process_frame(frame)
                k += frame_size
            
            # נרמול עוצמה לערוץ
            if np.max(np.abs(xfinal[:, ch])) > 0:
                xfinal[:, ch] = xfinal[:, ch] / np.max(np.abs(xfinal[:, ch]))
    else:
        # טיפול בקובץ מונו
        noise_suppressor = NoiseSuppressor(fs)
        frame_size = noise_suppressor.get_frame_size()
        xfinal = np.zeros(len(noisy_wav))

        k = 0
        while k + frame_size < len(noisy_wav):
            frame = noisy_wav[k : k + frame_size]
            xfinal[k : k + frame_size] = noise_suppressor.process_frame(frame)
            k += frame_size

        # נרמול עוצמה
        if np.max(np.abs(xfinal)) > 0:
            xfinal = xfinal / np.max(np.abs(xfinal))
    
    # שמירת התוצאה
    sf.write(output_path, xfinal, fs)
    print(f"הקובץ המעובד נשמר ב: {output_path}")

if __name__ == "__main__":
    # הגדרת הניתובים
    input_path = r"C:\Users\PC\Desktop\מיכלי\תכנות\working now\python-speech-enhancement\Noise3m_olympic.wav"
    output_path = r"C:\Users\PC\Desktop\מיכלי\תכנות\working now\python-speech-enhancement\Noise3m_olympic_processed.wav"
    
    denoise_single_file(input_path, output_path)