import soundfile as sf
def load_sound(filename):
    sig,fs=sf.read(filename)
    return sig,fs
def save_sound(filename, data,fs):
    sf.write(filename, data, fs)
    print(f"Output saved to: {filename}")