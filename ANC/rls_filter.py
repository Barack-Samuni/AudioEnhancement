import numpy as np
from tqdm import tqdm


class RLSFilter:
    def __init__(self, n_taps, lam=0.999, delta=10.0):
        self.n_taps = n_taps
        self.lam = lam
        self.delta = delta
        self.w = np.zeros(n_taps)
        self.P = (1.0 / delta) * np.eye(n_taps)

    def adapt(self, x, d):
        x = np.array(x)
        if x.shape[0] != self.n_taps:
            raise ValueError("Input vector length must match n_taps")

        # Gain vector
        Pi_x = self.P @ x
        k = Pi_x / (self.lam + x.T @ Pi_x)

        # Error
        y = self.w @ x
        e = d - y

        # Update weights
        self.w += k * e

        # Update inverse correlation matrix
        self.P = (self.P - np.outer(k, x) @ self.P) / self.lam

        return y, e

    def predict(self, x):
        return self.w @ np.array(x)

    def process(self, noisy_signal, noise):
        # wraps the whole process of filtering the signal
        # Ensure both have the same length and shape
        min_len = min(len(noisy_signal), len(noise))
        noise = noise[:min_len]
        noisy_signal = noisy_signal[:min_len]
        N = len(noisy_signal)
        print("Starting noise cancellation...")
        # pbar = tqdm(total=100)

        errors = []
        sig = []
        for i in tqdm(range(N - self.n_taps + 1)):
            x_vec = noisy_signal[i:i + self.n_taps]
            d = noise[i]
            y, e = self.adapt(x_vec, d)
            errors.append(e)
            sig.append(y)
            # pbar.update(i/(N - self.n_taps + 1))

        err_array = np.array(errors)
        sig = np.array(object=sig)
        return sig, err_array