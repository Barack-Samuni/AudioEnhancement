import numpy as np
from tqdm import tqdm


class RLSFilter:
    """
    Recursive Least Squares (RLS) adaptive filter implementation.
    Used for estimating a signal by recursively updating filter weights to
    minimize the weighted least squares error of the input signals.
    """

    def __init__(self, n_taps, lam=0.999, delta=10.0):
        """
        Initializes the RLS filter state.

        Args:
            n_taps (int): The number of filter coefficients (taps).
            lam (float): Forgetting factor, typically between 0.99 and 1.0.
            delta (float): Initial value for the inverse correlation matrix P.
        """
        self.n_taps = n_taps
        self.lam = lam
        self.delta = delta
        self.w = np.zeros(n_taps)  # Filter weights initialization
        self.P = (1.0 / delta) * np.eye(n_taps)  # Inverse correlation matrix initialization

    def adapt(self, x, d):
        """
        Updates filter weights based on a single input vector and desired output.

        Args:
            x (np.ndarray): Input vector of length n_taps.
            d (float): Desired signal sample at the current time step.

        Returns:
            tuple: (filter output y, estimation error e).
        """
        x = np.array(x)
        if x.shape[0] != self.n_taps:
            raise ValueError("Input vector length must match n_taps")

        # Gain vector calculation (k)
        Pi_x = self.P @ x
        k = Pi_x / (self.lam + x.T @ Pi_x)

        # Compute filter output (y) and estimation error (e)
        y = self.w @ x
        e = d - y

        # Update the weight vector (w)
        self.w += k * e

        # Update the inverse correlation matrix (P) using Riccati equation
        self.P = (self.P - np.outer(k, x) @ self.P) / self.lam

        return y, e

    def predict(self, x):
        """
        Predicts output for a given input using current weights without adaptation.

        Args:
            x (np.ndarray): Input vector of length n_taps.

        Returns:
            float: Predicted output value.
        """
        return self.w @ np.array(x)

    def process(self, noisy_signal, noise):
        """
        Processes entire signal arrays through the adaptive filter.

        Args:
            noisy_signal (np.ndarray): The primary signal containing the noise to be filtered.
            noise (np.ndarray): The reference noise signal.

        Returns:
            tuple: (processed signal array, error signal array).
        """
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
        # Sliding window iteration through the signals
        for i in tqdm(range(N - self.n_taps + 1)):
            x_vec = noisy_signal[i:i + self.n_taps]  # Input vector (regressor)
            d = noise[i]  # Desired signal target
            y, e = self.adapt(x_vec, d)
            errors.append(e)
            sig.append(y)
            # pbar.update(i/(N - self.n_taps + 1))

        err_array = np.array(errors)
        sig = np.array(object=sig)
        return sig, err_array