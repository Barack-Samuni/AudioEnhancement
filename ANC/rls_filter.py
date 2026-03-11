from typing import Tuple

import numpy as np
from tqdm import tqdm

import src.utils as ut
from src import files_handler as io_loader


class RLSFilter:
    """
    Recursive Least Squares (RLS) adaptive filter implementation.
    Used for estimating a signal by recursively updating filter weights to
    minimize the weighted least squares error of the input signals.
    """

    def __init__(self, n_taps: int, lam: float = 0.999, delta: float = 10.0):
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

    def adapt(self, x: np.ndarray, d: float) -> Tuple[np.ndarray, np.ndarray]:
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
        pi_x = self.P @ x
        k = pi_x / (self.lam + x.T @ pi_x)

        # Compute filter output (y) and estimation error (e)
        y = self.w @ x
        e = d - y

        # Update the weight vector (w)
        self.w += k * e

        # Update the inverse correlation matrix (P) using Riccati equation
        self.P = (self.P - np.outer(k, x) @ self.P) / self.lam

        return y, e

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts output for a given input using current weights without adaptation.

        Args:
            x (np.ndarray): Input vector of length n_taps.

        Returns:
            float: Predicted output value.
        """
        return self.w @ np.array(x)

    def process(self, noisy_signal: np.ndarray, noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes entire signal arrays through the adaptive filter.

        Args:
            noisy_signal (np.ndarray): The primary signal containing the noise to be filtered.
            noise (np.ndarray): The reference noise signal.

        Returns:
            tuple: (processed signal array, error signal array).

        Raises:
            ValueError: If inputs are invalid or signal too short for filter
        """
        # Input validation
        if not isinstance(noisy_signal, np.ndarray) or not isinstance(noise, np.ndarray):
            raise ValueError("Both noisy_signal and noise must be numpy arrays")

        if noisy_signal.size == 0 or noise.size == 0:
            raise ValueError("Input signals cannot be empty")

        noisy_signal = io_loader.stereo_to_mono(noisy_signal)
        noise = io_loader.stereo_to_mono(noise)

        noise, noisy_signal, min_len = ut.adjust_min_length(noise, noisy_signal)

        if min_len < self.n_taps:
            raise ValueError(f"Signal length ({min_len}) must be >= n_taps ({self.n_taps})")

        n = len(noisy_signal)
        print("Starting noise cancellation...")

        errors = []
        sig = []
        # Sliding window iteration through the signals
        for i in tqdm(range(n - self.n_taps + 1)):
            x_vec = noisy_signal[i : i + self.n_taps]  # Input vector (regressor)
            d = noise[i]  # Desired signal target
            y, e = self.adapt(x_vec, d)
            errors.append(e)
            sig.append(y)

        err_array = np.array(errors)
        noise_estimation = np.array(object=sig)
        return noise_estimation, err_array
