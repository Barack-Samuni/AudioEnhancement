import numpy as np

NLMS_FILTER_WINDOW = 1024
NLMS_MU = 0.1
RLS_N_TAPS = 64
DEFAULT_SAMPLE_RATE = 16000


def test_rls(resampled_noise: np.ndarray, resampled_sig: np.ndarray):
    """
    Executes the Recursive Least Squares (RLS) filter test.

    Args:
        resampled_noise (np.ndarray): The reference noise signal.
        resampled_sig (np.ndarray): The primary signal containing noise and target.
    """
    pass
    # rls_flit = RLSFilter(n_taps=RLS_N_TAPS)
    # noise_estimation, rls_error = rls_flit.process(noisy_signal=resampled_sig, noise=resampled_noise)
    # print("worked")
    # print(rls_error[0]+noise_estimation[0])


def test_nkf(fs_resample: int, resampled_noise: np.ndarray, resampled_sig: np.ndarray):
    """
    Executes the Neural Kalman Filter (NKF) test.

    Args:
        fs_resample (int): The audio sampling rate (Hz).
        resampled_noise (np.ndarray): The reference noise signal.
        resampled_sig (np.ndarray): The primary signal containing noise and target.
    """
    pass
    # nkf_result = process_nkf(sig=resampled_sig, noise=resampled_noise, fs_sig=fs_resample, fs_noise=fs_resample)
    # print("worked")
    # print(nkf_result[0])


def test_nlms(fs_resample: int, resampled_noise: np.ndarray, resampled_sig: np.ndarray):
    """
    Executes the Normalized The Least Mean Squares (NLMS) filter test.

    Args:
        fs_resample (int): The audio sampling rate (Hz).
        resampled_noise (np.ndarray): The reference noise signal.
        resampled_sig (np.ndarray): The primary signal containing noise and target.
    """
    pass
    # nlms_result = nlms_calculation(
    #     total_sig=resampled_sig,
    #     noise=resampled_noise,
    #     fs1=fs_resample,
    #     fs2=fs_resample,
    #     fs_resample=DEFAULT_SAMPLE_RATE,
    #     filter_window=NLMS_FILTER_WINDOW,
    #     mu=NLMS_MU,
    # )
    # print("worked")
    # print(nlms_result[0])
    #
