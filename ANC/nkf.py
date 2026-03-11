"""
Tencent is pleased to support the open source community by making NKF-AEC available.

Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.

Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
in compliance with the License. You may obtain a copy of the License at

https://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import src.utils as ut

RECOMMENDED_FS = 16000


class ComplexGRU(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True, dropout=0, bidirectional=False
    ):
        super().__init__()
        self.gru_r = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.gru_i = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_rr: Optional[torch.Tensor] = None,
        h_ir: Optional[torch.Tensor] = None,
        h_ri: Optional[torch.Tensor] = None,
        h_ii: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_rr, h_rr = self.gru_r(x.real, h_rr)
        f_ir, h_ir = self.gru_r(x.imag, h_ir)
        f_ri, h_ri = self.gru_i(x.real, h_ri)
        f_ii, h_ii = self.gru_i(x.imag, h_ii)
        y = torch.complex(f_rr - f_ii, f_ri + f_ir)
        return y, h_rr, h_ir, h_ri, h_ii


class ComplexDense(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.linear_real = nn.Linear(in_channel, out_channel, bias=bias)
        self.linear_imag = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_real = self.linear_real(x.real)
        y_imag = self.linear_imag(x.imag)
        return torch.complex(y_real, y_imag)


class ComplexPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.prelu(x.real), self.prelu(x.imag))


class KGNet(nn.Module):
    def __init__(self, layers, fc_dim, rnn_layers, rnn_dim, **kwargs):
        super().__init__()
        if "L" in kwargs:
            layers = kwargs.pop("L")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        self.L: int = int(layers)
        self.rnn_layers: int = int(rnn_layers)
        self.rnn_dim: int = int(rnn_dim)
        self.h_rr = None
        self.h_ir = None
        self.h_ri = None
        self.h_ii = None

        self.fc_in = nn.Sequential(ComplexDense(2 * self.L + 1, fc_dim, bias=True), ComplexPReLU())

        self.complex_gru = ComplexGRU(fc_dim, rnn_dim, rnn_layers, bidirectional=False)

        self.fc_out = nn.Sequential(
            ComplexDense(rnn_dim, fc_dim, bias=True), ComplexPReLU(), ComplexDense(fc_dim, self.L, bias=True)
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        layers = int(self.rnn_layers)
        hidden_dim = int(self.rnn_dim)
        bsz = int(batch_size)
        self.h_rr = torch.zeros(layers, bsz, hidden_dim, device=device)
        self.h_ir = torch.zeros(layers, bsz, hidden_dim, device=device)
        self.h_ri = torch.zeros(layers, bsz, hidden_dim, device=device)
        self.h_ii = torch.zeros(layers, bsz, hidden_dim, device=device)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        feat = self.fc_in(input_feature).unsqueeze(1)
        rnn_out, self.h_rr, self.h_ir, self.h_ri, self.h_ii = self.complex_gru.forward(
            feat,
            self.h_rr,
            self.h_ir,
            self.h_ri,
            self.h_ii,
        )
        kg = self.fc_out(rnn_out).permute(0, 2, 1)
        return kg


class NKF(nn.Module):
    def __init__(self, layers=4, **kwargs):
        super().__init__()
        if "L" in kwargs:
            layers = kwargs.pop("L")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")
        self.L: int = int(layers)
        self.kg_net = KGNet(self.L, fc_dim=18, rnn_layers=1, rnn_dim=18)
        self.stft = lambda x: torch.stft(
            x,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024),
            return_complex=True,
        )
        self.istft = lambda x_complex: torch.istft(
            x_complex,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024),
            return_complex=False,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        x = self.stft(x)
        y = self.stft(y)
        batch_size, freq_bins, time_steps = x.shape
        device = x.device
        total_bins = int(batch_size * freq_bins)
        time_steps_i = int(time_steps)
        h_prior = torch.zeros(total_bins, self.L, 1, dtype=torch.complex64, device=device)
        h_posterior = torch.zeros(total_bins, self.L, 1, dtype=torch.complex64, device=device)
        self.kg_net.init_hidden(total_bins, device)

        x = x.contiguous().view(total_bins, time_steps_i)
        y = y.contiguous().view(total_bins, time_steps_i)
        echo_hat = torch.zeros(total_bins, time_steps_i, dtype=torch.complex64, device=device)

        for frame_idx in tqdm(range(time_steps_i)):
            if frame_idx < self.L:
                xt = torch.cat(
                    [
                        torch.zeros(
                            total_bins,
                            self.L - frame_idx - 1,
                            dtype=torch.complex64,
                            device=device,
                        ),
                        x[:, : frame_idx + 1],
                    ],
                    dim=-1,
                )
            else:
                xt = x[:, frame_idx - self.L + 1 : frame_idx + 1]
            if xt.abs().mean() < 1e-5:
                continue

            dh = h_posterior - h_prior
            h_prior = h_posterior
            e = y[:, frame_idx] - torch.matmul(xt.unsqueeze(1), h_prior).squeeze()

            input_feature = torch.cat([xt, e.unsqueeze(1), dh.squeeze()], dim=1)
            kg = self.kg_net.forward(input_feature)
            h_posterior = h_prior + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))

            echo_hat[:, frame_idx] = torch.matmul(xt.unsqueeze(1), h_posterior).squeeze()

        s_hat = self.istft(y - echo_hat).squeeze()

        return s_hat


def process_nkf(sig: np.ndarray, noise: np.ndarray, fs_sig, fs_noise):
    """
    Processes audio using a Neural Kalman Filter (NKF) for noise cancellation.

    This function initializes the NKF model, loads pre-trained weights from a
    specific local path, and performs a forward pass to estimate the clean signal.

    Args:
        sig (np.ndarray): The primary noisy signal array.
        noise (np.ndarray): The reference noise signal array.
        sr (int, optional): The sampling rate of the signals. Defaults to 16,000Hz.

    Returns:
        torch.Tensor: The estimated clean signal (s_hat) produced by the model.

    Raises:
        IndexError: If the input signal and noise arrays have different lengths.
    """

    if fs_sig != fs_noise:
        raise IndexError("Both signal must have the same sample rate.")
    # Use relative path from the current file location
    if fs_sig != RECOMMENDED_FS:
        warnings.warn(
            f"To optimize nkf- we recommend using sample rate of 16Khz which in this case the fs is : {fs_sig}"
        )

    noise, sig, min_len = ut.adjust_min_length(noise, sig)

    model_path = Path(__file__).parent / "nkf_epoch70.pt"

    # Check if model file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = NKF(layers=4)
    numparams = 0
    for param in model.parameters():
        numparams += param.numel()
    print(f"Total number of parameters: {numparams:,}")

    try:
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    model.eval()

    noise_tensor = torch.from_numpy(noise).float()
    sig_tensor = torch.from_numpy(sig).float()

    with torch.no_grad():
        s_hat = model.forward(noise_tensor, sig_tensor)
    return s_hat
