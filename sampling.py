"""独立 DAC/ADC 均匀网格间的带限复 IQ 重采样。"""

from __future__ import annotations

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import czt


def sample_iq(iq: np.ndarray, start_sample: float, step: float, count: int) -> np.ndarray:
    """在 ADC 网格采样补零 DAC 波形；``step`` 为 DAC/ADC 样点步长比。"""

    if not np.isfinite(start_sample) or not np.isfinite(step) or step <= 0 or count < 1:
        raise ValueError("采样坐标须有限、速率和长度须为正")
    end = start_sample + step * (count - 1)
    left = 64 + int(np.ceil(max(0.0, -start_sample)))
    right = 64 + int(np.ceil(max(0.0, end - len(iq))))
    nfft = next_fast_len(left + len(iq) + right)
    spectrum = np.fft.fftshift(np.fft.fft(np.pad(iq, (left, right)), nfft))
    first = start_sample + left
    position = first + step * np.arange(count)
    values = czt(
        spectrum,
        m=count,
        w=np.exp(2j * np.pi * step / nfft),
        a=np.exp(-2j * np.pi * first / nfft),
    ) / nfft
    values *= np.exp(-2j * np.pi * (nfft // 2) * position / nfft)
    return np.asarray(values, dtype=np.complex128)
