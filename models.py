"""阶段 1 的只读数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Waveform:
    """双音波形及其物理元数据。

    ``samples`` 为无量纲复基带样点，``time_s`` 为秒，``energy``
    为离散积分得到的幅度平方秒。
    """

    samples: ComplexArray
    time_s: FloatArray
    envelope: FloatArray
    active_mask: BoolArray
    sample_rate_hz: float
    average_active_power: float
    energy: float


@dataclass(frozen=True)
class NoisySignal:
    """AWGN 注入结果；功率均表示复基带样点的幅度平方均值。"""

    samples: ComplexArray
    noise_samples: ComplexArray
    signal_power: float
    noise_power: float
    target_snr_db: float
    measured_snr_db: float


@dataclass(frozen=True)
class PropagationResult:
    """静态链路输出；数组为按 ``sample_rate_hz`` 采样的复基带 IQ。"""

    samples: ComplexArray
    clean_samples: ComplexArray
    noise_samples: ComplexArray
    sample_rate_hz: float
    measured_snr_db: float | None


@dataclass(frozen=True)
class CorrelationResult:
    """FFT 线性互相关结果。

    ``lags_samples`` 是相对模板起点的有符号物理样点延迟，不能用
    ``values`` 的数组下标代替。
    """

    values: ComplexArray
    magnitude: FloatArray
    lags_samples: IntArray
    fft_length: int


@dataclass(frozen=True)
class DelayEstimate:
    """整数峰值和 QLS 亚采样时延估计结果。"""

    delay_s: float
    integer_lag_samples: int
    fractional_offset_samples: float
    raw_fractional_offset_samples: float
    peak_array_index: int
    peak_magnitude: float
    qls_denominator: float
    qls_valid: bool
    boundary_hit: bool
    fraction_clipped: bool
