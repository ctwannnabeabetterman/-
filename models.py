"""阶段 1 的只读数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


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
