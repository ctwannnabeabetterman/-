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


@dataclass(frozen=True)
class QLSCalibration:
    """由当前波形离线生成的 QLS 周期偏差校正表，坐标单位为样点。"""

    signature: str
    true_fraction_samples: FloatArray
    raw_fraction_samples: FloatArray
    estimated_axis_samples: FloatArray
    bias_samples: FloatArray
    corrected_error_samples: FloatArray
    grid_points: int


@dataclass(frozen=True)
class LinkDelayMeasurement:
    """单向波形时延测量，时间单位为秒。"""

    raw_estimate: DelayEstimate
    corrected_delay_s: float
    lut_adjustment_samples: float
    measured_snr_db: float | None


@dataclass(frozen=True)
class TwoWayObservation:
    """一次双向交换保存的四个本地时间戳和波形测量。"""

    t_tx1_s: float
    t_rx0_s: float
    t_tx0_s: float
    t_rx1_s: float
    up_measurement: LinkDelayMeasurement
    down_measurement: LinkDelayMeasurement


@dataclass(frozen=True)
class TwoWayEstimate:
    """双向时间传递输出；clock_correction_s 应加到 AP1 本地时钟。"""

    clock_correction_s: float
    ap1_offset_estimate_s: float
    symmetric_propagation_delay_s: float
    forward_interval_s: float
    reverse_interval_s: float


@dataclass(frozen=True)
class ClockTrackingResult:
    """多轮同步的真值、估计、校正和残差时间序列，单位均为秒。"""

    epoch_true_s: FloatArray
    raw_offset_s: FloatArray
    estimated_offset_s: FloatArray
    applied_correction_s: FloatArray
    residual_before_s: FloatArray
    residual_after_s: FloatArray
    random_walk_accumulated_s: FloatArray


@dataclass(frozen=True)
class FrequencyReferenceObservation:
    """AP1 对名义参考信号的采样观测。"""

    samples: ComplexArray
    clean_samples: ComplexArray
    nominal_time_s: FloatArray
    true_observed_offset_hz: float
    measured_snr_db: float


@dataclass(frozen=True)
class FrequencyEstimate:
    """由分段相位最小二乘拟合得到的频偏结果。"""

    frequency_offset_hz: float
    phase_intercept_rad: float
    segment_times_s: FloatArray
    unwrapped_phase_rad: FloatArray
    fitted_phase_rad: FloatArray
    residual_phase_rms_rad: float
