"""同步、控制、波束赋形和统计流程的只读数据结构。"""

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
    """双音起点相对 ADC 首样点的本地时延；加 capture_start_local_s 得到时间戳。"""

    raw_estimate: DelayEstimate
    corrected_delay_s: float
    lut_adjustment_samples: float
    measured_snr_db: float | None
    capture_start_local_s: float = 0.0
    acquisition_score: float = 1.0


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
    reference_epoch_s: FloatArray
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


@dataclass(frozen=True)
class BeamformingMetrics:
    """联合接收功率相对三种参考功率的 dB 指标。"""

    gain_vs_single_ap_db: float
    gain_vs_incoherent_sum_db: float
    normalized_ideal_loss_db: float
    single_ap_reference_power: float
    incoherent_sum_power: float
    ideal_coherent_power: float


@dataclass(frozen=True)
class BeamformingResult:
    """一个同步状态下的两路贡献、合成波形和残余误差。"""

    state_name: str
    received_samples: ComplexArray
    ap0_contribution: ComplexArray
    ap1_contribution: ComplexArray
    signal_power: float
    residual_arrival_difference_s: float
    residual_frequency_offset_hz: float
    residual_phase_difference_rad: float
    metrics: BeamformingMetrics
    carrier_phase_difference_rad: float = 0.0
    waveform_coherence: float = 1.0


@dataclass(frozen=True)
class BeamformingPlantState:
    """同一数据历元下的未控制和已控制物理状态快照。

    该结构只属于仿真 plant；估计器和控制器不接收此结构。
    """

    data_epoch_s: float
    raw_clock_offset_s: float
    residual_clock_offset_s: float
    raw_frequency_offset_hz: float
    residual_frequency_offset_hz: float
    raw_ap1_phase_rad: float
    residual_ap1_phase_rad: float
    time_only_clock_offset_s: float | None = None


@dataclass(frozen=True)
class ChannelFeedbackResult:
    """导频观测经过延迟、量化反馈后在数据历元可用的复信道估计。"""

    pilot_epoch_s: float
    data_epoch_s: float
    raw_channel_estimates: ComplexArray
    feedback_channel_estimates: ComplexArray
    true_effective_channels_at_data: ComplexArray
    phase_error_at_data_rad: FloatArray
    tx_time_correction_s: float = 0.0
    measured_arrival_difference_s: float = 0.0


@dataclass(frozen=True)
class CRLBResult:
    """一个活动区每样点 SNR 对应的双音时延理论下界。"""

    std_s: float
    variance_s2: float
    signal_energy: float
    noise_psd: float
    noise_bandwidth_hz: float
    active_sample_snr_db: float
    pulse_energy_snr_linear: float
    mean_squared_angular_bandwidth_rad2_s2: float


@dataclass(frozen=True)
class MonteCarloResult:
    """各 SNR 点的时延、钟差和两 AP 相干合成统计。"""

    snr_db: FloatArray
    integer_peak_rmse_s: FloatArray
    qls_rmse_s: FloatArray
    lut_rmse_s: FloatArray
    crlb_std_s: FloatArray
    clock_offset_rmse_s: FloatArray
    coherent_gain_vs_incoherent_db: FloatArray
    residual_frequency_rmse_hz: FloatArray
    acquisition_failure_rate: FloatArray
