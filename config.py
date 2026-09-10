"""集中管理波形、plant 真值、算法运行和统计参数。"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WaveformConfig:
    """脉冲双音复基带波形配置；时间单位为秒，频率单位为 Hz。"""

    sample_rate_hz: float = 200e6
    tone_separation_hz: float = 40e6
    pulse_duration_s: float = 10e-6
    rise_fall_s: float = 50e-9
    carrier_frequency_hz: float = 5.8e9
    normalization: str = "unit_active_power"

    def __post_init__(self) -> None:
        numeric_values = (
            self.sample_rate_hz,
            self.tone_separation_hz,
            self.pulse_duration_s,
            self.rise_fall_s,
            self.carrier_frequency_hz,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("波形配置必须为有限数值")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz 必须大于 0")
        if not 0.0 < self.tone_separation_hz < self.sample_rate_hz:
            raise ValueError("tone_separation_hz 必须位于 (0, sample_rate_hz) 内")
        if self.pulse_duration_s <= 0.0:
            raise ValueError("pulse_duration_s 必须大于 0")
        if not 0.0 <= self.rise_fall_s <= self.pulse_duration_s / 2.0:
            raise ValueError("rise_fall_s 必须位于 [0, pulse_duration_s/2] 内")
        if self.carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz 必须大于 0")
        if self.normalization != "unit_active_power":
            raise ValueError("阶段 1 仅支持 unit_active_power 归一化")
        if round(self.pulse_duration_s * self.sample_rate_hz) < 2:
            raise ValueError("脉冲至少需要两个离散样点")


@dataclass(frozen=True)
class ChannelConfig:
    """静态单径复基带信道配置。"""

    propagation_delay_s: float = 37.25 / 200e6
    amplitude: float = 1.0
    phase_rad: float = 0.0
    snr_db: float | None = 30.0
    fractional_delay_guard_samples: int = 64

    def __post_init__(self) -> None:
        if not math.isfinite(self.propagation_delay_s) or self.propagation_delay_s < 0.0:
            raise ValueError("propagation_delay_s 必须为非负有限数")
        if not math.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("amplitude 必须为非负有限数")
        if not math.isfinite(self.phase_rad):
            raise ValueError("phase_rad 必须为有限数")
        if self.snr_db is not None and not math.isfinite(self.snr_db):
            raise ValueError("snr_db 必须为有限数或 None")
        if self.fractional_delay_guard_samples < 16:
            raise ValueError("fractional_delay_guard_samples 至少为 16")


@dataclass(frozen=True)
class TwoWayConfig:
    """一次双向时间传递历元的公开调度和粗搜索参数。"""

    tx1_local_time_s: float = 1e-3
    processing_delay_s: float = 20e-6
    coarse_up_delay_s: float = 50e-9
    coarse_down_delay_s: float = 50e-9
    gate_half_width_samples: float = 2.5
    receive_pretrigger_s: float = 2e-6
    receive_window_s: float | None = None
    acquisition_threshold: float = 0.45

    def __post_init__(self) -> None:
        values = (
            self.tx1_local_time_s,
            self.processing_delay_s,
            self.coarse_up_delay_s,
            self.coarse_down_delay_s,
            self.gate_half_width_samples,
            self.receive_pretrigger_s,
            self.acquisition_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("双向时间传递配置必须为有限数")
        if self.processing_delay_s < 0.0:
            raise ValueError("processing_delay_s 必须为非负数")
        if self.coarse_up_delay_s < 0.0 or self.coarse_down_delay_s < 0.0:
            raise ValueError("粗传播时延必须为非负数")
        if self.gate_half_width_samples < 1.0:
            raise ValueError("gate_half_width_samples 至少为 1")
        if self.receive_pretrigger_s < 0 or not 0 < self.acquisition_threshold < 1:
            raise ValueError("接收前置时间须非负，捕获门限须位于 (0, 1)")
        if self.receive_window_s is not None and (
            not math.isfinite(self.receive_window_s) or self.receive_window_s <= 0
        ):
            raise ValueError("接收窗口须为正有限秒数")


@dataclass(frozen=True)
class ClockTrackingConfig:
    """周期双向同步和随机游走配置。"""

    rounds: int = 20
    sync_interval_s: float = 50e-3
    correction_gain: float = 1.0
    random_walk_std_s_per_sqrt_s: float = 0.0

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds 至少为 1")
        if not math.isfinite(self.sync_interval_s) or self.sync_interval_s <= 0.0:
            raise ValueError("sync_interval_s 必须为正有限数")
        if not math.isfinite(self.correction_gain) or not 0.0 < self.correction_gain <= 1.0:
            raise ValueError("correction_gain 必须位于 (0, 1] 内")
        if (
            not math.isfinite(self.random_walk_std_s_per_sqrt_s)
            or self.random_walk_std_s_per_sqrt_s < 0.0
        ):
            raise ValueError("随机游走强度必须为非负有限数")


@dataclass(frozen=True)
class ClockPlantConfig:
    """AP1 未控制仿真时钟的真偏差和真频差。"""

    initial_offset_s: float = 100e-9
    fractional_frequency_offset: float = 0.2e-6

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.initial_offset_s, self.fractional_frequency_offset)
        ):
            raise ValueError("时钟 plant 配置必须为有限数")
        if 1.0 + self.fractional_frequency_offset <= 0.0:
            raise ValueError("未控制时钟速率必须大于 0")


@dataclass(frozen=True)
class FrequencySyncConfig:
    """软件等效频率参考、分段相位估计和跟踪参数。"""

    sample_rate_hz: float = 200e6
    reference_frequency_hz: float = 10e6
    observation_duration_s: float = 2e-3
    segment_duration_s: float = 50e-6
    cfo_hz: float = 500.0
    sample_clock_offset_fraction: float = 0.0
    initial_phase_rad: float = 0.7
    amplitude: float = 1.0
    snr_db: float = 30.0
    tracker_alpha: float = 0.85

    def __post_init__(self) -> None:
        values = (
            self.sample_rate_hz,
            self.reference_frequency_hz,
            self.observation_duration_s,
            self.segment_duration_s,
            self.cfo_hz,
            self.sample_clock_offset_fraction,
            self.initial_phase_rad,
            self.amplitude,
            self.snr_db,
            self.tracker_alpha,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("频率同步配置必须为有限数")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz 必须大于 0")
        if self.reference_frequency_hz <= 0.0:
            raise ValueError("reference_frequency_hz 必须大于 0")
        if abs(self.reference_frequency_hz + self.cfo_hz) >= self.sample_rate_hz / 2.0:
            raise ValueError("参考信号必须位于复基带 Nyquist 区间内")
        if self.sample_clock_offset_fraction <= -1.0:
            raise ValueError("sample_clock_offset_fraction 必须大于 -1")
        if self.observation_duration_s <= 0.0 or self.segment_duration_s <= 0.0:
            raise ValueError("观测时长和分段时长必须大于 0")
        if self.segment_duration_s > self.observation_duration_s:
            raise ValueError("segment_duration_s 不能超过 observation_duration_s")
        if round(self.segment_duration_s * self.sample_rate_hz) < 2:
            raise ValueError("每个频率估计分段至少需要两个样点")
        if self.amplitude <= 0.0:
            raise ValueError("amplitude 必须大于 0")
        if not 0.0 <= self.tracker_alpha < 1.0:
            raise ValueError("tracker_alpha 必须位于 [0, 1) 内")


@dataclass(frozen=True)
class BeamformingConfig:
    """两 AP 下行相干合成的传播、同步误差和功率配置。"""

    ap0_propagation_delay_s: float = 50e-9
    ap1_propagation_delay_s: float = 50e-9
    ap0_amplitude: float = 1.0
    ap1_amplitude: float = 1.0
    ap0_channel_phase_rad: float = 0.2
    ap1_channel_phase_rad: float = -0.6
    normalization: str = "per_ap_fixed"

    def __post_init__(self) -> None:
        values = (
            self.ap0_propagation_delay_s,
            self.ap1_propagation_delay_s,
            self.ap0_amplitude,
            self.ap1_amplitude,
            self.ap0_channel_phase_rad,
            self.ap1_channel_phase_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("波束赋形配置必须为有限数")
        if self.ap0_propagation_delay_s < 0.0 or self.ap1_propagation_delay_s < 0.0:
            raise ValueError("传播时延必须为非负数")
        if self.ap0_amplitude < 0.0 or self.ap1_amplitude < 0.0:
            raise ValueError("信道幅度必须为非负数")
        if self.normalization not in {"per_ap_fixed", "total_fixed"}:
            raise ValueError("normalization 必须为 per_ap_fixed 或 total_fixed")


@dataclass(frozen=True)
class OscillatorConfig:
    """AP1 本振的仿真真频偏和初始相位。"""

    frequency_offset_hz: float = 600.0
    initial_phase_rad: float = 1.1

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.frequency_offset_hz, self.initial_phase_rad)
        ):
            raise ValueError("本振配置必须为有限数")


@dataclass(frozen=True)
class PhaseFeedbackConfig:
    """RX 导频、反馈延迟和相位量化配置。"""

    pilot_symbols: int = 1024
    snr_db: float = 32.0
    feedback_delay_s: float = 100e-6
    phase_quantization_bits: int | None = 12
    channel_phase_rate_rad_per_s: float = 0.0
    alignment_enabled: bool = True
    alignment_window_margin_s: float = 2e-6

    def __post_init__(self) -> None:
        if self.pilot_symbols < 16:
            raise ValueError("pilot_symbols 至少为 16")
        values = (self.snr_db, self.feedback_delay_s, self.channel_phase_rate_rad_per_s,
                  self.alignment_window_margin_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("相位反馈配置必须为有限数")
        if self.feedback_delay_s < 0.0:
            raise ValueError("feedback_delay_s 必须为非负数")
        if self.alignment_window_margin_s <= 0:
            raise ValueError("到达差测量窗口余量必须为正")
        if self.phase_quantization_bits is not None and self.phase_quantization_bits < 2:
            raise ValueError("phase_quantization_bits 必须至少为 2 或 None")


@dataclass(frozen=True)
class MonteCarloConfig:
    """SNR 扫描、随机种子和统计次数配置。"""

    snr_db_values: tuple[float, ...] = tuple(float(value) for value in range(6, 37, 3))
    trials_per_snr: int = 100
    seed: int = 2023
    nominal_delay_samples: int = 24
    gate_half_width_samples: float = 2.5
    noise_bandwidth_hz: float | None = None
    clock_offset_truth_s: float = 100e-9
    processing_delay_s: float = 20e-6
    initial_cfo_limit_hz: float = 1000.0
    frequency_observation_s: float = 100e-6

    def __post_init__(self) -> None:
        if len(self.snr_db_values) == 0:
            raise ValueError("snr_db_values 不能为空")
        if not all(math.isfinite(value) for value in self.snr_db_values):
            raise ValueError("SNR 扫描值必须为有限数")
        if self.trials_per_snr < 1:
            raise ValueError("trials_per_snr 至少为 1")
        if self.nominal_delay_samples < 4:
            raise ValueError("nominal_delay_samples 至少为 4")
        if not math.isfinite(self.gate_half_width_samples) or self.gate_half_width_samples < 1.0:
            raise ValueError("gate_half_width_samples 至少为 1")
        if self.noise_bandwidth_hz is not None and (
            not math.isfinite(self.noise_bandwidth_hz) or self.noise_bandwidth_hz <= 0.0
        ):
            raise ValueError("noise_bandwidth_hz 必须为正有限数或 None")
        if not math.isfinite(self.clock_offset_truth_s):
            raise ValueError("clock_offset_truth_s 必须为有限数")
        if not math.isfinite(self.processing_delay_s) or self.processing_delay_s < 0.0:
            raise ValueError("processing_delay_s 必须为非负有限数")
        if not math.isfinite(self.initial_cfo_limit_hz) or not 0 <= self.initial_cfo_limit_hz < 25000:
            raise ValueError("初始频偏须位于 20 us 分段捕获区间 [0, 25000) Hz 内")
        if not math.isfinite(self.frequency_observation_s) or self.frequency_observation_s < 40e-6:
            raise ValueError("频率观测时间须至少 40 us")


@dataclass(frozen=True)
class JointTrackingConfig:
    """持续联合同步、漂移、掉帧及锁定判据。"""

    rounds: int = 100
    interval_s: float = 50e-3
    seed: int = 42023
    clock_rate_walk_std: float = 1e-12
    oscillator_walk_std_hz: float = 0.02
    holdover_points: int = 9
    rate_fit_window: int = 8
    max_arrival_error_s: float = 20e-12
    max_phase_error_rad: float = 0.1
    min_gain_db: float = 2.9
    dropout_rounds: tuple[int, ...] = ()
    frequency_step_round: int | None = None
    frequency_step_hz: float = 100.0

    def __post_init__(self) -> None:
        if self.rounds < 3 or self.holdover_points < 2 or self.rate_fit_window < 2:
            raise ValueError("联合跟踪至少三轮、每周期至少两个保持观测点")
        values = (self.interval_s, self.clock_rate_walk_std, self.oscillator_walk_std_hz,
                  self.max_arrival_error_s, self.max_phase_error_rad, self.min_gain_db,
                  self.frequency_step_hz)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("联合跟踪参数必须有限")
        if min(self.interval_s, self.max_arrival_error_s, self.max_phase_error_rad) <= 0:
            raise ValueError("周期和锁定误差阈值必须为正")
        if min(self.clock_rate_walk_std, self.oscillator_walk_std_hz) < 0:
            raise ValueError("随机游走标准差不能为负")
        if any(i < 0 or i >= self.rounds for i in self.dropout_rounds):
            raise ValueError("掉帧轮次超出范围")
        if self.frequency_step_round is not None and not 0 <= self.frequency_step_round < self.rounds:
            raise ValueError("频偏突变轮次超出范围")
