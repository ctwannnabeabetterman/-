"""集中管理阶段 1 的波形和静态信道参数。"""

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

    def __post_init__(self) -> None:
        values = (
            self.tx1_local_time_s,
            self.processing_delay_s,
            self.coarse_up_delay_s,
            self.coarse_down_delay_s,
            self.gate_half_width_samples,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("双向时间传递配置必须为有限数")
        if self.processing_delay_s < 0.0:
            raise ValueError("processing_delay_s 必须为非负数")
        if self.coarse_up_delay_s < 0.0 or self.coarse_down_delay_s < 0.0:
            raise ValueError("粗传播时延必须为非负数")
        if self.gate_half_width_samples < 1.0:
            raise ValueError("gate_half_width_samples 至少为 1")


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
    ap1_clock_offset_s: float = 12.25e-9
    ap1_cfo_hz: float = 600.0
    ap1_initial_phase_rad: float = 1.1
    normalization: str = "per_ap_fixed"

    def __post_init__(self) -> None:
        values = (
            self.ap0_propagation_delay_s,
            self.ap1_propagation_delay_s,
            self.ap0_amplitude,
            self.ap1_amplitude,
            self.ap0_channel_phase_rad,
            self.ap1_channel_phase_rad,
            self.ap1_clock_offset_s,
            self.ap1_cfo_hz,
            self.ap1_initial_phase_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("波束赋形配置必须为有限数")
        if self.ap0_propagation_delay_s < 0.0 or self.ap1_propagation_delay_s < 0.0:
            raise ValueError("传播时延必须为非负数")
        if self.ap0_amplitude < 0.0 or self.ap1_amplitude < 0.0:
            raise ValueError("信道幅度必须为非负数")
        if self.normalization not in {"per_ap_fixed", "total_fixed"}:
            raise ValueError("normalization 必须为 per_ap_fixed 或 total_fixed")
