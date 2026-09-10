"""两 AP 联合下行、四种同步状态和相干增益指标。"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from channel import apply_fractional_delay
from config import BeamformingConfig, WaveformConfig
from models import BeamformingMetrics, BeamformingPlantState, BeamformingResult
from phase_sync import complex_los_channel, compute_phase_weights


def _complex_pair(values: ArrayLike, name: str) -> NDArray[np.complex128]:
    pair = np.asarray(values, dtype=np.complex128)
    if pair.shape != (2,) or not np.all(np.isfinite(pair)):
        raise ValueError(f"{name} 必须为两个有限复数")
    return pair


def _power_ratio_db(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("参考功率必须大于 0")
    if numerator <= 0.0:
        return -math.inf
    return float(10.0 * np.log10(numerator / denominator))


def _ideal_loss_db(ideal_power: float, actual_power: float) -> float:
    if ideal_power <= 0.0:
        raise ValueError("理想相干功率必须大于 0")
    if actual_power <= 0.0:
        return math.inf
    return float(10.0 * np.log10(ideal_power / actual_power))


def combine_two_ap(
    baseband_samples: ArrayLike,
    sample_rate_hz: float,
    arrival_delays_s: ArrayLike,
    effective_channels: ArrayLike,
    weights: ArrayLike,
    residual_frequency_offset_hz: float,
    state_name: str,
) -> BeamformingResult:
    """按连续时间到达差和 AP1 残余频偏合成两路复基带信号。

    功率在两路完整重叠的区间内统计。单 AP 参考功率采用 AP0 未缩放的
    发射功率，因此 ``total_fixed`` 归一化的理想双 AP 增益为 3.01 dB。
    """

    samples = np.asarray(baseband_samples, dtype=np.complex128)
    delays = np.asarray(arrival_delays_s, dtype=np.float64)
    channels = _complex_pair(effective_channels, "effective_channels")
    tx_weights = _complex_pair(weights, "weights")
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("baseband_samples 必须为非空有限一维数组")
    if delays.shape != (2,) or not np.all(np.isfinite(delays)):
        raise ValueError("arrival_delays_s 必须为两个有限实数")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz 必须为正有限数")
    if not math.isfinite(residual_frequency_offset_hz):
        raise ValueError("residual_frequency_offset_hz 必须为有限数")

    delay_samples = delays * sample_rate_hz
    guard = 64
    left = guard + int(math.ceil(max(0.0, -float(np.min(delay_samples)))))
    right = guard + int(math.ceil(max(0.0, float(np.max(delay_samples)))))
    padded = np.pad(samples, (left, right)).astype(np.complex128)

    delayed0 = apply_fractional_delay(padded, delays[0], sample_rate_hz, guard_samples=guard)
    delayed1 = apply_fractional_delay(padded, delays[1], sample_rate_hz, guard_samples=guard)
    time_s = (np.arange(padded.size, dtype=np.float64) - left) / sample_rate_hz
    frequency_phase = np.exp(1j * 2.0 * np.pi * residual_frequency_offset_hz * time_s)
    contribution0 = channels[0] * tx_weights[0] * delayed0
    contribution1 = channels[1] * tx_weights[1] * delayed1 * frequency_phase
    combined = np.asarray(contribution0 + contribution1, dtype=np.complex128)

    starts = left + delay_samples
    stops = starts + samples.size
    common_start = max(0, int(math.ceil(float(np.max(starts)))))
    common_stop = min(padded.size, int(math.floor(float(np.min(stops)))))
    if common_stop <= common_start:
        raise ValueError("两路信号没有可用于功率统计的重叠区间")
    window = slice(common_start, common_stop)

    signal_power = float(np.mean(np.abs(combined[window]) ** 2))
    single_reference = float(
        np.mean(np.abs(channels[0] * delayed0[window]) ** 2)
    )
    incoherent_sum = float(
        np.mean(
            np.abs(contribution0[window]) ** 2 + np.abs(contribution1[window]) ** 2
        )
    )
    ideal_coherent = float(
        np.mean((np.abs(contribution0[window]) + np.abs(contribution1[window])) ** 2)
    )

    reference_index = 0.5 * (common_start + common_stop - 1)
    reference_time_s = (reference_index - left) / sample_rate_hz
    effective0 = channels[0] * tx_weights[0]
    effective1 = (
        channels[1]
        * tx_weights[1]
        * np.exp(1j * 2.0 * np.pi * residual_frequency_offset_hz * reference_time_s)
    )
    carrier_phase_difference = float(np.angle(effective1 * np.conj(effective0)))
    cross_power = np.vdot(contribution0[window], contribution1[window])
    phase_difference = float(np.angle(cross_power))
    coherence = float(abs(cross_power) / max(np.sqrt(
        np.vdot(contribution0[window], contribution0[window]).real *
        np.vdot(contribution1[window], contribution1[window]).real), np.finfo(float).tiny))

    metrics = BeamformingMetrics(
        gain_vs_single_ap_db=_power_ratio_db(signal_power, single_reference),
        gain_vs_incoherent_sum_db=_power_ratio_db(signal_power, incoherent_sum),
        normalized_ideal_loss_db=_ideal_loss_db(ideal_coherent, signal_power),
        single_ap_reference_power=single_reference,
        incoherent_sum_power=incoherent_sum,
        ideal_coherent_power=ideal_coherent,
    )
    return BeamformingResult(
        state_name=state_name,
        received_samples=combined,
        ap0_contribution=np.asarray(contribution0, dtype=np.complex128),
        ap1_contribution=np.asarray(contribution1, dtype=np.complex128),
        signal_power=signal_power,
        residual_arrival_difference_s=float(delays[1] - delays[0]),
        residual_frequency_offset_hz=float(residual_frequency_offset_hz),
        residual_phase_difference_rad=phase_difference,
        metrics=metrics,
        carrier_phase_difference_rad=carrier_phase_difference,
        waveform_coherence=coherence,
    )


def simulate_four_sync_states(
    baseband_samples: ArrayLike,
    waveform_config: WaveformConfig,
    config: BeamformingConfig,
    plant_state: BeamformingPlantState,
    *,
    channel_estimates: ArrayLike,
    tx_time_correction_s: float = 0.0,
) -> dict[str, BeamformingResult]:
    """从显式 plant 快照比较四种状态，不在此处接收控制器估计量。"""

    plant_values = (
        plant_state.data_epoch_s,
        plant_state.raw_clock_offset_s,
        plant_state.residual_clock_offset_s,
        plant_state.raw_frequency_offset_hz,
        plant_state.residual_frequency_offset_hz,
        plant_state.raw_ap1_phase_rad,
        plant_state.residual_ap1_phase_rad,
    )
    if not all(math.isfinite(value) for value in plant_values):
        raise ValueError("plant_state 必须只包含有限数")
    estimates = _complex_pair(channel_estimates, "channel_estimates")

    h0 = complex_los_channel(
        config.ap0_amplitude,
        config.ap0_propagation_delay_s,
        waveform_config.carrier_frequency_hz,
        config.ap0_channel_phase_rad,
    )
    h1_static = complex_los_channel(
        config.ap1_amplitude,
        config.ap1_propagation_delay_s,
        waveform_config.carrier_frequency_hz,
        config.ap1_channel_phase_rad,
    )
    raw_channels = np.array(
        [h0, h1_static * np.exp(1j * plant_state.raw_ap1_phase_rad)],
        dtype=np.complex128,
    )
    corrected_channels = np.array(
        [h0, h1_static * np.exp(1j * plant_state.residual_ap1_phase_rad)],
        dtype=np.complex128,
    )

    amplitude_scale = 1.0 if config.normalization == "per_ap_fixed" else 1.0 / np.sqrt(2.0)
    uncorrected_weights = np.full(2, amplitude_scale, dtype=np.complex128)
    phase_weights = compute_phase_weights(estimates, config.normalization)

    unsynchronized_delays = np.array(
        [
            config.ap0_propagation_delay_s,
            config.ap1_propagation_delay_s - plant_state.raw_clock_offset_s,
        ],
        dtype=np.float64,
    )
    time_corrected_delays = np.array(
        [
            config.ap0_propagation_delay_s,
            config.ap1_propagation_delay_s - plant_state.residual_clock_offset_s,
        ],
        dtype=np.float64,
    )

    states: dict[str, BeamformingResult] = {}
    states["unsynchronized"] = combine_two_ap(
        baseband_samples,
        waveform_config.sample_rate_hz,
        unsynchronized_delays,
        raw_channels,
        uncorrected_weights,
        plant_state.raw_frequency_offset_hz,
        "unsynchronized",
    )
    states["time_only"] = combine_two_ap(
        baseband_samples,
        waveform_config.sample_rate_hz,
        np.array([config.ap0_propagation_delay_s,
                  config.ap1_propagation_delay_s - (
                      plant_state.time_only_clock_offset_s
                      if plant_state.time_only_clock_offset_s is not None
                      else plant_state.residual_clock_offset_s)]),
        raw_channels,
        uncorrected_weights,
        plant_state.raw_frequency_offset_hz,
        "time_only",
    )
    states["time_frequency"] = combine_two_ap(
        baseband_samples,
        waveform_config.sample_rate_hz,
        time_corrected_delays,
        corrected_channels,
        uncorrected_weights,
        plant_state.residual_frequency_offset_hz,
        "time_frequency",
    )
    states["full_sync"] = combine_two_ap(
        baseband_samples,
        waveform_config.sample_rate_hz,
        time_corrected_delays + np.array([0., tx_time_correction_s]),
        corrected_channels,
        phase_weights,
        plant_state.residual_frequency_offset_hz,
        "full_sync",
    )
    return states
