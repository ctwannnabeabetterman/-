"""以两次真实波形到达估计构造四时间戳双向时间传递。"""

from __future__ import annotations

import numpy as np

from channel import propagate_static_link
from clock_model import LocalClock
from config import ChannelConfig, TwoWayConfig, WaveformConfig
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from lut_calibration import correct_qls_fraction, wrap_fractional_sample
from models import LinkDelayMeasurement, QLSCalibration, TwoWayEstimate, TwoWayObservation
from waveforms import generate_two_tone


def _measure_link_delay(
    waveform_config: WaveformConfig,
    link: ChannelConfig,
    coarse_delay_s: float,
    gate_half_width_samples: float,
    calibration: QLSCalibration,
    rng: np.random.Generator,
) -> LinkDelayMeasurement:
    """经静态链路和正式匹配滤波路径测量一次单向传播时延。"""

    waveform = generate_two_tone(waveform_config)
    received = propagate_static_link(
        waveform.samples,
        sample_rate_hz=waveform_config.sample_rate_hz,
        config=link,
        rng=rng,
    )
    correlation = fft_matched_filter(received.samples, waveform.samples)
    gate = DelaySearchGate(
        center_s=coarse_delay_s,
        half_width_s=gate_half_width_samples / waveform_config.sample_rate_hz,
    )
    raw = estimate_delay(correlation, waveform_config.sample_rate_hz, gate)
    if not raw.qls_valid:
        raise RuntimeError("单向时延测量的 QLS 插值无效")
    corrected_fraction = float(
        correct_qls_fraction(raw.fractional_offset_samples, calibration)
    )
    lut_adjustment = float(
        wrap_fractional_sample(corrected_fraction - raw.fractional_offset_samples)
    )
    corrected_delay_s = raw.delay_s + lut_adjustment / waveform_config.sample_rate_hz
    return LinkDelayMeasurement(
        raw_estimate=raw,
        corrected_delay_s=float(corrected_delay_s),
        lut_adjustment_samples=lut_adjustment,
        measured_snr_db=received.measured_snr_db,
    )


def simulate_two_way_exchange(
    waveform_config: WaveformConfig,
    two_way_config: TwoWayConfig,
    ap0_clock: LocalClock,
    ap1_clock: LocalClock,
    up_link: ChannelConfig,
    down_link: ChannelConfig,
    calibration: QLSCalibration,
    rng: np.random.Generator,
) -> TwoWayObservation:
    """模拟 AP1→AP0→AP1 交换并保存四个本地时间戳。"""

    t_tx1_local = two_way_config.tx1_local_time_s
    t_tx1_true = ap1_clock.true_time_for_reading(t_tx1_local)
    up_measurement = _measure_link_delay(
        waveform_config,
        up_link,
        two_way_config.coarse_up_delay_s,
        two_way_config.gate_half_width_samples,
        calibration,
        rng,
    )
    t_rx0_local = (
        ap0_clock.read_time(t_tx1_true)
        + ap0_clock.effective_rate * up_measurement.corrected_delay_s
    )

    t_tx0_local = t_rx0_local + two_way_config.processing_delay_s
    t_tx0_true = ap0_clock.true_time_for_reading(t_tx0_local)
    down_measurement = _measure_link_delay(
        waveform_config,
        down_link,
        two_way_config.coarse_down_delay_s,
        two_way_config.gate_half_width_samples,
        calibration,
        rng,
    )
    t_rx1_local = (
        ap1_clock.read_time(t_tx0_true)
        + ap1_clock.effective_rate * down_measurement.corrected_delay_s
    )

    return TwoWayObservation(
        t_tx1_s=float(t_tx1_local),
        t_rx0_s=float(t_rx0_local),
        t_tx0_s=float(t_tx0_local),
        t_rx1_s=float(t_rx1_local),
        up_measurement=up_measurement,
        down_measurement=down_measurement,
    )


def estimate_two_way(observation: TwoWayObservation) -> TwoWayEstimate:
    """按论文四时间戳公式估计 AP1 校正量和对称传播时延。"""

    forward_interval = observation.t_rx0_s - observation.t_tx1_s
    reverse_interval = observation.t_rx1_s - observation.t_tx0_s
    correction = 0.5 * (forward_interval - reverse_interval)
    propagation = 0.5 * (forward_interval + reverse_interval)
    return TwoWayEstimate(
        clock_correction_s=float(correction),
        ap1_offset_estimate_s=float(-correction),
        symmetric_propagation_delay_s=float(propagation),
        forward_interval_s=float(forward_interval),
        reverse_interval_s=float(reverse_interval),
    )
