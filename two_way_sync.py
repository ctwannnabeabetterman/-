"""以两次真实波形到达估计构造四时间戳双向时间传递。"""

from __future__ import annotations

import numpy as np

from acquisition import measure_capture, simulate_capture, sync_burst
from clock_model import LocalClock
from config import ChannelConfig, TwoWayConfig, WaveformConfig
from models import LinkDelayMeasurement, QLSCalibration, TwoWayEstimate, TwoWayObservation


def _measure_link_delay(
    waveform_config: WaveformConfig,
    link: ChannelConfig,
    coarse_delay_s: float,
    gate_half_width_samples: float,
    calibration: QLSCalibration,
    rng: np.random.Generator,
    tx_clock: LocalClock,
    rx_clock: LocalClock,
    tx_local_s: float,
    schedule: TwoWayConfig,
) -> LinkDelayMeasurement:
    """在接收机本地调度窗口内采样；算法只读 IQ 和首样点时间戳。"""
    burst, _, prefix = sync_burst(waveform_config)
    fs = waveform_config.sample_rate_hz
    burst_tx_local = tx_local_s - prefix/fs
    # The announced transmitter timestamp and public coarse range set the window;
    # neither the true clock offset nor the actual propagation delay centres it.
    rx_start_local = burst_tx_local + coarse_delay_s - schedule.receive_pretrigger_s
    duration = schedule.receive_window_s
    if duration is None:
        duration = len(burst)/fs + 2*schedule.receive_pretrigger_s
    tx_true = tx_clock.true_time_for_reading(burst_tx_local)
    rx_true = rx_clock.true_time_for_reading(rx_start_local)
    source_start = (rx_true-tx_true-link.propagation_delay_s)*fs*tx_clock.effective_rate
    capture = simulate_capture(burst, fs, start_local_s=rx_start_local,
        start_source_sample=source_start,
        sample_step=tx_clock.effective_rate/rx_clock.effective_rate,
        count=int(np.ceil(duration*fs)), amplitude=link.amplitude*np.exp(1j*link.phase_rad),
        snr_db=link.snr_db, rng=rng)
    return measure_capture(capture, waveform_config, calibration,
        threshold=schedule.acquisition_threshold, fine_half_width_samples=gate_half_width_samples)


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
    up_measurement = _measure_link_delay(
        waveform_config,
        up_link,
        two_way_config.coarse_up_delay_s,
        two_way_config.gate_half_width_samples,
        calibration,
        rng,
        ap1_clock, ap0_clock, t_tx1_local, two_way_config,
    )
    t_rx0_local = up_measurement.capture_start_local_s + up_measurement.corrected_delay_s

    # Reply preamble starts after reception of the full pulse and processing.
    _, _, prefix = sync_burst(waveform_config)
    t_tx0_local = (t_rx0_local + waveform_config.pulse_duration_s
                  + two_way_config.processing_delay_s + prefix/waveform_config.sample_rate_hz)
    down_measurement = _measure_link_delay(
        waveform_config,
        down_link,
        two_way_config.coarse_down_delay_s,
        two_way_config.gate_half_width_samples,
        calibration,
        rng,
        ap0_clock, ap1_clock, t_tx0_local, two_way_config,
    )
    t_rx1_local = down_measurement.capture_start_local_s + down_measurement.corrected_delay_s

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
