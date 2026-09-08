"""多轮同步及后续 Monte Carlo 实验的统一编排模块。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from clock_model import LocalClock
from config import ChannelConfig, ClockTrackingConfig, TwoWayConfig, WaveformConfig
from models import ClockTrackingResult, QLSCalibration
from two_way_sync import estimate_two_way, simulate_two_way_exchange


def run_clock_tracking(
    waveform_config: WaveformConfig,
    two_way_config: TwoWayConfig,
    tracking_config: ClockTrackingConfig,
    ap0_clock: LocalClock,
    ap1_clock: LocalClock,
    up_link: ChannelConfig,
    down_link: ChannelConfig,
    calibration: QLSCalibration,
    rng: np.random.Generator,
) -> ClockTrackingResult:
    """按固定本地间隔执行完整双向交换并更新 AP1 软件时钟。"""

    rounds = tracking_config.rounds
    epoch_true_s = np.empty(rounds, dtype=np.float64)
    raw_offset_s = np.empty(rounds, dtype=np.float64)
    estimated_offset_s = np.empty(rounds, dtype=np.float64)
    applied_correction_s = np.empty(rounds, dtype=np.float64)
    residual_before_s = np.empty(rounds, dtype=np.float64)
    residual_after_s = np.empty(rounds, dtype=np.float64)
    random_walk_accumulated_s = np.empty(rounds, dtype=np.float64)
    accumulated_walk = 0.0

    for round_index in range(rounds):
        if round_index > 0 and tracking_config.random_walk_std_s_per_sqrt_s > 0.0:
            increment = (
                tracking_config.random_walk_std_s_per_sqrt_s
                * np.sqrt(tracking_config.sync_interval_s)
                * rng.standard_normal()
            )
            ap1_clock.offset_s += float(increment)
            accumulated_walk += float(increment)

        tx1_local_time_s = (
            two_way_config.tx1_local_time_s
            + round_index * tracking_config.sync_interval_s
        )
        round_config = replace(two_way_config, tx1_local_time_s=tx1_local_time_s)
        current_true_time_s = ap1_clock.true_time_for_reading(tx1_local_time_s)
        epoch_true_s[round_index] = current_true_time_s
        raw_offset_s[round_index] = (
            ap1_clock.offset_s
            + ap1_clock.fractional_frequency_offset * current_true_time_s
        )
        residual_before_s[round_index] = (
            ap1_clock.read_time(current_true_time_s)
            - ap0_clock.read_time(current_true_time_s)
        )

        observation = simulate_two_way_exchange(
            waveform_config=waveform_config,
            two_way_config=round_config,
            ap0_clock=ap0_clock,
            ap1_clock=ap1_clock,
            up_link=up_link,
            down_link=down_link,
            calibration=calibration,
            rng=rng,
        )
        estimate = estimate_two_way(observation)
        correction = tracking_config.correction_gain * estimate.clock_correction_s
        estimated_offset_s[round_index] = estimate.ap1_offset_estimate_s
        applied_correction_s[round_index] = correction
        ap1_clock.apply_time_correction(correction)
        residual_after_s[round_index] = (
            ap1_clock.read_time(current_true_time_s)
            - ap0_clock.read_time(current_true_time_s)
        )
        random_walk_accumulated_s[round_index] = accumulated_walk

    return ClockTrackingResult(
        epoch_true_s=epoch_true_s,
        raw_offset_s=raw_offset_s,
        estimated_offset_s=estimated_offset_s,
        applied_correction_s=applied_correction_s,
        residual_before_s=residual_before_s,
        residual_after_s=residual_after_s,
        random_walk_accumulated_s=random_walk_accumulated_s,
    )
