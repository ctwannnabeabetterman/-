"""SNR 扫描下的时延估计、双向钟差和相干增益 Monte Carlo。"""

from __future__ import annotations

import numpy as np

from clock_model import LocalClock
from config import ChannelConfig, MonteCarloConfig, TwoWayConfig, WaveformConfig
from crlb import crlb_for_waveform
from models import MonteCarloResult, QLSCalibration
from two_way_sync import estimate_two_way, simulate_two_way_exchange
from waveforms import generate_two_tone


def run_delay_monte_carlo(
    waveform_config: WaveformConfig,
    calibration: QLSCalibration,
    config: MonteCarloConfig,
) -> MonteCarloResult:
    """在 ``6:3:36 dB`` 等可配 SNR 轴上执行固定种子的完整统计。

    每个 trial 都调用正式的四时间戳交换和双向估计器。上下行使用同一真传播
    时延和独立 AWGN，AP1 具有未知真钟差，处理时延通过时间戳公式抵消。估计
    钟差残差在载频处转为相位误差，用于统计两 AP 相干增益。
    """

    waveform = generate_two_tone(waveform_config)
    snr_axis = np.asarray(config.snr_db_values, dtype=np.float64)
    count = snr_axis.size
    integer_rmse = np.empty(count, dtype=np.float64)
    qls_rmse = np.empty(count, dtype=np.float64)
    lut_rmse = np.empty(count, dtype=np.float64)
    crlb_std = np.empty(count, dtype=np.float64)
    clock_rmse = np.empty(count, dtype=np.float64)
    coherent_gain = np.empty(count, dtype=np.float64)

    rng = np.random.default_rng(config.seed)
    fractions = rng.uniform(-0.5, 0.5, size=config.trials_per_snr)
    for snr_index, snr_db in enumerate(snr_axis):
        integer_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        qls_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        lut_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        clock_errors = np.empty(config.trials_per_snr, dtype=np.float64)

        for trial_index, fraction in enumerate(fractions):
            delay_samples = config.nominal_delay_samples + float(fraction)
            true_delay_s = delay_samples / waveform_config.sample_rate_hz
            link = ChannelConfig(
                propagation_delay_s=true_delay_s,
                amplitude=1.0,
                phase_rad=0.0,
                snr_db=float(snr_db),
            )
            two_way = TwoWayConfig(
                tx1_local_time_s=1e-3,
                processing_delay_s=config.processing_delay_s,
                coarse_up_delay_s=config.nominal_delay_samples
                / waveform_config.sample_rate_hz,
                coarse_down_delay_s=config.nominal_delay_samples
                / waveform_config.sample_rate_hz,
                gate_half_width_samples=config.gate_half_width_samples,
            )
            observation = simulate_two_way_exchange(
                waveform_config=waveform_config,
                two_way_config=two_way,
                ap0_clock=LocalClock(),
                ap1_clock=LocalClock(offset_s=config.clock_offset_truth_s),
                up_link=link,
                down_link=link,
                calibration=calibration,
                rng=rng,
            )
            clock_estimate = estimate_two_way(observation)
            raw_up = observation.up_measurement.raw_estimate
            integer_s = raw_up.integer_lag_samples / waveform_config.sample_rate_hz
            raw_s = raw_up.delay_s
            corrected_s = observation.up_measurement.corrected_delay_s
            integer_errors[trial_index] = integer_s - true_delay_s
            qls_errors[trial_index] = raw_s - true_delay_s
            lut_errors[trial_index] = corrected_s - true_delay_s
            clock_errors[trial_index] = (
                clock_estimate.ap1_offset_estimate_s - config.clock_offset_truth_s
            )

        integer_rmse[snr_index] = float(np.sqrt(np.mean(integer_errors**2)))
        qls_rmse[snr_index] = float(np.sqrt(np.mean(qls_errors**2)))
        lut_rmse[snr_index] = float(np.sqrt(np.mean(lut_errors**2)))
        clock_rmse[snr_index] = float(np.sqrt(np.mean(clock_errors**2)))
        residual_phase_rad = (
            2.0 * np.pi * waveform_config.carrier_frequency_hz * clock_errors
        )
        gain_linear = np.abs(1.0 + np.exp(1j * residual_phase_rad)) ** 2 / 2.0
        coherent_gain[snr_index] = float(10.0 * np.log10(np.mean(gain_linear)))
        crlb_std[snr_index] = crlb_for_waveform(
            waveform,
            waveform_config.tone_separation_hz,
            float(snr_db),
            config.noise_bandwidth_hz,
        ).std_s

    return MonteCarloResult(
        snr_db=snr_axis,
        integer_peak_rmse_s=integer_rmse,
        qls_rmse_s=qls_rmse,
        lut_rmse_s=lut_rmse,
        crlb_std_s=crlb_std,
        clock_offset_rmse_s=clock_rmse,
        coherent_gain_vs_incoherent_db=coherent_gain,
    )
