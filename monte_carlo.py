"""SNR 扫描下的时延估计、双向钟差和相干增益 Monte Carlo。"""

from __future__ import annotations

import math

import numpy as np

from channel import add_awgn, apply_fractional_delay
from config import MonteCarloConfig, WaveformConfig
from crlb import crlb_for_waveform
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from lut_calibration import correct_qls_fraction, wrap_fractional_sample
from models import DelayEstimate, MonteCarloResult, QLSCalibration
from waveforms import generate_two_tone


def _delay_estimates_s(
    received: np.ndarray,
    template: np.ndarray,
    sample_rate_hz: float,
    gate: DelaySearchGate,
    calibration: QLSCalibration,
) -> tuple[float, float, float]:
    """返回整数峰值、原始 QLS 和 LUT 校正时延，单位为秒。"""

    correlation = fft_matched_filter(received, template)
    estimate: DelayEstimate = estimate_delay(correlation, sample_rate_hz, gate)
    integer_s = estimate.integer_lag_samples / sample_rate_hz
    raw_s = estimate.delay_s
    if not estimate.qls_valid:
        return float(integer_s), float(raw_s), float(raw_s)
    corrected_fraction = float(
        correct_qls_fraction(estimate.fractional_offset_samples, calibration)
    )
    adjustment_samples = float(
        wrap_fractional_sample(corrected_fraction - estimate.fractional_offset_samples)
    )
    corrected_s = raw_s + adjustment_samples / sample_rate_hz
    return float(integer_s), float(raw_s), float(corrected_s)


def run_delay_monte_carlo(
    waveform_config: WaveformConfig,
    calibration: QLSCalibration,
    config: MonteCarloConfig,
) -> MonteCarloResult:
    """在 ``6:3:36 dB`` 等可配 SNR 轴上执行固定种子的完整统计。

    每个 trial 对同一对称传播时延生成两次独立噪声观测。两次 LUT 时延误差
    之差的一半构成双向钟差误差；该残差在载频处转为相位误差，用于统计
    每 AP 功率固定时相对非相干功率和的两 AP 相干增益。
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
    gate = DelaySearchGate(
        center_s=config.nominal_delay_samples / waveform_config.sample_rate_hz,
        half_width_s=config.gate_half_width_samples / waveform_config.sample_rate_hz,
    )
    trailing = config.nominal_delay_samples + 66
    tx_buffer = np.pad(waveform.samples, (0, trailing)).astype(np.complex128)

    for snr_index, snr_db in enumerate(snr_axis):
        integer_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        qls_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        lut_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        clock_errors = np.empty(config.trials_per_snr, dtype=np.float64)

        for trial_index, fraction in enumerate(fractions):
            delay_samples = config.nominal_delay_samples + float(fraction)
            true_delay_s = delay_samples / waveform_config.sample_rate_hz
            clean = apply_fractional_delay(
                tx_buffer,
                true_delay_s,
                waveform_config.sample_rate_hz,
            )
            active_start = max(0, int(math.floor(delay_samples)))
            active_stop = min(clean.size, int(math.ceil(delay_samples)) + waveform.samples.size)
            active_mask = np.zeros(clean.shape, dtype=np.bool_)
            active_mask[active_start:active_stop] = True

            first = add_awgn(clean, float(snr_db), active_mask, rng)
            second = add_awgn(clean, float(snr_db), active_mask, rng)
            integer_s, raw_s, corrected_s = _delay_estimates_s(
                first.samples,
                waveform.samples,
                waveform_config.sample_rate_hz,
                gate,
                calibration,
            )
            _, _, corrected_second_s = _delay_estimates_s(
                second.samples,
                waveform.samples,
                waveform_config.sample_rate_hz,
                gate,
                calibration,
            )
            integer_errors[trial_index] = integer_s - true_delay_s
            qls_errors[trial_index] = raw_s - true_delay_s
            lut_errors[trial_index] = corrected_s - true_delay_s
            clock_errors[trial_index] = 0.5 * (
                (corrected_s - true_delay_s) - (corrected_second_s - true_delay_s)
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
