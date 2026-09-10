"""SNR 扫描下的时延估计、双向钟差和相干增益 Monte Carlo。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from beamforming import combine_two_ap
from clock_model import LocalClock
from config import (
    BeamformingConfig,
    ChannelConfig,
    MonteCarloConfig,
    PhaseFeedbackConfig,
    TwoWayConfig,
    WaveformConfig,
    FrequencySyncConfig,
)
from crlb import crlb_for_waveform
from models import MonteCarloResult, QLSCalibration
from oscillator_model import LocalOscillator
from phase_sync import compute_phase_weights, simulate_channel_feedback
from two_way_sync import estimate_two_way, simulate_two_way_exchange
from waveforms import generate_two_tone
from frequency_sync import simulate_frequency_reference, estimate_frequency_offset
from acquisition import AcquisitionError


def run_delay_monte_carlo(
    waveform_config: WaveformConfig,
    calibration: QLSCalibration,
    config: MonteCarloConfig,
    *,
    beamforming_config: BeamformingConfig | None = None,
    phase_feedback_config: PhaseFeedbackConfig | None = None,
) -> MonteCarloResult:
    """在 ``6:3:36 dB`` 等可配 SNR 轴上执行固定种子的完整统计。

    每个 trial 都调用正式的四时间戳交换和双向估计器。上下行使用同一真传播
    时延和独立 AWGN，AP1 具有未知真钟差，处理时延通过时间戳公式抵消。估计
    钟差补偿后，每个 trial 继续生成 RX 导频、计算反馈权重并合成数据波形，
    由实际合成功率统计两 AP 相干增益。
    """

    waveform = generate_two_tone(waveform_config)
    beamforming = beamforming_config or BeamformingConfig()
    phase_feedback = phase_feedback_config or PhaseFeedbackConfig()
    snr_axis = np.asarray(config.snr_db_values, dtype=np.float64)
    count = snr_axis.size
    integer_rmse = np.empty(count, dtype=np.float64)
    qls_rmse = np.empty(count, dtype=np.float64)
    lut_rmse = np.empty(count, dtype=np.float64)
    crlb_std = np.empty(count, dtype=np.float64)
    clock_rmse = np.empty(count, dtype=np.float64)
    coherent_gain = np.empty(count, dtype=np.float64)
    frequency_rmse = np.empty(count, dtype=np.float64)
    failure_rates = np.empty(count, dtype=np.float64)

    seed_sequence = np.random.SeedSequence(config.seed)
    fraction_seed, timing_seed, phase_seed, frequency_seed = seed_sequence.spawn(4)
    fraction_rng = np.random.default_rng(fraction_seed)
    fractions = fraction_rng.uniform(-0.5, 0.5, size=config.trials_per_snr)
    trial_count = count * config.trials_per_snr
    timing_trial_seeds = timing_seed.spawn(trial_count)
    phase_trial_seeds = phase_seed.spawn(trial_count)
    frequency_trial_seeds = frequency_seed.spawn(trial_count)
    for snr_index, snr_db in enumerate(snr_axis):
        integer_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        qls_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        lut_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        clock_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        coherent_gain_linear = np.empty(config.trials_per_snr, dtype=np.float64)
        frequency_errors = np.empty(config.trials_per_snr, dtype=np.float64)
        failures = 0

        for trial_index, fraction in enumerate(fractions):
            seed_index = snr_index * config.trials_per_snr + trial_index
            timing_rng = np.random.default_rng(timing_trial_seeds[seed_index])
            phase_rng = np.random.default_rng(phase_trial_seeds[seed_index])
            frequency_rng = np.random.default_rng(frequency_trial_seeds[seed_index])
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
            try:
                observation = simulate_two_way_exchange(
                    waveform_config=waveform_config,
                    two_way_config=two_way,
                    ap0_clock=LocalClock(),
                    ap1_clock=LocalClock(offset_s=config.clock_offset_truth_s),
                    up_link=link,
                    down_link=link,
                    calibration=calibration,
                    rng=timing_rng,
                )
            except AcquisitionError:
                failures += 1
                integer_errors[trial_index] = qls_errors[trial_index] = lut_errors[trial_index] = np.nan
                clock_errors[trial_index] = frequency_errors[trial_index] = np.nan
                coherent_gain_linear[trial_index] = np.nan
                continue
            clock_estimate = estimate_two_way(observation)
            raw_up = observation.up_measurement.raw_estimate
            integer_s = raw_up.integer_lag_samples / waveform_config.sample_rate_hz
            raw_s = raw_up.delay_s
            corrected_s = observation.up_measurement.corrected_delay_s
            true_arrival_s = two_way.tx1_local_time_s-config.clock_offset_truth_s+true_delay_s
            capture_start_s = observation.up_measurement.capture_start_local_s
            integer_errors[trial_index] = capture_start_s + integer_s - true_arrival_s
            qls_errors[trial_index] = capture_start_s + raw_s - true_arrival_s
            lut_errors[trial_index] = capture_start_s + corrected_s - true_arrival_s
            clock_errors[trial_index] = (
                clock_estimate.ap1_offset_estimate_s - config.clock_offset_truth_s
            )
            residual_clock_s = (
                config.clock_offset_truth_s - clock_estimate.ap1_offset_estimate_s
            )
            oscillator = LocalOscillator(
                frequency_offset_hz=float(frequency_rng.uniform(-config.initial_cfo_limit_hz, config.initial_cfo_limit_hz)),
                initial_phase_rad=float(phase_rng.uniform(-np.pi, np.pi)),
            )
            trial_feedback = replace(phase_feedback, snr_db=float(snr_db))
            reference = FrequencySyncConfig(cfo_hz=oscillator.frequency_offset_hz,
                observation_duration_s=config.frequency_observation_s, segment_duration_s=20e-6,
                snr_db=float(snr_db))
            frequency_observation = simulate_frequency_reference(reference, frequency_rng)
            frequency_estimate = estimate_frequency_offset(frequency_observation.samples, reference)
            oscillator.apply_frequency_correction(frequency_estimate.frequency_offset_hz,
                effective_time_s=two_way.tx1_local_time_s+config.frequency_observation_s)
            frequency_errors[trial_index] = oscillator.residual_frequency_offset_hz
            pilot_epoch_s = two_way.tx1_local_time_s + config.frequency_observation_s + 1e-3
            try:
                feedback = simulate_channel_feedback(
                    waveform_config,
                    beamforming,
                    trial_feedback,
                    oscillator,
                    residual_clock_offset_s=residual_clock_s,
                    pilot_epoch_s=pilot_epoch_s,
                    rng=phase_rng, calibration=calibration,
                )
            except AcquisitionError:
                failures += 1
                coherent_gain_linear[trial_index] = np.nan
                continue
            weights = compute_phase_weights(
                feedback.feedback_channel_estimates,
                beamforming.normalization,
            )
            channel_frequency_offset_hz = (
                trial_feedback.channel_phase_rate_rad_per_s / (2.0 * np.pi)
            )
            full_sync = combine_two_ap(
                waveform.samples,
                waveform_config.sample_rate_hz,
                np.array(
                    [
                        beamforming.ap0_propagation_delay_s,
                        beamforming.ap1_propagation_delay_s - residual_clock_s + feedback.tx_time_correction_s,
                    ],
                    dtype=np.float64,
                ),
                feedback.true_effective_channels_at_data,
                weights,
                channel_frequency_offset_hz + oscillator.residual_frequency_offset_hz,
                "full_sync_monte_carlo",
            )
            coherent_gain_linear[trial_index] = 10.0 ** (
                full_sync.metrics.gain_vs_incoherent_sum_db / 10.0
            )

        if not np.any(np.isfinite(coherent_gain_linear)):
            raise AcquisitionError(f"all_trials_failed_at_{snr_db:g}_dB")
        integer_rmse[snr_index] = float(np.sqrt(np.nanmean(integer_errors**2)))
        qls_rmse[snr_index] = float(np.sqrt(np.nanmean(qls_errors**2)))
        lut_rmse[snr_index] = float(np.sqrt(np.nanmean(lut_errors**2)))
        clock_rmse[snr_index] = float(np.sqrt(np.nanmean(clock_errors**2)))
        frequency_rmse[snr_index] = float(np.sqrt(np.nanmean(frequency_errors**2)))
        failure_rates[snr_index] = failures/config.trials_per_snr
        coherent_gain[snr_index] = float(
            10.0 * np.log10(np.nanmean(coherent_gain_linear))
        )
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
        residual_frequency_rmse_hz=frequency_rmse,
        acquisition_failure_rate=failure_rates,
    )
