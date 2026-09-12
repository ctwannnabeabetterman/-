"""论文三种配置共用的波形级时间同步通信仿真。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acquisition import AcquisitionError, simulate_capture
from clock_model import LocalClock
from config import (
    ChannelConfig,
    FrequencySyncConfig,
    ThreeExperimentConfig,
    TwoWayConfig,
    WaveformConfig,
)
from crlb import crlb_for_waveform
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from frequency_sync import simulate_frequency_reference
from lut_calibration import (
    build_qls_lut,
    correct_qls_fraction,
    wrap_fractional_sample,
)
from models import QLSCalibration, ThreeExperimentResult, Waveform
from two_way_sync import estimate_two_way, simulate_two_way_exchange
from waveforms import generate_two_tone


@dataclass(frozen=True)
class _ExperimentProfile:
    key: str
    label: str
    time_link_mode: str
    frequency_link_mode: str
    propagation_delay_s: float


def _profiles(config: ThreeExperimentConfig) -> tuple[_ExperimentProfile, ...]:
    return (
        _ExperimentProfile(
            "cabled",
            "Cabled time-frequency transfer",
            "cabled",
            "cabled",
            config.cabled_time_delay_s,
        ),
        _ExperimentProfile(
            "wireless_time",
            "Wireless time, cabled frequency",
            "wireless",
            "cabled",
            config.wireless_time_delay_s,
        ),
        _ExperimentProfile(
            "wireless_time_frequency",
            "Wireless time-frequency transfer",
            "wireless",
            "wireless",
            config.wireless_time_delay_s,
        ),
    )


def _estimate_wireless_clock_rate(
    waveform_config: WaveformConfig,
    config: ThreeExperimentConfig,
    true_rate_offset: float,
    rng: np.random.Generator,
) -> float:
    """由连续 10 MHz 参考的两个带噪 IQ 窗口反演 AP1 时钟频差。

    论文的自混频频率参考是连续信号。这里保留两个有限 ADC 窗口，
    但用公开的本地窗口间隔保留连续载波在整个同步周期内的相位积累；
    这样无需为每个 trial 分配一千万个连续样点。
    """

    capture_duration_s = config.frequency_capture_duration_s / 2.0
    baseline_local_s = config.sync_interval_s
    phase_0 = float(rng.uniform(-np.pi, np.pi))
    phase_advance = (
        2.0
        * np.pi
        * config.frequency_reference_hz
        * baseline_local_s
        / (1.0 + true_rate_offset)
    )
    capture_phases = (
        phase_0,
        float(np.angle(np.exp(1j * (phase_0 + phase_advance)))),
    )
    coherent: list[complex] = []
    for phase in capture_phases:
        reference_config = FrequencySyncConfig(
            sample_rate_hz=waveform_config.sample_rate_hz,
            reference_frequency_hz=config.frequency_reference_hz,
            observation_duration_s=capture_duration_s,
            segment_duration_s=capture_duration_s / 2.0,
            cfo_hz=0.0,
            sample_clock_offset_fraction=true_rate_offset,
            initial_phase_rad=phase,
            snr_db=config.frequency_reference_snr_db,
            tracker_alpha=0.0,
        )
        observation = simulate_frequency_reference(reference_config, rng)
        derotated = observation.samples * np.exp(
            -1j
            * 2.0
            * np.pi
            * config.frequency_reference_hz
            * observation.nominal_time_s
        )
        coherent.append(complex(np.sum(derotated)))
    phase_difference = float(np.angle(coherent[1] * np.conj(coherent[0])))
    frequency_offset_hz = phase_difference / (2.0 * np.pi * baseline_local_s)
    observed_frequency_hz = config.frequency_reference_hz + frequency_offset_hz
    if observed_frequency_hz <= 0.0:
        raise RuntimeError("无线频率参考估计产生非正频率")
    return float(config.frequency_reference_hz / observed_frequency_hz - 1.0)


def _corrected_pulse_arrival_s(
    capture_samples: np.ndarray,
    capture_start_s: float,
    waveform: Waveform,
    calibration: QLSCalibration,
    expected_delay_s: float,
    gate_half_width_samples: float,
) -> float:
    """由接收 IQ 的匹配滤波、QLS 和 LUT 得到绝对到达时间。"""

    correlation = fft_matched_filter(capture_samples, waveform.samples)
    raw = estimate_delay(
        correlation,
        waveform.sample_rate_hz,
        DelaySearchGate(
            center_s=expected_delay_s,
            half_width_s=gate_half_width_samples / waveform.sample_rate_hz,
        ),
    )
    if not raw.qls_valid:
        raise AcquisitionError("beamforming_fine_peak_invalid")
    corrected_fraction = float(
        correct_qls_fraction(raw.fractional_offset_samples, calibration)
    )
    adjustment = float(
        wrap_fractional_sample(
            corrected_fraction - raw.fractional_offset_samples
        )
    )
    corrected_delay_s = raw.delay_s + adjustment / waveform.sample_rate_hz
    return float(capture_start_s + corrected_delay_s)


def _measure_beamforming_interarrival(
    waveform: Waveform,
    calibration: QLSCalibration,
    ap0_clock: LocalClock,
    ap1_clock: LocalClock,
    transmit_local_s: float,
    snr_db: float,
    config: ThreeExperimentConfig,
    rng: np.random.Generator,
) -> float:
    """两 AP 同一计划时刻发射，理想 RX ADC 分别测量脉冲到达差。"""

    fs = waveform.sample_rate_hz
    rx_start_s = transmit_local_s - config.receiver_margin_s
    capture_count = int(
        np.ceil(
            (config.beamforming_pulse_duration_s + 2.0 * config.receiver_margin_s)
            * fs
        )
    )
    expected_delay_s = config.receiver_margin_s
    arrivals: list[float] = []
    for clock in (ap0_clock, ap1_clock):
        tx_true_s = clock.true_time_for_reading(transmit_local_s)
        start_source_sample = (rx_start_s - tx_true_s) * fs * clock.effective_rate
        capture = simulate_capture(
            waveform.samples,
            fs,
            start_local_s=rx_start_s,
            start_source_sample=start_source_sample,
            sample_step=clock.effective_rate,
            count=capture_count,
            amplitude=1.0 + 0.0j,
            snr_db=snr_db,
            rng=rng,
        )
        arrivals.append(
            _corrected_pulse_arrival_s(
                capture.samples,
                capture.start_local_s,
                waveform,
                calibration,
                expected_delay_s,
                config.data_gate_half_width_samples,
            )
        )
    return float(arrivals[1] - arrivals[0])


def run_three_experiment_suite(
    waveform_config: WaveformConfig,
    calibration: QLSCalibration,
    config: ThreeExperimentConfig,
    *,
    beamforming_calibration: QLSCalibration | None = None,
) -> ThreeExperimentResult:
    """对三种链路配置逐 SNR 重复完整频率、时间和脉冲读出链。"""

    profiles = _profiles(config)
    snr_axis = np.asarray(config.snr_db_values, dtype=np.float64)
    profile_count = len(profiles)
    snr_count = snr_axis.size
    trial_count = config.trials_per_snr
    sync_waveform = generate_two_tone(waveform_config)
    beamforming_config = WaveformConfig(
        sample_rate_hz=waveform_config.sample_rate_hz,
        tone_separation_hz=config.beamforming_tone_separation_hz,
        pulse_duration_s=config.beamforming_pulse_duration_s,
        rise_fall_s=min(
            waveform_config.rise_fall_s,
            config.beamforming_pulse_duration_s / 2.0,
        ),
        carrier_frequency_hz=config.beamforming_carrier_frequency_hz,
    )
    beamforming_waveform = generate_two_tone(beamforming_config)
    data_lut = beamforming_calibration or build_qls_lut(
        beamforming_config, grid_points=calibration.grid_points
    )

    sample_shape = (profile_count, snr_count, trial_count)
    time_samples = np.full(sample_shape, np.nan, dtype=np.float64)
    beamforming_samples = np.full(sample_shape, np.nan, dtype=np.float64)
    rate_samples = np.full(sample_shape, np.nan, dtype=np.float64)
    failure_rate = np.zeros((profile_count, snr_count), dtype=np.float64)
    seeds = np.random.SeedSequence(config.seed).spawn(
        profile_count * snr_count * trial_count
    )

    for profile_index, profile in enumerate(profiles):
        for snr_index, snr_db in enumerate(snr_axis):
            failures = 0
            for trial_index in range(trial_count):
                seed_index = (
                    (profile_index * snr_count + snr_index) * trial_count
                    + trial_index
                )
                frequency_seed, timing_seed, data_seed = seeds[seed_index].spawn(3)
                frequency_rng = np.random.default_rng(frequency_seed)
                timing_rng = np.random.default_rng(timing_seed)
                data_rng = np.random.default_rng(data_seed)
                true_rate_offset = (
                    config.wireless_clock_rate_offset
                    if profile.frequency_link_mode == "wireless"
                    else 0.0
                )
                ap0_clock = LocalClock()
                ap1_clock = LocalClock(
                    offset_s=config.initial_clock_offset_s,
                    fractional_frequency_offset=true_rate_offset,
                )
                if profile.frequency_link_mode == "wireless":
                    rate_estimate = _estimate_wireless_clock_rate(
                        waveform_config,
                        config,
                        true_rate_offset,
                        frequency_rng,
                    )
                    ap1_clock.apply_frequency_correction(
                        rate_estimate, effective_true_time_s=0.0
                    )

                link = ChannelConfig(
                    propagation_delay_s=profile.propagation_delay_s,
                    amplitude=1.0,
                    phase_rad=0.0,
                    snr_db=float(snr_db),
                )
                schedule = TwoWayConfig(
                    tx1_local_time_s=1e-3,
                    processing_delay_s=20e-6,
                    coarse_up_delay_s=profile.propagation_delay_s,
                    coarse_down_delay_s=profile.propagation_delay_s,
                )
                try:
                    observation = simulate_two_way_exchange(
                        waveform_config,
                        schedule,
                        ap0_clock,
                        ap1_clock,
                        link,
                        link,
                        calibration,
                        timing_rng,
                    )
                    estimate = estimate_two_way(observation)
                    time_correction_s = estimate.clock_correction_s
                    ap1_clock.apply_time_correction(estimate.clock_correction_s)
                    beamforming_interarrival_s = _measure_beamforming_interarrival(
                        beamforming_waveform,
                        data_lut,
                        ap0_clock,
                        ap1_clock,
                        schedule.tx1_local_time_s + config.sync_interval_s,
                        float(snr_db),
                        config,
                        data_rng,
                    )
                    residual_rate = ap1_clock.effective_rate - 1.0
                    # 只有频率、双向时间传递和下游脉冲读出全部成功时，
                    # 该 trial 才进入统计，避免半条链路的样本污染曲线。
                    time_samples[profile_index, snr_index, trial_index] = time_correction_s
                    beamforming_samples[profile_index, snr_index, trial_index] = (
                        beamforming_interarrival_s
                    )
                    rate_samples[profile_index, snr_index, trial_index] = residual_rate
                except AcquisitionError:
                    failures += 1
            failure_rate[profile_index, snr_index] = failures / trial_count
            valid_mask = (
                np.isfinite(time_samples[profile_index, snr_index])
                & np.isfinite(beamforming_samples[profile_index, snr_index])
                & np.isfinite(rate_samples[profile_index, snr_index])
            )
            valid_count = int(np.sum(valid_mask))
            if valid_count < 2:
                raise RuntimeError(
                    f"{profile.key} 在 {snr_db:g} dB 的有效试验少于两个"
                )

    time_std = np.nanstd(time_samples, axis=2, ddof=1)
    beamforming_std = np.nanstd(beamforming_samples, axis=2, ddof=1)
    rate_rmse = np.sqrt(np.nanmean(rate_samples**2, axis=2))
    crlb_std = np.array(
        [
            crlb_for_waveform(
                sync_waveform,
                waveform_config.tone_separation_hz,
                float(snr_db),
            ).std_s
            for snr_db in snr_axis
        ],
        dtype=np.float64,
    )
    crlb_best = crlb_std * 10.0 ** (-config.snr_uncertainty_db / 20.0)
    return ThreeExperimentResult(
        profile_keys=tuple(profile.key for profile in profiles),
        profile_labels=tuple(profile.label for profile in profiles),
        time_link_modes=tuple(profile.time_link_mode for profile in profiles),
        frequency_link_modes=tuple(profile.frequency_link_mode for profile in profiles),
        snr_db=snr_axis,
        time_transfer_std_s=np.asarray(time_std, dtype=np.float64),
        beamforming_std_s=np.asarray(beamforming_std, dtype=np.float64),
        crlb_std_s=crlb_std,
        crlb_best_case_std_s=np.asarray(crlb_best, dtype=np.float64),
        residual_clock_rate_rmse=np.asarray(rate_rmse, dtype=np.float64),
        acquisition_failure_rate=failure_rate,
        time_transfer_samples_s=time_samples,
        beamforming_samples_s=beamforming_samples,
        residual_clock_rate_samples=rate_samples,
    )
