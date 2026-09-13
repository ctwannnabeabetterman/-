"""论文主实验之外的受控模型失配敏感性仿真。"""

from __future__ import annotations

import numpy as np

from clock_model import LocalClock
from config import ChannelConfig, FrequencySyncConfig, SensitivityConfig, TwoWayConfig, WaveformConfig
from frequency_sync import simulate_frequency_reference
from models import QLSCalibration, SensitivityResult
from two_way_sync import estimate_two_way, simulate_two_way_exchange


def _estimate_rate_from_two_reference_windows(
    waveform_config: WaveformConfig,
    config: SensitivityConfig,
    phase_noise_rad: float,
    rng: np.random.Generator,
) -> float:
    """只用两个可观测 10 MHz IQ 窗口估计采样钟速率偏差。"""

    base_phase = float(rng.uniform(-np.pi, np.pi))
    continuous_advance = (
        2.0
        * np.pi
        * config.frequency_reference_hz
        * config.sync_interval_s
        / (1.0 + config.true_clock_rate_offset)
    )
    coherent: list[complex] = []
    for phase in (
        base_phase + float(rng.normal(0.0, phase_noise_rad)),
        base_phase
        + continuous_advance
        + float(rng.normal(0.0, phase_noise_rad)),
    ):
        reference_config = FrequencySyncConfig(
            sample_rate_hz=waveform_config.sample_rate_hz,
            reference_frequency_hz=config.frequency_reference_hz,
            rf_tone_separation_hz=config.frequency_reference_hz,
            observation_duration_s=config.frequency_capture_duration_s,
            segment_duration_s=config.frequency_capture_duration_s / 2.0,
            cfo_hz=0.0,
            sample_clock_offset_fraction=config.true_clock_rate_offset,
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
    observed_frequency_hz = (
        config.frequency_reference_hz
        + phase_difference / (2.0 * np.pi * config.sync_interval_s)
    )
    return float(config.frequency_reference_hz / observed_frequency_hz - 1.0)


def run_sensitivity_suite(
    waveform_config: WaveformConfig,
    calibration: QLSCalibration,
    config: SensitivityConfig,
) -> SensitivityResult:
    """运行双向时延不对称与频率参考相位扰动两类独立扫描。"""

    asymmetry_s = np.asarray(config.asymmetry_ps, dtype=np.float64) * 1e-12
    clock_samples = np.empty((asymmetry_s.size, config.trials), dtype=np.float64)
    seed_sequences = np.random.SeedSequence(config.seed).spawn(
        asymmetry_s.size * config.trials
    )
    schedule = TwoWayConfig(
        tx1_local_time_s=1e-3,
        processing_delay_s=50e-3 - waveform_config.pulse_duration_s,
        coarse_up_delay_s=config.base_propagation_delay_s,
        coarse_down_delay_s=config.base_propagation_delay_s,
    )
    for axis_index, delta_s in enumerate(asymmetry_s):
        up_delay_s = config.base_propagation_delay_s + 0.5 * delta_s
        down_delay_s = config.base_propagation_delay_s - 0.5 * delta_s
        for trial_index in range(config.trials):
            seed_index = axis_index * config.trials + trial_index
            rng = np.random.default_rng(seed_sequences[seed_index])
            observation = simulate_two_way_exchange(
                waveform_config,
                schedule,
                LocalClock(),
                LocalClock(),
                ChannelConfig(
                    propagation_delay_s=float(up_delay_s),
                    snr_db=config.time_link_snr_db,
                ),
                ChannelConfig(
                    propagation_delay_s=float(down_delay_s),
                    snr_db=config.time_link_snr_db,
                ),
                calibration,
                rng,
            )
            clock_samples[axis_index, trial_index] = estimate_two_way(
                observation
            ).clock_correction_s

    phase_noise_rad = np.deg2rad(
        np.asarray(config.reference_phase_noise_deg, dtype=np.float64)
    )
    rate_errors = np.empty((phase_noise_rad.size, config.trials), dtype=np.float64)
    frequency_seeds = np.random.SeedSequence(config.seed + 1).spawn(
        phase_noise_rad.size * config.trials
    )
    for axis_index, phase_std_rad in enumerate(phase_noise_rad):
        for trial_index in range(config.trials):
            seed_index = axis_index * config.trials + trial_index
            rng = np.random.default_rng(frequency_seeds[seed_index])
            estimate = _estimate_rate_from_two_reference_windows(
                waveform_config,
                config,
                float(phase_std_rad),
                rng,
            )
            rate_errors[axis_index, trial_index] = (
                estimate - config.true_clock_rate_offset
            )

    rate_rmse = np.sqrt(np.mean(rate_errors**2, axis=1))
    return SensitivityResult(
        path_asymmetry_s=asymmetry_s,
        clock_bias_s=np.mean(clock_samples, axis=1),
        clock_std_s=np.std(clock_samples, axis=1, ddof=1),
        reference_phase_noise_rad=phase_noise_rad,
        clock_rate_rmse=np.asarray(rate_rmse, dtype=np.float64),
        holdover_timing_rmse_s=np.asarray(
            rate_rmse * config.sync_interval_s,
            dtype=np.float64,
        ),
    )
