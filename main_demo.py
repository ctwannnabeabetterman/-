"""两节点分布式时间、频率、相位同步的一键软件通信仿真。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
import platform
import shutil
from typing import Mapping, Sequence

import numpy as np

from beamforming import simulate_four_sync_states
from channel import propagate_static_link
from clock_model import LocalClock
from config import (
    BeamformingConfig,
    ChannelConfig,
    ClockPlantConfig,
    ClockTrackingConfig,
    FrequencySyncConfig,
    MonteCarloConfig,
    OscillatorConfig,
    PhaseFeedbackConfig,
    SensitivityConfig,
    ThreeExperimentConfig,
    TwoWayConfig,
    WaveformConfig,
)
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from experiment_suite import run_three_experiment_suite
from experiments import (
    estimate_clock_frequency_offset,
    reconstruct_raw_offset_estimate,
    run_clock_tracking,
)
from frequency_sync import (
    estimate_frequency_offset,
    simulate_frequency_reference,
    update_frequency_tracker,
)
from lut_calibration import (
    build_qls_lut,
    correct_qls_fraction,
    validate_qls_lut,
    wrap_fractional_sample,
)
from models import BeamformingPlantState, BeamformingResult, ClockTrackingResult
from monte_carlo import run_delay_monte_carlo
from oscillator_model import LocalOscillator
from phase_sync import simulate_channel_feedback
from plotting import (
    plot_model_mismatch_sensitivity,
    plot_qls_lut_validation,
    plot_system_signal_chain,
    plot_three_config_vs_crlb,
)
from reporting import build_manifest, write_report, write_result_tables
from results_io import write_json
from sensitivity import run_sensitivity_suite
from waveforms import generate_two_tone


@dataclass(frozen=True)
class DemoSettings:
    """一次 Demo 所需配置；真值只传给仿真 plant。"""

    mode: str
    seed: int
    waveform: WaveformConfig
    two_way: TwoWayConfig
    clock_plant: ClockPlantConfig
    clock_tracking: ClockTrackingConfig
    frequency: FrequencySyncConfig
    oscillator: OscillatorConfig
    beamforming: BeamformingConfig
    phase_feedback: PhaseFeedbackConfig
    monte_carlo: MonteCarloConfig
    three_experiment: ThreeExperimentConfig
    sensitivity: SensitivityConfig
    lut_grid_points: int
    frequency_rounds: int


def build_demo_settings(mode: str) -> DemoSettings:
    """构造快速或正式预设；两种模式使用相同物理链和算法。"""

    if mode not in {"fast_demo", "formal"}:
        raise ValueError("mode 必须为 fast_demo 或 formal")
    formal = mode == "formal"
    waveform = WaveformConfig(pulse_duration_s=10e-6)
    return DemoSettings(
        mode=mode,
        seed=2023,
        waveform=waveform,
        two_way=TwoWayConfig(
            processing_delay_s=50e-3 - waveform.pulse_duration_s
        ),
        clock_plant=ClockPlantConfig(
            initial_offset_s=100e-9,
            fractional_frequency_offset=0.2e-6,
        ),
        clock_tracking=ClockTrackingConfig(
            rounds=20,
            sync_interval_s=100e-3,
            correction_gain=1.0,
            random_walk_std_s_per_sqrt_s=0.5e-12,
        ),
        frequency=FrequencySyncConfig(
            observation_duration_s=2e-3 if formal else 1e-3,
            segment_duration_s=50e-6 if formal else 20e-6,
            cfo_hz=600.0,
            sample_clock_offset_fraction=0.0,
            initial_phase_rad=0.7,
            snr_db=28.0,
            tracker_alpha=0.55,
        ),
        oscillator=OscillatorConfig(
            frequency_offset_hz=600.0,
            initial_phase_rad=1.1,
        ),
        beamforming=BeamformingConfig(
            ap0_propagation_delay_s=50e-9,
            ap1_propagation_delay_s=75e-9,
            ap0_amplitude=1.0,
            ap1_amplitude=0.9,
            ap0_channel_phase_rad=0.2,
            ap1_channel_phase_rad=-0.6,
            normalization="per_ap_fixed",
        ),
        phase_feedback=PhaseFeedbackConfig(
            pilot_symbols=1024,
            snr_db=32.0,
            feedback_delay_s=100e-6,
            phase_quantization_bits=12,
            channel_phase_rate_rad_per_s=0.0,
        ),
        monte_carlo=MonteCarloConfig(
            snr_db_values=tuple(range(6, 37, 3)),
            trials_per_snr=1000 if formal else 100,
            seed=2023,
        ),
        three_experiment=ThreeExperimentConfig(
            trials_per_snr=1000 if formal else 100,
        ),
        sensitivity=SensitivityConfig(trials=300 if formal else 40),
        lut_grid_points=2001 if formal else 401,
        frequency_rounds=20 if formal else 12,
    )


def _correct_delay_s(raw_estimate, calibration, sample_rate_hz: float) -> float:
    if not raw_estimate.qls_valid:
        return float(raw_estimate.delay_s)
    corrected_fraction = float(
        correct_qls_fraction(raw_estimate.fractional_offset_samples, calibration)
    )
    adjustment = float(
        wrap_fractional_sample(
            corrected_fraction - raw_estimate.fractional_offset_samples
        )
    )
    return float(raw_estimate.delay_s + adjustment / sample_rate_hz)


def _run_frequency_rounds(
    settings: DemoSettings,
    oscillator: LocalOscillator,
    sample_clock_offset_fraction: float,
    start_epoch_s: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    true_values = np.empty(settings.frequency_rounds, dtype=np.float64)
    estimates = np.empty(settings.frequency_rounds, dtype=np.float64)
    tracked = np.empty(settings.frequency_rounds, dtype=np.float64)
    tracker_state = 0.0
    for index in range(settings.frequency_rounds):
        epoch_s = start_epoch_s + index * settings.frequency.observation_duration_s
        round_config = replace(
            settings.frequency,
            cfo_hz=oscillator.frequency_offset_hz,
            sample_clock_offset_fraction=sample_clock_offset_fraction,
            initial_phase_rad=oscillator.raw_phase_at(epoch_s),
        )
        observation = simulate_frequency_reference(round_config, rng)
        estimate = estimate_frequency_offset(observation.samples, round_config)
        tracker_state = update_frequency_tracker(
            tracker_state,
            estimate.frequency_offset_hz,
            0.0 if index == 0 else settings.frequency.tracker_alpha,
        )
        true_values[index] = observation.true_observed_offset_hz
        estimates[index] = estimate.frequency_offset_hz
        tracked[index] = tracker_state
    end_epoch_s = (
        start_epoch_s
        + settings.frequency_rounds * settings.frequency.observation_duration_s
    )
    return true_values, estimates, tracked, float(end_epoch_s)


def _clear_generated_output(destination: Path) -> None:
    """只清理本程序已知输出，保留目标目录内的未知用户文件。"""

    known_files = {
        "README.md",
        "run_config.json",
        "summary.json",
        "manifest.json",
        "beamforming_states.csv",
        "clock_tracking.csv",
        "frequency_tracking.csv",
        "joint_tracking.csv",
        "lut_bias.csv",
        "lut_training.csv",
        "monte_carlo.csv",
        "three_experiment_samples.csv",
        "three_experiment_summary.csv",
        "three_config_summary.csv",
    }
    for filename in known_files:
        path = destination / filename
        if path.is_file():
            path.unlink()
    for dirname in ("figures", "diagnostics", "lut_cache"):
        path = destination / dirname
        if path.is_dir():
            shutil.rmtree(path)


def _beamforming_summary(
    states: Mapping[str, BeamformingResult],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "arrival_difference_ps": value.residual_arrival_difference_s * 1e12,
            "residual_frequency_hz": value.residual_frequency_offset_hz,
            "residual_phase_deg": float(np.rad2deg(value.residual_phase_difference_rad)),
            "carrier_phase_deg": float(np.rad2deg(value.carrier_phase_difference_rad)),
            "waveform_coherence": value.waveform_coherence,
            "combined_power": value.signal_power,
            "gain_vs_single_ap_db": value.metrics.gain_vs_single_ap_db,
            "gain_vs_incoherent_sum_db": value.metrics.gain_vs_incoherent_sum_db,
            "normalized_ideal_loss_db": value.metrics.normalized_ideal_loss_db,
        }
        for name, value in states.items()
    }


def _diagnostic_columns(
    calibration,
    validation,
    waveform_config: WaveformConfig,
    clock_result: ClockTrackingResult,
    frequency_true: np.ndarray,
    frequency_estimate: np.ndarray,
    frequency_tracked: np.ndarray,
    beamforming_states: Mapping[str, BeamformingResult],
    monte_carlo,
    three_experiment,
) -> dict[str, dict[str, Sequence[object]]]:
    raw_training_bias = np.asarray(
        wrap_fractional_sample(
            calibration.raw_fraction_samples - calibration.true_fraction_samples
        )
    )
    return {
        "lut_training.csv": {
            "true_fraction_samples": calibration.true_fraction_samples,
            "raw_qls_fraction_samples": calibration.raw_fraction_samples,
            "raw_bias_ps": raw_training_bias / waveform_config.sample_rate_hz * 1e12,
        },
        "lut_validation.csv": {
            "true_fraction_samples": validation.true_fraction_samples,
            "raw_bias_ps": validation.raw_error_samples
            / waveform_config.sample_rate_hz
            * 1e12,
            "corrected_bias_ps": validation.corrected_error_samples
            / waveform_config.sample_rate_hz
            * 1e12,
        },
        "monte_carlo.csv": {
            "snr_db": monte_carlo.snr_db,
            "integer_peak_rmse_ps": monte_carlo.integer_peak_rmse_s * 1e12,
            "qls_rmse_ps": monte_carlo.qls_rmse_s * 1e12,
            "qls_lut_rmse_ps": monte_carlo.lut_rmse_s * 1e12,
            "crlb_std_ps": monte_carlo.crlb_std_s * 1e12,
            "acquisition_failure_rate": monte_carlo.acquisition_failure_rate,
        },
        "clock_tracking.csv": {
            "round": np.arange(1, clock_result.raw_offset_s.size + 1),
            "reference_epoch_ms": clock_result.reference_epoch_s * 1e3,
            "estimated_raw_offset_ps": reconstruct_raw_offset_estimate(clock_result) * 1e12,
            "residual_after_ps": clock_result.residual_after_s * 1e12,
        },
        "frequency_tracking.csv": {
            "round": np.arange(1, frequency_true.size + 1),
            "true_observed_offset_hz": frequency_true,
            "single_estimate_hz": frequency_estimate,
            "tracked_estimate_hz": frequency_tracked,
        },
        "beamforming_states.csv": {
            "state": list(beamforming_states),
            "arrival_difference_ps": [
                value.residual_arrival_difference_s * 1e12
                for value in beamforming_states.values()
            ],
            "gain_vs_incoherent_sum_db": [
                value.metrics.gain_vs_incoherent_sum_db
                for value in beamforming_states.values()
            ],
        },
        "three_config_trials.csv": {
            "case": [
                key
                for key in three_experiment.profile_keys
                for _ in range(
                    three_experiment.snr_db.size
                    * three_experiment.time_transfer_samples_s.shape[2]
                )
            ],
            "snr_db": np.tile(
                np.repeat(
                    three_experiment.snr_db,
                    three_experiment.time_transfer_samples_s.shape[2],
                ),
                len(three_experiment.profile_keys),
            ),
            "trial": np.tile(
                np.arange(1, three_experiment.time_transfer_samples_s.shape[2] + 1),
                len(three_experiment.profile_keys) * three_experiment.snr_db.size,
            ),
            "time_correction_ps": three_experiment.time_transfer_samples_s.reshape(-1)
            * 1e12,
            "beamforming_interarrival_ps": three_experiment.beamforming_samples_s.reshape(-1)
            * 1e12,
            "residual_clock_rate_ppm": three_experiment.residual_clock_rate_samples.reshape(-1)
            * 1e6,
        },
    }


def run_demo(
    settings: DemoSettings,
    output_dir: str | Path,
    *,
    diagnostics: bool = False,
) -> dict[str, object]:
    """运行估计驱动闭环、三配置统计和失配扫描，并生成正式结果。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _clear_generated_output(destination)
    figures_dir = destination / "figures"
    rng = np.random.default_rng(settings.seed)

    waveform = generate_two_tone(settings.waveform)
    calibration = build_qls_lut(
        settings.waveform, grid_points=settings.lut_grid_points
    )
    validation = validate_qls_lut(
        settings.waveform,
        calibration,
        validation_points=settings.lut_grid_points - 1,
    )

    true_delay_s = 37.25 / settings.waveform.sample_rate_hz
    delay_link = ChannelConfig(
        propagation_delay_s=true_delay_s,
        amplitude=1.0,
        phase_rad=0.35,
        snr_db=36.0,
    )
    delay_observation = propagate_static_link(
        waveform.samples,
        settings.waveform.sample_rate_hz,
        delay_link,
        rng,
    )
    correlation = fft_matched_filter(delay_observation.samples, waveform.samples)
    raw_delay = estimate_delay(
        correlation,
        settings.waveform.sample_rate_hz,
        DelaySearchGate(
            center_s=37.0 / settings.waveform.sample_rate_hz,
            half_width_s=2.5 / settings.waveform.sample_rate_hz,
        ),
    )
    corrected_delay_s = _correct_delay_s(
        raw_delay, calibration, settings.waveform.sample_rate_hz
    )

    ap0_clock = LocalClock()
    ap1_clock = LocalClock(
        offset_s=settings.clock_plant.initial_offset_s,
        fractional_frequency_offset=settings.clock_plant.fractional_frequency_offset,
    )
    synchronization_link = ChannelConfig(
        propagation_delay_s=50e-9,
        amplitude=1.0,
        phase_rad=0.0,
        snr_db=36.0,
    )
    clock_result = run_clock_tracking(
        settings.waveform,
        settings.two_way,
        settings.clock_tracking,
        ap0_clock,
        ap1_clock,
        synchronization_link,
        synchronization_link,
        calibration,
        rng,
    )
    clock_frequency_estimate = estimate_clock_frequency_offset(clock_result)
    time_only_clock = replace(ap1_clock)
    clock_control_epoch_s = float(clock_result.epoch_true_s[-1])
    ap1_clock.apply_frequency_correction(
        clock_frequency_estimate,
        effective_true_time_s=clock_control_epoch_s,
    )

    oscillator = LocalOscillator(
        frequency_offset_hz=settings.oscillator.frequency_offset_hz,
        initial_phase_rad=settings.oscillator.initial_phase_rad,
    )
    frequency_start_epoch_s = (
        clock_control_epoch_s + settings.clock_tracking.sync_interval_s
    )
    frequency_true, frequency_estimate, frequency_tracked, frequency_end_epoch_s = (
        _run_frequency_rounds(
            settings,
            oscillator,
            ap1_clock.effective_rate - 1.0,
            frequency_start_epoch_s,
            rng,
        )
    )
    oscillator.apply_frequency_correction(
        float(frequency_tracked[-1]), effective_time_s=frequency_end_epoch_s
    )
    pilot_epoch_s = frequency_end_epoch_s + 1e-3
    channel_feedback = simulate_channel_feedback(
        settings.waveform,
        settings.beamforming,
        settings.phase_feedback,
        oscillator,
        residual_clock_offset_s=ap1_clock.residual_offset_at(pilot_epoch_s),
        pilot_epoch_s=pilot_epoch_s,
        rng=rng,
        calibration=calibration,
    )
    data_epoch_s = channel_feedback.data_epoch_s
    phase_rate = settings.phase_feedback.channel_phase_rate_rad_per_s
    channel_phase_at_data = phase_rate * data_epoch_s
    channel_frequency_offset_hz = phase_rate / (2.0 * np.pi)
    plant_state = BeamformingPlantState(
        time_only_clock_offset_s=time_only_clock.residual_offset_at(data_epoch_s),
        data_epoch_s=data_epoch_s,
        raw_clock_offset_s=ap1_clock.raw_offset_at(data_epoch_s),
        residual_clock_offset_s=ap1_clock.residual_offset_at(data_epoch_s),
        raw_frequency_offset_hz=oscillator.frequency_offset_hz
        + channel_frequency_offset_hz,
        residual_frequency_offset_hz=oscillator.residual_frequency_offset_hz
        + channel_frequency_offset_hz,
        raw_ap1_phase_rad=oscillator.raw_phase_at(data_epoch_s)
        + channel_phase_at_data,
        residual_ap1_phase_rad=oscillator.phase_at(data_epoch_s)
        + channel_phase_at_data,
    )
    beamforming_states = simulate_four_sync_states(
        waveform.samples,
        settings.waveform,
        settings.beamforming,
        plant_state,
        channel_estimates=channel_feedback.feedback_channel_estimates,
        tx_time_correction_s=channel_feedback.tx_time_correction_s,
    )

    monte_carlo = run_delay_monte_carlo(
        settings.waveform,
        calibration,
        settings.monte_carlo,
        beamforming_config=settings.beamforming,
        phase_feedback_config=settings.phase_feedback,
    )
    three_experiment = run_three_experiment_suite(
        settings.waveform,
        calibration,
        settings.three_experiment,
    )
    sensitivity = run_sensitivity_suite(
        settings.waveform,
        calibration,
        settings.sensitivity,
    )

    raw_validation_bias_s = validation.raw_error_samples / settings.waveform.sample_rate_hz
    corrected_validation_bias_s = (
        validation.corrected_error_samples / settings.waveform.sample_rate_hz
    )
    beamforming_summary = _beamforming_summary(beamforming_states)
    threshold_next_index = min(1, three_experiment.snr_db.size - 1)
    profile_labels_cn = {
        "cabled": "有线时间 + 有线频率",
        "wireless_time": "无线时间 + 有线频率",
        "wireless_time_frequency": "无线时间 + 无线频率",
    }
    summary: dict[str, object] = {
        "mode": settings.mode,
        "seed": settings.seed,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "matplotlib": version("matplotlib"),
        },
        "simulation_scope": {
            "type": "complex-baseband software simulation",
            "hardware_measurement": False,
            "omitted_effects": [
                "RF front-end group delay",
                "ADC/DAC and FPGA timestamp quantization",
                "hardware phase noise and spurs",
                "temperature drift",
                "field multipath",
            ],
        },
        "waveform": {
            "sample_rate_msa_s": settings.waveform.sample_rate_hz / 1e6,
            "transmit_sample_rate_msa_s": settings.waveform.transmit_sample_rate_hz
            / 1e6,
            "tone_separation_mhz": settings.waveform.tone_separation_hz / 1e6,
            "pulse_duration_us": settings.waveform.pulse_duration_s * 1e6,
            "rise_fall_ns": settings.waveform.rise_fall_s * 1e9,
            "time_transfer_carrier_ghz": settings.waveform.carrier_frequency_hz / 1e9,
        },
        "snr_definition": {
            "name": "active-region complex AWGN sample SNR",
            "paper_measurement_equivalent": False,
        },
        "lut": {
            "signature": calibration.signature,
            "grid_points": calibration.grid_points,
            "validation_points": validation.true_fraction_samples.size,
            "validation_policy": "half-step grid disjoint from LUT training fractions",
            "raw_rmse_ps": float(np.sqrt(np.mean(raw_validation_bias_s**2)) * 1e12),
            "corrected_rmse_ps": float(
                np.sqrt(np.mean(corrected_validation_bias_s**2)) * 1e12
            ),
        },
        "delay_demo": {
            "true_delay_ns": true_delay_s * 1e9,
            "lut_corrected_delay_ns": corrected_delay_s * 1e9,
            "corrected_error_ps": (corrected_delay_s - true_delay_s) * 1e12,
        },
        "clock_tracking": {
            "final_residual_after_update_ps": float(clock_result.residual_after_s[-1] * 1e12),
            "estimated_fractional_frequency_offset": clock_frequency_estimate,
            "applied_fractional_frequency_correction": ap1_clock.fractional_frequency_correction,
        },
        "frequency_sync": {
            "rf_carrier_ghz": settings.frequency.rf_carrier_frequency_hz / 1e9,
            "rf_lower_tone_ghz": (
                settings.frequency.rf_carrier_frequency_hz
                - 0.5 * settings.frequency.rf_tone_separation_hz
            )
            / 1e9,
            "rf_upper_tone_ghz": (
                settings.frequency.rf_carrier_frequency_hz
                + 0.5 * settings.frequency.rf_tone_separation_hz
            )
            / 1e9,
            "self_mixed_reference_mhz": settings.frequency.reference_frequency_hz / 1e6,
            "final_true_observed_offset_hz": float(frequency_true[-1]),
            "final_tracked_estimate_hz": float(frequency_tracked[-1]),
            "applied_oscillator_correction_hz": oscillator.frequency_correction_hz,
            "final_residual_hz": oscillator.residual_frequency_offset_hz,
        },
        "state_separation": {
            "plant_truth_policy": "truth generates observations and post-run metrics only",
            "plant_at_data_epoch": {
                "raw_clock_offset_ps": plant_state.raw_clock_offset_s * 1e12,
                "residual_clock_offset_ps": plant_state.residual_clock_offset_s * 1e12,
                "raw_frequency_offset_hz": plant_state.raw_frequency_offset_hz,
                "residual_frequency_offset_hz": plant_state.residual_frequency_offset_hz,
            },
        },
        "beamforming": beamforming_summary,
        "monte_carlo_highest_snr": {
            "snr_db": float(monte_carlo.snr_db[-1]),
            "integer_peak_rmse_ps": float(monte_carlo.integer_peak_rmse_s[-1] * 1e12),
            "qls_rmse_ps": float(monte_carlo.qls_rmse_s[-1] * 1e12),
            "qls_lut_rmse_ps": float(monte_carlo.lut_rmse_s[-1] * 1e12),
            "crlb_std_ps": float(monte_carlo.crlb_std_s[-1] * 1e12),
            "clock_offset_rmse_ps": float(monte_carlo.clock_offset_rmse_s[-1] * 1e12),
            "acquisition_failure_rate": float(monte_carlo.acquisition_failure_rate[-1]),
        },
        "three_experiments": {
            "statistic": "sample standard deviation, ddof=1",
            "threshold_region": {
                "snr_db": float(three_experiment.snr_db[0]),
                "time_transfer_std_ps": {
                    key: float(three_experiment.time_transfer_std_s[index, 0] * 1e12)
                    for index, key in enumerate(three_experiment.profile_keys)
                },
                "two_way_clock_crlb_std_ps": float(
                    three_experiment.two_way_clock_crlb_std_s[0] * 1e12
                ),
                "acquisition_failure_rate": {
                    key: float(three_experiment.acquisition_failure_rate[index, 0])
                    for index, key in enumerate(three_experiment.profile_keys)
                },
                "next_snr_db": float(three_experiment.snr_db[threshold_next_index]),
                "next_time_transfer_std_ps": {
                    key: float(
                        three_experiment.time_transfer_std_s[index, threshold_next_index]
                        * 1e12
                    )
                    for index, key in enumerate(three_experiment.profile_keys)
                },
                "next_two_way_clock_crlb_std_ps": float(
                    three_experiment.two_way_clock_crlb_std_s[threshold_next_index]
                    * 1e12
                ),
            },
            "profiles": {
                key: {
                    "label": label,
                    "label_cn": profile_labels_cn[key],
                    "time_link": time_link,
                    "frequency_link": frequency_link,
                    "highest_snr_db": float(three_experiment.snr_db[-1]),
                    "time_transfer_std_ps": float(
                        three_experiment.time_transfer_std_s[index, -1] * 1e12
                    ),
                    "beamforming_std_ps": float(
                        three_experiment.beamforming_std_s[index, -1] * 1e12
                    ),
                    "residual_clock_rate_rmse_ppm": float(
                        three_experiment.residual_clock_rate_rmse[index, -1] * 1e6
                    ),
                    "acquisition_failure_rate": float(
                        three_experiment.acquisition_failure_rate[index, -1]
                    ),
                }
                for index, (key, label, time_link, frequency_link) in enumerate(
                    zip(
                        three_experiment.profile_keys,
                        three_experiment.profile_labels,
                        three_experiment.time_link_modes,
                        three_experiment.frequency_link_modes,
                    )
                )
            },
        },
        "sensitivity": {
            "max_abs_asymmetry_ps": float(
                np.max(np.abs(sensitivity.path_asymmetry_s)) * 1e12
            ),
            "max_abs_clock_bias_ps": float(
                np.max(np.abs(sensitivity.clock_bias_s)) * 1e12
            ),
            "max_reference_phase_noise_deg": float(
                np.max(np.rad2deg(sensitivity.reference_phase_noise_rad))
            ),
            "max_holdover_timing_rmse_ps": float(
                np.max(sensitivity.holdover_timing_rmse_s) * 1e12
            ),
        },
    }

    write_json(destination / "run_config.json", settings)
    write_json(destination / "summary.json", summary)
    diagnostic_columns = (
        _diagnostic_columns(
            calibration,
            validation,
            settings.waveform,
            clock_result,
            frequency_true,
            frequency_estimate,
            frequency_tracked,
            beamforming_states,
            monte_carlo,
            three_experiment,
        )
        if diagnostics
        else None
    )
    write_result_tables(
        destination,
        three_experiment,
        diagnostics=diagnostics,
        diagnostic_columns=diagnostic_columns,
    )

    figure_paths: list[Path] = []
    figure_paths += plot_system_signal_chain(
        waveform, settings.frequency, delay_observation.samples, figures_dir
    )
    figure_paths += plot_qls_lut_validation(
        correlation,
        raw_delay,
        calibration,
        validation,
        monte_carlo,
        settings.waveform.sample_rate_hz,
        figures_dir,
    )
    figure_paths += plot_three_config_vs_crlb(three_experiment, figures_dir)
    figure_paths += plot_model_mismatch_sensitivity(sensitivity, figures_dir)
    relative_figures = [path.relative_to(destination) for path in figure_paths]
    write_json(
        destination / "manifest.json",
        build_manifest(relative_figures, diagnostics=diagnostics),
    )
    write_report(destination / "README.md", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-AP distributed coherent synchronization software simulation"
    )
    parser.add_argument(
        "--mode",
        choices=("fast_demo", "formal"),
        default="fast_demo",
        help="fast_demo uses 100 trials/SNR; formal uses 1000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory; default is results/<mode>",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="write internal tables under results/<mode>/diagnostics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = build_demo_settings(args.mode)
    output_dir = args.output_dir or Path("results") / args.mode
    print(
        f"Running {settings.mode}: LUT={settings.lut_grid_points}, "
        f"trials/SNR={settings.monte_carlo.trials_per_snr}"
    )
    summary = run_demo(settings, output_dir, diagnostics=args.diagnostics)
    high_snr = summary["monte_carlo_highest_snr"]
    full_sync = summary["beamforming"]["full_sync"]
    print(
        f"{high_snr['snr_db']:.0f} dB: QLS+LUT RMSE="
        f"{high_snr['qls_lut_rmse_ps']:.3f} ps, CRLB="
        f"{high_snr['crlb_std_ps']:.3f} ps"
    )
    print(
        "Full sync: gain over incoherent sum="
        f"{full_sync['gain_vs_incoherent_sum_db']:.3f} dB"
    )
    print(f"Results written to: {Path(output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
