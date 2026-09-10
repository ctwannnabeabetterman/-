"""两节点分布式 AP 时间、频率、相位同步的一键软件 Demo。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
import platform
from typing import Sequence

import numpy as np

from beamforming import simulate_four_sync_states
from channel import propagate_static_link
from clock_model import LocalClock
from config import (
    JointTrackingConfig,
    BeamformingConfig,
    ChannelConfig,
    ClockPlantConfig,
    ClockTrackingConfig,
    FrequencySyncConfig,
    MonteCarloConfig,
    OscillatorConfig,
    PhaseFeedbackConfig,
    TwoWayConfig,
    WaveformConfig,
)
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
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
from lut_calibration import correct_qls_fraction, load_or_build_lut, wrap_fractional_sample
from models import (
    BeamformingPlantState,
    BeamformingResult,
    ClockTrackingResult,
    MonteCarloResult,
)
from monte_carlo import run_delay_monte_carlo
from oscillator_model import LocalOscillator
from phase_sync import simulate_channel_feedback
from plotting import (
    plot_clock_tracking,
    plot_coherent_gain,
    plot_correlation_qls,
    plot_frequency_tracking,
    plot_lut_bias,
    plot_received_waveforms,
    plot_residual_summary,
    plot_rmse_crlb,
    plot_spectrum,
    plot_time_waveform,
)
from results_io import write_csv_columns, write_json
from waveforms import generate_two_tone
from joint_sync import run_joint_tracking


@dataclass(frozen=True)
class DemoSettings:
    """一键运行所需全部配置；真值只传给仿真器。"""

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
    lut_grid_points: int
    frequency_rounds: int
    joint_tracking: JointTrackingConfig


def build_demo_settings(mode: str) -> DemoSettings:
    """构造 ``fast_demo`` 或 ``formal`` 预设，二者算法链完全相同。"""

    if mode not in {"fast_demo", "formal"}:
        raise ValueError("mode 必须为 fast_demo 或 formal")
    formal = mode == "formal"
    waveform = WaveformConfig(pulse_duration_s=10e-6 if formal else 2e-6)
    frequency = FrequencySyncConfig(
        observation_duration_s=2e-3 if formal else 1e-3,
        segment_duration_s=50e-6 if formal else 20e-6,
        cfo_hz=600.0,
        sample_clock_offset_fraction=0.0,
        initial_phase_rad=0.7,
        snr_db=28.0,
        tracker_alpha=0.55,
    )
    monte_carlo = MonteCarloConfig(
        snr_db_values=tuple(range(6, 37, 3)),
        trials_per_snr=1000 if formal else 100,
        seed=2023,
    )
    return DemoSettings(
        mode=mode,
        seed=2023,
        waveform=waveform,
        two_way=TwoWayConfig(),
        clock_plant=ClockPlantConfig(
            initial_offset_s=100e-9,
            fractional_frequency_offset=0.2e-6,
        ),
        clock_tracking=ClockTrackingConfig(
            rounds=20,
            sync_interval_s=50e-3,
            correction_gain=1.0,
            random_walk_std_s_per_sqrt_s=0.5e-12,
        ),
        frequency=frequency,
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
        monte_carlo=monte_carlo,
        lut_grid_points=2001 if formal else 401,
        frequency_rounds=20 if formal else 12,
        joint_tracking=JointTrackingConfig(rounds=100 if formal else 40),
    )


def _correct_delay_s(raw_estimate, calibration, sample_rate_hz: float) -> float:
    """用当前 LUT 校正一个有效 QLS 估计，返回秒。"""

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
    """由连续本振 plant 生成参考观测，并只用观测更新跟踪器。"""

    true_values = np.empty(settings.frequency_rounds, dtype=np.float64)
    estimates = np.empty(settings.frequency_rounds, dtype=np.float64)
    tracked = np.empty(settings.frequency_rounds, dtype=np.float64)
    tracker_state = 0.0
    for index in range(settings.frequency_rounds):
        round_epoch_s = start_epoch_s + index * settings.frequency.observation_duration_s
        round_config = replace(
            settings.frequency,
            cfo_hz=oscillator.frequency_offset_hz,
            sample_clock_offset_fraction=sample_clock_offset_fraction,
            initial_phase_rad=oscillator.raw_phase_at(round_epoch_s),
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
    end_epoch_s = start_epoch_s + settings.frequency_rounds * settings.frequency.observation_duration_s
    return true_values, estimates, tracked, float(end_epoch_s)


def _save_numeric_tables(
    output_dir: Path,
    calibration,
    waveform_config: WaveformConfig,
    clock_result: ClockTrackingResult,
    frequency_true: np.ndarray,
    frequency_estimate: np.ndarray,
    frequency_tracked: np.ndarray,
    beamforming_states: dict[str, BeamformingResult],
    monte_carlo: MonteCarloResult,
) -> list[Path]:
    """保存复现实验曲线所需的五张 CSV 表。"""

    raw_bias_samples = np.asarray(
        wrap_fractional_sample(
            calibration.raw_fraction_samples - calibration.true_fraction_samples
        ),
        dtype=np.float64,
    )
    paths = [
        write_csv_columns(
            output_dir / "lut_bias.csv",
            {
                "true_fraction_samples": calibration.true_fraction_samples,
                "raw_bias_ps": raw_bias_samples / waveform_config.sample_rate_hz * 1e12,
                "corrected_bias_ps": calibration.corrected_error_samples
                / waveform_config.sample_rate_hz
                * 1e12,
            },
        ),
        write_csv_columns(
            output_dir / "clock_tracking.csv",
            {
                "round": np.arange(1, clock_result.raw_offset_s.size + 1),
                "epoch_ms": clock_result.epoch_true_s * 1e3,
                "reference_epoch_ms": clock_result.reference_epoch_s * 1e3,
                "true_raw_offset_ps": clock_result.raw_offset_s * 1e12,
                "estimated_residual_before_ps": clock_result.estimated_offset_s * 1e12,
                "estimated_raw_offset_ps": reconstruct_raw_offset_estimate(clock_result)
                * 1e12,
                "applied_correction_ps": clock_result.applied_correction_s * 1e12,
                "residual_before_ps": clock_result.residual_before_s * 1e12,
                "residual_after_ps": clock_result.residual_after_s * 1e12,
            },
        ),
        write_csv_columns(
            output_dir / "frequency_tracking.csv",
            {
                "round": np.arange(1, frequency_true.size + 1),
                "true_observed_offset_hz": frequency_true,
                "single_estimate_hz": frequency_estimate,
                "tracked_estimate_hz": frequency_tracked,
                "residual_offset_hz": frequency_true - frequency_tracked,
            },
        ),
        write_csv_columns(
            output_dir / "beamforming_states.csv",
            {
                "state": list(beamforming_states.keys()),
                "arrival_difference_ps": [
                    value.residual_arrival_difference_s * 1e12
                    for value in beamforming_states.values()
                ],
                "residual_frequency_hz": [
                    value.residual_frequency_offset_hz
                    for value in beamforming_states.values()
                ],
                "residual_phase_deg": [
                    np.rad2deg(value.residual_phase_difference_rad)
                    for value in beamforming_states.values()
                ],
                "combined_power": [
                    value.signal_power for value in beamforming_states.values()
                ],
                "gain_vs_single_ap_db": [
                    value.metrics.gain_vs_single_ap_db
                    for value in beamforming_states.values()
                ],
                "gain_vs_incoherent_sum_db": [
                    value.metrics.gain_vs_incoherent_sum_db
                    for value in beamforming_states.values()
                ],
                "ideal_loss_db": [
                    value.metrics.normalized_ideal_loss_db
                    for value in beamforming_states.values()
                ],
            },
        ),
        write_csv_columns(
            output_dir / "monte_carlo.csv",
            {
                "snr_db": monte_carlo.snr_db,
                "integer_peak_rmse_ps": monte_carlo.integer_peak_rmse_s * 1e12,
                "qls_rmse_ps": monte_carlo.qls_rmse_s * 1e12,
                "qls_lut_rmse_ps": monte_carlo.lut_rmse_s * 1e12,
                "crlb_std_ps": monte_carlo.crlb_std_s * 1e12,
                "clock_offset_rmse_ps": monte_carlo.clock_offset_rmse_s * 1e12,
                "coherent_gain_vs_incoherent_db": monte_carlo.coherent_gain_vs_incoherent_db,
                "residual_frequency_rmse_hz": monte_carlo.residual_frequency_rmse_hz,
                "acquisition_failure_rate": monte_carlo.acquisition_failure_rate,
            },
        ),
    ]
    return paths


def run_demo(settings: DemoSettings, output_dir: str | Path) -> dict[str, object]:
    """执行时间、频率、相位同步闭环并生成数值结果和十组图。"""

    destination = Path(output_dir)
    figures_dir = destination / "figures"
    cache_dir = destination / "lut_cache"
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(settings.seed)

    waveform = generate_two_tone(settings.waveform)
    calibration = load_or_build_lut(
        settings.waveform,
        cache_dir,
        grid_points=settings.lut_grid_points,
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
        float(frequency_tracked[-1]),
        effective_time_s=frequency_end_epoch_s,
    )
    pilot_epoch_s = frequency_end_epoch_s + 1e-3
    residual_clock_at_pilot_s = ap1_clock.residual_offset_at(pilot_epoch_s)
    channel_feedback = simulate_channel_feedback(
        settings.waveform,
        settings.beamforming,
        settings.phase_feedback,
        oscillator,
        residual_clock_offset_s=residual_clock_at_pilot_s,
        pilot_epoch_s=pilot_epoch_s,
        rng=rng,
        calibration=calibration,
    )
    data_epoch_s = channel_feedback.data_epoch_s
    channel_phase_rate_rad_per_s = (
        settings.phase_feedback.channel_phase_rate_rad_per_s
    )
    channel_phase_at_data_rad = channel_phase_rate_rad_per_s * data_epoch_s
    channel_frequency_offset_hz = channel_phase_rate_rad_per_s / (2.0 * np.pi)
    plant_state = BeamformingPlantState(
        time_only_clock_offset_s=time_only_clock.residual_offset_at(data_epoch_s),
        data_epoch_s=data_epoch_s,
        raw_clock_offset_s=ap1_clock.raw_offset_at(data_epoch_s),
        residual_clock_offset_s=ap1_clock.residual_offset_at(data_epoch_s),
        raw_frequency_offset_hz=(
            oscillator.frequency_offset_hz + channel_frequency_offset_hz
        ),
        residual_frequency_offset_hz=(
            oscillator.residual_frequency_offset_hz + channel_frequency_offset_hz
        ),
        raw_ap1_phase_rad=(
            oscillator.raw_phase_at(data_epoch_s) + channel_phase_at_data_rad
        ),
        residual_ap1_phase_rad=(
            oscillator.phase_at(data_epoch_s) + channel_phase_at_data_rad
        ),
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
    joint_result = run_joint_tracking(settings.waveform, calibration, settings.clock_plant,
        settings.oscillator, settings.frequency, settings.beamforming, settings.phase_feedback,
        settings.joint_tracking, settings.two_way)
    joint_rows = joint_result["rows"]
    write_csv_columns(destination / "joint_tracking.csv",
        {key: [row[key] for row in joint_rows] for key in joint_rows[0]})

    raw_lut_bias = np.asarray(
        wrap_fractional_sample(
            calibration.raw_fraction_samples - calibration.true_fraction_samples
        ),
        dtype=np.float64,
    ) / settings.waveform.sample_rate_hz
    corrected_lut_bias = (
        calibration.corrected_error_samples / settings.waveform.sample_rate_hz
    )
    beamforming_summary = {
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
        for name, value in beamforming_states.items()
    }
    summary: dict[str, object] = {
        "mode": settings.mode,
        "seed": settings.seed,
        "joint_tracking": joint_result["summary"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "matplotlib": version("matplotlib"),
        },
        "state_separation": {
            "plant_truth_policy": "truth is used only to generate observations, unsynchronized baselines, and post-run metrics",
            "control_sources": {
                "time_offset": "two-way timestamp estimate",
                "sample_clock_rate": "slope of reconstructed two-way offset estimates",
                "oscillator_frequency": "tracked reference phase-slope estimate",
                "transmit_phase": "quantized RX pilot LS feedback",
                "transmit_delay": "RX two-burst timestamp difference feedback",
            },
            "plant_at_data_epoch": {
                "data_epoch_s": plant_state.data_epoch_s,
                "raw_clock_offset_ps": plant_state.raw_clock_offset_s * 1e12,
                "residual_clock_offset_ps": plant_state.residual_clock_offset_s * 1e12,
                "raw_frequency_offset_hz": plant_state.raw_frequency_offset_hz,
                "residual_frequency_offset_hz": plant_state.residual_frequency_offset_hz,
                "raw_ap1_phase_deg": float(
                    np.rad2deg(np.angle(np.exp(1j * plant_state.raw_ap1_phase_rad)))
                ),
                "residual_ap1_phase_deg": float(
                    np.rad2deg(
                        np.angle(np.exp(1j * plant_state.residual_ap1_phase_rad))
                    )
                ),
            },
        },
        "waveform": {
            "sample_rate_msa_s": settings.waveform.sample_rate_hz / 1e6,
            "tone_separation_mhz": settings.waveform.tone_separation_hz / 1e6,
            "pulse_duration_us": settings.waveform.pulse_duration_s * 1e6,
            "carrier_frequency_ghz_parameter_only": settings.waveform.carrier_frequency_hz
            / 1e9,
        },
        "snr_definition": {
            "monte_carlo_axis": "active-region signal power / complex AWGN sample power",
            "noise_bandwidth_hz": settings.monte_carlo.noise_bandwidth_hz
            or settings.waveform.sample_rate_hz,
            "crlb_formula": "var(tau) >= N0 / (2 * (pi*B)^2 * Es)",
        },
        "lut": {
            "signature": calibration.signature,
            "grid_points": calibration.grid_points,
            "raw_max_abs_bias_ps": float(np.max(np.abs(raw_lut_bias)) * 1e12),
            "raw_rmse_ps": float(np.sqrt(np.mean(raw_lut_bias**2)) * 1e12),
            "corrected_max_abs_bias_ps": float(
                np.max(np.abs(corrected_lut_bias)) * 1e12
            ),
            "corrected_rmse_ps": float(
                np.sqrt(np.mean(corrected_lut_bias**2)) * 1e12
            ),
        },
        "delay_demo": {
            "true_delay_ns": true_delay_s * 1e9,
            "integer_delay_ns": raw_delay.integer_lag_samples
            / settings.waveform.sample_rate_hz
            * 1e9,
            "raw_qls_delay_ns": raw_delay.delay_s * 1e9,
            "lut_corrected_delay_ns": corrected_delay_s * 1e9,
            "raw_error_ps": (raw_delay.delay_s - true_delay_s) * 1e12,
            "corrected_error_ps": (corrected_delay_s - true_delay_s) * 1e12,
        },
        "clock_tracking": {
            "initial_raw_offset_ps": float(clock_result.raw_offset_s[0] * 1e12),
            "final_raw_offset_ps": float(clock_result.raw_offset_s[-1] * 1e12),
            "final_estimated_raw_offset_ps": float(
                reconstruct_raw_offset_estimate(clock_result)[-1] * 1e12
            ),
            "final_residual_after_update_ps": float(
                clock_result.residual_after_s[-1] * 1e12
            ),
            "estimated_fractional_frequency_offset": clock_frequency_estimate,
            "applied_fractional_frequency_correction": ap1_clock.fractional_frequency_correction,
            "residual_fractional_frequency_offset": ap1_clock.effective_rate - 1.0,
        },
        "frequency_sync": {
            "final_true_observed_offset_hz": float(frequency_true[-1]),
            "final_single_estimate_hz": float(frequency_estimate[-1]),
            "final_tracked_estimate_hz": float(frequency_tracked[-1]),
            "applied_oscillator_correction_hz": oscillator.frequency_correction_hz,
            "final_residual_hz": oscillator.residual_frequency_offset_hz,
        },
        "channel_feedback": {
            "measured_arrival_difference_ps": channel_feedback.measured_arrival_difference_s * 1e12,
            "applied_tx_time_correction_ps": channel_feedback.tx_time_correction_s * 1e12,
            "pilot_epoch_s": channel_feedback.pilot_epoch_s,
            "data_epoch_s": channel_feedback.data_epoch_s,
            "feedback_delay_us": settings.phase_feedback.feedback_delay_s * 1e6,
            "phase_quantization_bits": settings.phase_feedback.phase_quantization_bits,
            "true_data_phase_rad": np.angle(
                channel_feedback.true_effective_channels_at_data
            ),
            "raw_pilot_estimated_phase_rad": np.angle(
                channel_feedback.raw_channel_estimates
            ),
            "feedback_phase_rad": np.angle(
                channel_feedback.feedback_channel_estimates
            ),
            "phase_error_at_data_deg": np.rad2deg(
                channel_feedback.phase_error_at_data_rad
            ),
        },
        "beamforming": beamforming_summary,
        "monte_carlo_highest_snr": {
            "residual_frequency_rmse_hz": float(monte_carlo.residual_frequency_rmse_hz[-1]),
            "acquisition_failure_rate": float(monte_carlo.acquisition_failure_rate[-1]),
            "snr_db": float(monte_carlo.snr_db[-1]),
            "integer_peak_rmse_ps": float(monte_carlo.integer_peak_rmse_s[-1] * 1e12),
            "qls_rmse_ps": float(monte_carlo.qls_rmse_s[-1] * 1e12),
            "qls_lut_rmse_ps": float(monte_carlo.lut_rmse_s[-1] * 1e12),
            "crlb_std_ps": float(monte_carlo.crlb_std_s[-1] * 1e12),
            "clock_offset_rmse_ps": float(
                monte_carlo.clock_offset_rmse_s[-1] * 1e12
            ),
            "coherent_gain_vs_incoherent_db": float(
                monte_carlo.coherent_gain_vs_incoherent_db[-1]
            ),
        },
    }

    write_json(destination / "run_config.json", settings)
    write_json(destination / "summary.json", summary)
    _save_numeric_tables(
        destination,
        calibration,
        settings.waveform,
        clock_result,
        frequency_true,
        frequency_estimate,
        frequency_tracked,
        beamforming_states,
        monte_carlo,
    )

    figure_paths: list[Path] = []
    figure_paths += plot_time_waveform(waveform, figures_dir)
    figure_paths += plot_spectrum(waveform, figures_dir)
    figure_paths += plot_correlation_qls(
        correlation, raw_delay, settings.waveform.sample_rate_hz, figures_dir
    )
    figure_paths += plot_lut_bias(
        calibration, settings.waveform.sample_rate_hz, figures_dir
    )
    figure_paths += plot_rmse_crlb(monte_carlo, figures_dir)
    figure_paths += plot_clock_tracking(clock_result, figures_dir)
    figure_paths += plot_frequency_tracking(
        frequency_true, frequency_estimate, frequency_tracked, figures_dir
    )
    figure_paths += plot_received_waveforms(
        beamforming_states, settings.waveform.sample_rate_hz, figures_dir
    )
    figure_paths += plot_coherent_gain(beamforming_states, figures_dir)
    figure_paths += plot_residual_summary(beamforming_states, figures_dir)

    manifest = {
        "mode": settings.mode,
        "figure_count": len(figure_paths),
        "figures": [str(path.relative_to(destination)) for path in figure_paths],
        "tables": [
            "lut_bias.csv",
            "clock_tracking.csv",
            "frequency_tracking.csv",
            "beamforming_states.csv",
            "monte_carlo.csv",
            "joint_tracking.csv",
        ],
        "summary": "summary.json",
        "configuration": "run_config.json",
    }
    write_json(destination / "manifest.json", manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-AP distributed coherent beamforming synchronization demo"
    )
    parser.add_argument(
        "--mode",
        choices=("fast_demo", "formal"),
        default="fast_demo",
        help="fast_demo uses 100 trials; formal uses 1000 trials and the 10 us pulse",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory; default is results/<mode>",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；成功时打印关键结果和输出目录。"""

    args = _parser().parse_args(argv)
    settings = build_demo_settings(args.mode)
    output_dir = args.output_dir or Path("results") / args.mode
    print(f"Running {settings.mode}: LUT={settings.lut_grid_points}, trials/SNR={settings.monte_carlo.trials_per_snr}")
    summary = run_demo(settings, output_dir)
    high_snr = summary["monte_carlo_highest_snr"]
    full_sync = summary["beamforming"]["full_sync"]
    print(
        "36 dB: QLS+LUT RMSE="
        f"{high_snr['qls_lut_rmse_ps']:.3f} ps, clock RMSE="
        f"{high_snr['clock_offset_rmse_ps']:.3f} ps"
    )
    print(
        "Full sync: gain over incoherent sum="
        f"{full_sync['gain_vs_incoherent_sum_db']:.3f} dB, ideal loss="
        f"{full_sync['normalized_ideal_loss_db']:.4f} dB"
    )
    print(f"Results written to: {Path(output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
