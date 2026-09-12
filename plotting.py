"""论文风格的十一组 Demo 图；每组同时保存 PNG 和 PDF。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import reconstruct_raw_offset_estimate
from models import (
    BeamformingResult,
    ClockTrackingResult,
    CorrelationResult,
    DelayEstimate,
    MonteCarloResult,
    QLSCalibration,
    QLSValidation,
    ThreeExperimentResult,
    Waveform,
)


_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
_STATE_ORDER = ("unsynchronized", "time_only", "time_frequency", "full_sync")
_STATE_LABELS = ("Unsynchronized", "Time", "Time + frequency", "Full sync")


def configure_paper_style() -> None:
    """配置白底、Times New Roman 和适合论文插图的线条字号。"""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(
    fig: plt.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    tight_layout: bool = True,
) -> list[Path]:
    """紧凑布局后把图保存为 180 dpi PNG 和矢量 PDF。"""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    paths = [directory / f"{stem}.png", directory / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_time_waveform(waveform: Waveform, output_dir: str | Path) -> list[Path]:
    """图 1：公式生成的完整脉冲包络与双音拍频局部波形。"""

    configure_paper_style()
    fig, (envelope_ax, waveform_ax) = plt.subplots(2, 1, figsize=(7.2, 5.3))
    time_us = waveform.time_s * 1e6
    envelope_scale = float(np.max(np.abs(waveform.samples)))
    envelope_ax.plot(time_us, np.abs(waveform.samples), color=_COLORS[0], lw=0.9, label="|s(t)|")
    envelope_ax.plot(time_us, waveform.envelope * envelope_scale, "k--", lw=1.0, label="Raised-cosine window")
    envelope_ax.set(ylabel="Magnitude", title="Formula-generated 40 MHz pulsed two-tone")
    envelope_ax.legend(loc="upper right", ncol=2)

    zoom_duration_us = min(0.6, float(time_us[-1]))
    zoom = time_us <= zoom_duration_us
    waveform_ax.plot(time_us[zoom], waveform.samples.real[zoom], color=_COLORS[0], lw=0.9)
    waveform_ax.axhline(0.0, color="black", lw=0.6)
    waveform_ax.set(
        xlabel="Time (µs)",
        ylabel="In-phase amplitude",
        title=f"First {zoom_duration_us:.1f} µs: beating of −20 MHz and +20 MHz",
    )
    return save_figure(fig, output_dir, "01_two_tone_time_waveform")


def plot_spectrum(
    waveform: Waveform,
    received_samples: Sequence[complex] | np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    """图 2：由公式 TX 样点和链路 RX 样点实际计算双音频谱。"""

    configure_paper_style()
    rx = np.asarray(received_samples, dtype=np.complex128)
    if rx.ndim != 1 or rx.size == 0 or not np.all(np.isfinite(rx)):
        raise ValueError("received_samples 必须为非空一维有限复数组")

    def spectrum_db(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spectrum = np.fft.fftshift(np.fft.fft(samples))
        frequency_mhz = np.fft.fftshift(
            np.fft.fftfreq(samples.size, d=1.0 / waveform.sample_rate_hz)
        ) / 1e6
        magnitude = np.abs(spectrum)
        magnitude_db = 20.0 * np.log10(
            np.maximum(magnitude / np.max(magnitude), 1e-6)
        )
        return frequency_mhz, magnitude_db

    tx_frequency_mhz, tx_db = spectrum_db(waveform.samples)
    rx_frequency_mhz, rx_db = spectrum_db(rx)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.plot(tx_frequency_mhz, tx_db, color=_COLORS[0], lw=0.9, label="Formula TX pulse")
    ax.plot(rx_frequency_mhz, rx_db, color=_COLORS[1], lw=0.8, alpha=0.75, label="RX after delay + AWGN")
    ax.set(
        xlim=(-50.0, 50.0),
        ylim=(-105.0, 3.0),
        xlabel="Baseband frequency (MHz)",
        ylabel="Normalized magnitude (dB)",
        title="FFT calculated from transmitted and received IQ samples",
    )
    ax.legend(loc="lower center", ncol=2)
    return save_figure(fig, output_dir, "02_two_tone_spectrum")


def plot_correlation_qls(
    correlation: CorrelationResult,
    estimate: DelayEstimate,
    sample_rate_hz: float,
    output_dir: str | Path,
) -> list[Path]:
    """图 3：匹配滤波峰值的三个样点与 QLS 抛物线。"""

    configure_paper_style()
    peak = estimate.peak_array_index
    lower = max(0, peak - 3)
    upper = min(correlation.magnitude.size, peak + 4)
    x_samples = correlation.lags_samples[lower:upper].astype(np.float64)
    scale = float(correlation.magnitude[peak])
    y = correlation.magnitude[lower:upper] / scale
    local_x = correlation.lags_samples[peak - 1 : peak + 2].astype(np.float64)
    local_y = correlation.magnitude[peak - 1 : peak + 2] / scale
    coefficients = np.polyfit(local_x, local_y, 2)
    dense_x = np.linspace(local_x[0], local_x[-1], 300)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x_samples, y, "o-", color=_COLORS[0], label="Matched-filter samples")
    ax.plot(dense_x, np.polyval(coefficients, dense_x), color=_COLORS[1], lw=1.4, label="Three-point QLS parabola")
    qls_samples = estimate.integer_lag_samples + estimate.fractional_offset_samples
    ax.axvline(qls_samples, color=_COLORS[2], ls="--", label=f"QLS = {qls_samples:.4f} samples")
    ax.set(
        xlabel=f"Delay lag (samples at {sample_rate_hz / 1e6:.0f} MSa/s)",
        ylabel="Normalized correlation magnitude",
        title="Matched-filter peak and local QLS fit",
    )
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "03_matched_filter_qls")


def plot_lut_bias(
    calibration: QLSCalibration,
    validation: QLSValidation,
    sample_rate_hz: float,
    output_dir: str | Path,
) -> list[Path]:
    """图 4：训练 LUT 与独立分数时延验证集上的校正结果。"""

    configure_paper_style()
    training_error = (
        (calibration.raw_fraction_samples - calibration.true_fraction_samples + 0.5) % 1.0
        - 0.5
    ) / sample_rate_hz * 1e12
    validation_raw = validation.raw_error_samples / sample_rate_hz * 1e12
    validation_corrected = validation.corrected_error_samples / sample_rate_hz * 1e12
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    axes[0].plot(
        calibration.true_fraction_samples,
        training_error,
        color=_COLORS[1],
        lw=1.0,
        label="LUT training scan: raw QLS bias",
    )
    axes[0].plot(
        validation.true_fraction_samples,
        validation_raw,
        color=_COLORS[3],
        lw=0.8,
        alpha=0.8,
        label="Held-out delays: raw QLS bias",
    )
    axes[0].set(ylabel="Error (ps)", title="QLS periodic bias before LUT correction")
    axes[0].legend(loc="best")
    axes[1].plot(
        validation.true_fraction_samples,
        validation_corrected,
        color=_COLORS[2],
        lw=1.0,
        label="Held-out delays after LUT",
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set(
        xlabel="True fractional delay (samples)",
        ylabel="Error (ps)",
        title="Independent validation after LUT correction",
    )
    axes[1].legend(loc="best")
    return save_figure(fig, output_dir, "04_qls_lut_bias")


def plot_rmse_crlb(result: MonteCarloResult, output_dir: str | Path) -> list[Path]:
    """图 5：三种时延估计 RMSE 和双音 CRLB 随 SNR 变化。"""

    configure_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    curves = (
        (result.integer_peak_rmse_s, "Integer peak", _COLORS[3], "o"),
        (result.qls_rmse_s, "QLS", _COLORS[1], "s"),
        (result.lut_rmse_s, "QLS + LUT", _COLORS[0], "^"),
        (result.crlb_std_s, "CRLB", "black", "x"),
        (result.clock_offset_rmse_s, "Two-way clock offset", _COLORS[2], "d"),
    )
    for values, label, color, marker in curves:
        ax.semilogy(result.snr_db, values * 1e12, marker=marker, color=color, lw=1.2, label=label)
    ax.set(xlabel="Active-sample SNR (dB)", ylabel="RMSE / standard deviation (ps)", title="Delay estimation accuracy and CRLB")
    ax.legend(loc="best", ncol=2)
    return save_figure(fig, output_dir, "05_delay_rmse_crlb")


def plot_clock_tracking(result: ClockTrackingResult, output_dir: str | Path) -> list[Path]:
    """图 6：AP1 真钟差、双向估计和补偿后残差。"""

    configure_paper_style()
    rounds = np.arange(1, result.raw_offset_s.size + 1)
    cumulative_raw_offset_estimate_s = reconstruct_raw_offset_estimate(result)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.5), sharex=True)
    axes[0].plot(rounds, result.raw_offset_s * 1e12, "o-", color=_COLORS[3], label="True raw offset")
    axes[0].plot(rounds, cumulative_raw_offset_estimate_s * 1e12, "s-", color=_COLORS[0], label="Cumulative two-way estimate")
    axes[0].set(ylabel="Clock offset (ps)", title="AP1 clock-offset tracking")
    axes[0].legend(loc="best")
    axes[1].plot(rounds, result.residual_after_s * 1e12, "^-", color=_COLORS[2], label="Residual after update")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set(xlabel="Synchronization round", ylabel="Residual offset (ps)")
    axes[1].legend(loc="best")
    return save_figure(fig, output_dir, "06_clock_offset_tracking")


def plot_frequency_tracking(
    true_hz: Sequence[float],
    measured_hz: Sequence[float],
    tracked_hz: Sequence[float],
    output_dir: str | Path,
) -> list[Path]:
    """图 7：真实、单次估计、指数跟踪和残余频偏。"""

    configure_paper_style()
    true_values = np.asarray(true_hz, dtype=np.float64)
    measured_values = np.asarray(measured_hz, dtype=np.float64)
    tracked_values = np.asarray(tracked_hz, dtype=np.float64)
    rounds = np.arange(1, true_values.size + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    axes[0].plot(rounds, true_values, "k-", lw=1.4, label="True observed offset")
    axes[0].plot(rounds, measured_values, "o", color=_COLORS[1], label="Single estimate")
    axes[0].plot(rounds, tracked_values, "s-", color=_COLORS[0], label="Exponential tracker")
    axes[0].set(ylabel="Frequency offset (Hz)", title="Software-equivalent frequency synchronization")
    axes[0].legend(loc="best", ncol=3)
    axes[1].plot(rounds, true_values - tracked_values, "d-", color=_COLORS[2])
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set(xlabel="Synchronization round", ylabel="Residual offset (Hz)")
    return save_figure(fig, output_dir, "07_frequency_tracking")


def plot_received_waveforms(
    states: Mapping[str, BeamformingResult],
    sample_rate_hz: float,
    output_dir: str | Path,
) -> list[Path]:
    """图 8：RX 未同步与完整同步时合成复包络实部。"""

    configure_paper_style()
    unsync = states["unsynchronized"].received_samples
    full = states["full_sync"].received_samples
    time_unsync_us = np.arange(unsync.size, dtype=np.float64) / sample_rate_hz * 1e6
    time_full_us = np.arange(full.size, dtype=np.float64) / sample_rate_hz * 1e6
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.plot(time_unsync_us, unsync.real, color=_COLORS[1], lw=0.9, alpha=0.85, label="Unsynchronized")
    ax.plot(time_full_us, full.real, color=_COLORS[0], lw=0.9, alpha=0.85, label="Full synchronization")
    ax.set(xlabel="Receiver sample time (µs)", ylabel="Combined in-phase amplitude", title="Received coherent waveform")
    ax.legend(loc="upper right")
    return save_figure(fig, output_dir, "08_received_waveforms")


def plot_coherent_gain(
    states: Mapping[str, BeamformingResult], output_dir: str | Path
) -> list[Path]:
    """图 9：四种同步状态相对非相干功率和的相干增益。"""

    configure_paper_style()
    values = [states[name].metrics.gain_vs_incoherent_sum_db for name in _STATE_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(_STATE_LABELS, values, color=_COLORS[:4])
    ax.axhline(10.0 * np.log10(2.0), color="black", ls="--", lw=1.0, label="Ideal = 3.01 dB")
    ax.bar_label(bars, fmt="%.2f dB", padding=3)
    ax.set(ylabel="Gain over incoherent power sum (dB)", title="Coherent gain under four synchronization states")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "09_four_state_coherent_gain")


def plot_residual_summary(
    states: Mapping[str, BeamformingResult], output_dir: str | Path
) -> list[Path]:
    """图 10：四状态残余到达、频率和相位误差分量。"""

    configure_paper_style()
    floor = 1e-6
    time_ps = [max(abs(states[name].residual_arrival_difference_s * 1e12), floor) for name in _STATE_ORDER]
    frequency_hz = [max(abs(states[name].residual_frequency_offset_hz), floor) for name in _STATE_ORDER]
    phase_deg = [max(abs(np.rad2deg(states[name].residual_phase_difference_rad)), floor) for name in _STATE_ORDER]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    datasets = (
        (time_ps, "|Arrival difference| (ps)", _COLORS[3]),
        (frequency_hz, "|Residual frequency| (Hz)", _COLORS[1]),
        (phase_deg, "|Residual phase| (deg)", _COLORS[0]),
    )
    for axis, (values, ylabel, color) in zip(axes, datasets):
        axis.bar(_STATE_LABELS, values, color=color, alpha=0.85)
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
    axes[0].set_title("Residual synchronization errors")
    axes[-1].tick_params(axis="x", rotation=12)
    return save_figure(fig, output_dir, "10_residual_error_summary")


def plot_three_experiment_precision(
    result: ThreeExperimentResult,
    output_dir: str | Path,
) -> list[Path]:
    """图 11：三种论文配置的完整链路时间与脉冲到达精度。"""

    configure_paper_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.7),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    for profile_index, (axis, label) in enumerate(
        zip(axes, result.profile_labels)
    ):
        axis.semilogy(
            result.snr_db,
            result.time_transfer_std_s[profile_index] * 1e12,
            "o-",
            color=_COLORS[0],
            lw=1.2,
            label="Two-way time transfer",
        )
        axis.semilogy(
            result.snr_db,
            result.beamforming_std_s[profile_index] * 1e12,
            "s--",
            color=_COLORS[1],
            lw=1.2,
            label="RX pulse interarrival",
        )
        axis.semilogy(
            result.snr_db,
            result.crlb_best_case_std_s * 1e12,
            ":",
            color="black",
            lw=1.2,
            label="CRLB at SNR + 3 dB",
        )
        axis.set(
            xlabel=f"Time-transfer SNR (dB)\n({chr(97 + profile_index)})",
            title=label,
        )
        axis.tick_params(which="both", direction="in")
    axes[0].set_ylabel("Sample standard deviation (ps)")
    axes[2].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Three communication-chain synchronization experiments")
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.07, right=0.99)
    return save_figure(
        fig,
        output_dir,
        "11_three_experiment_precision",
        tight_layout=False,
    )
