"""正式报告的四张读者导向图；每张同时保存 PNG 与 PDF。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

from config import FrequencySyncConfig
from frequency_sync import frequency_transfer_rf_tones_hz
from models import (
    CorrelationResult,
    DelayEstimate,
    MonteCarloResult,
    QLSCalibration,
    QLSValidation,
    SensitivityResult,
    ThreeExperimentResult,
    Waveform,
)


_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def configure_paper_style() -> None:
    """选择可显示中文的字体并设置适合报告阅读的绘图参数。"""

    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (
            name
            for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")
            if name in available
        ),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def save_figure(
    figure: plt.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    tight_layout: bool = True,
) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        figure.tight_layout()
    paths = [directory / f"{stem}.png", directory / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=200, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _spectrum_db(samples: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.fftshift(np.fft.fft(np.asarray(samples, dtype=np.complex128)))
    frequency_mhz = np.fft.fftshift(
        np.fft.fftfreq(spectrum.size, d=1.0 / sample_rate_hz)
    ) / 1e6
    magnitude = np.abs(spectrum)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude / np.max(magnitude), 1e-6))
    return frequency_mhz, magnitude_db


def plot_system_signal_chain(
    waveform: Waveform,
    frequency_config: FrequencySyncConfig,
    received_samples: Sequence[complex] | np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    """图 01：三条链路、有限双音脉冲及由实际 IQ 计算的频谱。"""

    configure_paper_style()
    received = np.asarray(received_samples, dtype=np.complex128)
    if received.ndim != 1 or received.size == 0 or not np.all(np.isfinite(received)):
        raise ValueError("received_samples 必须为非空一维有限复数数组")

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    if_axis, envelope_axis, spectrum_axis, chain_axis = axes.reshape(-1)

    display_rate_hz = 2e9
    display_duration_s = min(0.6e-6, float(waveform.time_s[-1]))
    display_time_s = np.arange(
        int(np.floor(display_duration_s * display_rate_hz)) + 1,
        dtype=np.float64,
    ) / display_rate_hz
    envelope = np.interp(display_time_s, waveform.time_s, waveform.envelope)
    half_separation_hz = 0.5 * 40e6
    display_if_hz = 100e6
    display_signal = envelope * np.real(
        np.exp(1j * 2 * np.pi * (display_if_hz - half_separation_hz) * display_time_s)
        + np.exp(1j * 2 * np.pi * (display_if_hz + half_separation_hz) * display_time_s)
    )
    display_signal /= np.max(np.abs(display_signal))
    if_axis.plot(display_time_s * 1e6, display_signal, color=_COLORS[0], lw=0.9)
    if_axis.plot(display_time_s * 1e6, envelope, "k--", lw=0.8, label="± 包络")
    if_axis.plot(display_time_s * 1e6, -envelope, "k--", lw=0.8)
    if_axis.set(
        xlabel="时间 (µs)",
        ylabel="归一化实幅度",
        title="(a) 40 MHz 双音时域示意：80/120 MHz 显示中频",
    )
    if_axis.legend(loc="upper right")

    envelope_axis.plot(waveform.time_s * 1e6, waveform.envelope, color=_COLORS[1], lw=1.2)
    envelope_axis.set(
        xlabel="时间 (µs)",
        ylabel="包络",
        ylim=(-0.05, 1.08),
        title="(b) 每次链路测量只发送一个 10 µs 脉冲",
    )
    envelope_axis.annotate(
        "50 ns 升沿",
        xy=(0.05, 1.0),
        xytext=(0.7, 0.55),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )

    tx_frequency_mhz, tx_db = _spectrum_db(waveform.samples, waveform.sample_rate_hz)
    rx_frequency_mhz, rx_db = _spectrum_db(received, waveform.sample_rate_hz)
    spectrum_axis.plot(tx_frequency_mhz, tx_db, color=_COLORS[0], lw=1.0, label="发送 IQ")
    spectrum_axis.plot(
        rx_frequency_mhz,
        rx_db,
        color=_COLORS[1],
        lw=0.8,
        alpha=0.75,
        label="传播、噪声与接收采样后的 IQ",
    )
    spectrum_axis.set(
        xlim=(-35.0, 35.0),
        ylim=(-100.0, 3.0),
        xlabel="相对 5.8 GHz 载频的频率 (MHz)",
        ylabel="归一化幅度 (dB)",
        title="(c) 实际有限 IQ 的 FFT：谱峰位于 ±20 MHz",
    )
    spectrum_axis.axvline(-20.0, color="black", ls=":", lw=0.7)
    spectrum_axis.axvline(20.0, color="black", ls=":", lw=0.7)
    spectrum_axis.legend(loc="lower center")

    low_hz, high_hz = frequency_transfer_rf_tones_hz(frequency_config)
    chain_axis.axis("off")
    chain_axis.set_title("(d) 三条信号链及其用途")
    rows = (
        (
            "时间传递",
            "5.8 GHz ± 20 MHz\n40 MHz 间隔，10 µs 脉冲",
            "IQ → 匹配滤波 → QLS\n→ LUT → 四时间戳",
        ),
        (
            "频率传递",
            f"{low_hz / 1e9:.3f}/{high_hz / 1e9:.3f} GHz 连续双音",
            "自混频 → 10 MHz 参考\n→ 采样钟速率估计",
        ),
        (
            "下游验证",
            "1.2 GHz，50 MHz 双音脉冲",
            "时间/频率/相位估计\n→ 两 AP 相干合成",
        ),
    )
    for row_index, (purpose, signal, processing) in enumerate(rows):
        y = 0.82 - row_index * 0.32
        chain_axis.text(0.02, y, purpose, weight="bold", transform=chain_axis.transAxes)
        chain_axis.text(0.24, y, signal, transform=chain_axis.transAxes, va="center")
        chain_axis.text(
            0.63,
            y,
            processing,
            transform=chain_axis.transAxes,
            va="center",
            fontsize=8.5,
        )
        chain_axis.annotate(
            "",
            xy=(0.61, y),
            xytext=(0.57, y),
            xycoords=chain_axis.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 0.9},
        )
    chain_axis.text(
        0.02,
        0.02,
        "说明：4.3/5.8 GHz 是 RF 参数；200 MSa/s 下的计算在等效复基带完成。",
        transform=chain_axis.transAxes,
        color="dimgray",
    )

    figure.suptitle("分布式相干同步 Demo：信号、参数与处理链", fontsize=13)
    figure.subplots_adjust(top=0.91)
    return save_figure(figure, output_dir, "01_system_signal_chain")


def plot_qls_lut_validation(
    correlation: CorrelationResult,
    estimate: DelayEstimate,
    calibration: QLSCalibration,
    validation: QLSValidation,
    monte_carlo: MonteCarloResult,
    sample_rate_hz: float,
    output_dir: str | Path,
) -> list[Path]:
    """图 02：QLS、LUT 确定性校准与带噪性能证据。"""

    configure_paper_style()
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    peak_axis, lut_axis, rmse_axis = axes

    peak = estimate.peak_array_index
    lower = max(0, peak - 3)
    upper = min(correlation.magnitude.size, peak + 4)
    x = correlation.lags_samples[lower:upper].astype(np.float64)
    reference = float(correlation.magnitude[peak])
    y = 20 * np.log10(np.maximum(correlation.magnitude[lower:upper] / reference, 1e-15))
    local_x = correlation.lags_samples[peak - 1 : peak + 2].astype(np.float64)
    local_y = 20 * np.log10(
        np.maximum(correlation.magnitude[peak - 1 : peak + 2] / reference, 1e-15)
    )
    coefficients = np.polyfit(local_x, local_y, 2)
    dense_x = np.linspace(local_x[0], local_x[-1], 300)
    peak_axis.plot(x, y, "o", color=_COLORS[0], label="匹配滤波离散样点")
    peak_axis.plot(dense_x, np.polyval(coefficients, dense_x), color=_COLORS[1], label="三点对数幅度 QLS")
    peak_axis.axvline(
        estimate.integer_lag_samples + estimate.fractional_offset_samples,
        color=_COLORS[2],
        ls="--",
        label="亚样点峰值",
    )
    peak_axis.set(
        xlabel="时延栅格 (sample)",
        ylabel="相对峰值 (dB)",
        title="(a) 5 ns 栅格内的 QLS 插值",
    )
    peak_axis.legend(loc="lower center")

    raw_bias_ps = validation.raw_error_samples / sample_rate_hz * 1e12
    corrected_bias_ps = validation.corrected_error_samples / sample_rate_hz * 1e12
    lut_axis.plot(
        validation.true_fraction_samples,
        raw_bias_ps,
        color=_COLORS[0],
        lw=1.0,
        label="原始 QLS 系统偏差",
    )
    lut_axis.plot(
        validation.true_fraction_samples,
        corrected_bias_ps,
        color=_COLORS[1],
        lw=1.0,
        label="LUT 校正残差",
    )
    lut_axis.set(
        xlabel="真实分数时延 (sample)",
        ylabel="无噪声偏差 (ps)",
        title="(b) 独立栅格上的确定性插值检查",
    )
    lut_axis.legend(loc="lower center")
    residual_axis = lut_axis.inset_axes([0.56, 0.57, 0.40, 0.34])
    residual_axis.plot(
        validation.true_fraction_samples,
        corrected_bias_ps,
        color=_COLORS[1],
        lw=0.9,
    )
    residual_axis.set_title("校正残差放大", fontsize=8)
    residual_axis.tick_params(labelsize=7)
    residual_axis.grid(True, alpha=0.2)

    rmse_axis.semilogy(
        monte_carlo.snr_db,
        monte_carlo.integer_peak_rmse_s * 1e12,
        "^-",
        color=_COLORS[3],
        label="整数峰",
    )
    rmse_axis.semilogy(
        monte_carlo.snr_db,
        monte_carlo.qls_rmse_s * 1e12,
        "s-",
        color=_COLORS[0],
        label="QLS",
    )
    rmse_axis.semilogy(
        monte_carlo.snr_db,
        monte_carlo.lut_rmse_s * 1e12,
        "o-",
        color=_COLORS[1],
        label="QLS + LUT",
    )
    rmse_axis.semilogy(
        monte_carlo.snr_db,
        monte_carlo.crlb_std_s * 1e12,
        "k--",
        lw=1.1,
        label="CRLB",
    )
    rmse_axis.set(
        xlabel="活动区复 AWGN SNR (dB)",
        ylabel="时延 RMSE (ps)",
        title="(c) 带噪 Monte Carlo 才是性能结果",
    )
    rmse_axis.legend(loc="lower left")

    figure.suptitle("QLS 与 LUT：从采样栅格到皮秒级时延估计", fontsize=13)
    figure.subplots_adjust(top=0.84)
    return save_figure(figure, output_dir, "02_qls_lut_validation")


def plot_three_config_vs_crlb(
    result: ThreeExperimentResult,
    output_dir: str | Path,
) -> list[Path]:
    """图 03：论文三种配置的软件等效时间同步结果与 CRLB。"""

    configure_paper_style()
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    markers = ("o", "s", "^")
    labels = ("有线时间 + 有线频率", "无线时间 + 有线频率", "无线时间 + 无线频率")
    for index, label in enumerate(labels):
        axis.semilogy(
            result.snr_db,
            result.time_transfer_std_s[index] * 1e12,
            marker=markers[index],
            color=_COLORS[index],
            lw=1.3,
            label=label,
        )
    axis.semilogy(
        result.snr_db,
        result.two_way_clock_crlb_std_s * 1e12,
        "k--",
        lw=1.3,
        label="双向钟差 CRLB（单向时延 CRLB / √2）",
    )
    first_snr = float(result.snr_db[0])
    second_snr = (
        float(result.snr_db[1]) if result.snr_db.size > 1 else first_snr + 1.0
    )
    threshold_edge = 0.5 * (first_snr + second_snr)
    axis.axvspan(
        first_snr - 0.6,
        threshold_edge,
        color="tab:red",
        alpha=0.08,
        zorder=0,
    )
    axis.annotate(
        "低 SNR 门限区\n捕获/错误峰粗差；CRLB 不包含此类错误",
        xy=(first_snr, float(np.max(result.time_transfer_std_s[:, 0]) * 1e12)),
        xytext=(second_snr + 1.0, 260.0),
        arrowprops={"arrowstyle": "->", "color": "dimgray", "lw": 0.9},
        fontsize=8.5,
        color="dimgray",
        ha="left",
    )
    axis.set(
        xlabel="时间传递接收端活动区复 AWGN SNR (dB)",
        ylabel="双向时间同步标准差 (ps)",
        title="论文三种配置的软件等效仿真与 CRLB",
    )
    axis.legend(loc="upper right")
    axis.text(
        0.01,
        0.02,
        "每点独立运行完整 IQ→匹配滤波→QLS→LUT→四时间戳链路；曲线不是硬件测量值。",
        transform=axis.transAxes,
        color="dimgray",
    )
    return save_figure(figure, output_dir, "03_three_config_vs_crlb")


def plot_model_mismatch_sensitivity(
    result: SensitivityResult,
    output_dir: str | Path,
) -> list[Path]:
    """图 04：双向不对称和参考相位扰动的受控敏感性。"""

    configure_paper_style()
    figure, (asymmetry_axis, phase_axis) = plt.subplots(1, 2, figsize=(10.2, 4.1))
    asymmetry_ps = result.path_asymmetry_s * 1e12
    asymmetry_axis.errorbar(
        asymmetry_ps,
        result.clock_bias_s * 1e12,
        yerr=result.clock_std_s * 1e12,
        fmt="o-",
        color=_COLORS[0],
        capsize=3,
        label="完整四时间戳仿真",
    )
    asymmetry_axis.plot(
        asymmetry_ps,
        0.5 * asymmetry_ps,
        "k--",
        label="理论：偏差 = 不对称量 / 2",
    )
    asymmetry_axis.set(
        xlabel="上行时延 − 下行时延 (ps)",
        ylabel="估计钟差偏差 (ps)",
        title="(a) 双向链路不对称",
    )
    asymmetry_axis.legend(loc="upper left")

    phase_axis.semilogy(
        np.rad2deg(result.reference_phase_noise_rad),
        result.holdover_timing_rmse_s * 1e12,
        "o-",
        color=_COLORS[1],
        label="100 ms 后时序漂移 RMSE",
    )
    phase_axis.set(
        xlabel="两个 10 MHz 观测窗口的相位扰动标准差 (°)",
        ylabel="保持期时序漂移 RMSE (ps)",
        title="(b) 无线频率参考相位扰动",
    )
    phase_axis.legend(loc="upper left")

    figure.suptitle("仿真扩展：模型假设失配会怎样影响同步", fontsize=13)
    figure.subplots_adjust(top=0.84)
    return save_figure(figure, output_dir, "04_model_mismatch_sensitivity")
