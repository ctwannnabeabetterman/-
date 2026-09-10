"""RX 导频复信道估计和两 AP 发射相位补偿。"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from channel import add_awgn, apply_fractional_delay
from config import BeamformingConfig, PhaseFeedbackConfig, WaveformConfig
from models import ChannelFeedbackResult
from oscillator_model import LocalOscillator
from acquisition import measure_capture, simulate_capture, sync_burst
from lut_calibration import build_qls_lut
from models import QLSCalibration


@lru_cache(maxsize=8)
def _alignment_lut(config: WaveformConfig) -> QLSCalibration:
    """复用离线标定，避免每个导频试验重复建表。"""
    return build_qls_lut(config, grid_points=2001)


def estimate_rx_arrival_difference(
    waveform_config: WaveformConfig, beamforming_config: BeamformingConfig,
    feedback_config: PhaseFeedbackConfig, residual_clock_offset_s: float,
    rng: np.random.Generator, calibration: QLSCalibration | None = None,
) -> float:
    """分时探测两路到达时间，以 RX 本地时间戳之差生成反馈。"""
    lut = calibration if calibration is not None else _alignment_lut(waveform_config)
    burst, _, prefix = sync_burst(waveform_config)
    fs = waveform_config.sample_rate_hz
    margin = feedback_config.alignment_window_margin_s
    count = len(burst) + int(np.ceil(2*margin*fs))
    delays = (beamforming_config.ap0_propagation_delay_s,
              beamforming_config.ap1_propagation_delay_s-residual_clock_offset_s)
    amplitudes = (beamforming_config.ap0_amplitude, beamforming_config.ap1_amplitude)
    observations = []
    for delay, amplitude in zip(delays, amplitudes):
        capture = simulate_capture(burst, fs, start_local_s=-margin,
            start_source_sample=(-margin-delay)*fs, sample_step=1., count=count,
            amplitude=complex(amplitude), snr_db=feedback_config.snr_db, rng=rng)
        measurement = measure_capture(capture, waveform_config, lut)
        observations.append(capture.start_local_s + measurement.corrected_delay_s-prefix/fs)
    return float(observations[1]-observations[0])


def complex_los_channel(
    amplitude: float,
    propagation_delay_s: float,
    carrier_frequency_hz: float,
    additional_phase_rad: float = 0.0,
) -> complex:
    """计算包含载频传播相位的单径复信道。

    先把 ``carrier_frequency_hz * propagation_delay_s`` 折回一个周期，
    避免对巨大相位直接求复指数造成精度损失。
    """

    values = (amplitude, propagation_delay_s, carrier_frequency_hz, additional_phase_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("LoS 信道参数必须为有限数")
    if amplitude < 0.0:
        raise ValueError("amplitude 必须为非负数")
    if propagation_delay_s < 0.0 or carrier_frequency_hz <= 0.0:
        raise ValueError("传播时延必须非负且载频必须为正")
    fractional_cycle = float(np.remainder(carrier_frequency_hz * propagation_delay_s, 1.0))
    phase_rad = -2.0 * np.pi * fractional_cycle + additional_phase_rad
    return complex(amplitude * np.exp(1j * phase_rad))


def estimate_channel_ls(pilot: ArrayLike, received: ArrayLike) -> complex:
    """由分时已知导频和接收复基带样点估计单个 AP 的复信道。"""

    pilot_iq = np.asarray(pilot, dtype=np.complex128)
    received_iq = np.asarray(received, dtype=np.complex128)
    if pilot_iq.ndim != 1 or pilot_iq.size == 0:
        raise ValueError("pilot 必须为非空一维数组")
    if received_iq.shape != pilot_iq.shape:
        raise ValueError("received 必须与 pilot 同形")
    if not np.all(np.isfinite(pilot_iq)) or not np.all(np.isfinite(received_iq)):
        raise ValueError("导频和接收样点必须为有限数")
    pilot_energy = float(np.vdot(pilot_iq, pilot_iq).real)
    if pilot_energy <= np.finfo(np.float64).tiny:
        raise ValueError("导频能量必须大于 0")
    return complex(np.vdot(pilot_iq, received_iq) / pilot_energy)


def compute_phase_weights(
    channel_estimates: ArrayLike,
    normalization: str = "per_ap_fixed",
) -> NDArray[np.complex128]:
    """计算信道相位共轭权重，并按指定发射功率方式归一化。"""

    channels = np.asarray(channel_estimates, dtype=np.complex128)
    if channels.ndim != 1 or channels.size == 0:
        raise ValueError("channel_estimates 必须为非空一维数组")
    if not np.all(np.isfinite(channels)) or np.any(np.abs(channels) == 0.0):
        raise ValueError("复信道估计必须为有限非零数")
    weights = np.exp(-1j * np.angle(channels)).astype(np.complex128)
    if normalization == "per_ap_fixed":
        return weights
    if normalization == "total_fixed":
        return np.asarray(weights / np.sqrt(channels.size), dtype=np.complex128)
    raise ValueError("normalization 必须为 per_ap_fixed 或 total_fixed")


def _quantize_channel_phase(
    channel_estimates: NDArray[np.complex128],
    phase_bits: int | None,
) -> NDArray[np.complex128]:
    """按可选均匀相位码本量化反馈，保留 LS 幅度。"""

    if phase_bits is None:
        return channel_estimates.copy()
    phase_step = 2.0 * np.pi / (2**phase_bits)
    quantized_phase = np.round(np.angle(channel_estimates) / phase_step) * phase_step
    return np.asarray(
        np.abs(channel_estimates) * np.exp(1j * quantized_phase),
        dtype=np.complex128,
    )


def simulate_channel_feedback(
    waveform_config: WaveformConfig,
    beamforming_config: BeamformingConfig,
    feedback_config: PhaseFeedbackConfig,
    oscillator: LocalOscillator,
    *,
    residual_clock_offset_s: float,
    pilot_epoch_s: float,
    rng: np.random.Generator,
    calibration: QLSCalibration | None = None,
) -> ChannelFeedbackResult:
    """通过与数据共享的时钟和本振轨迹生成 RX 导频反馈。

    ``pilot_epoch_s`` 是导频块中心真时刻。AP1 导频包含当前残余时间偏差、
    残余本振相位斜率和可配信道相位漂移。数据历元位于反馈延迟之后，因而
    估计会自然包含反馈陈旧误差，而非直接从数据历元真信道构造权重。
    """

    if not math.isfinite(residual_clock_offset_s) or not math.isfinite(pilot_epoch_s):
        raise ValueError("残余钟差和导频历元必须为有限数")
    count = feedback_config.pilot_symbols
    centered_time_s = (
        np.arange(count, dtype=np.float64) - 0.5 * (count - 1)
    ) / waveform_config.sample_rate_hz
    absolute_time_s = pilot_epoch_s + centered_time_s
    data_epoch_s = pilot_epoch_s + count/(2*waveform_config.sample_rate_hz) + feedback_config.feedback_delay_s
    measured_arrival = 0.0
    if feedback_config.alignment_enabled:
        measured_arrival = estimate_rx_arrival_difference(
            waveform_config, beamforming_config, feedback_config,
            residual_clock_offset_s, rng, calibration)
    tx_time_correction_s = -measured_arrival
    pilot = np.exp(
        1j * np.pi / 2.0 * rng.integers(0, 4, size=count)
    ).astype(np.complex128)

    h0_static = complex_los_channel(
        beamforming_config.ap0_amplitude,
        beamforming_config.ap0_propagation_delay_s,
        waveform_config.carrier_frequency_hz,
        beamforming_config.ap0_channel_phase_rad,
    )
    h1_static = complex_los_channel(
        beamforming_config.ap1_amplitude,
        beamforming_config.ap1_propagation_delay_s,
        waveform_config.carrier_frequency_hz,
        beamforming_config.ap1_channel_phase_rad,
    )
    delayed_ap1_pilot = apply_fractional_delay(
        pilot,
        beamforming_config.ap1_propagation_delay_s
        - beamforming_config.ap0_propagation_delay_s
        - residual_clock_offset_s + tx_time_correction_s,
        waveform_config.sample_rate_hz,
    )
    ap1_phase_rad = np.array(
        [oscillator.phase_at(float(time_s)) for time_s in absolute_time_s],
        dtype=np.float64,
    ) + feedback_config.channel_phase_rate_rad_per_s * absolute_time_s
    clean0 = np.asarray(h0_static * pilot, dtype=np.complex128)
    clean1 = np.asarray(
        h1_static * np.exp(1j * ap1_phase_rad) * delayed_ap1_pilot,
        dtype=np.complex128,
    )
    active_mask = np.ones(pilot.shape, dtype=np.bool_)
    received0 = add_awgn(clean0, feedback_config.snr_db, active_mask, rng)
    received1 = add_awgn(clean1, feedback_config.snr_db, active_mask, rng)
    raw_estimates = np.array(
        [
            estimate_channel_ls(pilot, received0.samples),
            estimate_channel_ls(pilot, received1.samples),
        ],
        dtype=np.complex128,
    )
    feedback_estimates = _quantize_channel_phase(
        raw_estimates, feedback_config.phase_quantization_bits
    )
    true_data_channels = np.array(
        [
            h0_static,
            h1_static
            * np.exp(
                1j
                * (
                    oscillator.phase_at(data_epoch_s)
                    + feedback_config.channel_phase_rate_rad_per_s * data_epoch_s
                )
            ),
        ],
        dtype=np.complex128,
    )
    phase_error = np.angle(
        feedback_estimates * np.conj(true_data_channels)
    ).astype(np.float64)
    return ChannelFeedbackResult(
        pilot_epoch_s=float(pilot_epoch_s),
        data_epoch_s=float(data_epoch_s),
        raw_channel_estimates=raw_estimates,
        feedback_channel_estimates=feedback_estimates,
        true_effective_channels_at_data=true_data_channels,
        phase_error_at_data_rad=phase_error,
        tx_time_correction_s=tx_time_correction_s,
        measured_arrival_difference_s=measured_arrival,
    )
