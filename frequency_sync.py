"""论文双音自混频硬件的软件等效频率估计与补偿。"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from channel import add_awgn
from config import FrequencySyncConfig
from models import FrequencyEstimate, FrequencyReferenceObservation


def simulate_frequency_reference(
    config: FrequencySyncConfig,
    rng: np.random.Generator,
) -> FrequencyReferenceObservation:
    """模拟 AP1 以偏差采样时钟接收 AP0 名义复基带参考。"""

    sample_count = round(config.observation_duration_s * config.sample_rate_hz)
    nominal_time_s = np.arange(sample_count, dtype=np.float64) / config.sample_rate_hz
    true_sample_time_s = nominal_time_s / (1.0 + config.sample_clock_offset_fraction)
    physical_frequency_hz = config.reference_frequency_hz + config.cfo_hz
    clean = config.amplitude * np.exp(
        1j
        * (
            2.0 * np.pi * physical_frequency_hz * true_sample_time_s
            + config.initial_phase_rad
        )
    )
    clean = np.asarray(clean, dtype=np.complex128)
    noisy = add_awgn(
        clean,
        snr_db=config.snr_db,
        active_mask=np.ones(clean.shape, dtype=np.bool_),
        rng=rng,
    )
    observed_offset_hz = (
        physical_frequency_hz / (1.0 + config.sample_clock_offset_fraction)
        - config.reference_frequency_hz
    )
    return FrequencyReferenceObservation(
        samples=noisy.samples,
        clean_samples=clean,
        nominal_time_s=nominal_time_s,
        true_observed_offset_hz=float(observed_offset_hz),
        measured_snr_db=noisy.measured_snr_db,
    )


def estimate_frequency_offset(
    samples: ArrayLike,
    config: FrequencySyncConfig,
) -> FrequencyEstimate:
    """对参考信号分段相干积累并拟合展开相位的线性斜率。"""

    iq = np.asarray(samples, dtype=np.complex128)
    if iq.ndim != 1 or iq.size == 0 or not np.all(np.isfinite(iq)):
        raise ValueError("samples 必须为非空一维有限复数组")
    segment_samples = round(config.segment_duration_s * config.sample_rate_hz)
    segment_count = iq.size // segment_samples
    if segment_count < 2:
        raise ValueError("频偏拟合至少需要两个完整分段")
    usable_count = segment_count * segment_samples
    nominal_time_s = np.arange(usable_count, dtype=np.float64) / config.sample_rate_hz
    derotated = iq[:usable_count] * np.exp(
        -1j * 2.0 * np.pi * config.reference_frequency_hz * nominal_time_s
    )
    coherent_sum = derotated.reshape(segment_count, segment_samples).sum(axis=1)
    segment_times_s = (
        np.arange(segment_count, dtype=np.float64) * segment_samples
        + 0.5 * (segment_samples - 1)
    ) / config.sample_rate_hz
    unwrapped_phase = np.unwrap(np.angle(coherent_sum))
    weights = np.abs(coherent_sum) ** 2
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("分段相干和能量必须大于 0")
    time_center = float(np.sum(weights * segment_times_s) / weight_sum)
    phase_center = float(np.sum(weights * unwrapped_phase) / weight_sum)
    centered_time = segment_times_s - time_center
    denominator = float(np.sum(weights * centered_time**2))
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("分段时间跨度不足以拟合频偏")
    slope_rad_per_s = float(
        np.sum(weights * centered_time * (unwrapped_phase - phase_center)) / denominator
    )
    intercept = phase_center - slope_rad_per_s * time_center
    fitted = intercept + slope_rad_per_s * segment_times_s
    residual_rms = float(np.sqrt(np.mean((unwrapped_phase - fitted) ** 2)))
    return FrequencyEstimate(
        frequency_offset_hz=slope_rad_per_s / (2.0 * np.pi),
        phase_intercept_rad=float(intercept),
        segment_times_s=segment_times_s,
        unwrapped_phase_rad=np.asarray(unwrapped_phase, dtype=np.float64),
        fitted_phase_rad=np.asarray(fitted, dtype=np.float64),
        residual_phase_rms_rad=residual_rms,
    )


def compensate_frequency(
    samples: ArrayLike,
    frequency_offset_hz: float,
    sample_rate_hz: float,
) -> NDArray[np.complex128]:
    """仅使用估计频偏旋转复基带 IQ，时间从当前数据块起点计。"""

    iq = np.asarray(samples, dtype=np.complex128)
    if iq.ndim != 1 or iq.size == 0 or not np.all(np.isfinite(iq)):
        raise ValueError("samples 必须为非空一维有限复数组")
    if not math.isfinite(frequency_offset_hz):
        raise ValueError("frequency_offset_hz 必须为有限数")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz 必须为正有限数")
    time_s = np.arange(iq.size, dtype=np.float64) / sample_rate_hz
    return np.asarray(
        iq * np.exp(-1j * 2.0 * np.pi * frequency_offset_hz * time_s),
        dtype=np.complex128,
    )


def update_frequency_tracker(previous_hz: float, measurement_hz: float, alpha: float) -> float:
    """用一阶指数平滑更新频偏状态。"""

    if not all(math.isfinite(value) for value in (previous_hz, measurement_hz, alpha)):
        raise ValueError("频率跟踪器输入必须为有限数")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha 必须位于 [0, 1) 内")
    return float(alpha * previous_hz + (1.0 - alpha) * measurement_hz)
