"""阶段 1 的分数时延、AWGN 和静态单径信道。"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import next_fast_len

from config import ChannelConfig
from models import NoisySignal, PropagationResult


def _complex_vector(samples: ArrayLike) -> NDArray[np.complex128]:
    """将输入转换为一维 complex128 复基带数组。"""

    vector = np.asarray(samples, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError("复基带样点必须是一维数组")
    if vector.size == 0:
        raise ValueError("复基带样点不能为空")
    if not np.all(np.isfinite(vector)):
        raise ValueError("复基带样点必须为有限数")
    return vector


def apply_fractional_delay(
    samples: ArrayLike,
    delay_s: float,
    sample_rate_hz: float,
    *,
    guard_samples: int = 64,
) -> NDArray[np.complex128]:
    """以频域相移实现线性分数时延，并返回与输入等长的 IQ。

    ``delay_s`` 为正时信号向更大的样点下标移动。内部在两端补零，
    防止 FFT 循环回绕；调用者应在输入尾部预留空间以保存完整延时波形。
    """

    vector = _complex_vector(samples)
    if not math.isfinite(delay_s):
        raise ValueError("delay_s 必须为有限数")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz 必须为正有限数")
    if guard_samples < 16:
        raise ValueError("guard_samples 至少为 16")

    delay_samples = delay_s * sample_rate_hz
    if delay_samples == 0.0:
        return vector.copy()

    nearest_integer = int(round(delay_samples))
    if abs(delay_samples - nearest_integer) <= 16.0 * np.finfo(np.float64).eps * max(
        1.0, abs(delay_samples)
    ):
        shifted = np.zeros_like(vector)
        if 0 <= nearest_integer < vector.size:
            shifted[nearest_integer:] = vector[: vector.size - nearest_integer]
        elif -vector.size < nearest_integer < 0:
            shifted[:nearest_integer] = vector[-nearest_integer:]
        return shifted

    padding = max(guard_samples, int(math.ceil(abs(delay_samples))) + guard_samples)
    padded = np.pad(vector, (padding, padding))
    transform_length = next_fast_len(padded.size)
    frequency_cycles_per_sample = np.fft.fftfreq(transform_length)
    phase_ramp = np.exp(-1j * 2.0 * np.pi * frequency_cycles_per_sample * delay_samples)
    shifted_padded = np.fft.ifft(np.fft.fft(padded, transform_length) * phase_ramp)
    shifted = shifted_padded[padding : padding + vector.size]
    return np.asarray(shifted, dtype=np.complex128)


def add_awgn(
    samples: ArrayLike,
    snr_db: float,
    active_mask: ArrayLike,
    rng: np.random.Generator,
) -> NoisySignal:
    """按活动区每样点功率 SNR 向复基带信号加入圆对称 AWGN。"""

    vector = _complex_vector(samples)
    mask = np.asarray(active_mask, dtype=np.bool_)
    if mask.shape != vector.shape or not np.any(mask):
        raise ValueError("active_mask 必须与样点同形且至少包含一个活动样点")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db 必须为有限数")

    signal_power = float(np.mean(np.abs(vector[mask]) ** 2))
    if signal_power <= 0.0:
        raise ValueError("活动区信号功率必须大于 0")
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_scale = math.sqrt(target_noise_power / 2.0)
    noise = noise_scale * (
        rng.standard_normal(vector.size) + 1j * rng.standard_normal(vector.size)
    )
    noise = np.asarray(noise, dtype=np.complex128)
    measured_noise_power = float(np.mean(np.abs(noise[mask]) ** 2))
    measured_snr_db = float(10.0 * np.log10(signal_power / measured_noise_power))

    return NoisySignal(
        samples=np.asarray(vector + noise, dtype=np.complex128),
        noise_samples=noise,
        signal_power=signal_power,
        noise_power=measured_noise_power,
        target_snr_db=snr_db,
        measured_snr_db=measured_snr_db,
    )


def propagate_static_link(
    tx_samples: ArrayLike,
    sample_rate_hz: float,
    config: ChannelConfig,
    rng: np.random.Generator,
) -> PropagationResult:
    """通过分数传播时延、复增益和可选 AWGN 生成接收 IQ。"""

    tx = _complex_vector(tx_samples)
    delay_samples = config.propagation_delay_s * sample_rate_hz
    trailing_samples = int(math.ceil(delay_samples)) + config.fractional_delay_guard_samples
    tx_observation = np.pad(tx, (0, trailing_samples))
    delayed = apply_fractional_delay(
        tx_observation,
        delay_s=config.propagation_delay_s,
        sample_rate_hz=sample_rate_hz,
        guard_samples=config.fractional_delay_guard_samples,
    )
    complex_gain = config.amplitude * np.exp(1j * config.phase_rad)
    clean = np.asarray(complex_gain * delayed, dtype=np.complex128)

    if config.snr_db is None:
        noise = np.zeros_like(clean)
        return PropagationResult(
            samples=clean.copy(),
            clean_samples=clean,
            noise_samples=noise,
            sample_rate_hz=sample_rate_hz,
            measured_snr_db=None,
        )

    tx_magnitude = np.abs(tx)
    support_threshold = np.sqrt(np.finfo(np.float64).eps) * float(np.max(tx_magnitude))
    support_indices = np.flatnonzero(tx_magnitude > support_threshold)
    if support_indices.size == 0:
        raise ValueError("非零信号才能按指定 SNR 加入噪声")
    active_start = max(0, int(math.floor(support_indices[0] + delay_samples)))
    active_stop = min(
        clean.size,
        int(math.ceil(support_indices[-1] + delay_samples)) + 1,
    )
    active_mask = np.zeros(clean.shape, dtype=np.bool_)
    active_mask[active_start:active_stop] = True
    noisy = add_awgn(clean, config.snr_db, active_mask, rng)
    return PropagationResult(
        samples=noisy.samples,
        clean_samples=clean,
        noise_samples=noisy.noise_samples,
        sample_rate_hz=sample_rate_hz,
        measured_snr_db=noisy.measured_snr_db,
    )
