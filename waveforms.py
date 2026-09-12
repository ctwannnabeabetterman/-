"""生成带有限升降沿的脉冲双音复基带波形。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config import WaveformConfig
from models import Waveform


def raised_cosine_envelope(time_s: NDArray[np.float64], config: WaveformConfig) -> NDArray[np.float64]:
    """在给定秒制时间轴上生成分段升余弦脉冲包络。"""

    envelope = np.zeros(time_s.shape, dtype=np.float64)
    inside = (time_s >= 0.0) & (time_s < config.pulse_duration_s)

    if config.rise_fall_s == 0.0:
        envelope[inside] = 1.0
        return envelope

    rising = inside & (time_s < config.rise_fall_s)
    plateau = inside & (time_s >= config.rise_fall_s) & (
        time_s < config.pulse_duration_s - config.rise_fall_s
    )
    falling = inside & (time_s >= config.pulse_duration_s - config.rise_fall_s)

    envelope[rising] = 0.5 * (
        1.0 - np.cos(np.pi * time_s[rising] / config.rise_fall_s)
    )
    envelope[plateau] = 1.0
    envelope[falling] = 0.5 * (
        1.0
        - np.cos(
            np.pi
            * (config.pulse_duration_s - time_s[falling])
            / config.rise_fall_s
        )
    )
    return envelope


def generate_two_tone(config: WaveformConfig) -> Waveform:
    """生成位于 ``±tone_separation_hz/2`` 的脉冲双音波形。"""

    sample_count = round(config.pulse_duration_s * config.sample_rate_hz)
    time_s = np.arange(sample_count, dtype=np.float64) / config.sample_rate_hz
    envelope = raised_cosine_envelope(time_s, config)
    half_separation_hz = config.tone_separation_hz / 2.0
    samples = envelope * (
        np.exp(-1j * 2.0 * np.pi * half_separation_hz * time_s)
        + np.exp(1j * 2.0 * np.pi * half_separation_hz * time_s)
    )
    samples = np.asarray(samples, dtype=np.complex128)

    active_mask = envelope > 0.0
    active_power = float(np.mean(np.abs(samples[active_mask]) ** 2))
    if not np.isfinite(active_power) or active_power <= 0.0:
        raise ValueError("波形活动区功率必须大于 0")
    samples /= np.sqrt(active_power)

    average_active_power = float(np.mean(np.abs(samples[active_mask]) ** 2))
    energy = float(np.sum(np.abs(samples) ** 2) / config.sample_rate_hz)
    return Waveform(
        samples=samples,
        time_s=time_s,
        envelope=envelope,
        active_mask=np.asarray(active_mask, dtype=np.bool_),
        sample_rate_hz=config.sample_rate_hz,
        average_active_power=average_active_power,
        energy=energy,
    )
