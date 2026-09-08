"""双音时延估计 CRLB 及复基带 SNR/噪声 PSD 映射。"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from models import CRLBResult, Waveform


def two_tone_mean_squared_angular_bandwidth(tone_separation_hz: float) -> float:
    """返回等幅 ``±B/2`` 双音的均方角带宽 ``(pi B)^2``。"""

    if not math.isfinite(tone_separation_hz) or tone_separation_hz <= 0.0:
        raise ValueError("tone_separation_hz 必须为正有限数")
    return float((np.pi * tone_separation_hz) ** 2)


def mean_squared_angular_bandwidth(samples: ArrayLike, sample_rate_hz: float) -> float:
    """由离散频谱计算 ``sum((2πf)^2|S|^2)/sum(|S|^2)``。"""

    vector = np.asarray(samples, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("samples 必须为非空有限一维数组")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz 必须为正有限数")
    spectrum = np.fft.fft(vector)
    spectral_energy = np.abs(spectrum) ** 2
    total = float(np.sum(spectral_energy))
    if total <= np.finfo(np.float64).tiny:
        raise ValueError("信号能量必须大于 0")
    frequency_hz = np.fft.fftfreq(vector.size, d=1.0 / sample_rate_hz)
    return float(np.sum((2.0 * np.pi * frequency_hz) ** 2 * spectral_energy) / total)


def noise_psd_from_active_sample_snr(
    active_signal_power: float,
    active_sample_snr_db: float,
    noise_bandwidth_hz: float,
) -> float:
    """把活动区每样点 SNR 映射为复基带噪声 PSD ``N0=Pn/Bn``。"""

    values = (active_signal_power, active_sample_snr_db, noise_bandwidth_hz)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("SNR 映射参数必须为有限数")
    if active_signal_power <= 0.0 or noise_bandwidth_hz <= 0.0:
        raise ValueError("信号功率和噪声带宽必须大于 0")
    noise_power = active_signal_power / (10.0 ** (active_sample_snr_db / 10.0))
    return float(noise_power / noise_bandwidth_hz)


def delay_crlb_std(
    signal_energy: float,
    noise_psd: float,
    mean_squared_angular_bandwidth_rad2_s2: float,
) -> float:
    """按 ``var(tau)>=N0/(2*zeta^2*Es)`` 返回标准差下界（秒）。"""

    values = (signal_energy, noise_psd, mean_squared_angular_bandwidth_rad2_s2)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Es、N0 和均方角带宽必须为正有限数")
    variance = noise_psd / (
        2.0 * mean_squared_angular_bandwidth_rad2_s2 * signal_energy
    )
    return float(math.sqrt(variance))


def crlb_for_waveform(
    waveform: Waveform,
    tone_separation_hz: float,
    snr_db: float,
    noise_bandwidth_hz: float | None = None,
) -> CRLBResult:
    """用活动区每样点 SNR 计算一个波形的双音解析 CRLB。"""

    bandwidth_hz = waveform.sample_rate_hz if noise_bandwidth_hz is None else noise_bandwidth_hz
    n0 = noise_psd_from_active_sample_snr(
        waveform.average_active_power,
        snr_db,
        bandwidth_hz,
    )
    zeta_squared = two_tone_mean_squared_angular_bandwidth(tone_separation_hz)
    std_s = delay_crlb_std(waveform.energy, n0, zeta_squared)
    return CRLBResult(
        std_s=std_s,
        variance_s2=std_s**2,
        signal_energy=waveform.energy,
        noise_psd=n0,
        noise_bandwidth_hz=float(bandwidth_hz),
        active_sample_snr_db=float(snr_db),
        pulse_energy_snr_linear=float(waveform.energy / n0),
        mean_squared_angular_bandwidth_rad2_s2=zeta_squared,
    )
