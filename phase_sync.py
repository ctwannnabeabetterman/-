"""RX 导频复信道估计和两 AP 发射相位补偿。"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
