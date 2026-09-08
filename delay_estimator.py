"""FFT 线性匹配滤波和三点 QLS 亚采样时延估计。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import next_fast_len

from models import CorrelationResult, DelayEstimate


@dataclass(frozen=True)
class DelaySearchGate:
    """以秒表示的粗时延搜索门，不包含任何隐藏真值。"""

    center_s: float
    half_width_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.center_s):
            raise ValueError("center_s 必须为有限数")
        if not math.isfinite(self.half_width_s) or self.half_width_s < 0.0:
            raise ValueError("half_width_s 必须为非负有限数")


def _complex_vector(samples: ArrayLike, name: str) -> np.ndarray:
    """将输入检查并转换为一维 complex128 数组。"""

    vector = np.asarray(samples, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} 必须为非空一维数组")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须只包含有限数")
    return vector


def fft_matched_filter(received: ArrayLike, template: ArrayLike) -> CorrelationResult:
    """用零填充 FFT 计算接收 IQ 与模板的完整线性互相关。"""

    rx = _complex_vector(received, "received")
    tx = _complex_vector(template, "template")
    linear_length = rx.size + tx.size - 1
    fft_length = next_fast_len(linear_length)
    circular = np.fft.ifft(
        np.fft.fft(rx, fft_length) * np.conj(np.fft.fft(tx, fft_length))
    )
    values = np.concatenate((circular[-(tx.size - 1) :], circular[: rx.size]))
    lags = np.arange(-(tx.size - 1), rx.size, dtype=np.int64)
    return CorrelationResult(
        values=np.asarray(values, dtype=np.complex128),
        magnitude=np.asarray(np.abs(values), dtype=np.float64),
        lags_samples=lags,
        fft_length=fft_length,
    )


def estimate_delay(
    correlation: CorrelationResult,
    sample_rate_hz: float,
    gate: DelaySearchGate | None = None,
) -> DelayEstimate:
    """在可选粗搜索门内寻找峰值并进行三点 QLS 插值。"""

    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz 必须为正有限数")
    magnitude = np.asarray(correlation.magnitude, dtype=np.float64)
    lags = np.asarray(correlation.lags_samples, dtype=np.int64)
    if magnitude.ndim != 1 or magnitude.size == 0 or magnitude.shape != lags.shape:
        raise ValueError("相关幅度与 lag 必须为同形非空一维数组")
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("相关幅度必须为有限数")

    if gate is None:
        candidate_indices = np.arange(magnitude.size, dtype=np.int64)
    else:
        delay_axis_s = lags.astype(np.float64) / sample_rate_hz
        lower = gate.center_s - gate.half_width_s
        upper = gate.center_s + gate.half_width_s
        candidate_indices = np.flatnonzero((delay_axis_s >= lower) & (delay_axis_s <= upper))
    if candidate_indices.size == 0:
        raise ValueError("搜索门内没有相关 lag 样点")

    peak_position_in_gate = int(np.argmax(magnitude[candidate_indices]))
    peak_index = int(candidate_indices[peak_position_in_gate])
    boundary_hit = (
        peak_index == 0
        or peak_index == magnitude.size - 1
        or peak_position_in_gate == 0
        or peak_position_in_gate == candidate_indices.size - 1
    )

    denominator = 0.0
    raw_fraction = 0.0
    fraction = 0.0
    qls_valid = False
    clipped = False
    if not boundary_hit:
        left = float(magnitude[peak_index - 1])
        center = float(magnitude[peak_index])
        right = float(magnitude[peak_index + 1])
        denominator = left - 2.0 * center + right
        local_scale = max(abs(left), abs(center), abs(right), np.finfo(np.float64).tiny)
        curvature_threshold = 64.0 * np.finfo(np.float64).eps * local_scale
        if denominator < -curvature_threshold:
            raw_fraction = 0.5 * (left - right) / denominator
            fraction = float(np.clip(raw_fraction, -0.5, 0.5))
            clipped = fraction != raw_fraction
            qls_valid = True

    integer_lag = int(lags[peak_index])
    delay_s = (integer_lag + fraction) / sample_rate_hz
    return DelayEstimate(
        delay_s=float(delay_s),
        integer_lag_samples=integer_lag,
        fractional_offset_samples=fraction,
        raw_fractional_offset_samples=raw_fraction,
        peak_array_index=peak_index,
        peak_magnitude=float(magnitude[peak_index]),
        qls_denominator=denominator,
        qls_valid=qls_valid,
        boundary_hit=boundary_hit,
        fraction_clipped=clipped,
    )
