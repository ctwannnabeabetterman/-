"""生成、缓存并应用双音 QLS 分数时延周期偏差 LUT。"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from channel import apply_fractional_delay
from config import WaveformConfig
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from models import QLSCalibration, QLSValidation
from waveforms import generate_two_tone


_ALGORITHM_VERSION = "qls-lut-v1"


@overload
def wrap_fractional_sample(value: float) -> float: ...


@overload
def wrap_fractional_sample(value: ArrayLike) -> NDArray[np.float64]: ...


def wrap_fractional_sample(value: float | ArrayLike) -> float | NDArray[np.float64]:
    """把样点分数周期折回半开区间 ``[-0.5, 0.5)``。"""

    array = np.asarray(value, dtype=np.float64)
    wrapped = (array + 0.5) % 1.0 - 0.5
    if array.ndim == 0:
        return float(wrapped)
    return np.asarray(wrapped, dtype=np.float64)


def calibration_signature(config: WaveformConfig, grid_points: int) -> str:
    """计算覆盖波形、采样率、网格和算法版本的稳定缓存签名。"""

    if grid_points < 5:
        raise ValueError("grid_points 至少为 5")
    payload = {
        "algorithm_version": _ALGORITHM_VERSION,
        "grid_points": int(grid_points),
        "waveform": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _periodic_bias_interpolation(
    values: NDArray[np.float64],
    axis: NDArray[np.float64],
    bias: NDArray[np.float64],
) -> NDArray[np.float64]:
    """在周期为一个样点的 LUT 上进行线性插值。"""

    wrapped_values = np.asarray(wrap_fractional_sample(values), dtype=np.float64)
    extended_axis = np.concatenate((axis - 1.0, axis, axis + 1.0))
    extended_bias = np.concatenate((bias, bias, bias))
    return np.interp(wrapped_values, extended_axis, extended_bias)


def correct_qls_fraction(
    raw_fraction_samples: float | ArrayLike,
    calibration: QLSCalibration,
) -> float | NDArray[np.float64]:
    """仅依据 QLS 估计位置查表，返回校正后的分数样点。"""

    raw = np.asarray(raw_fraction_samples, dtype=np.float64)
    interpolated_bias = _periodic_bias_interpolation(
        raw,
        calibration.estimated_axis_samples,
        calibration.bias_samples,
    )
    corrected = np.asarray(wrap_fractional_sample(raw - interpolated_bias), dtype=np.float64)
    if raw.ndim == 0:
        return float(corrected)
    return corrected


def _estimate_fractional_grid(
    config: WaveformConfig,
    true_fraction_samples: NDArray[np.float64],
) -> NDArray[np.float64]:
    """经分数时延信道、匹配滤波和 QLS 估计一组真实分数位置。"""

    waveform = generate_two_tone(config)
    nominal_integer = 24
    tx_buffer = np.pad(waveform.samples, (0, 64))
    gate = DelaySearchGate(
        center_s=nominal_integer / config.sample_rate_hz,
        half_width_s=2.5 / config.sample_rate_hz,
    )
    raw_fraction = np.empty(true_fraction_samples.shape, dtype=np.float64)
    for index, fraction in enumerate(true_fraction_samples):
        received = apply_fractional_delay(
            tx_buffer,
            delay_s=(nominal_integer + float(fraction)) / config.sample_rate_hz,
            sample_rate_hz=config.sample_rate_hz,
        )
        correlation = fft_matched_filter(received, waveform.samples)
        estimate = estimate_delay(correlation, config.sample_rate_hz, gate)
        if not estimate.qls_valid:
            raise RuntimeError(f"分数时延网格 {index} 的 QLS 插值无效")
        relative = (
            estimate.integer_lag_samples
            - nominal_integer
            + estimate.fractional_offset_samples
        )
        raw_fraction[index] = float(wrap_fractional_sample(relative))
    return raw_fraction


def build_qls_lut(config: WaveformConfig, grid_points: int = 2001) -> QLSCalibration:
    """用当前无噪声双音波形扫描一个完整分数样点周期。"""

    signature = calibration_signature(config, grid_points)
    true_fraction = np.linspace(-0.5, 0.5, grid_points, endpoint=False, dtype=np.float64)
    raw_fraction = _estimate_fractional_grid(config, true_fraction)

    bias_by_scan = np.asarray(
        wrap_fractional_sample(raw_fraction - true_fraction), dtype=np.float64
    )
    order = np.argsort(raw_fraction)
    estimated_axis = raw_fraction[order]
    sorted_bias = bias_by_scan[order]
    unique_axis, inverse = np.unique(estimated_axis, return_inverse=True)
    if unique_axis.size != estimated_axis.size:
        summed_bias = np.zeros(unique_axis.shape, dtype=np.float64)
        counts = np.zeros(unique_axis.shape, dtype=np.int64)
        np.add.at(summed_bias, inverse, sorted_bias)
        np.add.at(counts, inverse, 1)
        estimated_axis = unique_axis
        sorted_bias = summed_bias / counts

    provisional = QLSCalibration(
        signature=signature,
        true_fraction_samples=true_fraction,
        raw_fraction_samples=raw_fraction,
        estimated_axis_samples=np.asarray(estimated_axis, dtype=np.float64),
        bias_samples=np.asarray(sorted_bias, dtype=np.float64),
        corrected_error_samples=np.zeros(true_fraction.shape, dtype=np.float64),
        grid_points=grid_points,
    )
    corrected = np.asarray(correct_qls_fraction(raw_fraction, provisional), dtype=np.float64)
    corrected_error = np.asarray(
        wrap_fractional_sample(corrected - true_fraction), dtype=np.float64
    )
    return QLSCalibration(
        signature=signature,
        true_fraction_samples=true_fraction,
        raw_fraction_samples=raw_fraction,
        estimated_axis_samples=provisional.estimated_axis_samples,
        bias_samples=provisional.bias_samples,
        corrected_error_samples=corrected_error,
        grid_points=grid_points,
    )


def validate_qls_lut(
    config: WaveformConfig,
    calibration: QLSCalibration,
    validation_points: int = 2000,
) -> QLSValidation:
    """在与 LUT 训练点错开的独立网格上运行完整 QLS/LUT 链路。"""

    if validation_points < 5:
        raise ValueError("validation_points 至少为 5")
    step = 1.0 / validation_points
    true_fraction = -0.5 + (np.arange(validation_points) + 0.5) * step
    true_fraction = np.asarray(true_fraction, dtype=np.float64)
    raw_fraction = _estimate_fractional_grid(config, true_fraction)
    corrected_fraction = np.asarray(
        correct_qls_fraction(raw_fraction, calibration), dtype=np.float64
    )
    raw_error = np.asarray(
        wrap_fractional_sample(raw_fraction - true_fraction), dtype=np.float64
    )
    corrected_error = np.asarray(
        wrap_fractional_sample(corrected_fraction - true_fraction), dtype=np.float64
    )
    return QLSValidation(
        true_fraction_samples=true_fraction,
        raw_fraction_samples=raw_fraction,
        corrected_fraction_samples=corrected_fraction,
        raw_error_samples=raw_error,
        corrected_error_samples=corrected_error,
    )


def _save_lut(calibration: QLSCalibration, config: WaveformConfig, cache_dir: Path) -> None:
    """把数值数组和可读元数据写入同名 NPZ/JSON 文件。"""

    cache_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"qls_lut_{calibration.signature}"
    np.savez_compressed(
        cache_dir / f"{base_name}.npz",
        true_fraction_samples=calibration.true_fraction_samples,
        raw_fraction_samples=calibration.raw_fraction_samples,
        estimated_axis_samples=calibration.estimated_axis_samples,
        bias_samples=calibration.bias_samples,
        corrected_error_samples=calibration.corrected_error_samples,
    )
    metadata = {
        "algorithm_version": _ALGORITHM_VERSION,
        "signature": calibration.signature,
        "grid_points": calibration.grid_points,
        "waveform": asdict(config),
    }
    (cache_dir / f"{base_name}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_or_build_lut(
    config: WaveformConfig,
    cache_dir: str | Path,
    grid_points: int = 2001,
) -> QLSCalibration:
    """加载签名完全匹配的 LUT；不存在或无效时自动重建。"""

    directory = Path(cache_dir)
    signature = calibration_signature(config, grid_points)
    base_name = f"qls_lut_{signature}"
    npz_path = directory / f"{base_name}.npz"
    json_path = directory / f"{base_name}.json"
    if npz_path.exists() and json_path.exists():
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            if metadata.get("signature") != signature:
                raise ValueError("LUT 元数据签名不匹配")
            with np.load(npz_path, allow_pickle=False) as arrays:
                return QLSCalibration(
                    signature=signature,
                    true_fraction_samples=np.asarray(
                        arrays["true_fraction_samples"], dtype=np.float64
                    ),
                    raw_fraction_samples=np.asarray(
                        arrays["raw_fraction_samples"], dtype=np.float64
                    ),
                    estimated_axis_samples=np.asarray(
                        arrays["estimated_axis_samples"], dtype=np.float64
                    ),
                    bias_samples=np.asarray(arrays["bias_samples"], dtype=np.float64),
                    corrected_error_samples=np.asarray(
                        arrays["corrected_error_samples"], dtype=np.float64
                    ),
                    grid_points=int(metadata["grid_points"]),
                )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass

    calibration = build_qls_lut(config, grid_points)
    _save_lut(calibration, config, directory)
    return calibration
