"""阶段 3：QLS 周期偏差 LUT 的生成、校正和缓存测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from channel import apply_fractional_delay
from config import WaveformConfig
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from lut_calibration import (
    build_qls_lut,
    calibration_signature,
    correct_qls_fraction,
    load_or_build_lut,
    wrap_fractional_sample,
)
from waveforms import generate_two_tone


def raw_fraction_estimate(config: WaveformConfig, true_fraction: float) -> float:
    """通过正式估计路径生成独立测试点的原始分数估计。"""

    waveform = generate_two_tone(config)
    nominal_integer = 24
    tx_buffer = np.pad(waveform.samples, (0, 64))
    received = apply_fractional_delay(
        tx_buffer,
        delay_s=(nominal_integer + true_fraction) / config.sample_rate_hz,
        sample_rate_hz=config.sample_rate_hz,
    )
    correlation = fft_matched_filter(received, waveform.samples)
    gate = DelaySearchGate(
        center_s=nominal_integer / config.sample_rate_hz,
        half_width_s=2.5 / config.sample_rate_hz,
    )
    estimate = estimate_delay(correlation, config.sample_rate_hz, gate)
    relative = estimate.integer_lag_samples - nominal_integer + estimate.fractional_offset_samples
    return float(wrap_fractional_sample(relative))


class QLSCalibrationTests(unittest.TestCase):
    """验证 LUT 不泄漏运行时真值并能显著降低系统偏差。"""

    def test_wrap_uses_half_open_fractional_interval(self) -> None:
        values = np.array([-1.5, -0.5, 0.5, 1.5, 0.49])
        wrapped = wrap_fractional_sample(values)
        np.testing.assert_allclose(wrapped, [-0.5, -0.5, -0.5, -0.5, 0.49])

    def test_lut_correction_reduces_error_on_independent_fractional_points(self) -> None:
        config = WaveformConfig(pulse_duration_s=1e-6)
        calibration = build_qls_lut(config, grid_points=401)
        true_fractions = np.linspace(-0.495, 0.495, 81)
        raw = np.array([raw_fraction_estimate(config, value) for value in true_fractions])
        corrected = correct_qls_fraction(raw, calibration)
        raw_error = wrap_fractional_sample(raw - true_fractions)
        corrected_error = wrap_fractional_sample(corrected - true_fractions)

        self.assertLess(
            float(np.sqrt(np.mean(corrected_error**2))),
            0.05 * float(np.sqrt(np.mean(raw_error**2))),
        )
        self.assertLess(float(np.max(np.abs(corrected_error))), 5e-4)

    def test_signature_changes_when_waveform_parameters_change(self) -> None:
        base = WaveformConfig(pulse_duration_s=1e-6)
        changed = WaveformConfig(pulse_duration_s=1e-6, tone_separation_hz=30e6)

        self.assertNotEqual(
            calibration_signature(base, grid_points=201),
            calibration_signature(changed, grid_points=201),
        )

    def test_cache_round_trip_and_parameter_change_create_distinct_files(self) -> None:
        base = WaveformConfig(pulse_duration_s=1e-6)
        changed = WaveformConfig(pulse_duration_s=1e-6, rise_fall_s=25e-9)
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project_root) as temporary_directory:
            cache_dir = Path(temporary_directory)
            first = load_or_build_lut(base, cache_dir, grid_points=101)
            loaded = load_or_build_lut(base, cache_dir, grid_points=101)
            different = load_or_build_lut(changed, cache_dir, grid_points=101)

            self.assertEqual(first.signature, loaded.signature)
            self.assertNotEqual(first.signature, different.signature)
            self.assertEqual(len(list(cache_dir.glob("qls_lut_*.npz"))), 2)
            self.assertEqual(len(list(cache_dir.glob("qls_lut_*.json"))), 2)
            np.testing.assert_array_equal(first.bias_samples, loaded.bias_samples)


if __name__ == "__main__":
    unittest.main()
