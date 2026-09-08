"""阶段 2：FFT 线性匹配滤波和 QLS 亚采样估计测试。"""

from __future__ import annotations

import unittest

import numpy as np

from channel import apply_fractional_delay
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from models import CorrelationResult
from config import WaveformConfig
from waveforms import generate_two_tone


class FFTMatchedFilterTests(unittest.TestCase):
    """验证循环 FFT 输出到物理线性 lag 的转换。"""

    def test_recovers_positive_integer_lag(self) -> None:
        rng = np.random.default_rng(31)
        template = rng.normal(size=64) + 1j * rng.normal(size=64)
        delay_samples = 37
        received = np.pad(template, (delay_samples, 23))

        correlation = fft_matched_filter(received, template)

        peak = int(np.argmax(correlation.magnitude))
        self.assertEqual(int(correlation.lags_samples[peak]), delay_samples)

    def test_recovers_negative_integer_lag(self) -> None:
        rng = np.random.default_rng(32)
        template = rng.normal(size=80) + 1j * rng.normal(size=80)
        removed_samples = 9
        received = template[removed_samples:]

        correlation = fft_matched_filter(received, template)

        peak = int(np.argmax(correlation.magnitude))
        self.assertEqual(int(correlation.lags_samples[peak]), -removed_samples)


class QLSTests(unittest.TestCase):
    """验证 QLS 公式、搜索门和保护状态。"""

    def test_qls_finds_exact_peak_of_three_point_parabola(self) -> None:
        lags = np.arange(-2, 3, dtype=np.int64)
        true_fraction = 0.25
        magnitude = 4.0 - (lags.astype(np.float64) - true_fraction) ** 2
        correlation = CorrelationResult(
            values=magnitude.astype(np.complex128),
            magnitude=magnitude,
            lags_samples=lags,
            fft_length=8,
        )

        estimate = estimate_delay(correlation, sample_rate_hz=200e6)

        self.assertEqual(estimate.integer_lag_samples, 0)
        self.assertAlmostEqual(estimate.fractional_offset_samples, true_fraction, places=14)
        self.assertAlmostEqual(estimate.delay_s, true_fraction / 200e6, places=18)
        self.assertTrue(estimate.qls_valid)
        self.assertFalse(estimate.boundary_hit)

    def test_search_gate_rejects_stronger_echo_outside_expected_window(self) -> None:
        rng = np.random.default_rng(33)
        template = rng.normal(size=48) + 1j * rng.normal(size=48)
        received = np.zeros(180, dtype=np.complex128)
        received[30:78] += template
        received[100:148] += 1.5 * template
        correlation = fft_matched_filter(received, template)
        gate = DelaySearchGate(center_s=30 / 200e6, half_width_s=4 / 200e6)

        estimate = estimate_delay(correlation, sample_rate_hz=200e6, gate=gate)

        self.assertEqual(estimate.integer_lag_samples, 30)
        self.assertFalse(estimate.boundary_hit)

    def test_gate_boundary_disables_qls_instead_of_reading_outside_gate(self) -> None:
        lags = np.arange(5, dtype=np.int64)
        magnitude = np.array([0.0, 1.0, 2.0, 4.0, 3.9], dtype=np.float64)
        correlation = CorrelationResult(
            values=magnitude.astype(np.complex128),
            magnitude=magnitude,
            lags_samples=lags,
            fft_length=8,
        )
        gate = DelaySearchGate(center_s=2 / 10.0, half_width_s=1 / 10.0)

        estimate = estimate_delay(correlation, sample_rate_hz=10.0, gate=gate)

        self.assertEqual(estimate.integer_lag_samples, 3)
        self.assertTrue(estimate.boundary_hit)
        self.assertFalse(estimate.qls_valid)
        self.assertEqual(estimate.fractional_offset_samples, 0.0)

    def test_qls_improves_noiseless_two_tone_fractional_delay(self) -> None:
        config = WaveformConfig(pulse_duration_s=2e-6)
        waveform = generate_two_tone(config)
        integer_delay = 37
        fractional_delay = 0.25
        tx_buffer = np.pad(waveform.samples, (0, 80))
        received = apply_fractional_delay(
            tx_buffer,
            delay_s=(integer_delay + fractional_delay) / config.sample_rate_hz,
            sample_rate_hz=config.sample_rate_hz,
        )
        correlation = fft_matched_filter(received, waveform.samples)
        gate = DelaySearchGate(
            center_s=(integer_delay + fractional_delay) / config.sample_rate_hz,
            half_width_s=1.5 / config.sample_rate_hz,
        )

        estimate = estimate_delay(correlation, config.sample_rate_hz, gate)
        integer_error = abs(estimate.integer_lag_samples - integer_delay - fractional_delay)
        qls_error = abs(
            estimate.integer_lag_samples
            + estimate.fractional_offset_samples
            - integer_delay
            - fractional_delay
        )

        self.assertTrue(estimate.qls_valid)
        self.assertLess(qls_error, integer_error)


if __name__ == "__main__":
    unittest.main()
