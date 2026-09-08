"""阶段 7：RX 导频信道估计和发射相位权重测试。"""

from __future__ import annotations

import unittest

import numpy as np

from channel import add_awgn
from phase_sync import complex_los_channel, compute_phase_weights, estimate_channel_ls


class ChannelPhaseEstimationTests(unittest.TestCase):
    """验证复信道建模和 LS 导频估计。"""

    def test_noiseless_ls_estimate_equals_complex_channel(self) -> None:
        rng = np.random.default_rng(71)
        pilot = rng.choice([1.0, -1.0], size=256).astype(np.complex128)
        channel = 0.65 * np.exp(1j * 1.1)
        received = channel * pilot

        estimate = estimate_channel_ls(pilot, received)

        self.assertAlmostEqual(estimate.real, channel.real, places=14)
        self.assertAlmostEqual(estimate.imag, channel.imag, places=14)

    def test_high_snr_pilot_estimates_phase_within_one_degree(self) -> None:
        rng = np.random.default_rng(72)
        pilot = np.exp(1j * np.pi / 2.0 * np.arange(2048)).astype(np.complex128)
        channel = 0.8 * np.exp(-1j * 0.9)
        clean = channel * pilot
        noisy = add_awgn(clean, 30.0, np.ones(clean.shape, dtype=np.bool_), rng)

        estimate = estimate_channel_ls(pilot, noisy.samples)
        phase_error = float(np.angle(estimate * np.conj(channel)))

        self.assertLess(abs(np.rad2deg(phase_error)), 1.0)

    def test_los_channel_phase_is_periodic_over_one_wavelength(self) -> None:
        carrier_hz = 5.8e9
        wavelength_delay_s = 1.0 / carrier_hz
        first = complex_los_channel(1.0, 123.4e-9, carrier_hz, 0.2)
        second = complex_los_channel(
            1.0,
            123.4e-9 + wavelength_delay_s,
            carrier_hz,
            0.2,
        )

        self.assertLess(abs(first - second), 1e-12)


class PhaseWeightTests(unittest.TestCase):
    """验证相位共轭权重和两种功率归一化。"""

    def test_per_ap_weights_align_effective_channel_phases(self) -> None:
        channels = np.array(
            [np.exp(1j * 0.4), 0.7 * np.exp(-1j * 1.2)], dtype=np.complex128
        )

        weights = compute_phase_weights(channels, normalization="per_ap_fixed")

        np.testing.assert_allclose(np.angle(channels * weights), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.abs(weights), 1.0, atol=1e-14)

    def test_total_fixed_weights_have_unit_total_power(self) -> None:
        channels = np.array(
            [np.exp(1j * 0.4), 0.7 * np.exp(-1j * 1.2)], dtype=np.complex128
        )

        weights = compute_phase_weights(channels, normalization="total_fixed")

        self.assertAlmostEqual(float(np.sum(np.abs(weights) ** 2)), 1.0, places=14)
        np.testing.assert_allclose(np.angle(channels * weights), 0.0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
