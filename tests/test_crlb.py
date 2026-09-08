"""阶段 9：双音时延 CRLB 的公式、单位和 SNR 映射测试。"""

from __future__ import annotations

import unittest

import numpy as np

from crlb import (
    crlb_for_waveform,
    delay_crlb_std,
    mean_squared_angular_bandwidth,
    noise_psd_from_active_sample_snr,
    two_tone_mean_squared_angular_bandwidth,
)
from config import WaveformConfig
from waveforms import generate_two_tone


class DelayCRLBTests(unittest.TestCase):
    """验证论文公式和复基带离散噪声的量纲关系。"""

    def test_two_tone_mean_squared_bandwidth_matches_pi_b_squared(self) -> None:
        separation_hz = 40e6
        self.assertEqual(
            two_tone_mean_squared_angular_bandwidth(separation_hz),
            (np.pi * separation_hz) ** 2,
        )

    def test_delay_standard_deviation_matches_paper_formula(self) -> None:
        energy = 2e-6
        n0 = 5e-10
        zeta_squared = (np.pi * 40e6) ** 2
        expected = np.sqrt(n0 / (2.0 * zeta_squared * energy))

        self.assertAlmostEqual(
            delay_crlb_std(energy, n0, zeta_squared), expected, places=18
        )

    def test_active_sample_snr_maps_noise_power_to_psd(self) -> None:
        n0 = noise_psd_from_active_sample_snr(
            active_signal_power=1.0,
            active_sample_snr_db=10.0,
            noise_bandwidth_hz=200e6,
        )

        self.assertAlmostEqual(n0, 0.1 / 200e6, places=20)

    def test_numeric_bandwidth_of_bin_centered_two_tone_matches_analytic_value(self) -> None:
        sample_rate_hz = 200e6
        count = 1000
        time_s = np.arange(count) / sample_rate_hz
        samples = (
            np.exp(-1j * 2.0 * np.pi * 20e6 * time_s)
            + np.exp(1j * 2.0 * np.pi * 20e6 * time_s)
        )

        numeric = mean_squared_angular_bandwidth(samples, sample_rate_hz)

        self.assertAlmostEqual(numeric / (np.pi * 40e6) ** 2, 1.0, places=12)

    def test_crlb_decreases_by_ten_for_twenty_db_more_snr(self) -> None:
        waveform = generate_two_tone(WaveformConfig(pulse_duration_s=2e-6))
        low = crlb_for_waveform(waveform, tone_separation_hz=40e6, snr_db=10.0)
        high = crlb_for_waveform(waveform, tone_separation_hz=40e6, snr_db=30.0)

        self.assertAlmostEqual(low.std_s / high.std_s, 10.0, places=12)


if __name__ == "__main__":
    unittest.main()
