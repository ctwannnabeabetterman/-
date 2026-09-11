"""阶段 1：双音波形、分数时延和 AWGN 信道的行为测试。"""

from __future__ import annotations

import unittest

import numpy as np

from channel import add_awgn, apply_fractional_delay, propagate_static_link
from config import ChannelConfig, WaveformConfig
from waveforms import generate_two_tone, ideal_two_tone_line_spectrum


class WaveformConfigTests(unittest.TestCase):
    """验证集中配置和物理参数约束。"""

    def test_rejects_tone_separation_at_or_above_sample_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "tone_separation_hz"):
            WaveformConfig(sample_rate_hz=200e6, tone_separation_hz=200e6)


class TwoToneWaveformTests(unittest.TestCase):
    """验证脉冲双音波形的时域、频域和归一化。"""

    def test_generates_normalized_two_tone_with_raised_cosine_edges(self) -> None:
        cfg = WaveformConfig()

        waveform = generate_two_tone(cfg)

        self.assertEqual(waveform.samples.dtype, np.complex128)
        self.assertEqual(waveform.samples.size, round(cfg.pulse_duration_s * cfg.sample_rate_hz))
        self.assertAlmostEqual(waveform.time_s[1] - waveform.time_s[0], 1.0 / cfg.sample_rate_hz)
        self.assertAlmostEqual(float(waveform.envelope[0]), 0.0, places=15)
        self.assertLess(float(waveform.envelope[-1]), 0.03)
        self.assertAlmostEqual(waveform.average_active_power, 1.0, places=12)
        expected_energy = float(np.sum(np.abs(waveform.samples) ** 2) / cfg.sample_rate_hz)
        self.assertAlmostEqual(waveform.energy, expected_energy, places=18)

    def test_places_spectral_peaks_at_half_tone_separation(self) -> None:
        cfg = WaveformConfig()
        waveform = generate_two_tone(cfg)

        spectrum = np.fft.fftshift(np.fft.fft(waveform.samples))
        frequencies_hz = np.fft.fftshift(
            np.fft.fftfreq(waveform.samples.size, d=1.0 / cfg.sample_rate_hz)
        )
        strongest = np.argpartition(np.abs(spectrum), -2)[-2:]
        peak_frequencies_hz = np.sort(frequencies_hz[strongest])

        bin_width_hz = cfg.sample_rate_hz / waveform.samples.size
        np.testing.assert_allclose(
            peak_frequencies_hz,
            [-cfg.tone_separation_hz / 2.0, cfg.tone_separation_hz / 2.0],
            atol=bin_width_hz,
        )

    def test_ideal_line_spectrum_has_exactly_two_equal_tones(self) -> None:
        cfg = WaveformConfig(sample_rate_hz=200e6, tone_separation_hz=40e6)

        frequencies_hz, normalized_magnitudes = ideal_two_tone_line_spectrum(cfg)

        np.testing.assert_array_equal(frequencies_hz, np.array([-20e6, 20e6]))
        np.testing.assert_array_equal(normalized_magnitudes, np.ones(2))
        self.assertEqual(float(np.diff(frequencies_hz)[0]), cfg.tone_separation_hz)


class FractionalDelayTests(unittest.TestCase):
    """验证线性、非整数采样时延。"""

    def test_zero_delay_is_identity(self) -> None:
        rng = np.random.default_rng(7)
        samples = rng.normal(size=256) + 1j * rng.normal(size=256)

        shifted = apply_fractional_delay(samples, delay_s=0.0, sample_rate_hz=200e6)

        np.testing.assert_allclose(shifted, samples, atol=1e-12, rtol=1e-12)

    def test_integer_delay_matches_zero_filled_shift_without_wraparound(self) -> None:
        samples = np.zeros(128, dtype=np.complex128)
        samples[32:48] = 1.0 + 0.5j
        delay_samples = 11

        shifted = apply_fractional_delay(
            samples,
            delay_s=delay_samples / 200e6,
            sample_rate_hz=200e6,
        )
        expected = np.zeros_like(samples)
        expected[32 + delay_samples : 48 + delay_samples] = 1.0 + 0.5j

        np.testing.assert_allclose(shifted, expected, atol=1e-12, rtol=1e-12)
        self.assertLess(float(np.max(np.abs(shifted[:delay_samples]))), 1e-12)

    def test_fractional_delay_matches_analytic_bandlimited_pulse(self) -> None:
        sample_rate_hz = 200e6
        sample_index = np.arange(512, dtype=np.float64)
        center = 240.0
        width = 24.0
        tone_cycles_per_sample = 0.075
        delay_samples = 0.25
        samples = np.exp(-0.5 * ((sample_index - center) / width) ** 2) * np.exp(
            1j * 2.0 * np.pi * tone_cycles_per_sample * sample_index
        )
        expected_index = sample_index - delay_samples
        expected = np.exp(-0.5 * ((expected_index - center) / width) ** 2) * np.exp(
            1j * 2.0 * np.pi * tone_cycles_per_sample * expected_index
        )

        shifted = apply_fractional_delay(
            samples,
            delay_s=delay_samples / sample_rate_hz,
            sample_rate_hz=sample_rate_hz,
        )

        np.testing.assert_allclose(shifted[80:-80], expected[80:-80], atol=2e-10, rtol=2e-10)

    def test_negative_fractional_delay_advances_bandlimited_pulse(self) -> None:
        sample_rate_hz = 200e6
        sample_index = np.arange(512, dtype=np.float64)
        center = 260.0
        width = 25.0
        delay_samples = -0.4
        samples = np.exp(-0.5 * ((sample_index - center) / width) ** 2).astype(np.complex128)
        expected = np.exp(
            -0.5 * ((sample_index - delay_samples - center) / width) ** 2
        ).astype(np.complex128)

        shifted = apply_fractional_delay(
            samples,
            delay_s=delay_samples / sample_rate_hz,
            sample_rate_hz=sample_rate_hz,
        )

        np.testing.assert_allclose(shifted[80:-80], expected[80:-80], atol=2e-10, rtol=2e-10)


class NoiseAndChannelTests(unittest.TestCase):
    """验证复 AWGN 标定和静态单径传播。"""

    def test_complex_awgn_matches_requested_active_sample_snr(self) -> None:
        rng = np.random.default_rng(20260908)
        samples = np.ones(200_000, dtype=np.complex128)
        active_mask = np.ones(samples.size, dtype=np.bool_)

        result = add_awgn(samples, snr_db=18.0, active_mask=active_mask, rng=rng)

        self.assertAlmostEqual(result.signal_power, 1.0, places=12)
        self.assertLess(abs(result.measured_snr_db - 18.0), 0.08)
        self.assertLess(abs(np.mean(result.noise_samples)), 0.01)

    def test_static_link_applies_delay_amplitude_and_phase(self) -> None:
        sample_rate_hz = 200e6
        sample_index = np.arange(512, dtype=np.float64)
        center = 180.0
        width = 20.0
        delay_samples = 12.25
        tx = np.exp(-0.5 * ((sample_index - center) / width) ** 2).astype(np.complex128)
        link = ChannelConfig(
            propagation_delay_s=delay_samples / sample_rate_hz,
            amplitude=0.7,
            phase_rad=0.4,
            snr_db=None,
        )

        result = propagate_static_link(
            tx,
            sample_rate_hz=sample_rate_hz,
            config=link,
            rng=np.random.default_rng(11),
        )
        output_index = np.arange(result.samples.size, dtype=np.float64)
        expected = 0.7 * np.exp(1j * 0.4) * np.exp(
            -0.5 * ((output_index - delay_samples - center) / width) ** 2
        )

        np.testing.assert_allclose(
            result.clean_samples[80:-80], expected[80:-80], atol=2e-10, rtol=2e-10
        )
        self.assertTrue(np.array_equal(result.samples, result.clean_samples))
        self.assertIsNone(result.measured_snr_db)

    def test_static_link_defines_snr_over_shifted_transmit_activity_window(self) -> None:
        sample_rate_hz = 200e6
        tx = np.zeros(4096, dtype=np.complex128)
        tx[1000:3000] = 1.0
        delay_samples = 12.25
        target_snr_db = 15.0
        link = ChannelConfig(
            propagation_delay_s=delay_samples / sample_rate_hz,
            snr_db=target_snr_db,
        )

        result = propagate_static_link(
            tx,
            sample_rate_hz=sample_rate_hz,
            config=link,
            rng=np.random.default_rng(23),
        )
        active_start = 1000 + int(np.floor(delay_samples))
        active_stop = 3000 + int(np.ceil(delay_samples))
        signal_power = float(np.mean(np.abs(result.clean_samples[active_start:active_stop]) ** 2))
        noise_power = float(np.mean(np.abs(result.noise_samples[active_start:active_stop]) ** 2))
        measured_snr_db = 10.0 * np.log10(signal_power / noise_power)

        self.assertLess(abs(measured_snr_db - target_snr_db), 0.2)


if __name__ == "__main__":
    unittest.main()
