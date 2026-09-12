"""独立 ADC 网格、有限窗口和粗捕获的物理回归。"""
import unittest
import numpy as np
from acquisition import AcquisitionError, sample_iq, simulate_capture, measure_capture
from config import WaveformConfig, ChannelConfig, TwoWayConfig
from clock_model import LocalClock
from lut_calibration import build_qls_lut
from two_way_sync import simulate_two_way_exchange, estimate_two_way
from waveforms import generate_two_tone


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = WaveformConfig(pulse_duration_s=2e-6)
        cls.lut = build_qls_lut(cls.w, 1001)

    def test_uniform_grid_matches_samples_and_rate_error_changes_iq(self):
        x = np.exp(2j*np.pi*.1*np.arange(1024)) * np.hanning(1024)
        a = sample_iq(x, 0., 1., 1024)
        b = sample_iq(x, 0., 1.0001, 1024)
        np.testing.assert_allclose(a, x, atol=1e-9)
        self.assertGreater(np.max(abs(a-b)), .01)

    def test_finite_single_pulse_resolves_coarse_clock_offset(self):
        for offset in (-700e-9, 600e-9):
            obs = simulate_two_way_exchange(self.w, TwoWayConfig(), LocalClock(),
                LocalClock(offset_s=offset), ChannelConfig(snr_db=None),
                ChannelConfig(snr_db=None), self.lut, np.random.default_rng(1))
            self.assertLess(abs(estimate_two_way(obs).ap1_offset_estimate_s-offset), 1e-12)

    def test_missing_or_truncated_pulse_is_rejected(self):
        for cfg in (TwoWayConfig(receive_window_s=100e-9), TwoWayConfig(receive_pretrigger_s=0.)):
            with self.assertRaises(AcquisitionError):
                simulate_two_way_exchange(self.w, cfg, LocalClock(), LocalClock(offset_s=500e-9),
                    ChannelConfig(snr_db=None), ChannelConfig(snr_db=None),
                    self.lut, np.random.default_rng(2))

    def test_adc_clock_rate_affects_pulse_estimate(self):
        pulse = generate_two_tone(
            self.w, sample_rate_hz=self.w.transmit_sample_rate_hz
        ).samples
        step0 = self.w.transmit_sample_rate_hz / self.w.sample_rate_hz
        def capture(step):
            return simulate_capture(pulse, self.w.sample_rate_hz, start_local_s=0.,
                start_source_sample=-400., sample_step=step, count=2400,
                amplitude=1., snr_db=None, rng=np.random.default_rng(3))
        a = measure_capture(capture(step0), self.w, self.lut)
        b = measure_capture(capture(step0 * 1.00005), self.w, self.lut)
        self.assertGreater(abs(a.corrected_delay_s-b.corrected_delay_s), 10e-12)

    def test_table_i_uses_one_pulse_on_distinct_tx_rx_grids(self):
        tx = generate_two_tone(
            self.w, sample_rate_hz=self.w.transmit_sample_rate_hz
        )
        rx_template = generate_two_tone(self.w)
        self.assertEqual(tx.samples.size, round(self.w.pulse_duration_s * 400e6))
        self.assertEqual(
            rx_template.samples.size,
            round(self.w.pulse_duration_s * 200e6),
        )
