"""独立 ADC 网格、有限窗口和粗捕获的物理回归。"""
import unittest
import numpy as np
from acquisition import AcquisitionError, sample_iq, simulate_capture, sync_burst, measure_capture
from config import WaveformConfig, ChannelConfig, TwoWayConfig
from clock_model import LocalClock
from lut_calibration import build_qls_lut
from two_way_sync import simulate_two_way_exchange, estimate_two_way


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

    def test_coarse_code_resolves_many_two_tone_periods(self):
        for offset in (-700e-9, 600e-9):
            obs = simulate_two_way_exchange(self.w, TwoWayConfig(), LocalClock(),
                LocalClock(offset_s=offset), ChannelConfig(snr_db=None),
                ChannelConfig(snr_db=None), self.lut, np.random.default_rng(1))
            self.assertLess(abs(estimate_two_way(obs).ap1_offset_estimate_s-offset), 1e-12)

    def test_missing_or_truncated_burst_is_rejected(self):
        for cfg in (TwoWayConfig(receive_window_s=100e-9), TwoWayConfig(receive_pretrigger_s=0.)):
            with self.assertRaises(AcquisitionError):
                simulate_two_way_exchange(self.w, cfg, LocalClock(), LocalClock(offset_s=500e-9),
                    ChannelConfig(snr_db=None), ChannelConfig(snr_db=None),
                    self.lut, np.random.default_rng(2))

    def test_adc_clock_rate_affects_pulse_estimate(self):
        burst, _, _ = sync_burst(self.w)
        def capture(step):
            return simulate_capture(burst, self.w.sample_rate_hz, start_local_s=0.,
                start_source_sample=-200., sample_step=step, count=len(burst)+400,
                amplitude=1., snr_db=None, rng=np.random.default_rng(3))
        a = measure_capture(capture(1.), self.w, self.lut)
        b = measure_capture(capture(1.00005), self.w, self.lut)
        self.assertGreater(abs(a.corrected_delay_s-b.corrected_delay_s), 10e-12)
