"""图 12 软件实验的统计定义、测量符号和噪声边界。"""

import unittest
from dataclasses import replace

import numpy as np

from channel import apply_fractional_delay
from config import WaveformConfig
from lut_calibration import build_qls_lut
from paper_fig12 import Figure12Config, measure_interarrival, run_precision_case, sample_statistics
from waveforms import generate_two_tone


class Figure12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sync = WaveformConfig(pulse_duration_s=2e-6)
        cls.data = WaveformConfig(tone_separation_hz=50e6, pulse_duration_s=1e-6,
                                  carrier_frequency_hz=1.2e9)
        cls.lut = build_qls_lut(cls.sync, 401)
        cls.data_lut = build_qls_lut(cls.data, 401)

    def test_precision_is_sample_std_not_rmse(self):
        result = sample_statistics(np.array([99., 100., 101.]) * 1e-12)
        self.assertAlmostEqual(result['std_ps'], 1.)
        self.assertAlmostEqual(result['mean_ps'], 100.)
        self.assertGreater(result['rmse_ps'], 99.)

    def test_cross_correlation_measures_positive_and_negative_delay(self):
        base = np.pad(generate_two_tone(self.data).samples, (128, 128))
        for delay in (-23e-12, 23e-12):
            shifted = apply_fractional_delay(base, delay, self.data.sample_rate_hz)
            measured = measure_interarrival(base, shifted, self.data, self.data_lut)
            self.assertLess(abs(measured - delay), 0.5e-12)

    def test_independent_readout_noise_does_not_change_time_commands(self):
        cfg = Figure12Config(trials=15, snr_db_values=(36.,))
        clean = run_precision_case(self.sync, self.data, self.lut, self.data_lut,
                                   cfg, 3e-9, 0., 36., 4)
        noisy = run_precision_case(self.sync, self.data, self.lut, self.data_lut,
                                   replace(cfg, readout_jitter_s=100e-12), 3e-9, 0., 36., 4)
        np.testing.assert_array_equal(clean['correction_s'], noisy['correction_s'])
        self.assertGreater(np.std(noisy['interarrival_s']), np.std(clean['interarrival_s']) * 3.)

    def test_increasing_reference_jitter_increases_reported_precision(self):
        cfg = Figure12Config(trials=30, snr_db_values=(36.,), readout_jitter_s=0.)
        clean = run_precision_case(self.sync, self.data, self.lut, self.data_lut,
                                   cfg, 3e-9, 0., 36., 5)
        noisy = run_precision_case(self.sync, self.data, self.lut, self.data_lut,
                                   cfg, 3e-9, 100e-12, 36., 5)
        self.assertGreater(np.std(noisy['correction_s']), np.std(clean['correction_s']) * 5.)


if __name__ == '__main__':
    unittest.main()
