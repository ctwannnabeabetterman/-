"""论文三种时间/频率传递配置的完整通信链测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from config import ThreeExperimentConfig, WaveformConfig
from experiment_suite import _estimate_wireless_clock_rate, run_three_experiment_suite
from lut_calibration import build_qls_lut
from two_way_sync import simulate_two_way_exchange


class ThreeExperimentSuiteTests(unittest.TestCase):
    """每个配置和试验都必须执行波形驱动的四时间戳交换。"""

    def test_three_profiles_run_the_same_estimation_chain(self) -> None:
        waveform = WaveformConfig(pulse_duration_s=1e-6)
        calibration = build_qls_lut(waveform, grid_points=101)
        config = ThreeExperimentConfig(
            snr_db_values=(24.0,),
            trials_per_snr=3,
            seed=91,
            frequency_capture_duration_s=40e-6,
        )

        with patch(
            "experiment_suite.simulate_two_way_exchange",
            wraps=simulate_two_way_exchange,
        ) as exchange:
            result = run_three_experiment_suite(waveform, calibration, config)

        self.assertEqual(
            result.profile_keys,
            ("cabled", "wireless_time", "wireless_time_frequency"),
        )
        self.assertEqual(exchange.call_count, 9)
        self.assertEqual(result.time_transfer_std_s.shape, (3, 1))
        self.assertEqual(result.beamforming_std_s.shape, (3, 1))
        self.assertEqual(result.time_transfer_samples_s.shape, (3, 1, 3))
        self.assertTrue(np.all(np.isfinite(result.time_transfer_std_s)))
        self.assertTrue(np.all(np.isfinite(result.beamforming_std_s)))
        self.assertEqual(result.frequency_link_modes[-1], "wireless")

    def test_continuous_wireless_reference_uses_separated_iq_captures(self) -> None:
        waveform = WaveformConfig(pulse_duration_s=2e-6)
        config = ThreeExperimentConfig(
            snr_db_values=(24.0,),
            trials_per_snr=2,
            frequency_reference_snr_db=30.0,
            frequency_capture_duration_s=40e-6,
            sync_interval_s=50e-3,
        )
        true_rate = 0.2e-6
        errors = []
        for seed in range(20):
            estimate = _estimate_wireless_clock_rate(
                waveform,
                config,
                true_rate,
                np.random.default_rng(seed),
            )
            errors.append(estimate - true_rate)

        self.assertLess(float(np.sqrt(np.mean(np.square(errors)))), 0.005e-6)


if __name__ == "__main__":
    unittest.main()
