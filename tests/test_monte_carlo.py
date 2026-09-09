"""阶段 9：SNR 扫描 Monte Carlo 的输出和可复现性测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from config import MonteCarloConfig, WaveformConfig
from beamforming import combine_two_ap as real_combine_two_ap
from lut_calibration import build_qls_lut
from monte_carlo import run_delay_monte_carlo
from phase_sync import simulate_channel_feedback as real_simulate_channel_feedback
from two_way_sync import simulate_two_way_exchange as real_simulate_two_way_exchange


class MonteCarloTests(unittest.TestCase):
    """用小规模固定种子实验检查统计链路。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.waveform_config = WaveformConfig(pulse_duration_s=2e-6)
        cls.calibration = build_qls_lut(cls.waveform_config, grid_points=201)
        cls.config = MonteCarloConfig(
            snr_db_values=(12.0, 30.0),
            trials_per_snr=20,
            seed=90210,
        )

    def test_outputs_have_one_finite_value_per_snr(self) -> None:
        result = run_delay_monte_carlo(
            self.waveform_config, self.calibration, self.config
        )

        for values in (
            result.integer_peak_rmse_s,
            result.qls_rmse_s,
            result.lut_rmse_s,
            result.crlb_std_s,
            result.clock_offset_rmse_s,
            result.coherent_gain_vs_incoherent_db,
        ):
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(result.coherent_gain_vs_incoherent_db <= 3.010299957))

    def test_fixed_seed_reproduces_every_reported_array(self) -> None:
        first = run_delay_monte_carlo(
            self.waveform_config, self.calibration, self.config
        )
        second = run_delay_monte_carlo(
            self.waveform_config, self.calibration, self.config
        )

        for field in first.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))

    def test_high_snr_lut_is_better_than_integer_peak_and_crlb_falls(self) -> None:
        result = run_delay_monte_carlo(
            self.waveform_config, self.calibration, self.config
        )

        self.assertLess(result.lut_rmse_s[-1], result.integer_peak_rmse_s[-1])
        self.assertLess(result.crlb_std_s[-1], result.crlb_std_s[0])

    def test_clock_statistic_executes_one_four_timestamp_exchange_per_trial(self) -> None:
        config = MonteCarloConfig(
            snr_db_values=(30.0,),
            trials_per_snr=3,
            seed=94,
            clock_offset_truth_s=80e-9,
            processing_delay_s=17e-6,
        )
        with patch(
            "monte_carlo.simulate_two_way_exchange",
            wraps=real_simulate_two_way_exchange,
        ) as exchange:
            run_delay_monte_carlo(
                self.waveform_config,
                self.calibration,
                config,
            )

        self.assertEqual(exchange.call_count, config.trials_per_snr)

    def test_gain_executes_feedback_and_data_combining_per_trial(self) -> None:
        config = MonteCarloConfig(
            snr_db_values=(30.0,),
            trials_per_snr=3,
            seed=96,
        )
        with patch(
            "monte_carlo.simulate_channel_feedback",
            wraps=real_simulate_channel_feedback,
        ) as feedback, patch(
            "monte_carlo.combine_two_ap",
            wraps=real_combine_two_ap,
        ) as combine:
            run_delay_monte_carlo(
                self.waveform_config,
                self.calibration,
                config,
            )

        self.assertEqual(feedback.call_count, config.trials_per_snr)
        self.assertEqual(combine.call_count, config.trials_per_snr)

    def test_processing_delay_does_not_change_fixed_seed_clock_rmse(self) -> None:
        base = dict(snr_db_values=(30.0,), trials_per_snr=10, seed=95)
        short = run_delay_monte_carlo(
            self.waveform_config,
            self.calibration,
            MonteCarloConfig(**base, processing_delay_s=5e-6),
        )
        long = run_delay_monte_carlo(
            self.waveform_config,
            self.calibration,
            MonteCarloConfig(**base, processing_delay_s=100e-6),
        )

        np.testing.assert_allclose(
            short.clock_offset_rmse_s,
            long.clock_offset_rmse_s,
            rtol=0.0,
            atol=1e-16,
        )


if __name__ == "__main__":
    unittest.main()
