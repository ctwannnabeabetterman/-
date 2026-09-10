"""阶段 9：SNR 扫描 Monte Carlo 的输出和可复现性测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from config import MonteCarloConfig, PhaseFeedbackConfig, WaveformConfig
from beamforming import combine_two_ap as real_combine_two_ap
from lut_calibration import build_qls_lut
from monte_carlo import run_delay_monte_carlo
from phase_sync import simulate_channel_feedback as real_simulate_channel_feedback
from two_way_sync import simulate_two_way_exchange as real_simulate_two_way_exchange
from frequency_sync import estimate_frequency_offset as real_estimate_frequency_offset
from acquisition import AcquisitionError


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

    def test_phase_settings_do_not_change_delay_or_clock_statistics(self) -> None:
        config = MonteCarloConfig(
            snr_db_values=(24.0,),
            trials_per_snr=20,
            seed=97,
        )
        short_pilot = run_delay_monte_carlo(
            self.waveform_config,
            self.calibration,
            config,
            phase_feedback_config=PhaseFeedbackConfig(pilot_symbols=16),
        )
        long_pilot = run_delay_monte_carlo(
            self.waveform_config,
            self.calibration,
            config,
            phase_feedback_config=PhaseFeedbackConfig(pilot_symbols=1024),
        )

        for field in (
            "integer_peak_rmse_s",
            "qls_rmse_s",
            "lut_rmse_s",
            "clock_offset_rmse_s",
        ):
            np.testing.assert_array_equal(
                getattr(short_pilot, field),
                getattr(long_pilot, field),
            )

    def test_feedback_staleness_matches_two_signal_gain(self) -> None:
        phase_rate_rad_per_s = 100.0
        feedback_delay_s = 5e-3
        result = run_delay_monte_carlo(
            self.waveform_config,
            self.calibration,
            MonteCarloConfig(
                snr_db_values=(120.0,),
                trials_per_snr=3,
                seed=98,
            ),
            phase_feedback_config=PhaseFeedbackConfig(
                pilot_symbols=256,
                snr_db=120.0,
                feedback_delay_s=feedback_delay_s,
                phase_quantization_bits=None,
                channel_phase_rate_rad_per_s=phase_rate_rad_per_s,
            ),
        )
        expected_gain_db = 10.0 * np.log10(
            1.0 + np.cos(phase_rate_rad_per_s * feedback_delay_s)
        )

        self.assertAlmostEqual(
            result.coherent_gain_vs_incoherent_db[0],
            expected_gain_db,
            places=3,
        )

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

    def test_every_trial_estimates_a_nonzero_random_frequency(self) -> None:
        cfg = MonteCarloConfig(snr_db_values=(36.,), trials_per_snr=4)
        with patch('monte_carlo.estimate_frequency_offset', wraps=real_estimate_frequency_offset) as estimator:
            result = run_delay_monte_carlo(self.waveform_config, self.calibration, cfg)
        self.assertEqual(estimator.call_count, 4)
        self.assertTrue(all(abs(call.args[1].cfo_hz) > 0 for call in estimator.call_args_list))
        self.assertLess(result.residual_frequency_rmse_hz[0], 5.)

    def test_capture_failure_is_counted_and_all_failed_is_not_success(self) -> None:
        cfg = MonteCarloConfig(snr_db_values=(36.,), trials_per_snr=4)
        calls = 0
        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise AcquisitionError('test_missing_packet')
            return real_simulate_two_way_exchange(*args, **kwargs)
        with patch('monte_carlo.simulate_two_way_exchange', side_effect=fail_once):
            result = run_delay_monte_carlo(self.waveform_config, self.calibration, cfg)
        self.assertEqual(result.acquisition_failure_rate[0], .25)
        with patch('monte_carlo.simulate_two_way_exchange', side_effect=AcquisitionError('missing')):
            with self.assertRaises(AcquisitionError):
                run_delay_monte_carlo(self.waveform_config, self.calibration, cfg)


if __name__ == "__main__":
    unittest.main()
