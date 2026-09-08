"""阶段 9：SNR 扫描 Monte Carlo 的输出和可复现性测试。"""

from __future__ import annotations

import unittest

import numpy as np

from config import MonteCarloConfig, WaveformConfig
from lut_calibration import build_qls_lut
from monte_carlo import run_delay_monte_carlo


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


if __name__ == "__main__":
    unittest.main()
