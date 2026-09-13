"""论文主实验之外的模型失配敏感性测试。"""

from __future__ import annotations

import unittest

import numpy as np

from config import SensitivityConfig, WaveformConfig
from lut_calibration import build_qls_lut
from sensitivity import run_sensitivity_suite


class SensitivitySuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.waveform_config = WaveformConfig(pulse_duration_s=2e-6)
        cls.calibration = build_qls_lut(cls.waveform_config, grid_points=101)

    def test_two_way_asymmetry_bias_is_half_path_difference(self) -> None:
        config = SensitivityConfig(
            asymmetry_ps=(-80.0, -40.0, 0.0, 40.0, 80.0),
            reference_phase_noise_deg=(0.0, 0.5),
            trials=24,
            time_link_snr_db=80.0,
            frequency_reference_snr_db=80.0,
        )

        result = run_sensitivity_suite(
            self.waveform_config, self.calibration, config
        )

        np.testing.assert_allclose(
            result.clock_bias_s,
            0.5 * result.path_asymmetry_s,
            atol=3e-12,
        )

    def test_reference_phase_noise_increases_rate_and_holdover_error(self) -> None:
        config = SensitivityConfig(
            asymmetry_ps=(0.0,),
            reference_phase_noise_deg=(0.0, 0.1, 0.3, 1.0, 3.0),
            trials=300,
            time_link_snr_db=80.0,
            frequency_reference_snr_db=80.0,
        )

        result = run_sensitivity_suite(
            self.waveform_config, self.calibration, config
        )

        self.assertGreater(result.clock_rate_rmse[-1], result.clock_rate_rmse[0])
        self.assertGreater(
            result.holdover_timing_rmse_s[-1],
            result.holdover_timing_rmse_s[0],
        )
        np.testing.assert_allclose(
            result.holdover_timing_rmse_s,
            result.clock_rate_rmse * config.sync_interval_s,
            rtol=1e-12,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
