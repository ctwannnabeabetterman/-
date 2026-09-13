"""四张正式结果图的文件与命名契约。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import FrequencySyncConfig, WaveformConfig
from delay_estimator import estimate_delay, fft_matched_filter
from lut_calibration import build_qls_lut, validate_qls_lut
from models import MonteCarloResult, SensitivityResult, ThreeExperimentResult
from plotting import (
    plot_model_mismatch_sensitivity,
    plot_qls_lut_validation,
    plot_system_signal_chain,
    plot_three_config_vs_crlb,
)
from waveforms import generate_two_tone


class FormalPlotContractTests(unittest.TestCase):
    def test_four_formal_plots_write_named_png_pdf_pairs(self) -> None:
        waveform_config = WaveformConfig(pulse_duration_s=2e-6)
        waveform = generate_two_tone(waveform_config)
        correlation = fft_matched_filter(waveform.samples, waveform.samples)
        estimate = estimate_delay(correlation, waveform.sample_rate_hz)
        calibration = build_qls_lut(waveform_config, grid_points=21)
        validation = validate_qls_lut(
            waveform_config, calibration, validation_points=20
        )
        axis = np.array([12.0, 24.0, 36.0])
        monte_carlo = MonteCarloResult(
            snr_db=axis,
            integer_peak_rmse_s=np.array([2e-9, 1.5e-9, 1.2e-9]),
            qls_rmse_s=np.array([30e-12, 12e-12, 5e-12]),
            lut_rmse_s=np.array([20e-12, 7e-12, 2e-12]),
            crlb_std_s=np.array([18e-12, 6e-12, 2e-12]),
            clock_offset_rmse_s=np.array([13e-12, 4e-12, 1.4e-12]),
            coherent_gain_vs_incoherent_db=np.array([2.8, 2.95, 3.0]),
            residual_frequency_rmse_hz=np.array([2.0, 1.0, 0.5]),
            acquisition_failure_rate=np.zeros(3),
        )
        profiles = ("cabled", "wireless_time", "wireless_time_frequency")
        three_experiment = ThreeExperimentResult(
            profile_keys=profiles,
            profile_labels=("有线时间+有线频率", "无线时间+有线频率", "全无线"),
            time_link_modes=("cabled", "wireless", "wireless"),
            frequency_link_modes=("cabled", "cabled", "wireless"),
            snr_db=axis,
            time_transfer_std_s=np.array(
                [[18, 6, 2], [19, 6.5, 2.2], [22, 8, 3]], dtype=float
            ) * 1e-12,
            beamforming_std_s=np.array(
                [[30, 10, 3], [32, 11, 3.5], [40, 16, 7]], dtype=float
            ) * 1e-12,
            one_way_delay_crlb_std_s=np.array([18, 6, 2], dtype=float) * 1e-12,
            two_way_clock_crlb_std_s=np.array([13, 4.2, 1.4], dtype=float) * 1e-12,
            residual_clock_rate_rmse=np.ones((3, 3)) * 1e-9,
            acquisition_failure_rate=np.zeros((3, 3)),
            time_transfer_samples_s=np.zeros((3, 3, 2)),
            beamforming_samples_s=np.zeros((3, 3, 2)),
            residual_clock_rate_samples=np.zeros((3, 3, 2)),
        )
        sensitivity = SensitivityResult(
            path_asymmetry_s=np.array([-100, 0, 100], dtype=float) * 1e-12,
            clock_bias_s=np.array([-50, 0, 50], dtype=float) * 1e-12,
            clock_std_s=np.ones(3) * 1e-12,
            reference_phase_noise_rad=np.deg2rad([0, 0.5, 3]),
            clock_rate_rmse=np.array([1e-10, 2e-9, 1e-8]),
            holdover_timing_rmse_s=np.array([10, 200, 1000]) * 1e-12,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = []
            paths += plot_system_signal_chain(
                waveform, FrequencySyncConfig(), waveform.samples, output
            )
            paths += plot_qls_lut_validation(
                correlation,
                estimate,
                calibration,
                validation,
                monte_carlo,
                waveform.sample_rate_hz,
                output,
            )
            paths += plot_three_config_vs_crlb(three_experiment, output)
            paths += plot_model_mismatch_sensitivity(sensitivity, output)
            names = {path.name for path in paths}
            sizes = [path.stat().st_size for path in paths]

        self.assertEqual(len(paths), 8)
        self.assertEqual(
            names,
            {
                f"{stem}.{suffix}"
                for stem in (
                    "01_system_signal_chain",
                    "02_qls_lut_validation",
                    "03_three_config_vs_crlb",
                    "04_model_mismatch_sensitivity",
                )
                for suffix in ("png", "pdf")
            },
        )
        self.assertTrue(all(size > 0 for size in sizes))


if __name__ == "__main__":
    unittest.main()
