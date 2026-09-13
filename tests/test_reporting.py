"""读者导向中文报告与正式表格测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from models import ThreeExperimentResult
from reporting import render_report, write_result_tables


def _three_experiment_fixture() -> ThreeExperimentResult:
    snr = np.array([30.0, 36.0])
    return ThreeExperimentResult(
        profile_keys=("cabled", "wireless_time", "wireless_time_frequency"),
        profile_labels=("Cabled", "Wireless time", "Full wireless"),
        time_link_modes=("cabled", "wireless", "wireless"),
        frequency_link_modes=("cabled", "cabled", "wireless"),
        snr_db=snr,
        time_transfer_std_s=np.array([[3, 2], [3.2, 2.2], [4, 3]]) * 1e-12,
        beamforming_std_s=np.array([[6, 3], [6.2, 3.2], [10, 7]]) * 1e-12,
        one_way_delay_crlb_std_s=np.array([3, 2]) * 1e-12,
        two_way_clock_crlb_std_s=np.array([2.1, 1.4]) * 1e-12,
        residual_clock_rate_rmse=np.ones((3, 2)) * 1e-9,
        acquisition_failure_rate=np.zeros((3, 2)),
        time_transfer_samples_s=np.zeros((3, 2, 2)),
        beamforming_samples_s=np.zeros((3, 2, 2)),
        residual_clock_rate_samples=np.zeros((3, 2, 2)),
    )


class ReportingTests(unittest.TestCase):
    def test_report_explains_figures_and_simulation_boundary(self) -> None:
        summary = {
            "mode": "formal",
            "seed": 2023,
            "waveform": {
                "sample_rate_msa_s": 200.0,
                "transmit_sample_rate_msa_s": 400.0,
                "tone_separation_mhz": 40.0,
                "pulse_duration_us": 10.0,
                "rise_fall_ns": 50.0,
            },
            "lut": {
                "raw_rmse_ps": 52.0,
                "corrected_rmse_ps": 0.001,
            },
            "monte_carlo_highest_snr": {
                "snr_db": 36.0,
                "qls_lut_rmse_ps": 2.0,
                "crlb_std_ps": 2.0,
                "acquisition_failure_rate": 0.0,
            },
            "three_experiments": {
                "profiles": {
                    key: {
                        "label_cn": label,
                        "time_transfer_std_ps": value,
                        "beamforming_std_ps": value * 2,
                        "residual_clock_rate_rmse_ppm": 0.001,
                        "acquisition_failure_rate": 0.0,
                    }
                    for key, label, value in (
                        ("cabled", "有线时间 + 有线频率", 2.0),
                        ("wireless_time", "无线时间 + 有线频率", 2.2),
                        ("wireless_time_frequency", "无线时间 + 无线频率", 3.0),
                    )
                }
            },
            "beamforming": {
                "full_sync": {
                    "arrival_difference_ps": 4.0,
                    "gain_vs_incoherent_sum_db": 3.0,
                }
            },
            "sensitivity": {
                "max_abs_asymmetry_ps": 100.0,
                "max_abs_clock_bias_ps": 50.0,
                "max_reference_phase_noise_deg": 3.0,
                "max_holdover_timing_rmse_ps": 900.0,
            },
        }

        text = render_report(summary)

        for phrase in (
            "三种配置与 CRLB",
            "QLS 与 LUT",
            "模型失配",
            "不能代表论文硬件实验",
            "如何阅读四张图",
        ):
            self.assertIn(phrase, text)

    def test_default_tables_exclude_internal_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_result_tables(
                directory, _three_experiment_fixture(), diagnostics=False
            )
            names = sorted(path.name for path in Path(directory).rglob("*.csv"))

        self.assertEqual([path.name for path in paths], ["three_config_summary.csv"])
        self.assertEqual(names, ["three_config_summary.csv"])


if __name__ == "__main__":
    unittest.main()
