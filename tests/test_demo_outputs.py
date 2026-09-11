"""阶段 10：运行预设与结果文件写入测试。"""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import ClockTrackingConfig, MonteCarloConfig, PhaseFeedbackConfig
from main_demo import build_demo_settings, run_demo
from results_io import write_csv_columns, write_json


class DemoSettingsTests(unittest.TestCase):
    """验证快速与正式模式保留同一算法流程和规定统计规模。"""

    def test_fast_mode_uses_one_hundred_trials_and_complete_snr_axis(self) -> None:
        settings = build_demo_settings("fast_demo")

        self.assertEqual(settings.monte_carlo.trials_per_snr, 100)
        self.assertEqual(settings.monte_carlo.snr_db_values, tuple(range(6, 37, 3)))
        self.assertLess(settings.lut_grid_points, 2001)

    def test_formal_mode_uses_default_waveform_and_one_thousand_trials(self) -> None:
        settings = build_demo_settings("formal")

        self.assertEqual(settings.waveform.pulse_duration_s, 10e-6)
        self.assertEqual(settings.lut_grid_points, 2001)
        self.assertEqual(settings.monte_carlo.trials_per_snr, 1000)

    def test_small_end_to_end_run_applies_estimator_control_outputs(self) -> None:
        base = build_demo_settings("fast_demo")
        settings = replace(
            base,
            lut_grid_points=101,
            frequency_rounds=3,
            clock_tracking=ClockTrackingConfig(
                rounds=5,
                sync_interval_s=50e-3,
                correction_gain=1.0,
                random_walk_std_s_per_sqrt_s=0.0,
            ),
            phase_feedback=PhaseFeedbackConfig(
                pilot_symbols=128,
                snr_db=32.0,
                feedback_delay_s=100e-6,
                phase_quantization_bits=12,
            ),
            monte_carlo=MonteCarloConfig(
                snr_db_values=(30.0,),
                trials_per_snr=3,
                seed=2023,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = run_demo(settings, directory)
            result_guide = (Path(directory) / "README.md").read_text(
                encoding="utf-8"
            )

        clock = summary["clock_tracking"]
        frequency = summary["frequency_sync"]
        states = summary["beamforming"]
        self.assertEqual(
            clock["applied_fractional_frequency_correction"],
            clock["estimated_fractional_frequency_offset"],
        )
        self.assertEqual(
            frequency["applied_oscillator_correction_hz"],
            frequency["final_tracked_estimate_hz"],
        )
        self.assertLess(
            abs(states["full_sync"]["arrival_difference_ps"]),
            abs(states["unsynchronized"]["arrival_difference_ps"]),
        )
        self.assertLess(
            states["full_sync"]["normalized_ideal_loss_db"],
            states["time_frequency"]["normalized_ideal_loss_db"],
        )
        self.assertIn("时间同步主结果", result_guide)
        self.assertIn("QLS + LUT", result_guide)
        self.assertIn("下游相干合成验证", result_guide)

    def test_time_varying_channel_is_shared_by_pilot_and_data(self) -> None:
        base = build_demo_settings("fast_demo")
        settings = replace(
            base,
            lut_grid_points=101,
            frequency_rounds=3,
            clock_tracking=ClockTrackingConfig(
                rounds=5,
                sync_interval_s=50e-3,
                correction_gain=1.0,
                random_walk_std_s_per_sqrt_s=0.0,
            ),
            phase_feedback=PhaseFeedbackConfig(
                pilot_symbols=256,
                snr_db=100.0,
                feedback_delay_s=100e-6,
                phase_quantization_bits=None,
                channel_phase_rate_rad_per_s=10.0,
            ),
            monte_carlo=MonteCarloConfig(
                snr_db_values=(30.0,),
                trials_per_snr=2,
                seed=2023,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            summary = run_demo(settings, directory)

        self.assertLess(
            summary["beamforming"]["full_sync"]["normalized_ideal_loss_db"],
            0.05,
        )


class ResultWriterTests(unittest.TestCase):
    """验证 NumPy 数值可写入可读 JSON 和列式 CSV。"""

    def test_json_converts_numpy_arrays_and_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(
                Path(directory) / "summary.json",
                {"values": np.array([1.0, 2.0]), "count": np.int64(2)},
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, {"values": [1.0, 2.0], "count": 2})

    def test_csv_rejects_different_column_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_csv_columns(
                    Path(directory) / "bad.csv",
                    {"a": [1, 2], "b": [3]},
                )

    def test_csv_writes_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv_columns(
                Path(directory) / "table.csv",
                {"snr_db": [6, 9], "rmse_ps": [12.0, 8.0]},
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(rows[0], ["snr_db", "rmse_ps"])
        self.assertEqual(rows[2], ["9", "8.0"])


if __name__ == "__main__":
    unittest.main()
