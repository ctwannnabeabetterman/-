"""精简正式结果的机器可读验收测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from verify_results import verify_output


class ResultAcceptanceTests(unittest.TestCase):
    def _write_valid_result(self, root: Path) -> None:
        figures = root / "figures"
        figures.mkdir(parents=True)
        names: list[str] = []
        for stem in (
            "01_system_signal_chain",
            "02_qls_lut_validation",
            "03_three_config_vs_crlb",
            "04_model_mismatch_sensitivity",
        ):
            for suffix in ("png", "pdf"):
                path = figures / f"{stem}.{suffix}"
                path.write_bytes(b"result")
                names.append(str(path.relative_to(root)))
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "figure_count": 8,
                    "figures": names,
                    "tables": ["three_config_summary.csv"],
                    "diagnostics": [],
                }
            ),
            encoding="utf-8",
        )
        (root / "three_config_summary.csv").write_text(
            "case,snr_db\ncabled,36\n", encoding="utf-8"
        )
        summary = {
            "lut": {
                "raw_rmse_ps": 20.0,
                "corrected_rmse_ps": 0.01,
                "validation_points": 400,
                "validation_policy": "half-step grid disjoint from LUT training fractions",
            },
            "three_experiments": {
                "profiles": {
                    key: {
                        "time_transfer_std_ps": 3.0,
                        "beamforming_std_ps": 4.0,
                        "residual_clock_rate_rmse_ppm": 0.001,
                        "acquisition_failure_rate": 0.0,
                    }
                    for key in (
                        "cabled",
                        "wireless_time",
                        "wireless_time_frequency",
                    )
                }
            },
            "clock_tracking": {
                "final_residual_after_update_ps": 2.0,
                "estimated_fractional_frequency_offset": 2e-7,
                "applied_fractional_frequency_correction": 2e-7,
            },
            "frequency_sync": {
                "final_true_observed_offset_hz": 600.0,
                "final_tracked_estimate_hz": 599.0,
                "applied_oscillator_correction_hz": 599.0,
                "final_residual_hz": 1.0,
            },
            "state_separation": {
                "plant_at_data_epoch": {
                    "raw_clock_offset_ps": 100000.0,
                    "residual_clock_offset_ps": 2.0,
                    "raw_frequency_offset_hz": 600.0,
                    "residual_frequency_offset_hz": 1.0,
                }
            },
            "beamforming": {
                "unsynchronized": {"combined_power": 1.0},
                "full_sync": {
                    "arrival_difference_ps": 2.0,
                    "combined_power": 3.9,
                    "gain_vs_incoherent_sum_db": 3.0,
                    "normalized_ideal_loss_db": 0.001,
                },
            },
            "monte_carlo_highest_snr": {
                "acquisition_failure_rate": 0.0,
                "integer_peak_rmse_ps": 1400.0,
                "qls_rmse_ps": 24.0,
                "qls_lut_rmse_ps": 2.0,
                "crlb_std_ps": 2.0,
            },
            "sensitivity": {
                "max_abs_asymmetry_ps": 100.0,
                "max_abs_clock_bias_ps": 50.0,
                "max_reference_phase_noise_deg": 3.0,
                "max_holdover_timing_rmse_ps": 900.0,
            },
        }
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def test_valid_output_passes_every_named_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_result(root)
            report = verify_output(root)

        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))

    def test_missing_primary_table_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_result(root)
            (root / "three_config_summary.csv").unlink()
            report = verify_output(root)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["primary_table_present"])

    def test_excessive_full_sync_loss_fails_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_result(root)
            path = root / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["beamforming"]["full_sync"]["normalized_ideal_loss_db"] = 1.0
            path.write_text(json.dumps(summary), encoding="utf-8")
            report = verify_output(root)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["full_sync_ideal_loss"])


if __name__ == "__main__":
    unittest.main()
