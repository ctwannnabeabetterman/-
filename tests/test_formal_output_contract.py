"""正式结果目录的最小、可读输出契约。"""

from __future__ import annotations

from pathlib import Path
import unittest

from reporting import build_manifest


class FormalOutputContractTests(unittest.TestCase):
    def test_default_manifest_contains_four_figure_pairs_and_one_table(self) -> None:
        figure_paths = [
            Path("figures") / f"{stem}.{suffix}"
            for stem in (
                "01_system_signal_chain",
                "02_qls_lut_validation",
                "03_three_config_vs_crlb",
                "04_model_mismatch_sensitivity",
            )
            for suffix in ("png", "pdf")
        ]

        manifest = build_manifest(figure_paths, diagnostics=False)

        self.assertEqual(manifest["schema_version"], 4)
        self.assertEqual(manifest["figure_count"], 8)
        self.assertEqual(manifest["figures"], [str(path) for path in figure_paths])
        self.assertEqual(manifest["tables"], ["three_config_summary.csv"])
        self.assertNotIn("three_experiment_samples.csv", manifest["tables"])
        self.assertNotIn("joint_tracking.csv", manifest["tables"])

    def test_diagnostics_are_declared_separately(self) -> None:
        manifest = build_manifest([], diagnostics=True)

        self.assertEqual(manifest["tables"], ["three_config_summary.csv"])
        self.assertTrue(manifest["diagnostics"])
        self.assertTrue(
            all(path.startswith("diagnostics/") for path in manifest["diagnostics"])
        )


if __name__ == "__main__":
    unittest.main()
