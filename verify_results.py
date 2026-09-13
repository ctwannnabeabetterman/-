"""按版本化阈值验证精简后的正式 Demo 输出。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


_DEFAULT_THRESHOLDS = Path(__file__).with_name("acceptance_thresholds.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须为 JSON object")
    return value


def verify_output(
    output_dir: str | Path,
    thresholds_path: str | Path = _DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """验证三配置、QLS/LUT、闭环、敏感性和正式文件清单。"""

    root = Path(output_dir)
    thresholds = _read_json(Path(thresholds_path))
    summary = _read_json(root / "summary.json")
    manifest = _read_json(root / "manifest.json")
    expected_pairs = int(thresholds["expected_figure_pairs"])
    expected_files = [root / relative for relative in manifest.get("figures", [])]

    lut = summary["lut"]
    clock = summary["clock_tracking"]
    frequency = summary["frequency_sync"]
    beamforming = summary["beamforming"]
    monte_carlo = summary["monte_carlo_highest_snr"]
    profiles = summary["three_experiments"]["profiles"]
    plant = summary["state_separation"]["plant_at_data_epoch"]
    sensitivity = summary["sensitivity"]
    expected_profile_keys = {
        "cabled",
        "wireless_time",
        "wireless_time_frequency",
    }
    true_frequency_hz = abs(float(frequency["final_true_observed_offset_hz"]))
    residual_frequency_hz = abs(float(frequency["final_residual_hz"]))
    allowed_frequency_hz = float(
        thresholds["max_frequency_residual_fraction"]
    ) * max(true_frequency_hz, 1.0)
    qls_lut_rmse = float(monte_carlo["qls_lut_rmse_ps"])
    crlb_std = float(monte_carlo["crlb_std_ps"])
    half_relation_error = abs(
        float(sensitivity["max_abs_clock_bias_ps"])
        - 0.5 * float(sensitivity["max_abs_asymmetry_ps"])
    )

    checks = {
        "manifest_schema": int(manifest.get("schema_version", -1)) == 4,
        "primary_table_present": (
            manifest.get("tables") == ["three_config_summary.csv"]
            and (root / "three_config_summary.csv").is_file()
        ),
        "formal_diagnostics_absent": manifest.get("diagnostics", []) == [],
        "three_experiment_profiles_complete": set(profiles) == expected_profile_keys,
        "three_experiment_metrics_finite": all(
            math.isfinite(float(profile[metric]))
            for profile in profiles.values()
            for metric in (
                "time_transfer_std_ps",
                "beamforming_std_ps",
                "residual_clock_rate_rmse_ppm",
            )
        ),
        "three_experiment_capture_success": all(
            float(profile["acquisition_failure_rate"])
            <= float(thresholds["max_acquisition_failure_rate"])
            for profile in profiles.values()
        ),
        "figure_manifest_count": int(manifest.get("figure_count", -1))
        == 2 * expected_pairs,
        "figure_files_present": len(expected_files) == 2 * expected_pairs
        and all(path.is_file() and path.stat().st_size > 0 for path in expected_files),
        "lut_reduces_systematic_rmse": float(lut["corrected_rmse_ps"])
        < float(lut["raw_rmse_ps"]),
        "lut_uses_held_out_grid": int(lut["validation_points"]) > 0
        and "disjoint" in str(lut["validation_policy"]),
        "monte_carlo_estimator_order": qls_lut_rmse
        < float(monte_carlo["qls_rmse_ps"])
        < float(monte_carlo["integer_peak_rmse_ps"]),
        "qls_lut_tracks_crlb": qls_lut_rmse
        <= float(thresholds["max_qls_lut_crlb_ratio"]) * crlb_std,
        "clock_residual": abs(float(clock["final_residual_after_update_ps"]))
        <= float(thresholds["max_clock_residual_ps"]),
        "clock_rate_control_uses_estimate": math.isclose(
            float(clock["applied_fractional_frequency_correction"]),
            float(clock["estimated_fractional_frequency_offset"]),
            rel_tol=0.0,
            abs_tol=1e-18,
        ),
        "frequency_residual": residual_frequency_hz <= allowed_frequency_hz,
        "oscillator_control_uses_estimate": math.isclose(
            float(frequency["applied_oscillator_correction_hz"]),
            float(frequency["final_tracked_estimate_hz"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "time_control_reduces_plant_error": abs(
            float(plant["residual_clock_offset_ps"])
        )
        < abs(float(plant["raw_clock_offset_ps"])),
        "frequency_control_reduces_plant_error": abs(
            float(plant["residual_frequency_offset_hz"])
        )
        < abs(float(plant["raw_frequency_offset_hz"])),
        "rx_arrival_aligned": abs(
            float(beamforming["full_sync"]["arrival_difference_ps"])
        )
        <= float(thresholds["max_arrival_error_ps"]),
        "full_sync_power_improves": float(
            beamforming["full_sync"]["combined_power"]
        )
        > float(beamforming["unsynchronized"]["combined_power"]),
        "full_sync_coherent_gain": float(
            beamforming["full_sync"]["gain_vs_incoherent_sum_db"]
        )
        >= float(thresholds["min_full_sync_gain_vs_incoherent_db"]),
        "full_sync_ideal_loss": float(
            beamforming["full_sync"]["normalized_ideal_loss_db"]
        )
        <= float(thresholds["max_full_sync_ideal_loss_db"]),
        "asymmetry_half_relation": half_relation_error
        <= float(thresholds["max_asymmetry_half_relation_error_ps"]),
        "reference_phase_sensitivity": float(
            sensitivity["max_holdover_timing_rmse_ps"]
        )
        >= float(thresholds["min_reference_phase_sensitivity_ps"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "output_dir": str(root.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify generated demo results")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--thresholds", type=Path, default=_DEFAULT_THRESHOLDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify_output(args.output_dir, args.thresholds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
