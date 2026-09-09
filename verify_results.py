"""按版本化 JSON 阈值验证一组 Demo 输出。"""

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
    """验证数值趋势、残差、相干增益和全部图文件，返回结构化报告。"""

    root = Path(output_dir)
    thresholds = _read_json(Path(thresholds_path))
    summary = _read_json(root / "summary.json")
    manifest = _read_json(root / "manifest.json")
    expected_files = [root / relative for relative in manifest.get("figures", [])]
    expected_pairs = int(thresholds["expected_figure_pairs"])

    lut = summary["lut"]
    clock = summary["clock_tracking"]
    frequency = summary["frequency_sync"]
    beamforming = summary["beamforming"]
    monte_carlo = summary["monte_carlo_highest_snr"]
    plant = summary["state_separation"]["plant_at_data_epoch"]
    true_frequency_hz = abs(float(frequency["final_true_observed_offset_hz"]))
    residual_frequency_hz = abs(float(frequency["final_residual_hz"]))
    allowed_frequency_hz = thresholds["max_frequency_residual_fraction"] * max(
        true_frequency_hz, 1.0
    )

    checks = {
        "figure_manifest_count": int(manifest.get("figure_count", -1))
        == 2 * expected_pairs,
        "figure_files_present": len(expected_files) == 2 * expected_pairs
        and all(path.is_file() and path.stat().st_size > 0 for path in expected_files),
        "lut_reduces_systematic_rmse": float(lut["corrected_rmse_ps"])
        < float(lut["raw_rmse_ps"]),
        "monte_carlo_estimator_order": float(monte_carlo["qls_lut_rmse_ps"])
        < float(monte_carlo["qls_rmse_ps"])
        < float(monte_carlo["integer_peak_rmse_ps"]),
        "clock_residual": abs(float(clock["final_residual_after_update_ps"]))
        <= float(thresholds["max_clock_residual_ps"]),
        "frequency_residual": residual_frequency_hz <= allowed_frequency_hz,
        "time_control_reduces_plant_error": abs(
            float(plant["residual_clock_offset_ps"])
        )
        < abs(float(plant["raw_clock_offset_ps"])),
        "frequency_control_reduces_plant_error": abs(
            float(plant["residual_frequency_offset_hz"])
        )
        < abs(float(plant["raw_frequency_offset_hz"])),
        "clock_rate_control_uses_estimate": math.isclose(
            float(clock["applied_fractional_frequency_correction"]),
            float(clock["estimated_fractional_frequency_offset"]),
            rel_tol=0.0,
            abs_tol=1e-18,
        ),
        "oscillator_control_uses_estimate": math.isclose(
            float(frequency["applied_oscillator_correction_hz"]),
            float(frequency["final_tracked_estimate_hz"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "full_sync_power_improves": float(beamforming["full_sync"]["combined_power"])
        > float(beamforming["unsynchronized"]["combined_power"]),
        "full_sync_coherent_gain": float(
            beamforming["full_sync"]["gain_vs_incoherent_sum_db"]
        )
        >= float(thresholds["min_full_sync_gain_vs_incoherent_db"]),
        "full_sync_ideal_loss": float(
            beamforming["full_sync"]["normalized_ideal_loss_db"]
        )
        <= float(thresholds["max_full_sync_ideal_loss_db"]),
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
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=_DEFAULT_THRESHOLDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：打印机器可读报告，任一检查失败时返回非零退出码。"""

    args = _parser().parse_args(argv)
    report = verify_output(args.output_dir, args.thresholds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
