"""审查修复：plant 真值、控制状态和估计量的边界测试。"""

from __future__ import annotations

import inspect
import unittest
from dataclasses import fields

from beamforming import simulate_four_sync_states
from clock_model import LocalClock
from config import BeamformingConfig
from experiments import estimate_clock_frequency_offset, reconstruct_raw_offset_estimate
from models import ClockTrackingResult
from oscillator_model import LocalOscillator

import numpy as np


class ClockControlTests(unittest.TestCase):
    """验证频率控制使用正确量纲并在生效时保持时间连续。"""

    def test_clock_frequency_update_is_continuous_at_effective_epoch(self) -> None:
        clock = LocalClock(offset_s=100e-9, fractional_frequency_offset=0.25e-6)
        epoch_s = 2.0
        before = clock.read_time(epoch_s)

        clock.apply_frequency_correction(0.24e-6, effective_true_time_s=epoch_s)

        self.assertAlmostEqual(clock.read_time(epoch_s), before, places=15)
        self.assertAlmostEqual(clock.effective_rate - 1.0, 0.01e-6, places=15)

    def test_clock_rate_estimate_uses_only_observations_and_applied_commands(self) -> None:
        epochs = np.arange(5, dtype=np.float64)
        true_offset = 30e-9 + 0.2e-6 * epochs
        applied = np.empty(5, dtype=np.float64)
        estimated_residual = np.empty(5, dtype=np.float64)
        accumulated = 0.0
        for index, offset in enumerate(true_offset):
            estimated_residual[index] = offset + accumulated
            applied[index] = -estimated_residual[index]
            accumulated += applied[index]
        result = ClockTrackingResult(
            epoch_true_s=epochs,
            reference_epoch_s=epochs,
            raw_offset_s=np.full(5, np.nan),
            estimated_offset_s=estimated_residual,
            applied_correction_s=applied,
            residual_before_s=np.zeros(5),
            residual_after_s=np.zeros(5),
            random_walk_accumulated_s=np.zeros(5),
        )

        reconstructed = reconstruct_raw_offset_estimate(result)
        rate = estimate_clock_frequency_offset(result)

        np.testing.assert_allclose(reconstructed, true_offset, atol=1e-18)
        self.assertAlmostEqual(rate, 0.2e-6, places=15)

    def test_clock_rate_estimate_ignores_diagnostic_true_time(self) -> None:
        reference_epochs = np.arange(5, dtype=np.float64)
        observed_offset = 30e-9 + 0.2e-6 * reference_epochs
        base = ClockTrackingResult(
            epoch_true_s=reference_epochs,
            reference_epoch_s=reference_epochs,
            raw_offset_s=np.full(5, np.nan),
            estimated_offset_s=observed_offset,
            applied_correction_s=np.zeros(5),
            residual_before_s=np.zeros(5),
            residual_after_s=np.zeros(5),
            random_walk_accumulated_s=np.zeros(5),
        )
        altered_truth = ClockTrackingResult(
            epoch_true_s=reference_epochs * 2.0 + 100.0,
            reference_epoch_s=reference_epochs,
            raw_offset_s=base.raw_offset_s,
            estimated_offset_s=base.estimated_offset_s,
            applied_correction_s=base.applied_correction_s,
            residual_before_s=base.residual_before_s,
            residual_after_s=base.residual_after_s,
            random_walk_accumulated_s=base.random_walk_accumulated_s,
        )

        self.assertEqual(
            estimate_clock_frequency_offset(base),
            estimate_clock_frequency_offset(altered_truth),
        )


class OscillatorControlTests(unittest.TestCase):
    """验证本振 Hz 控制状态和载波相位连续性。"""

    def test_frequency_update_is_phase_continuous_and_changes_residual_hz(self) -> None:
        oscillator = LocalOscillator(
            frequency_offset_hz=600.0,
            initial_phase_rad=1.1,
        )
        epoch_s = 0.25
        before = oscillator.phase_at(epoch_s)

        oscillator.apply_frequency_correction(599.5, effective_time_s=epoch_s)

        self.assertAlmostEqual(oscillator.phase_at(epoch_s), before, places=12)
        self.assertAlmostEqual(oscillator.residual_frequency_offset_hz, 0.5)


class StateBoundaryTests(unittest.TestCase):
    """波束赋形接口不得接收裸时间/频率估计并在内部读取真值配置。"""

    def test_beamforming_api_accepts_explicit_plant_snapshot(self) -> None:
        parameters = inspect.signature(simulate_four_sync_states).parameters

        self.assertIn("plant_state", parameters)
        self.assertNotIn("estimated_time_correction_s", parameters)
        self.assertNotIn("estimated_frequency_offset_hz", parameters)

    def test_static_beamforming_config_cannot_store_dynamic_sync_truth(self) -> None:
        names = {field.name for field in fields(BeamformingConfig)}

        self.assertTrue(
            names.isdisjoint(
                {
                    "ap1_clock_offset_s",
                    "ap1_cfo_hz",
                    "ap1_initial_phase_rad",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
