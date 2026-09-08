"""阶段 6：参考信号频偏估计、跟踪和补偿测试。"""

from __future__ import annotations

import unittest

import numpy as np

from config import FrequencySyncConfig
from frequency_sync import (
    compensate_frequency,
    estimate_frequency_offset,
    simulate_frequency_reference,
    update_frequency_tracker,
)


class FrequencyEstimationTests(unittest.TestCase):
    """验证分段相位最小二乘频偏估计。"""

    def _config(
        self,
        cfo_hz: float,
        initial_phase_rad: float = 0.7,
        sample_clock_offset_fraction: float = 0.0,
    ) -> FrequencySyncConfig:
        return FrequencySyncConfig(
            sample_rate_hz=2e6,
            reference_frequency_hz=100e3,
            observation_duration_s=20e-3,
            segment_duration_s=0.5e-3,
            cfo_hz=cfo_hz,
            sample_clock_offset_fraction=sample_clock_offset_fraction,
            initial_phase_rad=initial_phase_rad,
            snr_db=40.0,
            tracker_alpha=0.8,
        )

    def test_estimates_positive_carrier_frequency_offset(self) -> None:
        config = self._config(cfo_hz=250.0)
        observation = simulate_frequency_reference(config, np.random.default_rng(61))

        estimate = estimate_frequency_offset(observation.samples, config)

        self.assertLess(abs(estimate.frequency_offset_hz - 250.0), 0.2)
        self.assertGreater(estimate.segment_times_s.size, 10)

    def test_frequency_estimate_is_independent_of_initial_phase(self) -> None:
        estimates = []
        for phase in (-2.5, 0.0, 2.3):
            config = self._config(cfo_hz=-175.0, initial_phase_rad=phase)
            observation = simulate_frequency_reference(config, np.random.default_rng(62))
            estimates.append(estimate_frequency_offset(observation.samples, config))

        for estimate in estimates:
            self.assertLess(abs(estimate.frequency_offset_hz + 175.0), 0.2)

    def test_compensation_reduces_residual_phase_slope(self) -> None:
        config = self._config(cfo_hz=400.0)
        observation = simulate_frequency_reference(config, np.random.default_rng(63))
        estimate = estimate_frequency_offset(observation.samples, config)

        compensated = compensate_frequency(
            observation.samples,
            frequency_offset_hz=estimate.frequency_offset_hz,
            sample_rate_hz=config.sample_rate_hz,
        )
        residual = estimate_frequency_offset(compensated, config)

        self.assertLess(abs(residual.frequency_offset_hz), abs(estimate.frequency_offset_hz) / 100.0)

    def test_sampling_clock_offset_appears_as_predictable_reference_frequency_error(self) -> None:
        epsilon = 50e-6
        config = self._config(cfo_hz=0.0, sample_clock_offset_fraction=epsilon)
        observation = simulate_frequency_reference(config, np.random.default_rng(64))
        estimate = estimate_frequency_offset(observation.samples, config)
        expected_hz = config.reference_frequency_hz / (1.0 + epsilon) - config.reference_frequency_hz

        self.assertLess(abs(estimate.frequency_offset_hz - expected_hz), 0.2)


class FrequencyTrackerTests(unittest.TestCase):
    """验证一阶指数频率跟踪器。"""

    def test_exponential_tracker_uses_configured_alpha(self) -> None:
        tracked = update_frequency_tracker(
            previous_hz=100.0,
            measurement_hz=110.0,
            alpha=0.8,
        )

        self.assertAlmostEqual(tracked, 102.0, places=12)

    def test_tracker_reduces_zero_mean_measurement_noise_after_settling(self) -> None:
        rng = np.random.default_rng(65)
        measurements = 200.0 + rng.normal(scale=4.0, size=200)
        tracked = np.empty(measurements.shape)
        state = measurements[0]
        for index, measurement in enumerate(measurements):
            state = update_frequency_tracker(state, float(measurement), alpha=0.9)
            tracked[index] = state

        self.assertLess(float(np.std(tracked[50:] - 200.0)), float(np.std(measurements[50:] - 200.0)))


if __name__ == "__main__":
    unittest.main()
