"""审查修复：导频、反馈延迟和数据共用本振相位轨迹测试。"""

from __future__ import annotations

import unittest

import numpy as np

from config import BeamformingConfig, PhaseFeedbackConfig, WaveformConfig
from oscillator_model import LocalOscillator
from phase_sync import simulate_channel_feedback


class PhaseFeedbackLoopTests(unittest.TestCase):
    """验证相位反馈由接收导频产生，并体现反馈陈旧误差。"""

    def test_zero_gap_static_channel_recovers_data_epoch_phase(self) -> None:
        result = simulate_channel_feedback(
            WaveformConfig(pulse_duration_s=2e-6),
            BeamformingConfig(),
            PhaseFeedbackConfig(
                pilot_symbols=256,
                snr_db=100.0,
                feedback_delay_s=0.0,
                phase_quantization_bits=None,
            ),
            LocalOscillator(frequency_offset_hz=0.0, initial_phase_rad=0.7),
            residual_clock_offset_s=0.0,
            pilot_epoch_s=0.2,
            rng=np.random.default_rng(91),
        )

        self.assertLess(np.max(np.abs(result.phase_error_at_data_rad)), 1e-5)

    def test_feedback_gap_exposes_residual_frequency_phase_drift(self) -> None:
        residual_hz = 100.0
        gap_s = 1e-3
        result = simulate_channel_feedback(
            WaveformConfig(pulse_duration_s=2e-6),
            BeamformingConfig(),
            PhaseFeedbackConfig(
                pilot_symbols=512,
                snr_db=120.0,
                feedback_delay_s=gap_s,
                phase_quantization_bits=None,
            ),
            LocalOscillator(frequency_offset_hz=residual_hz, initial_phase_rad=0.3),
            residual_clock_offset_s=0.0,
            pilot_epoch_s=0.2,
            rng=np.random.default_rng(92),
        )

        expected_ap1_error = -2.0 * np.pi * residual_hz * gap_s
        self.assertAlmostEqual(
            result.phase_error_at_data_rad[1], expected_ap1_error, places=5
        )

    def test_phase_feedback_is_reproducible_with_fixed_seed(self) -> None:
        arguments = dict(
            waveform_config=WaveformConfig(pulse_duration_s=2e-6),
            beamforming_config=BeamformingConfig(),
            feedback_config=PhaseFeedbackConfig(),
            oscillator=LocalOscillator(frequency_offset_hz=0.5, initial_phase_rad=0.7),
            residual_clock_offset_s=2e-12,
            pilot_epoch_s=0.2,
        )
        first = simulate_channel_feedback(
            **arguments, rng=np.random.default_rng(93)
        )
        second = simulate_channel_feedback(
            **arguments, rng=np.random.default_rng(93)
        )

        np.testing.assert_array_equal(
            first.feedback_channel_estimates, second.feedback_channel_estimates
        )


if __name__ == "__main__":
    unittest.main()
