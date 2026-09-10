"""持续漂移、掉帧保持、频偏突变和重新捕获的集成验证。"""
import unittest
from config import (WaveformConfig, ClockPlantConfig, OscillatorConfig, FrequencySyncConfig,
                    BeamformingConfig, PhaseFeedbackConfig, JointTrackingConfig)
from joint_sync import run_joint_tracking
from lut_calibration import build_qls_lut


class JointSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = WaveformConfig(pulse_duration_s=2e-6)
        cls.lut = build_qls_lut(cls.w, 401)

    def run_case(self, cfg):
        return run_joint_tracking(self.w, self.lut, ClockPlantConfig(), OscillatorConfig(),
            FrequencySyncConfig(cfo_hz=600., observation_duration_s=400e-6, segment_duration_s=20e-6,
                                snr_db=40.),
            BeamformingConfig(ap1_propagation_delay_s=75e-9),
            PhaseFeedbackConfig(snr_db=45.), cfg)

    def test_nonzero_cfo_and_clock_skew_lock_with_unequal_paths(self):
        result = self.run_case(JointTrackingConfig(rounds=12))
        self.assertEqual(result['summary']['acquisition_failure_rate'], 0.)
        self.assertTrue(all(row['locked'] for row in result['rows'][3:]))
        self.assertLess(result['summary']['steady_max_arrival_error_ps'], 20.)
        self.assertGreater(result['summary']['steady_min_gain_db'], 2.9)

    def test_dropout_with_frequency_step_is_scored_and_recovers(self):
        result = self.run_case(JointTrackingConfig(rounds=12, dropout_rounds=(5, 6),
                                                   frequency_step_round=5, frequency_step_hz=100.))
        self.assertFalse(result['rows'][5]['acquisition_success'])
        self.assertFalse(result['rows'][5]['locked'])
        self.assertTrue(result['rows'][8]['locked'])
        self.assertGreaterEqual(result['summary']['lock_loss_count'], 1)
        self.assertGreaterEqual(result['summary']['reacquisition_count'], 1)

    def test_joint_trajectory_is_reproducible(self):
        cfg = JointTrackingConfig(rounds=5)
        self.assertEqual(self.run_case(cfg), self.run_case(cfg))
