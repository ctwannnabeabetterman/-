"""不同目标路径下的波形对齐与 IQ 相位判据回归。"""
import unittest
import numpy as np
from beamforming import combine_two_ap
from config import BeamformingConfig, PhaseFeedbackConfig, WaveformConfig
from oscillator_model import LocalOscillator
from phase_sync import simulate_channel_feedback, compute_phase_weights
from waveforms import generate_two_tone


class RxAlignmentTests(unittest.TestCase):
    def test_twenty_five_ns_path_difference_is_estimated_and_compensated(self):
        w = WaveformConfig(pulse_duration_s=2e-6)
        b = BeamformingConfig(ap0_propagation_delay_s=50e-9, ap1_propagation_delay_s=75e-9)
        f = simulate_channel_feedback(w, b, PhaseFeedbackConfig(snr_db=100., phase_quantization_bits=None),
            LocalOscillator(initial_phase_rad=1.2), residual_clock_offset_s=3e-12,
            pilot_epoch_s=.1, rng=np.random.default_rng(91))
        self.assertLess(abs(f.measured_arrival_difference_s-(25e-9-3e-12)), 1e-12)
        x = generate_two_tone(w).samples
        args = (f.true_effective_channels_at_data, compute_phase_weights(f.feedback_channel_estimates), 0.)
        bad = combine_two_ap(x, w.sample_rate_hz, [50e-9, 75e-9-3e-12], *args, 'uncorrected')
        good = combine_two_ap(x, w.sample_rate_hz,
            [50e-9, 75e-9-3e-12+f.tx_time_correction_s], *args, 'corrected')
        self.assertGreater(abs(bad.residual_phase_difference_rad), 3.)
        self.assertLess(bad.metrics.gain_vs_incoherent_sum_db, -10.)
        self.assertLess(abs(good.residual_arrival_difference_s), 1e-12)
        self.assertGreater(good.metrics.gain_vs_incoherent_sum_db, 3.)

    def test_data_cannot_precede_end_of_phase_pilot(self):
        w = WaveformConfig(pulse_duration_s=2e-6)
        f = simulate_channel_feedback(w, BeamformingConfig(), PhaseFeedbackConfig(feedback_delay_s=0.),
            LocalOscillator(), residual_clock_offset_s=0., pilot_epoch_s=.1,
            rng=np.random.default_rng(92))
        self.assertGreaterEqual(f.data_epoch_s, .1+1024/(2*w.sample_rate_hz))
