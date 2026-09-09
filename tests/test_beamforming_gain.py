"""阶段 8：两 AP 相干合成、功率归一化和四状态比较测试。"""

from __future__ import annotations

import unittest

import numpy as np

from beamforming import combine_two_ap, simulate_four_sync_states
from config import BeamformingConfig, WaveformConfig
from models import BeamformingPlantState
from phase_sync import complex_los_channel, compute_phase_weights
from waveforms import generate_two_tone


class BeamformingPowerTests(unittest.TestCase):
    """验证相对单 AP、非相干功率和及理想合成指标。"""

    def test_equal_channels_per_ap_fixed_give_six_db_over_single_ap(self) -> None:
        samples = np.ones(1024, dtype=np.complex128)
        channels = np.ones(2, dtype=np.complex128)
        weights = compute_phase_weights(channels, "per_ap_fixed")

        result = combine_two_ap(
            samples,
            sample_rate_hz=1e6,
            arrival_delays_s=np.zeros(2),
            effective_channels=channels,
            weights=weights,
            residual_frequency_offset_hz=0.0,
            state_name="ideal",
        )

        self.assertAlmostEqual(result.metrics.gain_vs_single_ap_db, 6.020599913, places=8)
        self.assertAlmostEqual(result.metrics.gain_vs_incoherent_sum_db, 3.010299957, places=8)
        self.assertAlmostEqual(result.metrics.normalized_ideal_loss_db, 0.0, places=12)

    def test_equal_channels_total_fixed_give_three_db_over_single_ap(self) -> None:
        samples = np.ones(1024, dtype=np.complex128)
        channels = np.ones(2, dtype=np.complex128)
        weights = compute_phase_weights(channels, "total_fixed")

        result = combine_two_ap(
            samples,
            sample_rate_hz=1e6,
            arrival_delays_s=np.zeros(2),
            effective_channels=channels,
            weights=weights,
            residual_frequency_offset_hz=0.0,
            state_name="ideal",
        )

        self.assertAlmostEqual(result.metrics.gain_vs_single_ap_db, 3.010299957, places=8)
        self.assertAlmostEqual(result.metrics.gain_vs_incoherent_sum_db, 3.010299957, places=8)

    def test_opposite_channel_phases_without_weights_cancel(self) -> None:
        samples = np.ones(1024, dtype=np.complex128)

        result = combine_two_ap(
            samples,
            sample_rate_hz=1e6,
            arrival_delays_s=np.zeros(2),
            effective_channels=np.array([1.0, -1.0], dtype=np.complex128),
            weights=np.ones(2, dtype=np.complex128),
            residual_frequency_offset_hz=0.0,
            state_name="opposite",
        )

        self.assertLess(result.signal_power, 1e-24)


class FourStateBeamformingTests(unittest.TestCase):
    """验证四种状态只启用对应的估计补偿。"""

    def test_full_sync_removes_configured_time_frequency_and_phase_errors(self) -> None:
        waveform_config = WaveformConfig(pulse_duration_s=2e-6)
        waveform = generate_two_tone(waveform_config)
        config = BeamformingConfig(
            ap0_propagation_delay_s=50e-9,
            ap1_propagation_delay_s=50e-9,
            ap0_amplitude=1.0,
            ap1_amplitude=1.0,
            ap0_channel_phase_rad=0.2,
            ap1_channel_phase_rad=-0.6,
            normalization="per_ap_fixed",
        )
        plant_state = BeamformingPlantState(
            data_epoch_s=0.1,
            raw_clock_offset_s=12.25e-9,
            residual_clock_offset_s=0.0,
            raw_frequency_offset_hz=600.0,
            residual_frequency_offset_hz=0.0,
            raw_ap1_phase_rad=1.1,
            residual_ap1_phase_rad=1.1,
        )
        h0 = complex_los_channel(
            config.ap0_amplitude,
            config.ap0_propagation_delay_s,
            waveform_config.carrier_frequency_hz,
            config.ap0_channel_phase_rad,
        )
        h1 = complex_los_channel(
            config.ap1_amplitude,
            config.ap1_propagation_delay_s,
            waveform_config.carrier_frequency_hz,
            config.ap1_channel_phase_rad,
        ) * np.exp(1j * plant_state.residual_ap1_phase_rad)

        states = simulate_four_sync_states(
            waveform.samples,
            waveform_config,
            config,
            plant_state,
            channel_estimates=np.array([h0, h1], dtype=np.complex128),
        )

        self.assertEqual(
            set(states),
            {"unsynchronized", "time_only", "time_frequency", "full_sync"},
        )
        self.assertAlmostEqual(states["time_only"].residual_arrival_difference_s, 0.0, places=15)
        self.assertEqual(
            states["time_only"].residual_frequency_offset_hz,
            plant_state.raw_frequency_offset_hz,
        )
        self.assertAlmostEqual(states["time_frequency"].residual_frequency_offset_hz, 0.0)
        self.assertLess(abs(states["full_sync"].residual_phase_difference_rad), 1e-12)
        self.assertLess(states["full_sync"].metrics.normalized_ideal_loss_db, 0.01)
        self.assertGreater(
            states["full_sync"].signal_power,
            states["time_frequency"].signal_power,
        )


if __name__ == "__main__":
    unittest.main()
