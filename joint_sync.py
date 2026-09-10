"""持续运行的时间、频率、RX 延时/相位联合同步及保持期验证。"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from acquisition import AcquisitionError, sync_burst
from beamforming import combine_two_ap
from clock_model import LocalClock
from config import (BeamformingConfig, ChannelConfig, ClockPlantConfig, FrequencySyncConfig,
                    JointTrackingConfig, OscillatorConfig, PhaseFeedbackConfig, TwoWayConfig,
                    WaveformConfig)
from frequency_sync import estimate_frequency_offset, simulate_frequency_reference
from models import QLSCalibration
from oscillator_model import LocalOscillator
from phase_sync import complex_los_channel, compute_phase_weights, simulate_channel_feedback
from two_way_sync import estimate_two_way, simulate_two_way_exchange
from waveforms import generate_two_tone


def run_joint_tracking(waveform_config: WaveformConfig, calibration: QLSCalibration,
                       clock_config: ClockPlantConfig, oscillator_config: OscillatorConfig,
                       frequency_config: FrequencySyncConfig, beamforming: BeamformingConfig,
                       phase_config: PhaseFeedbackConfig, config: JointTrackingConfig,
                       schedule: TwoWayConfig | None = None) -> dict:
    """逐周期执行观测及控制，并在下一次同步前采样检验相干保持情况。

    原始物理值只负责生成 IQ 和评分。速率估计从本地时间戳和已发控制命令
    重建，捕获失败不会清零误差或跳过评分，而是沿用上次命令进入保持状态。
    """
    schedule = schedule or TwoWayConfig()
    seeds = np.random.SeedSequence(config.seed).spawn(4)
    drift_rng, timing_rng, frequency_rng, phase_rng = [np.random.default_rng(s) for s in seeds]
    clock = LocalClock(offset_s=clock_config.initial_offset_s,
                       fractional_frequency_offset=clock_config.fractional_frequency_offset)
    oscillator = LocalOscillator(oscillator_config.frequency_offset_hz, oscillator_config.initial_phase_rad)
    link = ChannelConfig(propagation_delay_s=50e-9, snr_db=36.)
    waveform = generate_two_tone(waveform_config)
    burst, _, _ = sync_burst(waveform_config)
    probe_time = 2*(len(burst)/waveform_config.sample_rate_hz + 2*phase_config.alignment_window_margin_s)
    reserved = (2*waveform_config.pulse_duration_s + schedule.processing_delay_s
                + frequency_config.observation_duration_s + probe_time
                + 2*phase_config.feedback_delay_s + phase_config.pilot_symbols/waveform_config.sample_rate_hz)
    if reserved >= config.interval_s*.9:
        raise ValueError("同步交换、频率观测及反馈不能装入指定周期")
    rate_epochs: list[float] = []
    raw_offsets: list[float] = []
    weights = np.ones(2, dtype=np.complex128)
    if beamforming.normalization == 'total_fixed':
        weights /= np.sqrt(2)
    tx_correction = 0.
    rows = []
    h0 = complex_los_channel(beamforming.ap0_amplitude, beamforming.ap0_propagation_delay_s,
                            waveform_config.carrier_frequency_hz, beamforming.ap0_channel_phase_rad)
    h1 = complex_los_channel(beamforming.ap1_amplitude, beamforming.ap1_propagation_delay_s,
                            waveform_config.carrier_frequency_hz, beamforming.ap1_channel_phase_rad)
    for index in range(config.rounds):
        epoch = .01 + index*config.interval_s
        if index:
            rate_change = config.clock_rate_walk_std*np.sqrt(config.interval_s)*drift_rng.standard_normal()
            # Changes to physical oscillators preserve the time/phase at the event.
            clock.offset_s -= rate_change*epoch
            clock.fractional_frequency_offset += rate_change
            frequency_change = config.oscillator_walk_std_hz*np.sqrt(config.interval_s)*drift_rng.standard_normal()
        else:
            frequency_change = 0.
        if config.frequency_step_round == index:
            frequency_change += config.frequency_step_hz
        oscillator.initial_phase_rad -= 2*np.pi*frequency_change*epoch
        oscillator.frequency_offset_hz += frequency_change
        failure = ''
        data_epoch = epoch
        time_command = 0.
        frequency_command = 0.
        try:
            if index in config.dropout_rounds:
                raise AcquisitionError('scheduled_packet_loss')
            observation = simulate_two_way_exchange(waveform_config,
                replace(schedule, tx1_local_time_s=epoch), LocalClock(), clock,
                link, link, calibration, timing_rng)
            estimate = estimate_two_way(observation)
            reference_epoch = .5*(observation.t_rx0_s+observation.t_tx0_s)
            raw_estimate = (estimate.ap1_offset_estimate_s-clock.time_correction_s
                            + clock.fractional_frequency_correction*reference_epoch)
            rate_estimate = clock.fractional_frequency_correction
            rate_epochs.append(reference_epoch)
            raw_offsets.append(raw_estimate)
            rate_epochs = rate_epochs[-config.rate_fit_window:]
            raw_offsets = raw_offsets[-config.rate_fit_window:]
            if len(rate_epochs) >= 2:
                t = np.asarray(rate_epochs)-np.mean(rate_epochs)
                offsets = np.asarray(raw_offsets)-np.mean(raw_offsets)
                rate_estimate = float(np.dot(t, offsets)/np.dot(t, t))
            effective_epoch = (clock.true_time_for_reading(observation.t_rx1_s)
                               + waveform_config.pulse_duration_s/clock.effective_rate)
            observable_end = (observation.t_tx0_s + max(0., estimate.symmetric_propagation_delay_s)
                              + waveform_config.pulse_duration_s)
            predicted_residual = (estimate.ap1_offset_estimate_s
                + (rate_estimate-clock.fractional_frequency_correction)*(observable_end-reference_epoch))
            time_command = -predicted_residual
            clock.apply_time_correction(time_command)
            clock.apply_frequency_correction(rate_estimate, effective_true_time_s=effective_epoch)
            reference = replace(frequency_config, cfo_hz=oscillator.residual_frequency_offset_hz,
                sample_clock_offset_fraction=clock.effective_rate-1.,
                initial_phase_rad=oscillator.phase_at(effective_epoch))
            samples = simulate_frequency_reference(reference, frequency_rng)
            frequency_estimate = estimate_frequency_offset(samples.samples, reference)
            frequency_end = effective_epoch+reference.observation_duration_s
            frequency_command = frequency_estimate.frequency_offset_hz
            oscillator.apply_frequency_correction(oscillator.frequency_correction_hz+frequency_command,
                                                   effective_time_s=frequency_end)
            pilot_epoch = (frequency_end+probe_time+phase_config.feedback_delay_s
                           + phase_config.pilot_symbols/(2*waveform_config.sample_rate_hz))
            feedback = simulate_channel_feedback(waveform_config, beamforming, phase_config,
                oscillator, residual_clock_offset_s=clock.residual_offset_at(pilot_epoch),
                pilot_epoch_s=pilot_epoch, rng=phase_rng, calibration=calibration)
            tx_correction = feedback.tx_time_correction_s
            weights = compute_phase_weights(feedback.feedback_channel_estimates, beamforming.normalization)
            data_epoch = feedback.data_epoch_s
        except AcquisitionError as exc:
            failure = str(exc)
        if data_epoch >= epoch+config.interval_s:
            raise ValueError("数据反馈到达晚于下一同步周期")
        delays_ps, phases, gains, clock_ps = [], [], [], []
        for time_s in np.linspace(data_epoch, epoch+config.interval_s-1e-6, config.holdover_points):
            residual_clock = clock.residual_offset_at(float(time_s))
            channel_phase = phase_config.channel_phase_rate_rad_per_s*time_s
            arrival_error = (beamforming.ap1_propagation_delay_s-beamforming.ap0_propagation_delay_s
                             - residual_clock+tx_correction)
            if abs(arrival_error) >= waveform_config.pulse_duration_s-1/waveform_config.sample_rate_hz:
                delays_ps.append(abs(arrival_error)*1e12)
                clock_ps.append(abs(residual_clock)*1e12)
                # Disjoint bursts have no coherent cross term; phase is undefined.
                phases.append(np.pi)
                gains.append(0.)
                continue
            result = combine_two_ap(waveform.samples, waveform_config.sample_rate_hz,
                [beamforming.ap0_propagation_delay_s,
                 beamforming.ap1_propagation_delay_s-residual_clock+tx_correction],
                [h0, h1*np.exp(1j*(oscillator.phase_at(float(time_s))+channel_phase))],
                weights, oscillator.residual_frequency_offset_hz+phase_config.channel_phase_rate_rad_per_s/(2*np.pi),
                'joint_holdover')
            delays_ps.append(abs(result.residual_arrival_difference_s)*1e12)
            phases.append(abs(result.residual_phase_difference_rad))
            gains.append(result.metrics.gain_vs_incoherent_sum_db)
            clock_ps.append(abs(residual_clock)*1e12)
        locked = (max(delays_ps) <= config.max_arrival_error_s*1e12
                  and max(clock_ps) <= config.max_arrival_error_s*1e12
                  and max(phases) <= config.max_phase_error_rad and min(gains) >= config.min_gain_db)
        rows.append(dict(round=index, epoch_s=epoch, acquisition_success=not bool(failure),
            failure_reason=failure, locked=locked, time_command_ps=time_command*1e12,
            clock_rate_command=clock.fractional_frequency_correction,
            frequency_increment_hz=frequency_command, frequency_command_hz=oscillator.frequency_correction_hz,
            residual_frequency_hz=oscillator.residual_frequency_offset_hz,
            tx_delay_command_ps=tx_correction*1e12, max_clock_error_ps=max(clock_ps),
            max_arrival_error_ps=max(delays_ps), max_phase_error_deg=np.rad2deg(max(phases)),
            min_gain_db=min(gains)))
    states = np.array([r['locked'] for r in rows], dtype=bool)
    first = int(np.flatnonzero(states)[0]) if np.any(states) else None
    longest = run = 0
    for locked in states:
        run = run+1 if locked else 0
        longest = max(longest, run)
    return dict(rows=rows, summary=dict(rounds=config.rounds, duration_s=config.rounds*config.interval_s,
        holdover_points=config.holdover_points, first_lock_round=first,
        lock_loss_count=int(np.sum(states[:-1] & ~states[1:])),
        reacquisition_count=max(0, int(np.sum(~states[:-1] & states[1:]))-(0 if states[0] else 1)),
        longest_locked_duration_s=longest*config.interval_s,
        locked_fraction=float(np.mean(states)),
        acquisition_failure_rate=float(np.mean([not r['acquisition_success'] for r in rows])),
        steady_max_arrival_error_ps=max(r['max_arrival_error_ps'] for r in rows[2:]),
        steady_max_clock_error_ps=max(r['max_clock_error_ps'] for r in rows[2:]),
        steady_max_phase_error_deg=max(r['max_phase_error_deg'] for r in rows[2:]),
        steady_min_gain_db=min(r['min_gain_db'] for r in rows[2:])))
