"""论文 Fig. 12 形式的三配置软件实验；硬件抖动为显式假设而非实测拟合。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import platform

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from beamforming import combine_two_ap
from channel import add_awgn, apply_fractional_delay
from clock_model import LocalClock
from config import BeamformingConfig, ChannelConfig, PhaseFeedbackConfig, TwoWayConfig, WaveformConfig
from crlb import crlb_for_waveform
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from lut_calibration import correct_qls_fraction, load_or_build_lut, wrap_fractional_sample
from models import QLSCalibration
from oscillator_model import LocalOscillator
from phase_sync import compute_phase_weights, simulate_channel_feedback
from plotting import configure_paper_style
from results_io import write_csv_columns, write_json
from two_way_sync import estimate_two_way, simulate_two_way_exchange
from waveforms import generate_two_tone


@dataclass(frozen=True)
class Figure12Config:
    """图 12 对应实验参数；所有时间为秒，抖动数值为未标定的软件假设。"""

    trials: int = 1000
    snr_db_values: tuple[float, ...] = tuple(float(x) for x in range(6, 37, 3))
    seed: int = 120923
    interval_s: float = 50e-3
    processing_delay_s: float = 20e-6
    initial_offset_s: float = 100e-9
    start_epoch_s: float = 1e-3
    pilot_after_exchange_s: float = 1e-3
    feedback_delay_s: float = 100e-6
    pilot_symbols: int = 256
    pilot_snr_db: float = 40.
    readout_snr_db: float = 60.
    readout_jitter_s: float = 5e-12
    wireless_reference_jitter_s: float = 10e-12
    cabled_delay_s: float = 0.9144 / 2e8
    wireless_delay_s: float = 0.90 / 299792458.
    snr_uncertainty_db: float = 3.

    def __post_init__(self) -> None:
        if self.trials < 2 or not self.snr_db_values:
            raise ValueError('至少两个有效测量，SNR 轴不可为空')
        values = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(self.snr_db_values)):
            raise ValueError('实验参数必须有限')
        if min(self.readout_jitter_s, self.wireless_reference_jitter_s) < 0:
            raise ValueError('抖动标准差不能为负')
        if self.feedback_delay_s < 0 or self.interval_s <= 0:
            raise ValueError('反馈延迟须非负且同步周期须为正')


def sample_statistics(values_s: np.ndarray) -> dict[str, float]:
    """将秒制样本汇总为 ps；precision 使用 ddof=1 标准差，不混用 RMSE。"""
    values = np.asarray(values_s, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError('统计至少需要两个有限一维样本')
    return {'std_ps': float(np.std(values, ddof=1) * 1e12),
            'mean_ps': float(np.mean(values) * 1e12),
            'rmse_ps': float(np.sqrt(np.mean(values**2)) * 1e12)}


def measure_interarrival(reference: np.ndarray, secondary: np.ndarray,
                         waveform: WaveformConfig, lut: QLSCalibration) -> float:
    """由两路观测互相关测量 AP1 相对 AP0 的到达差，正值表示 AP1 晚到。"""
    estimate = estimate_delay(
        fft_matched_filter(secondary, reference), waveform.sample_rate_hz,
        DelaySearchGate(center_s=0., half_width_s=2.5 / waveform.sample_rate_hz))
    if not estimate.qls_valid:
        raise RuntimeError('波束赋形读出互相关峰不在公开的零时差搜索门内')
    correction = float(wrap_fractional_sample(
        correct_qls_fraction(estimate.fractional_offset_samples, lut)
        - estimate.fractional_offset_samples))
    return float(estimate.delay_s + correction / waveform.sample_rate_hz)


def run_precision_case(sync_config: WaveformConfig, data_config: WaveformConfig,
                       sync_lut: QLSCalibration, data_lut: QLSCalibration,
                       config: Figure12Config, link_delay_s: float,
                       reference_jitter_s: float, snr_db: float, seed: int) -> dict[str, np.ndarray]:
    """持续运行 N+1 轮并丢弃捕获首轮；控制只读取时间戳或导频估计。

    参考抖动在同步与数据两个时刻独立抽样，是白时间抖动等效模型。
    readout_jitter_s 是独立的差分读出抖动，只作用于观测支路。
    """
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    timing_rng, jitter_rng, phase_rng, readout_rng = streams
    clock = LocalClock(offset_s=config.initial_offset_s)
    link = ChannelConfig(propagation_delay_s=link_delay_s, snr_db=snr_db)
    beam_config = BeamformingConfig(ap0_propagation_delay_s=0., ap1_propagation_delay_s=0.)
    feedback_config = PhaseFeedbackConfig(pilot_symbols=config.pilot_symbols,
        snr_db=config.pilot_snr_db, feedback_delay_s=config.feedback_delay_s,
        phase_quantization_bits=12)
    data = generate_two_tone(data_config)
    output = {name: np.empty(config.trials) for name in (
        'correction_s', 'interarrival_s', 'residual_clock_s', 'gain_db', 'measured_snr_db')}
    for index in range(config.trials + 1):
        epoch = config.start_epoch_s + index * config.interval_s
        clock.offset_s = config.initial_offset_s + reference_jitter_s * jitter_rng.standard_normal()
        schedule = TwoWayConfig(tx1_local_time_s=epoch,
            processing_delay_s=config.processing_delay_s,
            coarse_up_delay_s=0., coarse_down_delay_s=0.)
        observation = simulate_two_way_exchange(sync_config, schedule, LocalClock(), clock,
                                                link, link, sync_lut, timing_rng)
        estimate = estimate_two_way(observation)
        clock.apply_time_correction(estimate.clock_correction_s)
        # 仿真 plant 推进到新的参考抖动样本；不以真值构造控制命令。
        clock.offset_s = config.initial_offset_s + reference_jitter_s * jitter_rng.standard_normal()
        pilot_epoch = observation.t_tx0_s + config.pilot_after_exchange_s
        residual = clock.residual_offset_at(pilot_epoch)
        oscillator = LocalOscillator(frequency_offset_hz=0.,
                                      initial_phase_rad=float(phase_rng.uniform(-np.pi, np.pi)))
        feedback = simulate_channel_feedback(data_config, beam_config, feedback_config,
            oscillator, residual_clock_offset_s=residual, pilot_epoch_s=pilot_epoch, rng=phase_rng)
        weights = compute_phase_weights(feedback.feedback_channel_estimates)
        combined = combine_two_ap(data.samples, data_config.sample_rate_hz,
            np.array([0., -residual]), feedback.true_effective_channels_at_data,
            weights, 0., 'fig12_full_sync')
        # 差分仪器抖动及 AWGN 作用于实际已生成的两路数据 IQ。
        difference_jitter = config.readout_jitter_s * readout_rng.standard_normal()
        rx0 = combined.ap0_contribution
        rx1 = apply_fractional_delay(combined.ap1_contribution, difference_jitter,
                                     data_config.sample_rate_hz)
        mask = np.abs(rx0) > 0.05 * np.max(np.abs(rx0))
        rx0 = add_awgn(rx0, config.readout_snr_db, mask, readout_rng).samples
        rx1 = add_awgn(rx1, config.readout_snr_db, mask, readout_rng).samples
        arrival = measure_interarrival(rx0, rx1, data_config, data_lut)
        if index:
            output['correction_s'][index - 1] = estimate.clock_correction_s
            output['interarrival_s'][index - 1] = arrival
            output['residual_clock_s'][index - 1] = residual
            output['gain_db'][index - 1] = combined.metrics.gain_vs_incoherent_sum_db
            output['measured_snr_db'][index - 1] = observation.up_measurement.measured_snr_db
    return output


def run_figure12(output_dir: str | Path, config: Figure12Config,
                 lut_grid_points: int = 2001) -> dict:
    """保存三联图、逐轮测量、汇总 CSV 与可复现配置 JSON。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sync = WaveformConfig()
    data = WaveformConfig(tone_separation_hz=50e6, pulse_duration_s=1e-6,
                           carrier_frequency_hz=1.2e9)
    sync_lut = load_or_build_lut(sync, output_dir / 'lut_cache', lut_grid_points)
    data_lut = load_or_build_lut(data, output_dir / 'lut_cache', lut_grid_points)
    cases = [('cabled', 'Cabled time-frequency transfer', config.cabled_delay_s, 0.),
             ('wireless_time', 'Wireless time transfer', config.wireless_delay_s, 0.),
             ('wireless_time_frequency', 'Wireless time-frequency transfer',
              config.wireless_delay_s, config.wireless_reference_jitter_s)]
    rows = []
    seeds = np.random.SeedSequence(config.seed).spawn(len(cases) * len(config.snr_db_values))
    for case_index, (key, title, delay, jitter) in enumerate(cases):
        for snr_index, snr in enumerate(config.snr_db_values):
            seed = int(seeds[case_index * len(config.snr_db_values) + snr_index].generate_state(1)[0])
            samples = run_precision_case(sync, data, sync_lut, data_lut, config,
                                          delay, jitter, snr, seed)
            time_stats = sample_statistics(samples['correction_s'])
            beam_stats = sample_statistics(samples['interarrival_s'])
            nominal_crlb = crlb_for_waveform(generate_two_tone(sync), sync.tone_separation_hz, snr).std_s
            best_crlb = nominal_crlb * 10. ** (-config.snr_uncertainty_db / 20.)
            row = dict(case=key, snr_db=snr, time_std_ps=time_stats['std_ps'],
                beamforming_std_ps=beam_stats['std_ps'], time_mean_ps=time_stats['mean_ps'],
                beamforming_mean_ps=beam_stats['mean_ps'], crlb_nominal_ps=nominal_crlb * 1e12,
                crlb_best_case_ps=best_crlb * 1e12,
                mean_gain_db=10. * np.log10(np.mean(10. ** (samples['gain_db'] / 10.))),
                measured_snr_mean_db=float(np.mean(samples['measured_snr_db'])))
            rows.append(row)
            write_csv_columns(output_dir / 'samples' / f'{key}_{snr:g}dB.csv', samples)
        print(f'Completed {key}: {config.trials} samples/SNR', flush=True)
    write_csv_columns(output_dir / 'figure12_summary.csv',
                      {name: [row[name] for row in rows] for name in rows[0]})
    configure_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), sharey=True, gridspec_kw={'wspace': 0.035})
    for i, (ax, (key, title, _, _)) in enumerate(zip(axes, cases)):
        subset = [r for r in rows if r['case'] == key]
        snr = [r['snr_db'] for r in subset]
        for name, style, label in [('time_std_ps', '-', 'Time transfer'),
                                  ('beamforming_std_ps', '--', 'Beamforming'),
                                  ('crlb_best_case_ps', ':', f'CRLB (+{config.snr_uncertainty_db:g} dB)')]:
            ax.semilogy(snr, [r[name] for r in subset], style, color='#0072BD',
                        linewidth=1.4, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'Pre-processing SNR (dB, simulated)\n({chr(97+i)})')
        ax.set_xlim(6, 36)
        ax.set_xticks([10, 20, 30])
        ax.set_ylim(1, 200)
        ax.grid(True, which='both', linestyle=':', alpha=0.35)
        ax.tick_params(which='both', direction='in')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Standard deviation (ps)')
    axes[2].legend(loc='upper right', frameon=False, fontsize=8)
    fig.suptitle('Fig. 12-style software experiment — illustrative jitter model', fontsize=11)
    fig.text(0.5, -0.025,
             f'Assumed: readout jitter {config.readout_jitter_s * 1e12:g} ps; '
             f'wireless reference jitter {config.wireless_reference_jitter_s * 1e12:g} ps/epoch. No measured hardware curves.',
             ha='center', fontsize=8)
    # 手动留白避免共享轴 tight_layout 警告；图注随 bbox_inches 一并保存。
    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.07, right=0.99)
    for suffix in ('png', 'pdf'):
        fig.savefig(output_dir / f'figure12_simulation.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    metadata = dict(config=asdict(config), sync_waveform=asdict(sync), data_waveform=asdict(data),
        python=platform.python_version(), numpy=np.__version__, lut_grid_points=lut_grid_points,
        statistic='sample standard deviation ddof=1, first acquisition excluded',
        reference_jitter_model='independent time-offset samples at sync and data epochs; assumed, not measured',
        snr_definition='configured active-sample complex AWGN SNR; not calibrated to paper RMS preprocessing SNR',
        crlb_definition='single-pulse AWGN CRLB evaluated at nominal SNR + 3 dB; not a hardware floor',
        readout_definition='200 MSa/s complex IQ cross-correlation; paper uses 20 GSa/s real oscilloscope samples',
        frequency_definition='ideal mean syntonization; no RF self-mixing circuit or frequency acquisition in this experiment',
        rows=rows)
    write_json(output_dir / 'figure12_metadata.json', metadata)
    return metadata


def main() -> None:
    """命令行一键生成可复现的图 12 软件实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/final_review'))
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=120923)
    parser.add_argument('--reference-jitter-ps', type=float, default=10.)
    parser.add_argument('--readout-jitter-ps', type=float, default=5.)
    args = parser.parse_args()
    run_figure12(args.output_dir, Figure12Config(trials=args.trials, seed=args.seed,
        wireless_reference_jitter_s=args.reference_jitter_ps * 1e-12,
        readout_jitter_s=args.readout_jitter_ps * 1e-12))


if __name__ == '__main__':
    main()
