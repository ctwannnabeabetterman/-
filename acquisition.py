"""有限窗口中的独立 DAC/ADC 采样与单个双音脉冲精测。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import WaveformConfig
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from lut_calibration import correct_qls_fraction, wrap_fractional_sample
from models import LinkDelayMeasurement, QLSCalibration
from sampling import sample_iq
from waveforms import generate_two_tone


class AcquisitionError(RuntimeError):
    """无法从当前接收 IQ 获得完整且可信的同步脉冲。"""


@dataclass(frozen=True)
class Capture:
    """ADC 输出；接收端只可见 IQ 和第一个样点的本地时间戳。"""

    samples: np.ndarray
    start_local_s: float
    sample_rate_hz: float
    measured_snr_db: float | None = None


def simulate_capture(iq: np.ndarray, sample_rate_hz: float, *,
                     start_local_s: float, start_source_sample: float,
                     sample_step: float, count: int, amplitude: complex,
                     snr_db: float | None, rng: np.random.Generator) -> Capture:
    """物理信道在独立 ADC 网格生成 IQ，缺失脉冲时仍存在接收噪声。"""
    samples = amplitude * sample_iq(iq, start_source_sample, sample_step, count)
    measured_snr = None
    if snr_db is not None:
        # Both probe components use unit active power. Do not dilute the SNR
        # with the silent acquisition guard or the finite receiver window.
        noise_power = abs(amplitude)**2 / 10**(snr_db/10)
        noise = np.sqrt(noise_power/2) * (
            rng.standard_normal(count) + 1j*rng.standard_normal(count))
        measured_snr = float(snr_db + 10*np.log10(noise_power/max(np.mean(abs(noise)**2), 1e-300)))
        samples += noise
    return Capture(samples, float(start_local_s), sample_rate_hz, measured_snr)


def measure_capture(
    capture: Capture,
    config: WaveformConfig,
    lut: QLSCalibration,
    *,
    threshold: float = 0.45,
) -> LinkDelayMeasurement:
    """从一个有限双音脉冲完成相关峰、对数幅度 QLS 和 LUT 精测。"""
    rx = capture.samples
    pulse = generate_two_tone(config).samples
    if len(rx) < len(pulse) + 2:
        raise AcquisitionError("window_too_short")
    correlation = fft_matched_filter(rx, pulse)
    max_valid_lag = len(rx) - len(pulse)
    if max_valid_lag < 2:
        raise AcquisitionError("sync_pulse_truncated")
    # 粗 PPS 只需保证完整脉冲落入有限窗口。有限 10 us 包络使正确的
    # 中央相关峰成为全局最大值，随后 QLS 只读取该峰左右两个栅格点。
    gate = DelaySearchGate(
        center_s=0.5 * max_valid_lag / capture.sample_rate_hz,
        half_width_s=0.5 * max_valid_lag / capture.sample_rate_hz,
    )
    raw = estimate_delay(correlation, capture.sample_rate_hz, gate)
    if not raw.qls_valid:
        raise AcquisitionError("fine_peak_invalid")
    integer_start = raw.integer_lag_samples
    if integer_start < 0 or integer_start + len(pulse) > len(rx):
        raise AcquisitionError("sync_pulse_truncated")
    energy = float(np.sum(np.abs(rx[integer_start : integer_start + len(pulse)]) ** 2))
    score = float(
        raw.peak_magnitude
        / np.sqrt(max(energy * np.vdot(pulse, pulse).real, 1e-300))
    )
    if score < threshold:
        raise AcquisitionError("pulse_not_detected")
    adjustment = float(wrap_fractional_sample(
        correct_qls_fraction(raw.fractional_offset_samples, lut) - raw.fractional_offset_samples))
    return LinkDelayMeasurement(raw, raw.delay_s + adjustment/capture.sample_rate_hz,
                                adjustment, capture.measured_snr_db, capture.start_local_s, score)
