"""有限窗口中的独立 DAC/ADC 采样及短码粗捕获、双音精测。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import czt

from config import WaveformConfig
from delay_estimator import DelaySearchGate, estimate_delay, fft_matched_filter
from lut_calibration import correct_qls_fraction, wrap_fractional_sample
from models import LinkDelayMeasurement, QLSCalibration
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


def sync_burst(config: WaveformConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """固定已知短码消除双音整周期歧义；返回 IQ、短码和双音起始下标。"""
    code = np.random.default_rng(71023).choice([-1., 1.], size=128)
    code = np.convolve(code, np.hanning(5), mode="full").astype(np.complex128)
    code /= np.sqrt(np.mean(np.abs(code)**2))
    prefix = code.size + 32
    return np.concatenate((code, np.zeros(32), generate_two_tone(config).samples)), code, prefix


def sample_iq(iq: np.ndarray, start_sample: float, step: float, count: int) -> np.ndarray:
    """在 ADC 网格采样补零 DAC 的带限插值；step=发射时钟速率/接收速率。"""
    if not np.isfinite(start_sample) or not np.isfinite(step) or step <= 0 or count < 1:
        raise ValueError("采样坐标须有限、速率和长度须为正")
    # CZT evaluates the signed Fourier series on an arbitrary uniform ADC grid.
    # Include the whole requested window in the zero padding to avoid periodic echoes.
    end = start_sample + step * (count - 1)
    left = 64 + int(np.ceil(max(0., -start_sample)))
    right = 64 + int(np.ceil(max(0., end - len(iq))))
    nfft = next_fast_len(left + len(iq) + right)
    spectrum = np.fft.fftshift(np.fft.fft(np.pad(iq, (left, right)), nfft))
    first = start_sample + left
    position = first + step * np.arange(count)
    values = czt(spectrum, m=count, w=np.exp(2j*np.pi*step/nfft),
                 a=np.exp(-2j*np.pi*first/nfft)) / nfft
    values *= np.exp(-2j*np.pi*(nfft//2)*position/nfft)
    return np.asarray(values, dtype=np.complex128)


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


def measure_capture(capture: Capture, config: WaveformConfig, lut: QLSCalibration,
                    *, threshold: float = 0.45, fine_half_width_samples: float = 2.5
                    ) -> LinkDelayMeasurement:
    """仅由接收样点找到短码，随后以局部双音峰完成 QLS/LUT 精测。"""
    _, code, prefix = sync_burst(config)
    rx = capture.samples
    if len(rx) < len(code):
        raise AcquisitionError("window_too_short")
    correlation = fft_matched_filter(rx, code)
    valid = (correlation.lags_samples >= 0) & (correlation.lags_samples <= len(rx)-len(code))
    magnitudes = correlation.magnitude[valid]
    index = int(np.argmax(magnitudes))
    energy = float(np.sum(np.abs(rx[index:index+len(code)])**2))
    score = float(magnitudes[index] / np.sqrt(max(energy*np.vdot(code, code).real, 1e-300)))
    if score < threshold:
        raise AcquisitionError("preamble_not_detected")
    coarse = estimate_delay(correlation, capture.sample_rate_hz,
                            DelaySearchGate(index/capture.sample_rate_hz, 2.5/capture.sample_rate_hz))
    sync_start = coarse.delay_s * capture.sample_rate_hz + prefix
    pulse = generate_two_tone(config).samples
    if sync_start < 1 or sync_start + len(pulse) > len(rx)-1:
        raise AcquisitionError("sync_pulse_truncated")
    raw = estimate_delay(fft_matched_filter(rx, pulse), capture.sample_rate_hz,
                         DelaySearchGate(sync_start/capture.sample_rate_hz,
                                         fine_half_width_samples/capture.sample_rate_hz))
    if not raw.qls_valid:
        raise AcquisitionError("fine_peak_invalid")
    adjustment = float(wrap_fractional_sample(
        correct_qls_fraction(raw.fractional_offset_samples, lut) - raw.fractional_offset_samples))
    return LinkDelayMeasurement(raw, raw.delay_s + adjustment/capture.sample_rate_hz,
                                adjustment, capture.measured_snr_db, capture.start_local_s, score)
