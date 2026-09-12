"""把仿真数据稳定写为 UTF-8 JSON 和列式 CSV。"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def to_serializable(value: Any) -> Any:
    """递归转换 dataclass、NumPy 数组和标量为 JSON 原生类型。"""

    if is_dataclass(value) and not isinstance(value, type):
        return to_serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> Path:
    """以 UTF-8、两空格缩进写入 JSON，并返回绝对或传入形式的路径。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(to_serializable(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def write_csv_columns(
    path: str | Path,
    columns: Mapping[str, Sequence[Any] | np.ndarray],
) -> Path:
    """按映射插入顺序写入等长列；首行为字段名。"""

    if not columns:
        raise ValueError("columns 不能为空")
    normalized = {name: list(values) for name, values in columns.items()}
    lengths = {len(values) for values in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("所有 CSV 列必须等长")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(normalized.keys())
        writer.writerows(zip(*normalized.values()))
    return destination


def write_result_readme(
    path: str | Path,
    summary: Mapping[str, Any],
) -> Path:
    """为单次运行生成以时间同步为主线的简短结果说明。"""

    waveform = summary["waveform"]
    delay = summary["delay_demo"]
    clock = summary["clock_tracking"]
    lut = summary["lut"]
    high_snr = summary["monte_carlo_highest_snr"]
    full_sync = summary["beamforming"]["full_sync"]
    profiles = summary["three_experiments"]["profiles"]
    profile_lines = "\n".join(
        "- {label}：时间传递标准差 {time:.3f} ps，50 MHz 脉冲到达差标准差 "
        "{beam:.3f} ps，残余采样钟速率 RMSE {rate:.6f} ppm，捕获失败率 "
        "{failure:.3%}。".format(
            label=profile["label"],
            time=profile["time_transfer_std_ps"],
            beam=profile["beamforming_std_ps"],
            rate=profile["residual_clock_rate_rmse_ppm"],
            failure=profile["acquisition_failure_rate"],
        )
        for profile in profiles.values()
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# 本次 Demo 结果

本目录只对应一次 `{summary['mode']}` 运行，固定随机种子为 `{summary['seed']}`。主目标是复现脉冲双音双向时间同步；频率、RX 到达对齐和相位反馈用于验证时间同步结果能进入下游相干合成链路。

## 时间同步主结果

- 波形：{waveform['sample_rate_msa_s']:.0f} MSa/s，两个基带音点位于 ±{waveform['tone_separation_mhz'] / 2:.0f} MHz，间隔 {waveform['tone_separation_mhz']:.0f} MHz，脉冲长度 {waveform['pulse_duration_us']:.0f} µs。
- QLS/LUT 独立验证：留出 {lut['validation_points']} 个未参与 LUT 构建的分数时延；原始 QLS RMSE {lut['raw_rmse_ps']:.3f} ps，校正后 {lut['corrected_rmse_ps']:.3f} ps，最大绝对残差 {lut['corrected_max_abs_bias_ps']:.3f} ps。
- 单次时延：真值 {delay['true_delay_ns']:.6f} ns，QLS + LUT 估计 {delay['lut_corrected_delay_ns']:.6f} ns，误差 {delay['corrected_error_ps']:.3f} ps。
- {high_snr['snr_db']:.0f} dB Monte Carlo：QLS + LUT 时延 RMSE {high_snr['qls_lut_rmse_ps']:.3f} ps，CRLB 标准差 {high_snr['crlb_std_ps']:.3f} ps，完整双向钟差 RMSE {high_snr['clock_offset_rmse_ps']:.3f} ps。
- 多轮控制末次更新后钟差残差：{clock['final_residual_after_update_ps']:.3f} ps。

图 01–06 和 `lut_bias.csv`、`lut_training.csv`、`clock_tracking.csv`、`monte_carlo.csv` 是时间同步复现的主要证据。图 01 的波形由双音公式逐样点生成；图 02 对发送 IQ 和经过信道、AWGN、接收采样后的 IQ 做 FFT，没有直接绘制理想谱线。有限脉冲会使 ±20 MHz 两个音点与脉冲包络频谱卷积，所以实际主瓣具有有限宽度和旁瓣。

## 论文三种实验配置的等效软件链路

每个统计点都依次运行接收 IQ、匹配滤波、整数峰、QLS、LUT、四时间戳双向校时，再用公式生成的 50 MHz、1 µs 脉冲经过 DAC 网格、信道、AWGN、RX ADC 和同一到达时刻估计器。全无线频率配置另外从连续 10 MHz 参考上相隔 50 ms 的两个带噪 IQ 窗口估计相位差和采样钟速率。以下为最高 SNR 点：

{profile_lines}

图 11 和 `three_experiment_summary.csv` 给出完整 6:3:36 dB 曲线；`three_experiment_samples.csv` 保存每个 trial 的原始结果。三种配置是论文拓扑的复基带等效模型，其中“cabled”与“wireless”选择传播时延和频率参考来源，不包含线缆/RF 前端群时延、无线多径和自混频硬件。

## 下游相干合成验证

- 完整同步到达差：{full_sync['arrival_difference_ps']:.3f} ps。
- 相对两 AP 非相干功率和的增益：{full_sync['gain_vs_incoherent_sum_db']:.3f} dB。
- 相对当前幅度条件下理想相干和的损失：{full_sync['normalized_ideal_loss_db']:.6f} dB。

图 07–10、`frequency_tracking.csv`、`beamforming_states.csv` 和 `joint_tracking.csv` 用于检查同步后的频率、相位和持续保持。完整机器可读指标见 `summary.json`，参数见 `run_config.json`，文件清单见 `manifest.json`。

这些结果属于静态单径、复基带、纯软件仿真，不等同于论文硬件测量值。
"""
    destination.write_text(report, encoding="utf-8")
    return destination
