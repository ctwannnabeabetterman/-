"""面向读者的正式结果报告与最小输出清单。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from models import ThreeExperimentResult
from results_io import write_csv_columns


DIAGNOSTIC_TABLES = (
    "diagnostics/lut_training.csv",
    "diagnostics/lut_validation.csv",
    "diagnostics/monte_carlo.csv",
    "diagnostics/clock_tracking.csv",
    "diagnostics/frequency_tracking.csv",
    "diagnostics/beamforming_states.csv",
    "diagnostics/three_config_trials.csv",
)


def build_manifest(
    figure_paths: Sequence[str | Path],
    *,
    diagnostics: bool,
) -> dict[str, object]:
    """构造正式输出清单；内部诊断项使用独立命名空间。"""

    figures = [str(Path(path)) for path in figure_paths]
    return {
        "schema_version": 4,
        "figure_count": len(figures),
        "figures": figures,
        "tables": ["three_config_summary.csv"],
        "diagnostics": list(DIAGNOSTIC_TABLES) if diagnostics else [],
        "summary": "summary.json",
        "configuration": "run_config.json",
        "guide": "README.md",
    }


def write_result_tables(
    output_dir: str | Path,
    result: ThreeExperimentResult,
    *,
    diagnostics: bool,
    diagnostic_columns: Mapping[str, Mapping[str, Sequence[object]]] | None = None,
) -> list[Path]:
    """写一张正式汇总表，并按需把内部数据隔离到 diagnostics。"""

    root = Path(output_dir)
    paths = [
        write_csv_columns(
            root / "three_config_summary.csv",
            {
                "case": [key for key in result.profile_keys for _ in result.snr_db],
                "time_link": [mode for mode in result.time_link_modes for _ in result.snr_db],
                "frequency_link": [
                    mode for mode in result.frequency_link_modes for _ in result.snr_db
                ],
                "snr_db": np.tile(result.snr_db, len(result.profile_keys)),
                "time_transfer_std_ps": result.time_transfer_std_s.reshape(-1) * 1e12,
                "beamforming_interarrival_std_ps": result.beamforming_std_s.reshape(-1)
                * 1e12,
                "two_way_clock_crlb_std_ps": np.tile(
                    result.two_way_clock_crlb_std_s * 1e12,
                    len(result.profile_keys),
                ),
                "residual_clock_rate_rmse_ppm": result.residual_clock_rate_rmse.reshape(-1)
                * 1e6,
                "acquisition_failure_rate": result.acquisition_failure_rate.reshape(-1),
            },
        )
    ]
    if diagnostics and diagnostic_columns:
        for filename, columns in diagnostic_columns.items():
            paths.append(write_csv_columns(root / "diagnostics" / filename, columns))
    return paths


def render_report(summary: Mapping[str, object]) -> str:
    """生成无需阅读代码即可理解的中文正式结果说明。"""

    waveform = summary["waveform"]
    lut = summary["lut"]
    monte_carlo = summary["monte_carlo_highest_snr"]
    three_experiments = summary["three_experiments"]
    profiles = three_experiments["profiles"]
    threshold = three_experiments["threshold_region"]
    full_sync = summary["beamforming"]["full_sync"]
    sensitivity = summary["sensitivity"]
    threshold_std = list(threshold["time_transfer_std_ps"].values())
    threshold_failure = list(threshold["acquisition_failure_rate"].values())
    next_std = list(threshold["next_time_transfer_std_ps"].values())
    profile_rows = "\n".join(
        "| {label} | {time:.3f} | {beam:.3f} | {rate:.6f} | {failure:.3%} |".format(
            label=profile.get("label_cn", profile.get("label", key)),
            time=profile["time_transfer_std_ps"],
            beam=profile["beamforming_std_ps"],
            rate=profile["residual_clock_rate_rmse_ppm"],
            failure=profile["acquisition_failure_rate"],
        )
        for key, profile in profiles.items()
    )
    return f"""# 分布式相干同步软件仿真结果

本目录来自一次 `{summary['mode']}` 运行，随机种子为 `{summary['seed']}`。核心问题是：40 MHz 双音脉冲经过完整接收链后，QLS 与 LUT 能否支持皮秒级双向时间同步，以及这些估计能否进入下游相干合成。

> **结论边界：**这里给出的是复基带软件仿真，不能代表论文硬件实验，也不能把低于论文测量值的数字解释为硬件性能更优。

## 先看结论

- 时间传递采用 5.8 GHz RF 参数、±20 MHz 复基带音点、10 µs 单脉冲、{waveform['transmit_sample_rate_msa_s']:.0f} MSa/s 发射网格和 {waveform['sample_rate_msa_s']:.0f} MSa/s 接收网格。
- 在 {monte_carlo['snr_db']:.0f} dB 仿真 SNR 下，QLS + LUT 时延 RMSE 为 {monte_carlo['qls_lut_rmse_ps']:.3f} ps，同一定义下 CRLB 为 {monte_carlo['crlb_std_ps']:.3f} ps，捕获失败率为 {monte_carlo['acquisition_failure_rate']:.3%}。
- 完整同步后的两 AP 到达差为 {full_sync['arrival_difference_ps']:.3f} ps，相对非相干功率和的增益为 {full_sync['gain_vs_incoherent_sum_db']:.3f} dB。

## 如何阅读四张图

1. **图 01：系统、信号与链路。**图区分 5.8 GHz 时间传递、4.295/4.305 GHz 频率传递和 1.2 GHz 下游验证。频谱由实际有限 IQ 的 FFT 得到；10 µs 包络使两个谱峰具有有限主瓣和旁瓣。
2. **图 02：QLS 与 LUT。**左图说明三个匹配滤波样点怎样插值得到亚样点峰；中图只是无噪声的确定性 LUT 插值检查；右图的带噪 Monte Carlo 才是性能结果。
3. **图 03：三种配置与 CRLB。**这是主结果，逐个 SNR 比较有线时间+有线频率、无线时间+有线频率、无线时间+无线频率和 CRLB。最低 SNR 点属于捕获与峰值选择的门限效应区，不能按局部 CRLB 解读。
4. **图 04：模型失配。**该扩展展示上下行时延不对称和 10 MHz 参考相位扰动怎样破坏理想仿真结果。

## QLS 与 LUT 验证

200 MSa/s 的采样间隔为 5 ns。原始 QLS 在一个采样栅格内存在周期性系统偏差；独立无噪声验证网格上的原始 RMSE 为 {lut['raw_rmse_ps']:.3f} ps，LUT 校正残差为 {lut['corrected_rmse_ps']:.6f} ps。这个残差只说明查表插值一致，不能代替带噪性能。

## 三配置最高 SNR 结果

| 软件等效配置 | 时间同步标准差 (ps) | 下游脉冲到达差标准差 (ps) | 采样钟速率残差 RMSE (ppm) | 捕获失败率 |
|---|---:|---:|---:|---:|
{profile_rows}

三个配置每个统计点都运行 IQ 捕获、匹配滤波、整数峰、对数幅度 QLS、LUT 和四时间戳双向校时。频率为无线的配置还运行 4.295/4.305 GHz 双音的复包络自混频与 10 MHz 参考观测。`three_config_summary.csv` 保存所有 SNR 汇总，不在正式目录堆放逐试验样本。

### 低 SNR 门限效应

在 {threshold['snr_db']:.0f} dB 时，三种配置的时间同步标准差为 {min(threshold_std):.3f}–{max(threshold_std):.3f} ps，而双向钟差 CRLB 为 {threshold['two_way_clock_crlb_std_ps']:.3f} ps，捕获失败率为 {min(threshold_failure):.3%}–{max(threshold_failure):.3%}。低 SNR 下少量捕获失败和搜索门内错误峰会主导样本标准差；CRLB 是局部无偏时延估计的下界，不包含捕获错误和粗差。在 {threshold['next_snr_db']:.0f} dB 时，三种配置已回到 {min(next_std):.3f}–{max(next_std):.3f} ps，对应 CRLB 为 {threshold['next_two_way_clock_crlb_std_ps']:.3f} ps。该门限效应是仿真链路的实际输出，不应删点或隐藏。

## 模型失配结果

- 扫描到 ±{sensitivity['max_abs_asymmetry_ps']:.0f} ps 的上下行不对称时，最大钟差偏差为 {sensitivity['max_abs_clock_bias_ps']:.3f} ps，符合“双向不对称量的一半”关系。
- 两个 10 MHz 观测窗口的相位扰动标准差扫描到 {sensitivity['max_reference_phase_noise_deg']:.1f}° 时，100 ms 保持期时序漂移 RMSE 达 {sensitivity['max_holdover_timing_rmse_ps']:.3f} ps。

这说明静态对称 AWGN 仿真得到的皮秒数字依赖明确假设。真实系统还需要校准 RF 群时延，并测量 ADC/DAC 量化、FPGA 时间戳量化、振荡器与混频器相噪、杂散、温漂和多径。

## 文件说明

- `figures/`：四张正式图的 PNG 与 PDF；
- `three_config_summary.csv`：三配置全 SNR 汇总；
- `summary.json`：机器可读指标；
- `run_config.json`：本次参数；
- `manifest.json`：防止旧文件混入的输出清单。
"""


def write_report(path: str | Path, summary: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(summary), encoding="utf-8")
    return destination
