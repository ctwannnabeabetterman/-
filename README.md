# 分布式阵列皮秒级时间同步与相干合成 Demo

本项目是 *Wireless Picosecond Time Synchronization for Distributed Antenna Arrays* 的纯 Python 复基带通信仿真。它复现论文的时间同步算法链和三种实验拓扑，并验证同步估计可以进入下游相干合成。

这里没有 SDR 或 RF 硬件。结果用于检查公式、估计器、协议和参数趋势，不能当作论文硬件测量值。

## 三类信号不能混淆

| 用途 | 论文 RF 参数 | 软件仿真中的实际信号 |
|---|---|---|
| 时间传递 | 5.8 GHz，双音间隔 40 MHz | ±20 MHz 复基带双音，10 µs 单脉冲，50 ns 升降沿 |
| 频率传递 | 4.295/4.305 GHz 连续双音 | 两路相对 4.3 GHz 的复包络，自混频得到连续 10 MHz 参考 |
| 波束赋形读出 | 1.2 GHz，双音间隔 50 MHz | ±25 MHz 复基带脉冲、接收端到达差与相干功率 |

200 MSa/s 不能直接采样 4.3 或 5.8 GHz。代码把 GHz 载频保存为物理参数，在复基带中生成、传播、加噪、采样和估计。这是标准的等效通信仿真，不会制造数字混叠伪信号。

## 时间同步链

每次链路测量只发送一个脉冲：

```text
s(t) = A w(t) [exp(-jπBt) + exp(+jπBt)]
B = 40 MHz
```

`w(t)` 是长度 10 µs、两端各 50 ns 升余弦边沿的包络，中间保持单位幅度。程序没有给整个脉冲乘 Hann、Hamming 或 Blackman 窗。

完整数据流为：

```text
400 MSa/s 发射样点
    → 分数传播时延、复增益和 AWGN
    → 200 MSa/s 接收样点
    → FFT 线性匹配滤波
    → 整数峰
    → 三点对数幅度 QLS
    → 周期偏差 LUT
    → 四时间戳双向钟差估计与校正
```

200 MSa/s 的采样间隔为 5 ns。皮秒精度来自匹配滤波峰形的亚样点估计和系统偏差校正，并不是把采样率误写成皮秒级。

QLS 使用峰值附近三个点的对数幅度：

```text
y[k] = 20 log10 |r_mf[k]|
μ = 0.5 (y[-1] - y[+1]) / (y[-1] - 2y[0] + y[+1])
τ_hat = (integer_lag + μ) / fs
```

LUT 在一个采样周期内扫描分数时延，反演 QLS 的周期性系统偏差。无噪声 LUT 残差只用于检查插值一致性；实际精度必须看带噪 Monte Carlo。

双向交换使用四个本地时间戳：

```text
clock_correction = [(t_RX0 - t_TX1) - (t_RX1 - t_TX0)] / 2
propagation_delay = [(t_RX0 - t_TX1) + (t_RX1 - t_TX0)] / 2
```

对称链路下，回复处理时间抵消；上下行传播时延不对称会以时延差的一半进入钟差估计。正式结果的图 04 会直接验证这个限制。

## 论文三种配置

正式仿真在 6–36 dB、步长 3 dB 的 SNR 轴上运行：

1. 有线时间传递 + 有线频率参考；
2. 无线时间传递 + 有线频率参考；
3. 无线时间传递 + 无线频率参考。

正式模式每个 SNR、每种配置运行 1000 次。每次都执行接收 IQ、匹配滤波、QLS、LUT、四时间戳校时和下游脉冲读出。无线频率配置还执行 4.295/4.305 GHz 双音复包络自混频和两次 10 MHz 参考观测。

CRLB 与代码的活动区每样点复 AWGN SNR 使用同一定义。论文 Fig. 12 的 SNR 来自硬件预处理测量，两者只能比较趋势，不能逐点等同。

## 正式结果怎样阅读

`results/formal/README.md` 会先给出结论，再依次解释四张图：

1. `01_system_signal_chain`：信号参数、双音波形、实际 IQ 频谱和三条链路；
2. `02_qls_lut_validation`：QLS 插值、LUT 确定性校准和带噪 RMSE；
3. `03_three_config_vs_crlb`：三种配置的软件时间同步结果与 CRLB；
4. `04_model_mismatch_sensitivity`：链路不对称和 10 MHz 参考相位扰动。

正式目录只保留四组 PNG/PDF、一张 `three_config_summary.csv`、摘要、参数、清单和中文说明。逐试验样本及内部追踪数据默认不生成。

## 运行

固定使用现有环境：

```powershell
cd "E:\研究生\研究生科研相关\分布式系统波束赋形\distributed_beamforming_demo"
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" main_demo.py --mode fast_demo
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" main_demo.py --mode formal
```

只有定位算法内部问题时才生成诊断表：

```powershell
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" main_demo.py --mode fast_demo --diagnostics
```

一键测试、运行和验收：

```powershell
.\verify.ps1 -Mode fast_demo
.\verify.ps1 -Mode formal
```

## 代码结构

| 文件 | 单一职责 |
|---|---|
| `waveforms.py` | 10 µs、40 MHz 双音脉冲公式 |
| `sampling.py`, `acquisition.py`, `channel.py` | 发射/接收栅格、信道和有限窗口捕获 |
| `delay_estimator.py`, `lut_calibration.py` | 匹配滤波、QLS 和 LUT |
| `two_way_sync.py`, `clock_model.py` | 四时间戳时间传递与本地钟控制 |
| `frequency_sync.py`, `oscillator_model.py` | 双音自混频 10 MHz 参考与本振控制 |
| `phase_sync.py`, `beamforming.py` | RX 导频反馈、到达对齐和两 AP 相干合成 |
| `monte_carlo.py` | QLS/LUT 的独立带噪统计 |
| `experiment_suite.py` | 三种论文拓扑的完整通信链 |
| `sensitivity.py` | 不对称链路与参考相位扰动扫描 |
| `plotting.py`, `reporting.py` | 四张正式图、中文报告与最小输出 |
| `main_demo.py` | 配置和顶层编排 |

## 仿真边界

当前模型没有真实 RF 前端群时延、ADC/DAC 量化、FPGA 时间戳量化、收发切换抖动、混频器杂散、真实振荡器相噪、温漂和现场多径。图 04 只对其中两个关键假设做受控扫描。若进入硬件实践，下一步应首先建立线缆/无线通道群时延标定和 10 MHz 参考相噪测量流程。
