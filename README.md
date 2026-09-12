# 分布式阵列皮秒级时间同步与相干合成 Demo

本项目是论文 *Wireless Picosecond Time Synchronization for Distributed Antenna Arrays* 的纯 Python 复基带通信仿真。主线是一次发送一个 10 µs、40 MHz 间隔的双音脉冲，经发射采样、传播、AWGN、接收采样、FFT 匹配滤波、三点对数幅度 QLS、LUT 校正和四时间戳双向交换，实现皮秒量级时延估计。频率传递和相干合成用于检验时间同步估计能否驱动后续链路。

载频与信号用途必须分清：

| 链路 | 论文 RF 参数 | 纯软件实际计算的信号 |
|---|---|---|
| 时间传递 | 5.8 GHz 载频，双音间隔 40 MHz | 复基带 ±20 MHz，10 µs 单脉冲，50 ns 升降沿 |
| 频率传递 | 4.295 GHz 与 4.305 GHz 连续双音 | 两路相对 4.3 GHz 的复包络，自混频得到 10 MHz |
| 波束赋形读出 | 1.2 GHz 载频，双音间隔 50 MHz | 复基带 ±25 MHz 脉冲和 RX 到达差估计 |

4.3 GHz 属于频率传递，5.8 GHz 属于时间传递。200 MSa/s 无法直接采样 4.3/5.8 GHz，因此载频作为 RF 元数据和传播相位参数；采样、噪声、匹配滤波及估计在等效复基带中进行。这保留了包络时延和差频参考，也避免数字仿真中的混叠伪信号。

## 论文参数落地

默认时间传递参数来自论文 Table I：

- TX 数字波形采样率 400 MSa/s，RX 采样率 200 MSa/s；
- 双音间隔 40 MHz，即复基带音点位于 ±20 MHz；
- 单脉冲持续 10 µs，升降沿 50 ns；
- 第一、第二个双向脉冲起点相隔 50 ms，一次 epoch 从首脉冲起点到回复脉冲终点为 50.01 ms；
- 重同步间隔采用 Table I 的 100 ms。

论文正文另有“每 50 ms 重同步”的表述，与 Table I 的 100 ms 不一致。本实现将 50 ms 用作一次双向交换内的回复间隔，将 100 ms 用作相邻同步 epoch 的间隔，并在配置中分别保存为 `reply_interval_s` 与 `sync_interval_s`。正文把波束赋形测试脉冲写成 1 µs，而 Table I 写成 10 µs；三配置统计采用 Table I 的 10 µs。该选择不改变 40 MHz 时间传递波形。

## 时间同步链

发射脉冲由公式逐样点生成：

```text
s(t) = A w(t) [exp(-j*pi*B*t) + exp(+j*pi*B*t)]
```

其中 `B=40 MHz`，`w(t)` 是 50 ns 升降沿包络，`A` 使活动区平均功率为 1。代码先在 400 MSa/s DAC 网格生成有限脉冲，再由带限均匀重采样器在独立 200 MSa/s ADC 网格生成接收 IQ。接收窗口只依据发送公告时间、公开粗传播时延和前置余量设置，估计器看不到真实钟差或真实传播时延。

每次时延测量只发送一个双音脉冲。代码不再添加论文中没有的 BPSK 前导。有限 10 µs 包络提供全局脉冲位置，粗 PPS/调度保证脉冲落入接收窗口；匹配滤波峰附近三个采样点负责亚样点估计。

FFT 匹配滤波计算线性相关：

```text
r_mf = IFFT(FFT(r) * conj(FFT(s)))
```

三点 QLS 使用匹配滤波的**对数幅度**：

```text
y[k] = 20*log10(|r_mf[k]|)
mu = 0.5 * (y[-1] - y[+1]) / (y[-1] - 2*y[0] + y[+1])
tau_hat = (integer_lag + mu) / fs
```

这是复现论文 Fig. 4 约 73 ps 周期偏差的关键。若直接对线性幅度拟合，同一波形的峰值偏差只有约 32 ps，与论文曲线不符。当前 400→200 MSa/s 无噪声独立网格验证得到原始 QLS 峰值偏差约 73.65 ps。

LUT 在 `[-0.5, 0.5)` 内扫描真实分数样点，保存“QLS 估计位置→系统偏差”的周期映射。正式模式用 2001 个训练点，验收结果来自训练点之间的独立半步网格。LUT 只校正确知插值偏差；噪声下精度必须看 Monte Carlo，不能引用无噪声 LUT 残差。

双向交换保存四个本地时间戳：

```text
Delta_01 = [(t_RX0 - t_TX1) - (t_RX1 - t_TX0)] / 2
tau_01   = [(t_RX0 - t_TX1) + (t_RX1 - t_TX0)] / 2
```

接收时间戳由 ADC 首样点本地时间加 QLS/LUT 脉冲起点估计得到。对称链路下，50 ms 回复等待在差分中抵消；非对称上下行时延会以时延差的一半进入钟差估计。

## 频率传递链

论文在 AP0 发送 4.295/4.305 GHz 连续双音，AP1 接收、放大、滤波和自混频，输出 10 MHz 时钟参考。本实现用两路相对 4.3 GHz 的复包络表示 RF 音调：

```text
x_low(t)  = exp(j*2*pi*(-5 MHz)*t)
x_high(t) = exp(j*2*pi*(+5 MHz)*t)
x_ref(t)  = x_high(t) * conj(x_low(t)) = exp(j*2*pi*10 MHz*t)
```

代码随后加入等效复 AWGN，以 AP1 采样钟读取 10 MHz 参考，并通过分段相干积累、相位展开和加权直线拟合估计频差。该链路实现双音到 10 MHz 的数学自混频过程，但不声称复现实物混频器、放大器、滤波器和时钟缓冲器的相噪、杂散或温漂。

## 三种实验与相干合成

程序按论文的三个拓扑运行同一估计链：

1. 有线时间传递 + 有线频率参考；
2. 无线时间传递 + 有线频率参考；
3. 无线时间传递 + 无线双音自混频参考。

每个 SNR、每个 trial 都运行波形接收、匹配滤波、QLS/LUT、四时间戳校时和下游脉冲读出。SNR 轴为 6:3:36 dB，正式模式每点 1000 次。输出 `12_paper_figure12_three_experiment_precision` 采用与论文 Fig. 12 相近的三面板布局，同时保存 trial 级 CSV。当前信道为静态单径 AWGN，结果用于验证算法链和趋势，不能解释为优于论文硬件测量。

相干合成比较 `unsynchronized`、`time_only`、`time_frequency` 和 `full_sync` 四个状态。时间、采样钟频差、本振频偏、RX 到达差和导频相位权重均来自对应估计器与控制状态；仿真真值只用于生成 IQ、未同步基准和事后评分。

## 运行与输出

固定使用现有 Python 环境：

```powershell
cd "E:\研究生\研究生科研相关\分布式系统波束赋形\distributed_beamforming_demo"
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" main_demo.py --mode fast_demo
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" main_demo.py --mode formal
```

`fast_demo` 与 `formal` 使用同一 10 µs 物理波形和算法链。快速模式只缩小 LUT 网格和随机试验数；正式模式使用 2001 点 LUT、6:3:36 dB、每点 1000 次。

默认结果位于 `results/<mode>/`：

```text
README.md                         本次运行的指标和结论
run_config.json                   完整参数
summary.json                      机器可读摘要
lut_training.csv                  LUT 训练网格
lut_bias.csv                      独立验证网格
monte_carlo.csv                   QLS/LUT/CRLB/双向钟差统计
three_experiment_summary.csv      三配置曲线
three_experiment_samples.csv      三配置每个 trial
figures/01...06                   时间同步主图
figures/07...10                   频率与相干合成验证
figures/12_paper_figure12...      三配置 Fig. 12 风格结果
```

双音频谱图由实际有限 IQ 的 FFT 计算。10 µs 时间窗会把理想冲激谱线卷积成有限宽主瓣和旁瓣，因此图上应看到以 ±20 MHz 为中心的两个窄峰，而不会是数学意义上零宽度的两个点。

## 代码结构

| 文件 | 职责 |
|---|---|
| `waveforms.py` | 10 µs 脉冲双音公式与包络 |
| `sampling.py` | 400 MSa/s DAC 到 200 MSa/s ADC 带限重采样 |
| `acquisition.py` | 有限窗口单脉冲接收与检测 |
| `delay_estimator.py` | FFT 匹配滤波、对数幅度 QLS |
| `lut_calibration.py` | 400→200 MSa/s 分数栅格扫描和 LUT |
| `two_way_sync.py` | 单脉冲四时间戳双向时间传递 |
| `frequency_sync.py` | 4.295/4.305 GHz 复包络自混频与 10 MHz 估计 |
| `experiment_suite.py` | 三种论文拓扑的 trial 级通信链 |
| `phase_sync.py`, `beamforming.py` | RX 到达/相位反馈和两 AP 合成 |
| `main_demo.py` | 参数编排、运行、结果输出 |

## 验证

```powershell
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" -m unittest discover -s tests -v
.\verify.ps1 -Mode fast_demo
```

测试覆盖单脉冲捕获、不同 TX/RX 栅格、论文约 73 ps QLS 偏差、独立 LUT 验证、处理时延抵消、非对称链路、无真值频差估计、频率自混频、导频反馈、三配置 trial 链及结果阈值。

## 仿真边界

本项目不连接 SDR，也不生成被 200 MSa/s 混叠的 4.3/5.8 GHz 实信号。它没有 ADC/DAC 量化、FPGA 时间戳量化、真实收发切换、RF 群时延、硬件相噪、杂散、温漂和无线多径。CRLB 使用活动区每样点复 AWGN SNR；论文 Fig. 12 的 SNR 来自预处理测量，两者不能逐点等同。论文全无线实验在高 SNR 下约 10 ps 的平台含硬件频率传递链相噪，本软件 AWGN 等效模型不会自动产生相同平台。
