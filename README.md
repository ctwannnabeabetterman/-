# 两节点分布式 AP 相干波束赋形同步 Demo

这是一个纯 Python、复基带的软件仿真，演示 AP0 主节点、AP1 从节点和单天线 RX 之间的时间、频率、相位同步闭环。时间同步参考论文 *Wireless Picosecond Time Synchronization for Distributed Antenna Arrays* 的脉冲双音、双向时间传递、匹配滤波、三点 QLS 和周期偏差 LUT 方法。频率同步使用论文射频自混频参考的等效软件模型；本项目不模拟自混频电路，也不连接 USRP。

Demo 比较四种状态：

- `unsynchronized`：不补偿 AP1 的时间、频率或相位；
- `time_only`：只应用双向时间传递得到的时钟校正；
- `time_frequency`：再应用参考信号相位斜率得到的频偏校正；
- `full_sync`：再应用 RX 导频 LS 复信道估计得到的相位权重。

仿真真值、估计器输出和控制状态使用不同的数据结构。真值只用于 plant 生成 IQ、未同步基准和运行后误差统计；时间、采样时钟频率、本振频率和发射相位分别由双向估计、钟差斜率估计、参考信号跟踪和 RX 导频反馈驱动。`simulate_four_sync_states()` 不再接收裸时间或频率估计量，只读取控制生效后的 `BeamformingPlantState` 快照。

## 直接运行

项目固定使用现有环境，不需要也不会修改环境：

```powershell
cd "E:\研究生\研究生科研相关\分布式系统波束赋形\distributed_beamforming_demo"
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" main_demo.py --mode fast_demo
```

`fast_demo` 使用 2 µs 脉冲、401 点 LUT 和每个 SNR 100 次 Monte Carlo，覆盖完整算法链。正式模式使用题目默认的 10 µs 脉冲、2001 点 LUT 和每个 SNR 1000 次 Monte Carlo：

```powershell
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" main_demo.py --mode formal
```

可以指定结果目录：

```powershell
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" main_demo.py --mode fast_demo --output-dir results\my_run
```

测试、运行和阈值验收可以由一条命令完成：

```powershell
.\verify.ps1 -Mode fast_demo
```

随机种子固定为 `2023`。同一模式和代码版本会生成相同的数值 CSV 与 JSON。Matplotlib 的 PDF 元数据可能随运行时间变化，不作为逐字节复现对象。

## 输出文件

默认输出到 `results/fast_demo/` 或 `results/formal/`。`results/` 已加入 `.gitignore`，运行结果不会污染源码提交。

```text
results/<mode>/
├── run_config.json              # 本次运行的全部配置
├── summary.json                 # 运行环境、状态来源、主要结论和单位化指标
├── manifest.json                # 输出文件清单
├── lut_bias.csv                 # LUT 扫描真值、原始偏差、校正偏差
├── clock_tracking.csv           # 多轮钟差真值、估计、校正和残差
├── frequency_tracking.csv       # 真实、单次估计、跟踪和残余频偏
├── beamforming_states.csv       # 四状态同步误差、功率和增益
├── monte_carlo.csv              # 6:3:36 dB Monte Carlo 与 CRLB
├── lut_cache/                   # 按波形签名缓存的 NPZ/JSON LUT
└── figures/
    ├── 01_two_tone_time_waveform.{png,pdf}
    ├── 02_two_tone_spectrum.{png,pdf}
    ├── 03_matched_filter_qls.{png,pdf}
    ├── 04_qls_lut_bias.{png,pdf}
    ├── 05_delay_rmse_crlb.{png,pdf}
    ├── 06_clock_offset_tracking.{png,pdf}
    ├── 07_frequency_tracking.{png,pdf}
    ├── 08_received_waveforms.{png,pdf}
    ├── 09_four_state_coherent_gain.{png,pdf}
    └── 10_residual_error_summary.{png,pdf}
```

所有图使用白色背景、Times New Roman、明确的物理单位，并同时保存 PNG 和 PDF。内部时间量统一用秒；CSV 和图片按可读量级转换为 ns 或 ps。

## 算法与符号约定

### 时钟模型

节点时钟为

```text
T_n(t) = t + delta_n + epsilon_n * t + nu_n(t)
```

AP0 取 `delta_0 = epsilon_0 = 0`。`ClockPlantConfig` 保存未控制真值，`LocalClock.offset_s` 和 `fractional_frequency_offset` 只供事件仿真器生成观测。算法通过多轮双向残差估计重建原始钟差曲线，以每次交换的 AP0 接收/回复时间戳中点作为可观测参考历元，拟合斜率得到无量纲采样时钟频差，并更新独立的 `time_correction_s` 和 `fractional_frequency_correction`。`epoch_true_s` 只用于仿真诊断和控制生效时刻，不会进入频差估计器。

### 脉冲双音与分数时延

复基带波形为

```text
s(t) = w(t) [exp(-j*pi*B*t) + exp(+j*pi*B*t)]
```

默认 `B=40 MHz`、`fs=200 MSa/s`、脉冲宽度 `10 µs`、升降沿 `50 ns`。`5.8 GHz` 载频只用于传播相位和残余时间误差到载频相位的换算，不会以 `200 MSa/s` 直接采样。

任意分数时延由补零 FFT 相移实现。正时延表示接收信号向更大的样点下标移动；整数时延走严格的零填充移位路径，避免循环回绕。

### 匹配滤波、QLS 与 LUT

匹配滤波执行完整线性互相关：

```text
r_mf = IFFT(FFT(r) * conj(FFT(s)))
```

结果使用物理 `lags_samples` 轴；数组下标不等于传播时延。粗搜索门来自公开的先验传播时延，不使用仿真真值微调峰值位置。三点 QLS 为

```text
mu = 0.5 * (y[-1] - y[+1]) / (y[-1] - 2*y[0] + y[+1])
tau_hat = (integer_lag + mu) / fs
```

峰位于边界或曲率接近零时禁用插值。LUT 在 `[-0.5, 0.5)` 扫描无噪声真分数时延，以“估计位置到偏差”的周期插值校正 QLS。缓存签名包含算法版本、`B`、`fs`、脉冲宽度、包络和网格数；任一参数变化都会使用新的缓存。

### 双向时间传递

保存 AP1 发射、AP0 接收、AP0 回复和 AP1 接收四个本地时间戳。每个接收时刻都来自真实波形的匹配滤波和 LUT 校正结果。采用

```text
Delta_01 = [(t_RX0 - t_TX1) - (t_RX1 - t_TX0)] / 2
tau_01   = [(t_RX0 - t_TX1) + (t_RX1 - t_TX0)] / 2
```

本项目定义 `delta_1 = T_1 - T_0`，因此在 AP0 理想时 `Delta_01 = -delta_1`。代码把 `Delta_01` 作为应加到 AP1 软件时钟的校正量，同时把 `-Delta_01` 保存为 AP1 钟差估计。处理时延在差分公式中抵消。

公式依赖链路互易和上下行传播时延对称。`up_link` 与 `down_link` 是独立参数，测试明确验证了非对称时延会以一半时延差进入钟差结果。

### 频率同步

AP0 周期发送一个名义 10 MHz 复参考。AP1 的载波频偏和采样时钟偏差共同改变观测频率。接收信号先按名义频率去旋，再分段相干积累、展开相位，并对相位随时间做加权最小二乘直线拟合：

```text
frequency_offset = phase_slope / (2*pi)
```

可选指数跟踪器为

```text
f_track[k] = alpha*f_track[k-1] + (1-alpha)*f_est[k]
```

这是论文自混频锁频硬件的基带等效抽象，只验证频偏可观测性、估计符号、跟踪和数字补偿，不声称复现硬件相噪、混频杂散或 PLL 电路动态。

Hz 制载波频偏不会写入无量纲的 `LocalClock`。项目使用单独的 `LocalOscillator` 保存本振真频偏、估计控制量和连续相位；最后一个跟踪估计通过 `apply_frequency_correction()` 真正更新本振状态，导频和波束赋形数据随后读取该状态的残余频偏和连续相位。

### RX 相位反馈与相干合成

两路静态 LoS 复信道为

```text
h_i = a_i * exp(-j*2*pi*f_c*tau_i + j*theta_i)
```

AP1 初始本振相位来自 `LocalOscillator`。RX 分时接收 QPSK 导频，用 LS 估计 `h_0`、`h_1`。AP1 导频与数据共用同一残余钟差、本振频偏和连续相位轨迹；模型还包含可配置 AWGN、反馈延迟、反馈相位量化和信道相位变化率。信道相位变化率在数据历元同时表现为累积相位 `phase_rate * t` 和等效频偏 `phase_rate / (2*pi)`。数据历元位于导频中心之后，因此残余频偏和信道漂移会自然形成反馈陈旧误差。反馈后计算

```text
w_i = exp(-j*angle(h_i_hat))
```

`per_ap_fixed` 保持每个 AP 的发射功率，两个等幅 AP 理想时相对单 AP 增益为 `6.02 dB`，相对两 AP 非相干功率和增益为 `3.01 dB`。`total_fixed` 令每个权重再除以 `sqrt(2)`，保持总发射功率和单 AP 基准相同，理想时相对单 AP 增益为 `3.01 dB`。

`normalized_ideal_loss_db` 定义为当前相同幅度、时域重叠情况下的理想相干功率除以实际合成功率。值越接近 `0 dB` 越好。

### CRLB 和 SNR 定义

解析双音均方角带宽为

```text
zeta_f_squared = (pi*B)^2
var(tau_hat) >= N0 / (2*zeta_f_squared*Es)
```

各量在代码中分别保存：

- 活动区每样点 SNR：`P_signal_active / P_noise_sample`，也是 Monte Carlo 横轴；
- 复基带噪声带宽：默认 `B_n = fs`；
- 噪声功率谱密度：`N0 = P_noise_sample / B_n`；
- 脉冲能量：`Es = sum(|s[n]|^2) / fs`；
- 脉冲能量 SNR：`Es / N0`；
- 匹配滤波输出 SNR：未拿来替代横轴 SNR。

因此 CRLB 和 Monte Carlo 使用相同的输入噪声定义。论文中的预处理 SNR 与这里的活动区每样点 SNR 不应直接混为同一数值。

Monte Carlo 的每个 trial 都执行正式的 `simulate_two_way_exchange()` 和 `estimate_two_way()`，包含未知 AP1 钟差、四个本地时间戳、处理时延以及独立上下行 AWGN。`clock_offset_rmse_ps` 是完整双向时间传递闭环的统计量。每个 trial 随后还会生成带随机 AP1 初相的 RX 导频、执行 LS 相位反馈，并用补偿后的钟差实际合成两路数据波形；`coherent_gain_vs_incoherent_db` 来自这些波形级合成结果的线性功率平均，不使用钟差到增益的解析捷径。

## 快速模式的参考结果

在固定种子 `2023` 的当前实现中，`fast_demo` 的一次完整运行得到：

- 无噪声 401 点 LUT：原始 QLS 最大系统偏差约 `32.56 ps`，LUT 训练网格残差为浮点精度量级；
- 36 dB Monte Carlo：整数峰值 RMSE 约 `1404.9 ps`，QLS 约 `23.8 ps`，QLS+LUT 约 `4.55 ps`，CRLB 标准差约 `4.46 ps`；
- 36 dB 完整四时间戳双向钟差 RMSE 约 `3.16 ps`，波形级导频反馈与数据合成增益约 `3.00 dB`；
- 完整同步后相对两 AP 非相干功率和的增益约 `3.00 dB`，归一化理想损失接近 `0 dB`；
- Demo 时钟跟踪末轮补偿残差约 `2 ps`，采样时钟频差由观测斜率闭环校正；本振频率跟踪把约 `600 Hz` 偏移降到约 `0.01 Hz`。

这些数值用于回归和趋势检查。它们不是论文硬件 `2.26 ps` 实验结果的拟合目标；脉冲长度、SNR 定义、模拟信道、硬件噪声和测量链不同。

## 配置入口

所有物理参数集中在 [config.py](config.py)：

| 配置类 | 主要内容 | 关键默认值 |
|---|---|---|
| `WaveformConfig` | 采样率、双音间隔、脉冲、载频参数 | 200 MSa/s、40 MHz、10 µs、5.8 GHz |
| `ChannelConfig` | 分数时延、幅度、相位、SNR | 静态单径 AWGN |
| `TwoWayConfig` | 四时间戳调度、粗门 | 20 µs 处理时延、±2.5 样点门 |
| `ClockPlantConfig` | 未控制 AP1 时钟真值 | 100 ns、0.2 ppm |
| `ClockTrackingConfig` | 轮数、同步间隔、增益、随机游走 | 20 轮、50 ms |
| `FrequencySyncConfig` | 参考频率、观测段、CFO、SFO、跟踪器 | 10 MHz、600 Hz |
| `OscillatorConfig` | 未控制 AP1 本振真值 | 600 Hz、1.1 rad |
| `BeamformingConfig` | 两路 RX 静态信道和功率归一化 | `per_ap_fixed` |
| `PhaseFeedbackConfig` | 导频、SNR、反馈延迟和量化 | 1024、32 dB、100 µs、12 bit |
| `MonteCarloConfig` | SNR 轴、次数、种子、噪声带宽 | 6:3:36 dB、100、2023 |

`build_demo_settings()` 只组合两套运行预设。算法函数不在内部改变配置。

## 模块结构

| 文件 | 职责 |
|---|---|
| `waveforms.py` | 升余弦包络脉冲双音 |
| `channel.py` | 线性分数时延、复增益、复 AWGN |
| `delay_estimator.py` | FFT 线性匹配滤波、物理 lag、三点 QLS |
| `lut_calibration.py` | 周期偏差扫描、签名缓存、运行时校正 |
| `clock_model.py` | 仿射本地时钟与独立算法校正状态 |
| `oscillator_model.py` | 连续相位本振真值与 Hz 制控制状态 |
| `two_way_sync.py` | 波形驱动的四时间戳双向时间传递 |
| `experiments.py` | 多轮时间同步与随机游走跟踪 |
| `frequency_sync.py` | 分段相位频偏估计、补偿、指数跟踪 |
| `phase_sync.py` | LoS 复信道、RX 导频 LS、相位权重 |
| `beamforming.py` | 两路到达、CFO、相位及四状态合成 |
| `crlb.py` | SNR/PSD 映射、均方带宽和时延 CRLB |
| `monte_carlo.py` | SNR 扫描及六项统计 |
| `plotting.py` | 十组论文风格 PNG/PDF 图 |
| `results_io.py` | JSON/CSV 序列化 |
| `main_demo.py` | 配置、闭环编排、结果保存和 CLI |
| `verify_results.py` | 机器可读结果阈值验收 |
| `verify.ps1` | 测试、Demo 和结果验收的一键入口 |

## 测试

项目使用 Python 标准库 `unittest`，不依赖 `pytest`：

```powershell
& "C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe" -m unittest discover -s tests -v
```

测试覆盖分数时延的正负号和零填充、相关 lag、QLS 边界保护、LUT 周期插值和缓存签名、处理时延抵消、链路非对称项、时钟漂移、频率估计与补偿、信道 LS、两种功率归一化、CRLB 量纲和 Monte Carlo 可复现性。额外回归测试确保频差估计不读取诊断真时间、非零信道相位漂移在导频和数据中一致，并确认每次 Monte Carlo 试验实际执行导频反馈和数据合成。

`test_closed_loop_state.py` 检查 plant/控制接口隔离、时钟和本振校正连续性；`test_phase_feedback.py` 检查导频与数据共用本振轨迹及反馈延迟；端到端测试检查实际施加的控制值严格等于估计器输出。`acceptance_thresholds.json` 保存版本化阈值，`verify_results.py` 检查全部图、LUT 趋势、Monte Carlo 估计器次序、残余误差和相干增益。

项目声明 Python `>=3.10,<3.14`。`requirements.txt` 固定直接依赖，`requirements-lock.txt` 记录当前 Python 3.12 验证环境中本项目所需的最小传递依赖集合。开发约定见 `CONTRIBUTING.md`。

## 与论文硬件实验的边界

当前版本保留论文时间同步算法的关键基带链路，但做了以下软件化处理：

- 信道为静态单径 LoS 和 AWGN，没有多径、遮挡、移动目标和天线互耦；
- 收发切换、线缆、RF 前端群时延和温漂未建模；
- 频率同步是相位斜率估计，不包含双音自混频电路、模拟 PLL、相噪和杂散；
- RX 导频反馈包含可配固定延迟和相位量化，但没有反馈丢包、随机网络排队或闭环协议重传；
- 默认信道在一个导频到数据区间内静态，可用相位变化率做一阶漂移实验，但没有完整移动多径模型；
- 时间戳为浮点秒制事件，不包含 FPGA 计数器量化、DMA 和操作系统延迟；
- CRLB 使用理想已知波形、AWGN 和无干扰假设；LUT 训练网格上的接近零残差不代表噪声下估计无误差。
- 论文的三种 SDR/测试配置没有伪装成软件配置预设；当前两个模式只用于计算量切换，并不代表论文硬件配置。

## 数值注意事项

- 双音自相关存在周期性局部峰，实际系统必须从几何、协议或粗同步获得足够窄的搜索门；
- QLS 分母接近零、峰值在相关数组或门边界时不会强行插值；
- LUT 的自变量是估计分数位置，且按一个样点周期插值，跨越 `±0.5` 时同时处理整数 lag；
- 频域时延两端补零，避免把 FFT 循环移位误当作线性传播；
- `5.8 GHz * tau` 先折回一个载频周期再求复指数，以减少巨大相位的浮点精度损失；
- 完全相消时 dB 增益为负无穷、理想损失为正无穷，这是功率比定义的自然结果；默认场景不会精确落在该奇点。

## 迁移到 USRP 的接口方案

算法模块只接收复数 IQ、采样率、时间戳和配置，便于把仿真数据源替换为硬件适配层：

1. `WaveformConfig` 生成的 `complex128` 波形量化为 USRP 所需的 `complex64` 或定点格式，并在 FPGA/主机端保留 burst 标识；
2. 用 `send_burst(iq, scheduled_time)` 和 `receive_burst(expected_time, sample_count)` 适配 UHD 定时收发，把返回的硬件时间戳和 IQ 交给 `fft_matched_filter()`；
3. 用硬件时钟控制接口实现 `apply_time_correction()` 和 `apply_frequency_correction()`，记录实际可用的时间步进、NCO 分辨率和命令生效时刻；
4. 把 `simulate_frequency_reference()` 替换为自混频参考 ADC 或 USRP 参考通道采样，保持 `estimate_frequency_offset()` 的复 IQ 接口；
5. 通过 RX 导频上报链路提供 `estimate_channel_ls()` 的观测，反馈消息必须携带测量时刻和有效期；
6. 为每个 RF 通道离线标定固定 TX/RX 群时延，并把校准量放入配置，不写进估计器；
7. 在硬件阶段增加 PPS/10 MHz 锁定状态、溢出、丢包、late command、温度和增益日志，再开展非对称、多径和移动实验。

迁移时优先保持现有函数的物理单位和 lag 约定，通过录制 IQ 的回放测试验证硬件适配层。这样可把算法误差与射频、时钟、驱动和调度误差分开定位。
