# 开发与验证

项目要求 Python 3.10–3.13。当前固定验证环境为 `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe`，不得由项目脚本自动修改。

## 修改流程

1. 把仿真 plant 真值、估计器输出和控制器状态保存在不同结构中。真值只能生成观测和计算统计指标。
2. 新增或修复算法前先加入能够失败的最小 `unittest`，实现后运行相关测试和全量测试。
3. 所有物理量内部使用 SI 单位；显示和 CSV 导出时再转换为 MHz、ns、ps 或度。
4. 新的随机实验必须接受 `numpy.random.Generator` 或固定种子。
5. 修改波形、采样率、包络或 LUT 算法后确认缓存签名变化。

## 一键验证

```powershell
.\verify.ps1 -Mode fast_demo
```

脚本依次运行全部单元测试、生成 Demo 结果并按 `acceptance_thresholds.json` 验收。正式统计使用：

```powershell
.\verify.ps1 -Mode formal
```

`requirements.txt` 是直接依赖，`requirements-lock.txt` 记录当前验证环境中本项目所需的最小传递依赖版本。提交代码时不要加入 `results/`、缓存、图片或 CSV。
