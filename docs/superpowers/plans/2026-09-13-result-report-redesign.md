# Result Report Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cluttered eleven-figure result dump with a four-figure, reader-first formal report centered on QLS/LUT and the paper's three configurations versus CRLB.

**Architecture:** Keep waveform, acquisition, synchronization, frequency, phase, and beamforming models as the scientific core. Add one focused sensitivity experiment, centralize formal report writing, reduce `main_demo.py` to orchestration, and make diagnostic tables opt-in. Delete the redundant joint-tracking execution path and old single-purpose plots.

**Tech Stack:** Python 3, NumPy, SciPy, Matplotlib, standard-library `unittest`, JSON/CSV, PowerShell verification.

---

### Task 1: Lock the new formal-output contract

**Files:**
- Create: `tests/test_formal_output_contract.py`
- Modify: `main_demo.py`
- Modify: `verify_results.py`
- Modify: `acceptance_thresholds.json`

- [ ] **Step 1: Write the failing output-contract test**

```python
EXPECTED_FIGURES = {
    f"figures/{stem}.{suffix}"
    for stem in (
        "01_system_signal_chain",
        "02_qls_lut_validation",
        "03_three_config_vs_crlb",
        "04_model_mismatch_sensitivity",
    )
    for suffix in ("png", "pdf")
}

def test_formal_manifest_contains_only_reader_facing_outputs(self):
    manifest = build_manifest(self.output_dir, diagnostics=False)
    self.assertEqual(set(manifest["figures"]), EXPECTED_FIGURES)
    self.assertEqual(manifest["tables"], ["three_config_summary.csv"])
```

- [ ] **Step 2: Run the test and verify it fails because `build_manifest` is absent**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest tests.test_formal_output_contract -v`  
Expected: FAIL with an import/name error for `build_manifest`.

- [ ] **Step 3: Implement the manifest contract and `--diagnostics` CLI flag**

```python
def build_manifest(figure_paths, *, diagnostics):
    tables = ["three_config_summary.csv"]
    if diagnostics:
        tables.extend(DIAGNOSTIC_TABLES)
    return {
        "schema_version": 4,
        "figure_count": len(figure_paths),
        "figures": [str(path) for path in figure_paths],
        "tables": tables,
    }
```

- [ ] **Step 4: Update machine checks to require four figure pairs, one formal table, finite three-profile metrics, acquisition accounting, QLS/LUT ordering, CRLB proximity, and coherent-gain evidence**

- [ ] **Step 5: Run the contract test and commit**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest tests.test_formal_output_contract -v`  
Expected: PASS.

### Task 2: Add the model-mismatch sensitivity experiment

**Files:**
- Create: `sensitivity.py`
- Create: `tests/test_sensitivity.py`
- Modify: `config.py`
- Modify: `models.py`

- [ ] **Step 1: Write failing tests for delay asymmetry and reference-phase perturbation**

```python
def test_two_way_asymmetry_bias_is_half_path_difference(self):
    result = run_sensitivity_suite(self.waveform, self.lut, self.config)
    expected = 0.5 * result.asymmetry_s
    np.testing.assert_allclose(result.clock_bias_s, expected, atol=3e-12)

def test_reference_phase_noise_increases_rate_error(self):
    result = run_sensitivity_suite(self.waveform, self.lut, self.config)
    self.assertGreater(result.rate_rmse[-1], result.rate_rmse[0])
```

- [ ] **Step 2: Run the tests and verify missing API failures**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest tests.test_sensitivity -v`  
Expected: FAIL because `sensitivity.py` does not exist.

- [ ] **Step 3: Implement the smallest deterministic sweep**

```python
@dataclass(frozen=True)
class SensitivityConfig:
    asymmetry_ps: tuple[float, ...] = (-100, -50, 0, 50, 100)
    reference_phase_noise_deg: tuple[float, ...] = (0, 0.1, 0.3, 1.0, 3.0)
    trials: int = 300
    seed: int = 91723
```

The asymmetry path runs the real four-timestamp estimator. The frequency path perturbs the two observable 10 MHz reference windows and reports clock-rate RMSE plus the resulting downstream timing drift over one 100 ms synchronization interval.

- [ ] **Step 4: Run sensitivity tests and commit**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest tests.test_sensitivity -v`  
Expected: PASS.

### Task 3: Replace eleven plots with four reader-facing figures

**Files:**
- Modify: `plotting.py`
- Create: `tests/test_plotting_contract.py`

- [ ] **Step 1: Write a failing plot-contract test**

```python
def test_each_formal_plot_writes_png_and_pdf(self):
    paths = plot_formal_results(self.fixture, self.output_dir)
    self.assertEqual(len(paths), 8)
    self.assertEqual({path.suffix for path in paths}, {".png", ".pdf"})
```

- [ ] **Step 2: Verify the new plotting API is absent**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest tests.test_plotting_contract -v`  
Expected: FAIL because `plot_formal_results` is absent.

- [ ] **Step 3: Implement the four figures**

```python
def plot_formal_results(data: FormalPlotData, output_dir: Path) -> list[Path]:
    paths = []
    paths += plot_system_signal_chain(data, output_dir)
    paths += plot_qls_lut_validation(data, output_dir)
    paths += plot_three_config_vs_crlb(data.three_experiment, output_dir)
    paths += plot_model_mismatch_sensitivity(data.sensitivity, output_dir)
    return paths
```

Use Chinese titles, explicit units, direct signal-role labels, and captions inside each figure. The three-configuration plot contains only the three time-synchronization curves and CRLB; downstream metrics remain in the table.

- [ ] **Step 4: Delete obsolete figure functions and verify no references remain**

Run: `rg "plot_clock_tracking|plot_frequency_tracking|plot_received_waveforms|plot_coherent_gain|plot_residual_summary" -g "*.py"`  
Expected: no matches.

- [ ] **Step 5: Run plot tests and commit**

### Task 4: Centralize the report and remove redundant execution/output paths

**Files:**
- Create: `reporting.py`
- Create: `tests/test_reporting.py`
- Modify: `main_demo.py`
- Modify: `results_io.py`
- Delete: `joint_sync.py`
- Delete: `tests/test_joint_sync.py`

- [ ] **Step 1: Write failing tests for the Chinese report and diagnostic separation**

```python
def test_report_explains_each_figure_and_simulation_boundary(self):
    text = render_report(self.summary)
    for phrase in ("三种配置与 CRLB", "QLS 与 LUT", "模型失配", "不能代表硬件实验"):
        self.assertIn(phrase, text)

def test_default_tables_exclude_internal_trials(self):
    paths = write_result_tables(self.output, self.data, diagnostics=False)
    self.assertEqual([path.name for path in paths], ["three_config_summary.csv"])
```

- [ ] **Step 2: Run tests and verify missing reporting API failures**

- [ ] **Step 3: Implement `render_report`, `write_result_tables`, `build_manifest`, and concise core summary construction**

- [ ] **Step 4: Rebuild `main_demo.run_demo` as orchestration of waveform/LUT validation, estimator Monte Carlo, one closed-loop case, three profiles, sensitivity, reporting, and four plots**

- [ ] **Step 5: Delete the joint-tracking call, module, test, CSV, summary fields, thresholds, and documentation references**

- [ ] **Step 6: Run focused and full tests, then commit**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest discover -s tests -v`  
Expected: all tests PASS.

### Task 5: Regenerate, visually audit, document, and publish

**Files:**
- Modify: `README.md`
- Modify: `results/formal/**`
- Modify: `verify.ps1` if its expected paths change

- [ ] **Step 1: Update the project README to show the four-output reading order and separate paper parameters from simulation assumptions**

- [ ] **Step 2: Resolve the exact formal result path and delete its prior generated contents before running**

```powershell
$resultRoot = (Resolve-Path -LiteralPath '.\results\formal').Path
if (-not $resultRoot.EndsWith('distributed_beamforming_demo\results\formal')) { throw 'unexpected result path' }
Get-ChildItem -LiteralPath $resultRoot -Force | Remove-Item -Recurse -Force
```

- [ ] **Step 3: Run the formal simulation and acceptance checker**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe main_demo.py --mode formal`  
Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe verify_results.py results/formal`  
Expected: exit code 0 and `"passed": true`.

- [ ] **Step 4: Inspect all four PNGs at original resolution and correct any clipped labels, ambiguous legends, misleading scales, or unreadable Chinese text**

- [ ] **Step 5: Check repository cleanliness and result manifest**

Run: `git diff --check`  
Run: `git status --short`  
Run: `git ls-files | rg "__pycache__|\.pyc$|three_experiment_samples\.csv|joint_tracking\.csv"`  
Expected: no whitespace errors and no tracked cache/raw/redundant outputs.

- [ ] **Step 6: Commit and push `main`**

```powershell
git add -A
git commit -m "refactor(demo): 聚焦三配置时间同步结果"
git push origin main
```
