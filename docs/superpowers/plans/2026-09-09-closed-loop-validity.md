# Closed-loop Validity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the three verified closed-loop validity gaps while preserving the runnable pure-software demo.

**Architecture:** Keep simulation truth inside plant generation and post-run diagnostics. Add a reference-clock epoch to clock observations, construct one effective time-varying channel state for data, and calculate Monte Carlo gain through RX pilot feedback plus waveform combining.

**Tech Stack:** Python 3.12, NumPy, SciPy, unittest

**Status:** Implemented and verified on 2026-09-09; 74 unit tests and both result-acceptance modes pass.

---

### Task 1: Observable clock-rate time base

**Files:**
- Modify: `models.py`
- Modify: `experiments.py`
- Modify: `tests/test_closed_loop_state.py`

- [ ] **Step 1: Write the failing test**

Add `reference_epoch_s` to the test fixture and assert that replacing only `epoch_true_s` does not change `estimate_clock_frequency_offset()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest tests.test_closed_loop_state -v`

Expected: the estimate changes when only the diagnostic true-time array changes.

- [ ] **Step 3: Write minimal implementation**

Store `0.5 * (observation.t_rx0_s + observation.t_tx0_s)` for every exchange and regress against this AP0 timestamp midpoint.

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command and expect all cases to pass.

### Task 2: Shared channel phase trajectory

**Files:**
- Modify: `main_demo.py`
- Modify: `tests/test_demo_outputs.py`

- [ ] **Step 1: Write the failing test**

Run a small end-to-end configuration with a nonzero channel phase rate and assert that full-sync ideal loss remains below 0.05 dB.

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest tests.test_demo_outputs.DemoSettingsTests.test_time_varying_channel_is_shared_by_pilot_and_data -v`

Expected: the current data model omits the accumulated channel phase and fails the ideal-loss bound.

- [ ] **Step 3: Write minimal implementation**

At `data_epoch_s`, add `channel_phase_rate_rad_per_s * data_epoch_s` to raw and residual AP1 phases and add `channel_phase_rate_rad_per_s / (2*pi)` to raw and residual frequency offsets.

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command and expect the ideal-loss assertion to pass.

### Task 3: Waveform-level Monte Carlo coherent gain

**Files:**
- Modify: `monte_carlo.py`
- Modify: `main_demo.py`
- Modify: `tests/test_monte_carlo.py`

- [ ] **Step 1: Write the failing test**

Patch the real pilot-feedback and waveform-combining functions with wrapping spies, run three trials at one SNR, and assert that each function is called three times.

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\ct183\anaconda3\envs\gpu_torch\python.exe -m unittest tests.test_monte_carlo.MonteCarloTests.test_gain_executes_feedback_and_data_combining_per_trial -v`

Expected: both call counts are zero because the current implementation uses an analytic formula.

- [ ] **Step 3: Write minimal implementation**

For each trial, generate RX pilot feedback from a random AP1 phase, compute conjugate phase weights, combine the two delayed data waveforms, record the full-sync gain, and average gains in linear power before converting to dB.

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command and expect both call-count assertions to pass.

### Task 4: Full verification and publication

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the observable clock epoch and Monte Carlo definition**

State that clock-rate regression uses AP0 timestamps and that Monte Carlo coherent gain now comes from per-trial pilot feedback and waveform synthesis.

- [ ] **Step 2: Run all checks**

Run `verify.ps1 -Mode fast_demo`, then run `main_demo.py --mode formal --output-dir results/formal` and `verify_results.py results/formal`. Expect 72 or more passing tests and 13 passing result checks in both modes.

- [ ] **Step 3: Commit and push**

Commit the tested source, tests, and documentation, then push `main` to `origin/main`.
