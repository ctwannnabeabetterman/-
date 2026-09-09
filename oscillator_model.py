"""AP1 本振真频偏与算法频率控制状态。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class LocalOscillator:
    """把仿真真值与控制量分开保存的连续相位本振模型。"""

    frequency_offset_hz: float = 0.0
    initial_phase_rad: float = 0.0
    frequency_correction_hz: float = 0.0
    phase_correction_rad: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.frequency_offset_hz,
            self.initial_phase_rad,
            self.frequency_correction_hz,
            self.phase_correction_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("本振参数必须为有限数")

    @property
    def residual_frequency_offset_hz(self) -> float:
        """返回应用控制后的载波频偏，单位为 Hz。"""

        return float(self.frequency_offset_hz - self.frequency_correction_hz)

    def raw_phase_at(self, true_time_s: float) -> float:
        """返回未应用控制的连续仿真真相位。"""

        if not math.isfinite(true_time_s):
            raise ValueError("true_time_s 必须为有限数")
        return float(self.initial_phase_rad + 2.0 * np.pi * self.frequency_offset_hz * true_time_s)

    def phase_at(self, true_time_s: float) -> float:
        """返回应用频率控制后的连续相位。"""

        if not math.isfinite(true_time_s):
            raise ValueError("true_time_s 必须为有限数")
        return float(
            self.initial_phase_rad
            + self.phase_correction_rad
            + 2.0 * np.pi * self.residual_frequency_offset_hz * true_time_s
        )

    def apply_frequency_correction(
        self,
        correction_hz: float,
        *,
        effective_time_s: float,
    ) -> None:
        """应用 Hz 制本振校正，并保持生效真时刻的相位连续。"""

        if not math.isfinite(correction_hz) or not math.isfinite(effective_time_s):
            raise ValueError("频率校正和生效时刻必须为有限数")
        phase_before = self.phase_at(effective_time_s)
        self.frequency_correction_hz = correction_hz
        phase_after = self.phase_at(effective_time_s)
        self.phase_correction_rad += phase_before - phase_after
