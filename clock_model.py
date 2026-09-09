"""节点仿射本地时钟和算法补偿状态。"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class LocalClock:
    """节点本地时钟。

    ``offset_s`` 和 ``fractional_frequency_offset`` 是只供仿真器使用的真值；
    算法通过两个 correction 字段更新补偿，不能直接修改真值。
    """

    offset_s: float = 0.0
    fractional_frequency_offset: float = 0.0
    time_correction_s: float = 0.0
    fractional_frequency_correction: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.offset_s,
                self.fractional_frequency_offset,
                self.time_correction_s,
                self.fractional_frequency_correction,
            )
        ):
            raise ValueError("时钟参数必须为有限数")
        if self.effective_rate <= 0.0:
            raise ValueError("校正后的时钟速率必须大于 0")

    @property
    def effective_rate(self) -> float:
        """返回校正后本地秒相对真秒的比例。"""

        return 1.0 + self.fractional_frequency_offset - self.fractional_frequency_correction

    def read_time(self, true_time_s: float) -> float:
        """在隐藏真时间处读取校正后的本地时间，单位为秒。"""

        if not math.isfinite(true_time_s):
            raise ValueError("true_time_s 必须为有限数")
        return self.effective_rate * true_time_s + self.offset_s + self.time_correction_s

    def true_time_for_reading(self, local_time_s: float) -> float:
        """供事件仿真器把本地调度时刻反解为隐藏真时间。"""

        if not math.isfinite(local_time_s):
            raise ValueError("local_time_s 必须为有限数")
        return (local_time_s - self.offset_s - self.time_correction_s) / self.effective_rate

    def apply_time_correction(self, correction_s: float) -> None:
        """累加由双向时间传递估计出的时钟校正量。"""

        if not math.isfinite(correction_s):
            raise ValueError("correction_s 必须为有限数")
        self.time_correction_s += correction_s

    def _apply_frequency_correction(
        self,
        correction_fraction: float,
        effective_true_time_s: float | None,
    ) -> None:
        """更新速率；给出生效真时刻时保持该时刻的本地时间连续。"""

        if not math.isfinite(correction_fraction):
            raise ValueError("correction_fraction 必须为有限数")
        if 1.0 + self.fractional_frequency_offset - correction_fraction <= 0.0:
            raise ValueError("频率校正会产生非正时钟速率")
        if effective_true_time_s is not None:
            if not math.isfinite(effective_true_time_s):
                raise ValueError("effective_true_time_s 必须为有限数或 None")
            old_reading = self.read_time(effective_true_time_s)
            self.fractional_frequency_correction = correction_fraction
            self.time_correction_s += old_reading - self.read_time(effective_true_time_s)
            return
        self.fractional_frequency_correction = correction_fraction

    def apply_frequency_correction(
        self,
        correction_fraction: float,
        *,
        effective_true_time_s: float | None = None,
    ) -> None:
        """应用无量纲时钟速率校正，并可在生效时保持时间连续。"""

        self._apply_frequency_correction(correction_fraction, effective_true_time_s)

    def raw_offset_at(self, true_time_s: float) -> float:
        """返回未应用算法控制时 AP1 相对理想时钟的仿真真偏差。"""

        if not math.isfinite(true_time_s):
            raise ValueError("true_time_s 必须为有限数")
        return float(self.offset_s + self.fractional_frequency_offset * true_time_s)

    def residual_offset_at(self, true_time_s: float) -> float:
        """返回全部已应用控制生效后的物理时间残差。"""

        return float(self.read_time(true_time_s) - true_time_s)
