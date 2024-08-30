""" Configuration Handling Module

"""

import dataclasses


__all__ = [
    "Config",
]


@dataclasses.dataclass(frozen=True)
class Config:
    # Stores the peak measured voltage of the device's audio output for calibration
    device_output_peak: float   # In volts

    stereo_to_mono_output_frequency: float  # In hertz
    stereo_to_mono_output_peak: float  # In volts

    pre_amplifier_output_frequency: float
    pre_amplifier_output_peak: float

    tone_control_low_output_frequency: float
    tone_control_mid_output_frequency: float
    tone_control_high_output_frequency: float
