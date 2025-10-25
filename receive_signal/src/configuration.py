from dataclasses import dataclass


@dataclass
class ConfigParams:
    fs_hz: int
    fc_hz: int
    fsymb_hz: int
