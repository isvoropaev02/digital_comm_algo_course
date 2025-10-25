from dataclasses import dataclass


@dataclass
class ConfigParams:
    fs_hz: int
    fc_hz: int
    f_symb_hz: int
