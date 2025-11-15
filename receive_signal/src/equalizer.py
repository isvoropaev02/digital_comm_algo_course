from src.arinc_constants import PROBE_SEQ
from src.psk_modem import PSKModem

import numpy as np
from scipy.signal import decimate


class LMSEqualizer:
    def __init__(self, fs_hz: int, fsymb_hz: int, mu: float = 0.1, eq_len: int = 3) -> None:
        psk_modem = PSKModem(bits_per_sample=1)
        self.__mod_probe_seq = psk_modem.modulate(PROBE_SEQ)
        self.__samples_per_symb = int(fs_hz / fsymb_hz)
        self.__mu = mu

    @property
    def samples_per_symb(self) -> int:
        return self.__samples_per_symb

    @property
    def mu(self) -> float:
        return self.__mu

    def process(self, in_signal: np.ndarray) -> None:
        pass
