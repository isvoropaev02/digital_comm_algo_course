from arinc_constants import PREAMBULE_A, PREAMBULE_M
from psk_modem import PSKModem
from shape_filter import ShapeFilter

import numpy as np


class PreambuleDetector:
    def __init__(self, fs_hz: int, fsymb_hz: int) -> None:
        self.__psk_modem = PSKModem(bits_per_sample=1)
        self.__fs_hz = fs_hz
        self.__fsymb_hz = fsymb_hz
        self.__samples_per_symb = int(fs_hz / fsymb_hz)
        self.__shp_filter = ShapeFilter(samples_per_symbol=self.__samples_per_symb)

    @property
    def samples_per_symb(self) -> int:
        return self.__samples_per_symb

    def process(self) -> None:
        pass

    def __find_data_params(self) -> tuple:
        data_rate_hz = 1800
        mod_order = 3
        chips_per_symb = 3

        return (data_rate_hz, mod_order, chips_per_symb)

    def __get_clock_synch(self, in_signal: np.ndarray) -> np.ndarray:
        # 1. restoring preambule A signal
        preamb_a_symbs = self.__psk_modem.modulate(PREAMBULE_A)
        preamb_a_symbs_upsampled = np.zeros(self.__samples_per_symb * preamb_a_symbs.shape[0], dtype=np.complex64)
        preamb_a_symbs_upsampled[:: self.__samples_per_symb] = preamb_a_symbs
        preamb_a_signal = self.__shp_filter.apply_filter(preamb_a_symbs_upsampled)

        # 2. finding correlation

        return preamb_a_symbs
