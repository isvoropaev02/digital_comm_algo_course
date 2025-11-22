from src.arinc_constants import PREAMBULE_A, PREAMBULE_M, M_SHIFT_LUT, DATA_RATE_LUT
from src.psk_modem import PSKModem

import numpy as np
from numpy.fft import fft, ifft


class PreambuleDetector:
    def __init__(self, fs_hz: int, fsymb_hz: int) -> None:
        self.__psk_modem = PSKModem(bits_per_sample=1)
        self.__samples_per_symb = int(fs_hz / fsymb_hz)

    @property
    def samples_per_symb(self) -> int:
        return self.__samples_per_symb

    def process(self, in_signal: np.ndarray) -> tuple:
        a0_idx, a1_idx = tuple(self.__get_preambule_a_start_pos(in_signal))
        if (a1_idx - a0_idx) != self.__samples_per_symb * 127:
            print("[WARNING] samples_per_symb*127 not equal to a1_idx-a0_idx")

        m_shift = self.__get_preambule_m_shift(in_signal, start_sample_id=a1_idx)  # for corr with [M, M]
        id_shift = np.argmin(np.abs(m_shift - M_SHIFT_LUT))
        out_data_rate = DATA_RATE_LUT[id_shift % 4]
        out_interleaver_dur = 1.8 if (id_shift // 2 == 0) else 4.2
        out_symbols = in_signal[a0_idx :: self.__samples_per_symb]
        print("shift: ", m_shift, " id: ", id_shift)
        print("interl: ", out_interleaver_dur, " dr: ", out_data_rate)
        return (out_symbols, out_data_rate, out_interleaver_dur)

    def __get_preambule_a_start_pos(self, in_signal: np.ndarray) -> np.ndarray:
        preambule_a_modulated = np.repeat(self.__psk_modem.modulate(PREAMBULE_A), self.__samples_per_symb).astype(np.complex128)
        corr_a = np.correlate(in_signal, preambule_a_modulated, "full")
        max_indices = np.argpartition(np.abs(corr_a), -2)[-2:]
        return np.sort(max_indices)

    def __get_preambule_m_shift(self, in_signal: np.ndarray, start_sample_id: int) -> int:
        preambule_m_modulated = np.repeat(self.__psk_modem.modulate(PREAMBULE_M), self.__samples_per_symb).astype(np.complex128)
        signal_m_part = in_signal[start_sample_id : (start_sample_id + (2 * 127 + 15) * self.__samples_per_symb)]
        ## fft correlation - !TODO
        # corr_m = ifft(fft(signal_m_part, norm="ortho") * np.conj(fft(preambule_m_modulated, norm="ortho")), norm="ortho")
        ## simple correlation
        corr_m = np.correlate(signal_m_part, np.concat([preambule_m_modulated, preambule_m_modulated]), "full")
        max_idx = np.argmax(np.abs(corr_m))
        shift = int((self.__samples_per_symb * 127 * 2 - max_idx) / self.__samples_per_symb)
        return shift
