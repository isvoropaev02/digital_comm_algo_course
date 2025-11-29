from src.arinc_constants import PREAMBULE_A, PREAMBULE_M, M_SHIFT_LUT, DATA_RATE_LUT, PROBE_SEQ
from src.psk_modem import PSKModem

import numpy as np
from numpy.fft import fft, ifft


class PreambuleDetector:
    def __init__(self, fs_hz: int, fsymb_hz: int) -> None:
        self.__pmod = PSKModem(bits_per_sample=1)
        self.__samples_per_symb = int(fs_hz / fsymb_hz)

    @property
    def samples_per_symb(self) -> int:
        return self.__samples_per_symb

    def process(self, in_signal: np.ndarray) -> tuple:
        a0_idx, a1_idx = tuple(self.__get_preambule_a_start_pos(in_signal))
        if (a1_idx - a0_idx) != self.__samples_per_symb * 127:
            print("[WARNING] samples_per_symb*127 not equal to a1_idx-a0_idx")
            print("[WARNING] a1_idx is set to a0_idx + 127*num_samples_per_symb")
            a1_idx = a0_idx + 127 * self.__samples_per_symb

        m_shift = self.__get_preambule_m_shift(in_signal, start_sample_id=a1_idx)  # for corr with [M, M]
        id_shift = np.argmin(np.abs(m_shift - M_SHIFT_LUT))
        out_data_rate = DATA_RATE_LUT[id_shift % 4]
        out_interleaver_dur = 1800 if (id_shift // 2 == 0) else 4200  # ms
        out_symbols = in_signal[a0_idx :: self.__samples_per_symb]

        # ### debug
        # shifted_m_seq = np.roll(PREAMBULE_M, -M_SHIFT_LUT[id_shift])
        # full_preambule = self.__pmod.modulate(
        #     np.concatenate([PREAMBULE_A, PREAMBULE_A, shifted_m_seq, shifted_m_seq[:15], np.tile(PROBE_SEQ, 9)])
        # )
        # # full_preambule = self.__pmod.modulate(PREAMBULE_A)
        # corr_a = np.correlate(out_symbols, full_preambule, "valid")
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(np.arange(len(corr_a)), np.abs(corr_a))
        # plt.title("pr detector corr out")
        # plt.grid()
        # plt.show()
        # print("shift: ", m_shift, " id: ", id_shift)
        # print("interl: ", out_interleaver_dur, " dr: ", out_data_rate)

        return (out_symbols, out_data_rate, out_interleaver_dur, M_SHIFT_LUT[id_shift])

    def __get_preambule_a_start_pos(self, in_signal: np.ndarray) -> np.ndarray:
        preambule_a_modulated = np.repeat(self.__pmod.modulate(PREAMBULE_A), self.__samples_per_symb).astype(np.complex128)
        corr_a = np.correlate(in_signal, preambule_a_modulated, "full")
        max_indices = np.argpartition(np.abs(corr_a), -2)[-2:]
        return np.sort(max_indices) - (preambule_a_modulated.shape[0] - 1)

    def __get_preambule_m_shift(self, in_signal: np.ndarray, start_sample_id: int) -> int:
        preambule_m_modulated = np.repeat(self.__pmod.modulate(PREAMBULE_M), self.__samples_per_symb).astype(np.complex128)
        signal_m_part = in_signal[start_sample_id : (start_sample_id + (2 * len(PREAMBULE_M) + 15) * self.__samples_per_symb)]
        ## fft correlation - !TODO
        # corr_m = ifft(fft(signal_m_part, norm="ortho") * np.conj(fft(preambule_m_modulated, norm="ortho")), norm="ortho")
        ## simple correlation
        corr_m = np.correlate(signal_m_part, np.concat([preambule_m_modulated, preambule_m_modulated]), "full")

        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(np.arange(len(corr_m)) - (2 * len(PREAMBULE_M) * self.__samples_per_symb - 1), np.abs(corr_m))
        # plt.title("Pr detector correlation")
        # plt.grid()
        # plt.show()
        max_idx = np.argmax(np.abs(corr_m)) - (2 * len(PREAMBULE_M) * self.__samples_per_symb - 1)
        shift = int((self.__samples_per_symb * len(PREAMBULE_M) - max_idx) / self.__samples_per_symb)
        return shift


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a = np.zeros(8)
    a[:3] = 1
    plt.figure()
    plt.plot(np.abs(np.correlate(a, a, mode="full")))
    plt.title("correlation")
    plt.grid()
    plt.show()
