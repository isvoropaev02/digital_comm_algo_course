from src.arinc_constants import PROBE_SEQ, PREAMBULE_A, PREAMBULE_M
from src.psk_modem import PSKModem

import numpy as np
from scipy.signal import decimate


class Equalizer:
    def __init__(self, fs_hz: int, fsymb_hz: int) -> None:
        psk_modem = PSKModem(bits_per_sample=1)
        self.__mod_probe_seq = psk_modem.modulate(PROBE_SEQ)
        self.__samples_per_symb = int(fs_hz / fsymb_hz)
        self.__mu = 0.1
        self.__lmb = 0.98
        self.__filt_order = 15
        self.__dfe_order = 6
        self.__pmod = PSKModem(bits_per_sample=1)
        self.__eq_w = np.zeros(self.__filt_order)

    @property
    def samples_per_symb(self) -> int:
        return self.__samples_per_symb

    @property
    def mu(self) -> float:
        return self.__mu

    @property
    def lmb(self) -> float:
        return self.__lmb

    def process(self, in_samples: np.ndarray, shift: int) -> None:
        ref_samples = self.__form_ref_samples(shift=shift)
        train_samples = in_samples[: ref_samples.shape[0]]
        self.__train_filter(train_samples, ref_samples)
        samples_wo_preambule = in_samples[ref_samples.shape[0] :]

    def __form_ref_samples(self, shift: int) -> np.ndarray:
        shifted_m_seq = np.roll(PREAMBULE_M, -shift)
        full_preambule = np.concatenate([PREAMBULE_A, PREAMBULE_A, shifted_m_seq, shifted_m_seq[:15], np.tile(PROBE_SEQ, 9)])
        return self.__pmod.modulate(full_preambule)

    def __train_filter(self, in_samples: np.ndarray, ref_samples: np.ndarray) -> None:  # using RLS algorithm
        assert in_samples.shape == ref_samples.shape
        x_arr = np.zeros(self.__eq_w.shape[0], dtype=np.complex64)
        for j_smp in range(len(in_samples)):
            x_arr[-1] = in_samples[0]
            y = np.inner(np.conj(self.__eq_w), x_arr)
            err = y - ref_samples[j_smp]
            self.__eq_w = self.__eq_w - self.__mu * x_arr[-1] * np.conj(err)
            x_arr = np.roll(x_arr, -1)
