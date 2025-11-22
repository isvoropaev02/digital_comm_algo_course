from src.arinc_constants import PROBE_SEQ, PREAMBULE_A, PREAMBULE_M
from src.psk_modem import PSKModem

import numpy as np


class Equalizer:
    def __init__(self, mu: float = 0.1, filter_order: int = 15) -> None:
        self.__mu = 0.1
        self.__eq_w = np.zeros(filter_order)
        self.__x_buff = np.zeros(self.__eq_w.shape[0], dtype=np.complex64)
        self.__pmod = PSKModem(bits_per_sample=1)
        self.__probe_seq_mod = self.__pmod.modulate(PROBE_SEQ)

    @property
    def mu(self) -> float:
        return self.__mu

    @property
    def filter_order(self) -> float:
        return self.__eq_w.shape[0]

    def process(self, in_samples: np.ndarray, shift: int, data_len: int) -> np.ndarray:
        ref_samples = self.__form_ref_samples(shift=shift)
        train_samples = in_samples[: ref_samples.shape[0]]
        self.__train_filter(train_samples, ref_samples)
        samples_wo_preambule = in_samples[ref_samples.shape[0] :]
        return self.__process_equalization(samples_wo_preambule, data_len)

    def __form_ref_samples(self, shift: int) -> np.ndarray:
        shifted_m_seq = np.roll(PREAMBULE_M, -shift)
        full_preambule = np.concatenate([PREAMBULE_A, PREAMBULE_A, shifted_m_seq, shifted_m_seq[:15], np.tile(PROBE_SEQ, 9)])
        return self.__pmod.modulate(full_preambule)

    def __train_filter(self, in_samples: np.ndarray, ref_samples: np.ndarray) -> None:  # using RLS algorithm
        assert in_samples.shape == ref_samples.shape
        for j_smp in range(len(in_samples)):
            self.__x_buff[-1] = in_samples[0]
            y = np.inner(np.conj(self.__eq_w), self.__x_buff)
            err = y - ref_samples[j_smp]
            self.__eq_w = self.__eq_w - self.__mu * self.__x_buff[-1] * np.conj(err)
            self.__x_buff = np.roll(self.__x_buff, -1)

    def __equalize_data_group(self, in_samples: np.ndarray) -> np.ndarray:
        assert in_samples.shape[0] == 45
        in_data = in_samples[:30]
        out_data = np.zeros_like(in_data)
        for j_smp in range(len(in_data)):
            self.__x_buff[-1] = in_samples[0]
            out_data[j_smp] = np.inner(np.conj(self.__eq_w), self.__x_buff)
            self.__x_buff = np.roll(self.__x_buff, -1)
        self.__train_filter(in_samples[30:], self.__probe_seq_mod)
        return out_data

    def __process_equalization(self, in_samples: np.ndarray, data_len: int) -> np.ndarray:
        num_data_groups = int(data_len / 45)
        out_data_samples = np.zeros(30 * num_data_groups, dtype=np.complex64)
        for j_gr in range(num_data_groups):
            out_data_samples[j_gr * 30 : (j_gr + 1) * 30] = self.__equalize_data_group(in_samples[j_gr * 30 : (j_gr + 1) * 30])
        return out_data_samples
