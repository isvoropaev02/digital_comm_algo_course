from src.arinc_constants import PROBE_SEQ, PREAMBULE_A, PREAMBULE_M
from src.psk_modem import PSKModem

import numpy as np


class Equalizer:
    def __init__(self, mu: float = 0.1, filter_order: int = 15) -> None:
        self.__mu = mu
        self.__eq_w = np.zeros(filter_order, dtype=np.complex64)
        self.__x_buff = np.zeros(self.__eq_w.shape[0], dtype=np.complex64)
        self.__pmod = PSKModem(bits_per_sample=1)
        self.__probe_seq_mod = self.__pmod.modulate(PROBE_SEQ)

    def reset_filters(self) -> None:
        self.__init__(self.mu, self.__eq_w.shape[0])

    @property
    def mu(self) -> float:
        return self.__mu

    @property
    def filter_order(self) -> float:
        return self.__eq_w.shape[0]

    def process(self, in_samples: np.ndarray, shift: int, data_len: int) -> np.ndarray:
        ref_samples = self.__form_ref_samples(shift=shift)

        # debug
        corr_a = np.correlate(in_samples, ref_samples, "valid")
        import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(np.arange(len(corr_a)), np.abs(corr_a))
        # plt.title("Equalizer_correlation")
        # plt.grid()
        # plt.show()

        train_samples = in_samples[: ref_samples.shape[0]]
        self.__train_filter(train_samples, ref_samples)
        samples_wo_preambule = in_samples[ref_samples.shape[0] :]
        return self.__process_equalization(samples_wo_preambule, data_len)

    def __form_ref_samples(self, shift: int) -> np.ndarray:
        shifted_m_seq = np.roll(PREAMBULE_M, -shift)
        full_preambule = np.concatenate([PREAMBULE_A, PREAMBULE_A, shifted_m_seq, shifted_m_seq[:15], np.tile(PROBE_SEQ, 9)])
        return self.__pmod.modulate(full_preambule)

    def __train_filter(self, in_samples: np.ndarray, ref_samples: np.ndarray) -> None:
        assert in_samples.shape == ref_samples.shape
        # print("Start training")
        err = 0.0 + 0.0j
        for j_smp in range(len(in_samples)):
            self.__x_buff[-1] = in_samples[j_smp]
            y = np.inner(np.conj(self.__eq_w), self.__x_buff)
            err = ref_samples[j_smp] - y
            # print(np.abs(err))
            self.__eq_w = self.__eq_w + self.__mu * self.__x_buff * np.conj(err)
            self.__x_buff = np.roll(self.__x_buff, -1)
        # print("End training")
        print(np.abs(err))

    def __equalize_data_group(self, in_samples: np.ndarray) -> np.ndarray:
        assert in_samples.shape[0] == 45, f"got: {in_samples.shape[0]}"
        in_data = in_samples[:30]
        out_data = np.zeros_like(in_data)
        # self.__x_buff = np.zeros_like(self.__x_buff)
        for j_smp in range(len(in_data)):
            self.__x_buff[-1] = in_data[j_smp]
            out_data[j_smp] = np.inner(np.conj(self.__eq_w), self.__x_buff)
            self.__x_buff = np.roll(self.__x_buff, -1)
        # self.__x_buff = np.zeros_like(self.__x_buff)
        self.__train_filter(in_samples[30:], self.__probe_seq_mod)
        return out_data

    def __process_equalization(self, in_samples: np.ndarray, data_len: int) -> np.ndarray:
        num_data_groups = int(data_len / 45)
        out_data_samples = np.zeros(30 * num_data_groups, dtype=np.complex64)
        for j_gr in range(num_data_groups):
            out_data_samples[j_gr * 30 : (j_gr + 1) * 30] = self.__equalize_data_group(in_samples[j_gr * 45 : (j_gr + 1) * 45])
        return out_data_samples

    def run_ut(self) -> None:
        self.reset_filters()
        in_samples = np.concat(
            [self.__form_ref_samples(shift=72) * (0.5 + 0.5j), self.__form_ref_samples(shift=72) * (0.5 + 0.5j)]
        )
        in_samples += (np.random.randn(in_samples.shape[0]) + 1j * np.random.randn(in_samples.shape[0])) * np.sqrt(0.01 / 2)
        ref_samples = self.__form_ref_samples(shift=72)

        # debug
        corr_a = np.correlate(in_samples, ref_samples, "valid")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(np.arange(len(corr_a)), np.abs(corr_a))
        plt.title("Equalizer_correlation")
        plt.grid()
        plt.show()

        train_samples = in_samples[: ref_samples.shape[0]]
        self.__train_filter(train_samples, ref_samples)
        samples_wo_preambule = in_samples[ref_samples.shape[0] :]
        out_data = np.zeros_like(samples_wo_preambule)
        for j_smp in range(len(samples_wo_preambule)):
            self.__x_buff[-1] = samples_wo_preambule[j_smp]
            out_data[j_smp] = np.inner(np.conj(self.__eq_w), self.__x_buff)
            self.__x_buff = np.roll(self.__x_buff, -1)

        plt.figure()
        plt.scatter(np.real(out_data), np.imag(out_data), label="res")
        plt.scatter(np.real(ref_samples), np.imag(ref_samples), label="ref")
        plt.ylim([-1, 1])
        plt.show()
