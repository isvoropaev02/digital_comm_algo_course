import numpy as np


class PSKModem:
    def __init__(self, bits_per_sample: int = 1) -> None:
        if bits_per_sample not in {1, 2, 3}:
            raise ValueError(f"Invalid bits_per_sample={bits_per_sample}, the value should be one of [1, 2, 3]")
        self.__bits_per_sample = bits_per_sample
        self.__lut_table = self.__gen_lut_tables(bits_per_sample)
        self.__base_phase = 2 * np.pi / (2**bits_per_sample)

    def __gen_lut_tables(self, bits_per_sample: int = 1) -> np.ndarray:
        match bits_per_sample:
            case 1:  # bpsk
                return np.array([0, 1], dtype=np.int8)
            case 2:  # qpsk
                return np.array([0, 1, 3, 2], dtype=np.int8)
            case 3:  # 8psk
                return np.array([0, 1, 3, 2, 7, 6, 4, 5], dtype=np.int8)
            case _:
                return np.array([0], dtype=np.int8)

    def __from_int_to_bits(self, val: int) -> np.ndarray:
        out = np.zeros(self.__bits_per_sample, dtype=np.uint8)
        for j_pos in range(self.__bits_per_sample):
            out[j_pos] = 1 & (val >> self.__bits_per_sample - 1 - j_pos)
        return out

    @property
    def bits_per_sample(self) -> int:
        return self.__bits_per_sample

    @property
    def lut_table(self) -> np.ndarray:
        return self.__lut_table

    @property
    def base_phase(self) -> float:
        return self.__base_phase

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % self.__bits_per_sample != 0:
            raise ValueError(f"Number of bits ({len(bits)}) should be the multiple of {self.__bits_per_sample}")

        num_symbols = len(bits) // self.__bits_per_sample
        out_phases = np.zeros(num_symbols, dtype=np.int8)

        for j_samlpe in range(num_symbols):
            start_idx = j_samlpe * self.__bits_per_sample
            end_idx = start_idx + self.__bits_per_sample
            sample_idx = 0
            for j, bit in enumerate(bits[start_idx:end_idx]):
                sample_idx += bit * (1 << (self.__bits_per_sample - 1 - j))
                out_phases[j_samlpe] = self.__lut_table[sample_idx]

        # return np.exp(1j * self.__base_phase * out_phases, dtype=np.complex64)
        return out_phases


if __name__ == "__main__":
    modulators = {
        "bpsk": PSKModem(bits_per_sample=1),
        "qpsk": PSKModem(bits_per_sample=2),
        "8psk": PSKModem(bits_per_sample=3),
    }

    for name, modulator in modulators.items():
        print(f"\n{name}:")
        print(f"Bits per sample: {modulator.bits_per_sample}")

    # Тестирование работы модулятора/демодулятора
    print("\n" + "=" * 50)
    print("Testing:")

    test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1])
    print(f"Message: {test_bits}")

    samples_dict = dict()
    for name, modulator in modulators.items():
        print(f"\n{name}:")
        samples_dict[name] = modulator.modulate(test_bits)
        print(f"Samples: {samples_dict[name]}")

    # bits_dict = dict()
    # for name, modulator in modulators.items():
    #     print(f"\n{name}:")
    #     bits_dict[name] = modulator.demodulate(samples_dict[name] * (1 - 1j))
    #     print(f"Received bits: {bits_dict[name]}")
    #     print(f"Correct: {np.array_equal(test_bits, bits_dict[name])}")
