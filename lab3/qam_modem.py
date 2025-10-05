import numpy as np
import matplotlib.pyplot as plt

class QAMModem:
    """
    ETSI TS 138 211 V18.2.0 Section 5.1: QAM mapper
    """
    
    def __init__(self, bits_per_sample: int = 2):
        if bits_per_sample not in {1, 2, 4, 6, 8}:
            raise ValueError(f"Invalid bits_per_sample={bits_per_sample}, the value should be one of [1, 2, 4, 6, 8, 10]")
        self.__bits_per_sample = bits_per_sample
        self.__lut_table = self.__gen_lut_tables(bits_per_sample)

    def __gen_lut_tables(self, bits_per_sample: int = 2) -> np.ndarray:
        match bits_per_sample:
            case 1: # pi/2-bpsk
                print("[WARNING] pi/2-bpsk not supported yet")
                return self.__gen_lut_tables(2)
            case 2: # qpsk
                return np.array([((1-2*(1&(id>>1))) + 1j*(1-2*(1&(id>>0)))) / np.sqrt(2) for id in range(2**2)], dtype=np.complex64)
            case 4: # qam16
                return np.array([((1 - 2*(1&(id>>3))) * (2 - (1 - 2*(1&(id>>1))))
                                  + 1j * (1 - 2*(1&(id>>2))) * (2 - (1 - 2*(1&(id>>0)))))
                                  / np.sqrt(10) for id in range(2**4)], dtype=np.complex64)
            case 6: # qam64
                return np.array([((1 - 2*((i >> 5) & 1)) * (4 - (1 - 2*((i >> 3) & 1)) * (2 - (1 - 2*((i >> 1) & 1)))) + 
                                1j * (1 - 2*((i >> 4) & 1)) * (4 - (1 - 2*((i >> 2) & 1)) * (2 - (1 - 2*((i >> 0) & 1))))) / np.sqrt(42)
                                for i in range(2**6)], dtype=np.complex64)
            case 8: # qam256
                return np.array([((1 - 2*((i >> 7) & 1)) * (8 - (1 - 2*((i >> 5) & 1)) * (4 - (1 - 2*((i >> 3) & 1)) * (2 - (1 - 2*((i >> 1) & 1))))) + 
                                1j * (1 - 2*((i >> 6) & 1)) * (8 - (1 - 2*((i >> 4) & 1)) * (4 - (1 - 2*((i >> 2) & 1)) * (2 - (1 - 2*((i >> 0) & 1)))))) / np.sqrt(170)
                                for i in range(2**8)], dtype=np.complex64)

            case 10: # qam1024
                return np.array([((1 - 2*((i >> 9) & 1)) * (16 - (1 - 2*((i >> 7) & 1)) * (8 - (1 - 2*((i >> 5) & 1)) * (4 - (1 - 2*((i >> 3) & 1)) * (2 - (1 - 2*((i >> 1) & 1)))))) + 
                                1j * (1 - 2*((i >> 8) & 1)) * (16 - (1 - 2*((i >> 6) & 1)) * (8 - (1 - 2*((i >> 4) & 1)) * (4 - (1 - 2*((i >> 2) & 1)) * (2 - (1 - 2*((i >> 0) & 1))))))) / np.sqrt(682)
                                for i in range(2**10)], dtype=np.complex64)
            case _:
                return np.array([0], dtype=np.complex64)

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
    

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % self.__bits_per_sample != 0:
            raise ValueError(f"Number of bits ({len(bits)}) should be the multiple of {self.__bits_per_sample}")
        
        # Преобразование битов в десятичные индексы
        num_symbols = len(bits) // self.__bits_per_sample
        out_samples = np.zeros(num_symbols, dtype=np.complex64)
        
        for j_samlpe in range(num_symbols):
            start_idx = j_samlpe * self.__bits_per_sample
            end_idx = start_idx + self.__bits_per_sample
            sample_idx = 0
            for j, bit in enumerate(bits[start_idx:end_idx]):
                sample_idx += bit * (1 << (self.__bits_per_sample - 1 - j))
            out_samples[j_samlpe] = self.__lut_table[sample_idx]
        return out_samples
    
    def demodulate(self, samples: np.ndarray, method: str = 'hard') -> np.ndarray:
        if method == 'hard':
            return self._hard_demodulate(samples)
        elif method == 'soft':
            return self._soft_demodulate(samples)
        else:
            raise ValueError("Method must be in ['hard', 'soft']")
    
    def _hard_demodulate(self, samples: np.ndarray) -> np.ndarray:
        distances = np.abs(samples[:, np.newaxis] - self.__lut_table)
        closest_indices = np.argmin(distances, axis=1)
        return np.array([self.__from_int_to_bits(id) for id in closest_indices], dtype=np.int8).flatten()
    
    def _soft_demodulate(self, samples: np.ndarray) -> np.ndarray:
        distances = np.abs(samples[:, np.newaxis] - self.__lut_table)
        closest_indices = np.argmin(distances, axis=1)
        out_llrs = np.zeros(self.__bits_per_sample * closest_indices.shape[0], dtype=np.float32)
        for j_sample in range(closest_indices.shape[0]):
            out_bits = self.__from_int_to_bits(closest_indices[j_sample])
            for j_bit in range(out_bits.shape[0]):
                out_llrs[j_sample * self.__bits_per_sample + j_bit] \
                        = np.exp(-distances[j_sample]**2) if out_bits == 0 else -np.exp(-distances[j_sample]**2)
        return out_llrs

# Демонстрация работы улучшенной визуализации
if __name__ == "__main__":
    # Создание модуляторов разных порядков и демонстрация визуализации
    modulators = {
        'QPSK': QAMModem(bits_per_sample=2),
        '16-QAM': QAMModem(bits_per_sample=4),
        '64-QAM': QAMModem(bits_per_sample=6)
    }
    
    print("Демонстрация визуализации созвездий с подписями битовых комбинаций:")
    
    for name, modulator in modulators.items():
        print(f"\n{name}:")
        print(f"Бит на символ: {modulator.bits_per_sample}")
    
    # Тестирование работы модулятора/демодулятора
    print("\n" + "="*50)
    print("Тестирование модуляции/демодуляции:")
    
    modulator_16qam = QAMModem(bits_per_sample=4)
    
    # Генерация тестовых битов
    test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # Два символа 16-QAM
    print(f"Исходные биты: {test_bits}")
    
    # Модуляция
    symbols = modulator_16qam.modulate(test_bits)
    print(f"Модулированные символы: {symbols}")
    
    # Покажем, каким битовым комбинациям соответствуют символы
    bits_per_sample = modulator_16qam.bits_per_sample
    for i, symbol in enumerate(symbols):
        start_bit = i * bits_per_sample
        end_bit = start_bit + bits_per_sample
        symbol_bits = test_bits[start_bit:end_bit]
        bit_string = ''.join(str(bit) for bit in symbol_bits)
        print(f"Символ {i}: {symbol:.3f} соответствует битам {bit_string}")
    
    # Демодуляция
    demod_bits = modulator_16qam.demodulate(symbols, method='hard')
    print(f"Демодулированные биты: {demod_bits}")
    print(f"Биты совпадают: {np.array_equal(test_bits, demod_bits)}")