import numpy as np
from typing import Dict

class DPSKModem:
    def __init__(self, bits_per_sample: int = 1) -> None:
        if bits_per_sample not in {1, 2, 3}:
            raise ValueError(f"Invalid bits_per_sample={bits_per_sample}, the value should be one of [1, 2, 3]")
        self.__bits_per_sample = bits_per_sample
        self.__lut_table = self.__gen_lut_tables(bits_per_sample)
        self.__base_phase = 2 * np.pi / (2**bits_per_sample)
        self.__ref_mod_phase_id = 0

    def __gen_lut_tables(self, bits_per_sample: int = 1) -> np.ndarray:
        match bits_per_sample:
            case 1: # dbpsk
                return np.array([0, 1], dtype=np.int8)
            case 2: # dqpsk
                return np.array([0, 1, 3, 2], dtype=np.int8)
            case 3: # d8psk
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
    

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % self.__bits_per_sample != 0:
            raise ValueError(f"Number of bits ({len(bits)}) should be the multiple of {self.__bits_per_sample}")
        
        num_symbols = len(bits) // self.__bits_per_sample
        out_phases = np.zeros(num_symbols + 1, dtype=np.int8) # +1 for ref phase
        out_phases[0] = self.__lut_table[self.__ref_mod_phase_id]
        
        for j_samlpe in range(num_symbols):
            start_idx = j_samlpe * self.__bits_per_sample
            end_idx = start_idx + self.__bits_per_sample
            sample_idx = 0
            for j, bit in enumerate(bits[start_idx:end_idx]):
                sample_idx += bit * (1 << (self.__bits_per_sample - 1 - j))
                out_phases[j_samlpe + 1] = self.__lut_table[sample_idx]
        
        

        self.__ref_mod_phase_id = (self.__ref_mod_phase_id + 1) % (2 ** self.__bits_per_sample)
        return np.exp(1j * self.__base_phase * np.diff(out_phases, prepend=np.array([0], dtype=np.int8)), dtype=np.complex64)

    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        restored_phases = (np.round(np.angle(np.cumprod(samples)[1:]) / self.__base_phase)
                           % (2**self.__bits_per_sample)).astype(np.int8)
        return np.array([self.__from_int_to_bits(np.where(self.__lut_table == int_phase)[0][0]) for int_phase in restored_phases], dtype=np.int8).flatten()
    


if __name__ == "__main__":
    modulators = {
        'dbpsk': DPSKModem(bits_per_sample=1),
        'dqpsk': DPSKModem(bits_per_sample=2),
        'd8psk': DPSKModem(bits_per_sample=3)
    }
    
    for name, modulator in modulators.items():
        print(f"\n{name}:")
        print(f"Bits per sample: {modulator.bits_per_sample}")
    
    # Тестирование работы модулятора/демодулятора
    print("\n" + "="*50)
    print("Testing:")
    
    test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1])
    print(f"Message: {test_bits}")
    
    samples_dict = dict()
    for name, modulator in modulators.items():
        print(f"\n{name}:")
        samples_dict[name] = modulator.modulate(test_bits)
        print(f"Samples: {samples_dict[name]}")
    
    bits_dict = dict()
    for name, modulator in modulators.items():
        print(f"\n{name}:")
        bits_dict[name] = modulator.demodulate(samples_dict[name])
        print(f"Received bits: {bits_dict[name]}")
        print(f"Correct: {np.array_equal(test_bits, bits_dict[name])}")