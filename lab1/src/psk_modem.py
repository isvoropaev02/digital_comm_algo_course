import numpy as np
from copy import deepcopy

class PSKModem:

    __bit_symbol_map = dict()
    __name = ''
    __bits_per_symb = 0

    def __init__(self, name: str) -> None:
        if name == 'bpsk':
            self.__bits_per_symb = 1
            self.__bit_symbol_map[(1,)] = self.__bit_symbol_map.get((1,), np.pi)
            self.__bit_symbol_map[(0,)] = self.__bit_symbol_map.get((0,), 0)
        elif name == 'qpsk':
            self.__bits_per_symb = 2
            self.__bit_symbol_map[(0,1)] = self.__bit_symbol_map.get((0,1), np.pi/2)
            self.__bit_symbol_map[(0,0)] = self.__bit_symbol_map.get((0,0), 0)
            self.__bit_symbol_map[(1,0)] = self.__bit_symbol_map.get((1,0), 3*np.pi/2)
            self.__bit_symbol_map[(1,1)] = self.__bit_symbol_map.get((1,1), np.pi)
        elif name == '8psk':
            self.__bits_per_symb = 3
            self.__bit_symbol_map[(0,0,0)] = self.__bit_symbol_map.get((0,0,0), 0)
            self.__bit_symbol_map[(0,0,1)] = self.__bit_symbol_map.get((0,0,1), np.pi/4)
            self.__bit_symbol_map[(0,1,1)] = self.__bit_symbol_map.get((0,1,1), np.pi/2)
            self.__bit_symbol_map[(0,1,0)] = self.__bit_symbol_map.get((0,1,0), 3*np.pi/4)
            self.__bit_symbol_map[(1,1,0)] = self.__bit_symbol_map.get((1,1,0), np.pi)
            self.__bit_symbol_map[(1,1,1)] = self.__bit_symbol_map.get((1,1,1), 5*np.pi/4)
            self.__bit_symbol_map[(1,0,1)] = self.__bit_symbol_map.get((1,0,1), 3*np.pi/2)
            self.__bit_symbol_map[(1,0,0)] = self.__bit_symbol_map.get((1,0,0), 7*np.pi/4)
        else:
            raise(ValueError('No match for modulator '+name+' from [\'bpsk\', \'qpsk\',\'8psk\'])'))
        self.__name = name
        
    
    def map_symbols(self, mes: np.ndarray) -> np.ndarray:
        len_mes = len(mes)
        if len_mes % self.__bits_per_symb != 0:
            raise ValueError('Length of message must be divisible by '+str(self.__bits_per_symb)+' for ' + self.__name + ' modulation')
        output_size = len(mes) // self.__bits_per_symb
        output = np.zeros(output_size, dtype=np.complex128)
        for j_symbol in range(output_size):
            angle = self.__bit_symbol_map.get(tuple(mes[self.__bits_per_symb*j_symbol:self.__bits_per_symb*j_symbol+self.__bits_per_symb]), 0.)
            output[j_symbol] = np.exp(1j*angle)
        return output
    
    def demap_symbols(self, input: np.ndarray) -> np.ndarray:
        bit_message = np.zeros(len(input)*self.__bits_per_symb, dtype=int)
        for j_symb in range(len(input)):
            bit_message[j_symb*self.__bits_per_symb:(j_symb+1)*self.__bits_per_symb] = \
                np.array(list(self.__demap_single_symbol(input[j_symb])))
        return bit_message
    
    def modulate(self, mes: np.ndarray, fs_hz: float, fc_hz:float, symb_rate_hz: float) -> np.ndarray:
        len_mes = len(mes)
        if len_mes % self.__bits_per_symb != 0:
            raise(ValueError('Length of message must be divisible by '+str(self.__bits_per_symb)+' for ' + self.__name + 'modulation'))
        num_symb = len(mes) // self.__bits_per_symb
        samples_per_symb = int(fs_hz/symb_rate_hz)
        output_signal = np.zeros(int(num_symb*samples_per_symb))
        time_samples = np.arange(0, samples_per_symb, 1)
        for j_symb in range(num_symb):
            phase = self.__bit_symbol_map.get(tuple(mes[self.__bits_per_symb*j_symb:self.__bits_per_symb*j_symb+self.__bits_per_symb]), 0.)
            output_signal[j_symb*samples_per_symb:(j_symb+1)*samples_per_symb] = np.cos(phase+2*np.pi*fc_hz*time_samples/fs_hz)
        return output_signal
    
    def demodulate(self, signal: np.ndarray, fs_hz, fc_hz, symb_rate_hz) -> tuple[np.ndarray, dict]:
        samples_per_symb = int(fs_hz/symb_rate_hz)
        num_symb = int(len(signal)/samples_per_symb)
        output_bits = np.zeros(num_symb*self.__bits_per_symb, dtype=int)
        time_samples = np.arange(0, samples_per_symb, 1)
        h_dict = deepcopy(self.__bit_symbol_map)
        for j_symb in range(num_symb):
            bits = np.array([])
            h_target = 0
            for key_bits, ref_phase in self.__bit_symbol_map.items():
                ref_wave = np.cos(ref_phase+2*np.pi*fc_hz*time_samples/fs_hz)
                h_current = np.dot(signal[j_symb*samples_per_symb:(j_symb+1)*samples_per_symb], ref_wave) / samples_per_symb
                h_dict[key_bits] = h_current
                if len(bits) == 0:
                    bits = np.array(list(key_bits))
                    h_target = h_current
                else:
                    if h_current > h_target:
                        bits = np.array(list(key_bits))
                        h_target = h_current
            output_bits[j_symb*self.__bits_per_symb:(j_symb+1)*self.__bits_per_symb] = bits
        return output_bits, h_dict
    
    def padded_symbols(self, mes: np.ndarray, fs_hz, symb_rate_hz) -> np.ndarray:
        symbols = self.map_symbols(mes)
        samples_per_symb = int(fs_hz/symb_rate_hz)
        output_lp_signal = np.zeros(len(symbols)*samples_per_symb, dtype=complex)
        for j_symb in range(len(symbols)):
            output_lp_signal[j_symb*samples_per_symb] = symbols[j_symb]
        return output_lp_signal
    
    def __demap_single_symbol(self, in_symb):
        dist = -1
        out_bits = (0, 0, 0)
        for key_bits, ref_phase in self.__bit_symbol_map.items():
            if dist == -1:
                temp_symbol = np.exp(1j*ref_phase)
                out_bits = key_bits
                dist = (np.real(in_symb)-np.real(temp_symbol))**2+(np.imag(in_symb)-np.imag(temp_symbol))**2
                continue
            temp_symbol =  np.exp(1j*ref_phase)
            temp_dist = (np.real(in_symb)-np.real(temp_symbol))**2+(np.imag(in_symb)-np.imag(temp_symbol))**2
            if temp_dist < dist:
                out_bits = key_bits
                dist = temp_dist
        return out_bits
    
    @property
    def get_bits_per_symbol(self):
        return self.__bits_per_symb


# bit_message = np.random.randint(low=0, high=2, size=24)
# modem_block = PSKModem("qpsk")
# symbols = modem_block.map_symbols(bit_message)
# signal1 = modem_block.modulate(bit_message, 1000, 50, 10)
# signal2 = modem_block.modulate(bit_message, 1000, 50, 15)
# time_array1 = np.arange(0, len(signal1), 1) / 1000
# time_array2 = np.arange(0, len(signal2), 1) / 1000
# rec_bits1, metrics1 = modem_block.demodulate(signal1, 1000, 50, 10)
# rec_bits2, metrics2 = modem_block.demodulate(signal2, 1000, 50, 15)
# print('Bit message: ', *bit_message)
# print('Modulation symbols: ', *np.round(symbols, 2))
# print('Demodulated bits1: ', *rec_bits1)
# print('Demodulated metrics1: ', metrics1, 2)
# print('Demodulated bits2: ', *rec_bits2)
# print('Demodulated metrics2: ', metrics2)