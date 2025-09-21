import numpy as np
from configuration import*
# import matplotlib.pyplot as plt

class PSKModem:

    __bit_symbol_map = dict()
    __name = ''
    __bits_per_symb = 0

    def __init__(self, name) -> None:
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
        output = np.zeros(output_size, dtype=complex)
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
    
    def modulate(self, mes: np.ndarray, fs_Hz, f_carrier_Hz, f_symb_Hz) -> np.ndarray:
        len_mes = len(mes)
        if len_mes % self.__bits_per_symb != 0:
            raise(ValueError('Length of message must be divisible by '+str(self.__bits_per_symb)+' for ' + self.__name + 'modulation'))
        num_symb = len(mes) // self.__bits_per_symb
        samples_per_symb = int(fs_Hz/f_symb_Hz)
        total_samples = int(num_symb*samples_per_symb)
        output_signal = np.zeros(total_samples)
        time_samples = np.arange(0, samples_per_symb, 1)
        for j_symb in range(num_symb):
            phase = self.__bit_symbol_map.get(tuple(mes[self.__bits_per_symb*j_symb:self.__bits_per_symb*j_symb+self.__bits_per_symb]))
            output_signal[j_symb*samples_per_symb:(j_symb+1)*samples_per_symb] = np.cos(phase+2*np.pi*f_carrier_Hz*time_samples/fs_Hz)
        return output_signal
    
    def demodulate(self, signal: np.ndarray, fs_Hz, f_carrier_Hz, f_symb_Hz) -> np.ndarray:
        samples_per_symb = int(fs_Hz/f_symb_Hz)
        num_symb = int(len(signal)/samples_per_symb)
        output_bits = np.zeros(num_symb*self.__bits_per_symb, dtype=int)
        time_samples = np.arange(0, samples_per_symb, 1)
        for j_symb in range(num_symb):
            bits = np.array([])
            h_target = np.array([])
            for key_bits, ref_phase in self.__bit_symbol_map.items():
                ref_wave = np.cos(ref_phase+2*np.pi*f_carrier_Hz*time_samples/fs_Hz)
                h_current = np.dot(signal[j_symb*samples_per_symb:(j_symb+1)*samples_per_symb], ref_wave)
                if len(bits) == 0:
                    bits = np.array(list(key_bits))
                    h_target = h_current
                else:
                    if h_current > h_target:
                        bits = np.array(list(key_bits))
                        h_target = h_current
            output_bits[j_symb*self.__bits_per_symb:(j_symb+1)*self.__bits_per_symb] = bits
        return output_bits
    
    def padded_symbols(self, mes: np.ndarray, fs_Hz, f_symb_Hz) -> np.ndarray:
        symbols = self.map_symbols(mes)
        samples_per_symb = int(fs_Hz/f_symb_Hz)
        output_lp_signal = np.zeros(len(symbols)*samples_per_symb, dtype=complex)
        for j_symb in range(len(symbols)):
            output_lp_signal[j_symb*samples_per_symb] = symbols[j_symb]
        return output_lp_signal
    
    def get_bits_per_symbol(self):
        return self.__bits_per_symb
    
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


class FSKModem:
    __f_dev = 0
    __phi_0 = 0

    def __init__(self, f_symb_Hz, m=1):
        self.__f_dev = m / 4 * f_symb_Hz


    def CPFSK_complex_envelope(self, in_bits, f_symb_Hz, fs_Hz):
        current_phi = 0
        samples_per_symbol = int(fs_Hz/f_symb_Hz)
        total_samples = int(len(in_bits)*samples_per_symbol)
        output_signal = np.zeros(total_samples, dtype=complex)
        q_pulse = np.arange(0, samples_per_symbol,1)/(2*samples_per_symbol)
        for j_symb in range(len(in_bits)):
            I_n = 1 - 2*in_bits[j_symb]
            temp_phases = current_phi + 4*np.pi*self.__f_dev/f_symb_Hz*I_n*q_pulse
            output_signal[j_symb*samples_per_symbol:(j_symb+1)*samples_per_symbol] = np.exp(1j*temp_phases)
            current_phi = temp_phases[-1]
        return output_signal
    
    
    def GMSK_complex_envelope(self, in_bits, f_symb_Hz, fs_Hz):
        from filter_cls import GaussFilter
        h_gaus = GaussFilter(1/f_symb_Hz, int(fs_Hz/f_symb_Hz)).h_impulse_response
        delta_t = 1/fs_Hz
        samples_per_symbol = int(fs_Hz/f_symb_Hz)
        total_samples = int(len(h_gaus)+(len(in_bits)-1)*samples_per_symbol)
        output_signal = np.ones(total_samples, dtype=complex)
        phases = np.zeros(total_samples, dtype=float)
        for j_symb in range(len(in_bits)):
            I_n = 1 - 2*in_bits[j_symb]
            for j_sample in range(len(h_gaus)):
                phases[j_symb*samples_per_symbol + j_sample] += 4*np.pi*self.__f_dev/f_symb_Hz*I_n*h_gaus[j_sample]*delta_t
        output_signal = np.exp(np.cumsum(1j*phases))
        return output_signal
    
    
    def modulate(self, in_bits, f_symb_Hz, fs_Hz, f_carrier_Hz, mod_type='CPFSK'):
        if mod_type == 'CPFSK':
            bb_signal = self.CPFSK_complex_envelope(in_bits, f_symb_Hz, fs_Hz)
        elif mod_type == 'GMSK':
            bb_signal = self.GMSK_complex_envelope(in_bits, f_symb_Hz, fs_Hz)
        else:
            bb_signal = []
            print("No such modulation type supported"+mod_type)
        time_samples = np.arange(0, len(bb_signal), 1)/fs_Hz
        rf_signal = np.real(bb_signal*np.exp(1j*2*np.pi*f_carrier_Hz*time_samples))
        return rf_signal


# mm = FSKModem(10)
# bb_signal = mm.GMSK_complex_envelope(in_bits=[1, 1, 1, 1, 1, 1, 1, 1], f_symb_Hz=10, fs_Hz=1000)
# plt.plot(np.angle(bb_signal))
# plt.show()