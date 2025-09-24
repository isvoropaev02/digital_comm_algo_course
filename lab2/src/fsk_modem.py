import numpy as np

class GaussFilter:
    __pass_band_hz = 0
    h_impulse_response = np.array([])
    __time_sample_ids = np.array([])

    def __init__(self, symbol_time, samples_per_symbol, BT=0.3) -> None:
        alpha = 1/BT*np.sqrt(np.log(2)/2)
        self.__pass_band_hz = BT/symbol_time
        self.__time_sample_ids = np.linspace(-2, 2, 4*samples_per_symbol) # 4 signal lengths
        self.h_impulse_response = np.sqrt(np.pi)/alpha*np.exp(-(np.pi*self.__time_sample_ids/alpha)**2)

    @property
    def get_pass_band_3dB(self) -> float:
        return self.__pass_band_hz
    
    def get_abs_time(self, symbol_rate):
        return self.__time_sample_ids/symbol_rate

class FSKModem:
    __f_dev = 0

    def __init__(self, f_symb_hz: float, m=1):
        self.__f_dev = m / 4 * f_symb_hz

    @property
    def fdev_hz(self) -> float:
        return self.__f_dev

    def FSK_complex_envelope(self, in_bits: np.ndarray, f_symb_hz: float, fs_hz: float) -> np.ndarray:
        samples_per_symbol = int(fs_hz/f_symb_hz)
        total_samples = int(len(in_bits)*samples_per_symbol)
        output_signal = np.zeros(total_samples, dtype=np.complex128)
        for j_symb in range(len(in_bits)):
            I_n = 1 - 2*in_bits[j_symb]
            temp_phases = 2 * np.pi * self.__f_dev / fs_hz * I_n * np.arange(0, samples_per_symbol, 1)
            output_signal[j_symb*samples_per_symbol:(j_symb+1)*samples_per_symbol] = np.exp(1j*temp_phases)
        return output_signal

    def CPFSK_complex_envelope(self, in_bits: np.ndarray, f_symb_hz: float, fs_hz: float) -> np.ndarray:
        current_phi = 0
        samples_per_symbol = int(fs_hz/f_symb_hz)
        total_samples = int(len(in_bits)*samples_per_symbol)
        output_signal = np.zeros(total_samples, dtype=np.complex128)
        q_pulse = np.arange(0, samples_per_symbol,1)/(2*samples_per_symbol)
        for j_symb in range(len(in_bits)):
            I_n = 1 - 2*in_bits[j_symb]
            temp_phases = current_phi + 4*np.pi*self.__f_dev/f_symb_hz*I_n*q_pulse
            output_signal[j_symb*samples_per_symbol:(j_symb+1)*samples_per_symbol] = np.exp(1j*temp_phases)
            current_phi = temp_phases[-1]
        return output_signal
    
    
    def GMSK_complex_envelope(self, in_bits: np.ndarray, f_symb_hz: float, fs_hz: float) -> np.ndarray:
        h_gaus = GaussFilter(1/f_symb_hz, int(fs_hz/f_symb_hz)).h_impulse_response
        delta_t = 1/fs_hz
        samples_per_symbol = int(fs_hz/f_symb_hz)
        total_samples = int(len(h_gaus)+(len(in_bits)-1)*samples_per_symbol)
        output_signal = np.ones(total_samples, dtype=np.complex128)
        phases = np.zeros(total_samples, dtype=np.float64)
        for j_symb in range(len(in_bits)):
            I_n = 1 - 2*in_bits[j_symb]
            for j_sample in range(len(h_gaus)):
                phases[j_symb*samples_per_symbol + j_sample] += 4*np.pi*self.__f_dev/f_symb_hz*I_n*h_gaus[j_sample]*delta_t
        output_signal = np.exp(np.cumsum(1j*phases))
        return output_signal
    
    
    def modulate(self, in_bits: np.ndarray, f_symb_hz: float, fs_hz: float, fc_hz: float, mod_type: str ='CPFSK'):
        if mod_type == 'CPFSK':
            bb_signal = self.CPFSK_complex_envelope(in_bits, f_symb_hz, fs_hz)
        elif mod_type == 'GMSK':
            bb_signal = self.GMSK_complex_envelope(in_bits, f_symb_hz, fs_hz)
        elif mod_type == 'FSK':
            bb_signal = self.FSK_complex_envelope(in_bits, f_symb_hz, fs_hz)
        else:
            bb_signal = []
            print("No such modulation type supported" + mod_type)
        time_samples = np.arange(0, len(bb_signal), 1)/fs_hz
        rf_signal = np.real(bb_signal*np.exp(1j*2*np.pi*fc_hz*time_samples))
        return rf_signal


# mm = FSKModem(10)
# bb_signal = mm.GMSK_complex_envelope(in_bits=[1, 1, 1, 1, 1, 1, 1, 1], f_symb_hz=10, fs_hz=1000)
# plt.plot(np.angle(bb_signal))
# plt.show()