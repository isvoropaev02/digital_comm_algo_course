import numpy as np

class GaussFilter:
    __pass_band_Hz = 0
    h_impulse_response = np.array([])
    __time_samples = np.array([])

    def __init__(self, symbol_time, samples_per_symbol, BT=0.3) -> None:
        alpha = 1/BT*np.sqrt(np.log(2)/2)
        self.__pass_band_Hz = BT/symbol_time
        self.__time_samples = np.linspace(-2, 2, 4*samples_per_symbol) # 4 signal lengths
        self.h_impulse_response = np.sqrt(np.pi)/alpha*np.exp(-(np.pi*self.__time_samples/alpha)**2)

    def get_pass_band_3dB(self) -> float:
        return self.__pass_band_Hz
    
    def get_abs_time(self, symbol_rate):
        return self.__time_samples/symbol_rate
