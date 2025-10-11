import numpy as np

class OFDMModem:
    def __init__(self, num_sc: int = 1) -> None:
        self.__num_sc = num_sc
        # self.__cp_len = int(num_sc * 4.69 / 71.35) # https://www.sharetechnote.com/html/5G/5G_FrameStructure.html
        self.__cp_len = max(int(num_sc * 0.2), 1)
    
    @property
    def num_sc(self) -> int:
        return self.__num_sc
    
    @property
    def cp_len(self) -> int:
        return self.__cp_len
    

    def ofdm_modulate(self, samples: np.ndarray) -> np.ndarray:
        td_signal = np.fft.ifft(samples, norm='ortho')
        return np.concat([td_signal[len(td_signal)-self.__cp_len:], td_signal])
    
    def ofdm_demodulate(self, signal: np.ndarray) -> np.ndarray:
        # trimed_signal = signal[(self.__cp_len // 2) : len(signal) + (-self.__cp_len // 2)]
        trimed_signal = signal[self.__cp_len:]
        return np.fft.fft(trimed_signal, norm='ortho')


if __name__ == "__main__":
    num_sc = 4
    ofdm_block = OFDMModem(num_sc=num_sc)
    print("num_sc: ", ofdm_block.num_sc)
    print("cp_len: ", ofdm_block.cp_len)
    test_signal = ((1 - 2*np.random.randint(low=0, high=2, size=num_sc))
                   + 1j * (1 - 2*np.random.randint(low=0, high=2, size=num_sc))) / np.sqrt(2)
    
    print("test samples: ", (test_signal))
    
    td_signal = ofdm_block.ofdm_modulate(test_signal)
    print("td signal: ", (td_signal))

    samples_received = ofdm_block.ofdm_demodulate(td_signal)
    print("rec samples: ", (samples_received))
    print(f"Correct: {np.allclose(samples_received, test_signal, atol=1e-10)}")