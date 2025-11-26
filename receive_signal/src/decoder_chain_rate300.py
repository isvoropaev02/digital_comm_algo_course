from src.psk_modem import PSKModem
from src.arinc_constants import SCR_SEQ_HEX
from sk_dsp_comm.fec_conv import FECConv

import numpy as np


def hex_to_bit_array_msb_start(hex_str: str) -> np.ndarray:
    decimal_value = int(hex_str, 16)
    bit_string = bin(decimal_value)[2:]
    bit_string = bit_string.zfill(len(hex_str) * 4)
    return np.flip(np.array([int(bit) for bit in bit_string], dtype=np.int8))


class Rate300DecChain:
    def __init__(self) -> None:
        self.__pmod = PSKModem(bits_per_sample=1)
        self.__scr_seq = self.__pmod.modulate(hex_to_bit_array_msb_start(SCR_SEQ_HEX))
        assert self.__scr_seq.shape[0] == 120, "scr_seq should have the size=120"
        self.__conv_decoder = FECConv(G=("133", "171"), Depth=7)

    def process(self, in_samples: np.ndarray) -> np.ndarray:
        descr_out = self.__descramble(in_samples)
        deinterlv_out = self.__deinterleave(descr_out)
        demap_bits_out = self.__demap_with_combining(deinterlv_out)
        info_bits = self.__fec_decode(demap_bits_out)
        return info_bits

    def __descramble(self, in_samples: np.ndarray) -> np.ndarray:
        scr_seq_ext = np.resize(self.__scr_seq, in_samples.shape[0])
        return in_samples * scr_seq_ext

    def __deinterleave(self, in_samples: np.ndarray) -> np.ndarray:
        assert in_samples.shape[0] == 40 * 54, "incorrect in_samples shape in interleaver"
        intrlv_table = np.empty(shape=(40, 54), dtype=in_samples.dtype)
        # filling the table
        j_col = 0
        for j_sml in range(in_samples.shape[0]):
            intrlv_table[j_sml % 40, j_col] = in_samples[j_sml]
            j_col = (j_col - 17) % 54
        # reading the table
        out = np.empty_like(in_samples)
        for j_col in range(54):
            for j_row in range(40):
                out[40 * j_col + (j_row * 9) % 40] = intrlv_table[j_row, j_col]
        return out

    def __demap_with_combining(self, in_samples: np.ndarray) -> np.ndarray:
        combined_samples = (in_samples[::2] + in_samples[1::2]) / 2
        return self.__pmod.demodulate(combined_samples)

    def __fec_decode(self, in_bits: np.ndarray) -> np.ndarray:
        return self.__conv_decoder.viterbi_decoder(in_bits, metric_type="hard")
