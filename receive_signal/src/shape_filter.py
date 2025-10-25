import numpy as np

B = 0.31
TIME_OFFSET_SYMB = 4


class ShapeFilter:
    def __init__(self, samples_per_symbol: int) -> None:
        self.__time_offset_num_samples = TIME_OFFSET_SYMB * samples_per_symbol
        self.__t_rel = np.linspace(-TIME_OFFSET_SYMB, TIME_OFFSET_SYMB, 8 * samples_per_symbol, endpoint=False)
        self.__h_ir = (
            (1 - B)
            / (1 - (4 * B * self.__t_rel) ** 2)
            * (np.sinc((1 - B) * self.__t_rel) + 4 * B / (np.pi * (1 - B)) * np.cos(np.pi * (1 + B) * self.__t_rel))
        )

    @property
    def h_ir(self) -> np.ndarray:
        return self.__h_ir

    @property
    def time_offset_samples(self) -> int:
        return self.__time_offset_num_samples

    def apply_filter(self, in_samples: np.ndarray) -> np.ndarray:
        return np.convolve(in_samples, self.__h_ir, mode="full")
