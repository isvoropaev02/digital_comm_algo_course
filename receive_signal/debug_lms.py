import numpy as np
import matplotlib.pyplot as plt


def lms_filter(u: np.ndarray, d: np.ndarray, filter_length: int, mu=0.01):
    assert u.shape[0] == d.shape[0]
    # Инициализация вектора весов нулями
    w = np.zeros(filter_length, dtype=np.complex64)

    # Массивы для хранения истории
    errors = np.zeros(len(u))
    weights_history = []

    # # Основной цикл LMS алгоритма
    # for n in range(filter_length - 1, len(u)):
    #     u_n = u[n - filter_length + 1 : n + 1]
    #     y_n = np.dot(w.conj(), u_n)
    #     e_n = d[n] - y_n
    #     w = w + mu * u_n * e_n.conjugate()

    #     # Сохраняем абсолютное значение ошибки
    #     errors[n] = np.abs(e_n)
    #     weights_history.append(w.copy())
    # Основной цикл LMS алгоритма
    u_n = np.zeros_like(w)
    for n in range(0, len(u)):
        u_n[-1] = u[n]
        y_n = np.dot(w.conj(), u_n)
        e_n = d[n] - y_n
        w = w + mu * u_n * e_n.conjugate()
        u_n = np.roll(u_n, -1)

        # Сохраняем абсолютное значение ошибки
        errors[n] = np.abs(e_n)
        weights_history.append(w.copy())
    return w, errors, weights_history


# Генерация тестовых данных для проверки
def generate_test_data(signal_length=1000, filter_length=5):
    # Генерируем случайные истинные веса
    true_weights = np.random.randn(filter_length) + 1j * np.random.randn(filter_length)
    true_weights = true_weights / np.linalg.norm(true_weights)

    # Генерируем входной сигнал (комплексный белый шум)
    u = np.random.randn(signal_length) + 1j * np.random.randn(signal_length)

    # Вычисляем желаемый выход (без шума)
    d = np.convolve(u, true_weights, mode="valid")

    # Обрезаем входной сигнал до соответствующей длины
    u = u[filter_length - 1 :]

    return u, d, true_weights


# Пример использования
if __name__ == "__main__":
    filter_length = 15
    signal_length = 127 * 3
    mu = 0.05  # Шаг обучения

    u, d, true_weights = generate_test_data(signal_length, filter_length)
    w_final, errors, weights_history = lms_filter(u, d, filter_length, mu)

    # График сходимости
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(errors[filter_length - 1 :])
    plt.title("Сходимость LMS алгоритма")
    plt.xlabel("Итерация")
    plt.ylabel("|e(n)|")
    plt.grid(True)
    plt.yscale("log")

    plt.subplot(2, 2, 2)
    plt.plot(10 * np.log10(errors[filter_length - 1 :] ** 2))
    plt.title("Ошибка в dB")
    plt.xlabel("Итерация")
    plt.ylabel("MSE (dB)")
    plt.grid(True)

    # Сравнение истинных и найденных весов
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(filter_length), np.real(true_weights), "bo-", label="Истинные (Re)")
    plt.plot(np.arange(filter_length), np.real(np.conj(w_final[::-1])), "ro--", label="LMS (Re)")
    plt.plot(np.arange(filter_length), np.imag(true_weights), "go-", label="Истинные (Im)")
    plt.plot(np.arange(filter_length), np.imag(np.conj(w_final[::-1])), "mo--", label="LMS (Im)")
    plt.title("Сравнение весов")
    plt.xlabel("Коэффициент")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)

    # Траектория обучения
    if filter_length >= 2:
        weights_history = np.array(weights_history)
        plt.subplot(2, 2, 4)
        plt.plot(np.real(weights_history[:, -1]), -np.imag(weights_history[:, -1]), "b-", alpha=0.7)
        plt.plot(np.real(true_weights[0]), np.imag(true_weights[0]), "ro", markersize=10, label="Истинные")
        plt.plot(np.real(np.conj(w_final[-1])), np.imag(np.conj(w_final[-1])), "go", markersize=8, label="Конечные")
        plt.title("Траектория обучения веса w0")
        plt.xlabel("Re(w0)")
        plt.ylabel("Im(w0)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
