import matplotlib.pyplot as plt
import numpy as np


def function1(X: np.ndarray) -> float:
    """
    funkce desáté mocniny
    Globální minimum x = (0,...,0)
    f(x) = 0
    """
    return np.sum(np.power(X, 10), axis=-1)


def function2(X: np.ndarray) -> float:
    """
    Rosenbrock function
    Globální minimum x = (1,...,1)
    f(x) = 0
    """
    return np.sum(100 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (1 - X[..., :-1]) ** 2, axis=-1)


def function3(X: np.ndarray) -> float:
    """
    Ackleyho funkce
    Globální minimum x = (0,...,0)
    f(x) = 0
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = X.shape[-1]

    sum1 = np.sum(X ** 2, axis=-1)
    sum2 = np.sum(np.cos(c * X), axis=-1)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)

    return term1 + term2 + a + np.e


def function4(X: np.ndarray) -> float:
    """
    Rastriginova funkce
    Globální minimum x = (0,...,0)
    f(x) = 0
    """
    n = X.shape[-1]
    return 10 * n + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=-1)


def function5(X: np.ndarray) -> float:
    """
    Funkce Peaks
    Globální minimum x = (0.228,−1.626)
    f(x) = −6.551
    """
    x, y = X[..., 0], X[..., 1]
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)


def main() -> None:
    """
    Vizualizace testovacích funkcí

    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    Z1 = function1(np.stack([X, Y], axis=-1))
    Z2 = function2(np.stack([X, Y], axis=-1))
    Z3 = function3(np.stack([X, Y], axis=-1))
    Z4 = function4(np.stack([X, Y], axis=-1))
    Z = [Z1, Z2, Z3, Z4]

    titles = ['Funkce desáté mocniny', 'Rosenbrockova funkce', 'Ackleyho funkce', 'Rastriginova funkce']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={'projection': '3d'})
    fig.suptitle('Testovací funkce')

    for ax, z, title in zip(axes.flat, Z, titles):
        image = ax.plot_surface(X, Y, z, cmap='twilight_shifted')
        ax.set_title(title)
        plt.colorbar(image, orientation='horizontal', shrink=0.5)

    plt.show()


if __name__ == '__main__':
    main()
