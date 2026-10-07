from collections.abc import Callable
import matplotlib.animation as anim
from benchmark.test_functions import *
from itertools import combinations
from dataclasses import dataclass
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy import stats
import numpy as np


# Možnost uložení parametrů testování
@dataclass
class test_parameters:
    swarm_size: int
    max_iter: int
    tolerance: float
    patience: int


# Možnost uložení parametrů jednotlivých algoritmů
@dataclass
class alg_parameters:
    pso_params: list
    cso_params: list
    slpso_params: list
    gpso_params: list


def get_bounds(function: Callable[[np.ndarray], float], dimension: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Meze prohledávaného prostoru

    Parametry:
        function: testovací funkce
        dimension: dimenze

    return: dolní mez, horní mez, globální minimum, hodnota funkce v bodě globálního minima
    """

    if function.__name__ == 'function1':
        lb, ub = -5.12, 5.12
        global_min = 0
        glob_min_fit = 0

    elif function.__name__ == 'function2':
        lb, ub = -2.048, 2.048
        global_min = 1
        glob_min_fit = 0

    elif function.__name__ == 'function3':
        lb, ub = -30, 30
        global_min = 0
        glob_min_fit = 0

    elif function.__name__ == 'function4':
        lb, ub = -5.12, 5.12
        global_min = 0
        glob_min_fit = 0

    elif function.__name__ == 'function5':
        lb, ub = -5, 5
        global_min = np.array([0.228, -1.626])
        glob_min_fit = 0
        return np.full(dimension, lb), np.full(dimension, ub), global_min, glob_min_fit

    else:
        raise ValueError('Neznámá testovací funkce')

    return np.full(dimension, lb), np.full(dimension, ub), np.full(dimension, global_min), glob_min_fit


def swarm_generation(swarm_size: int, lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generování hejna částic

    Parametry:
        swarm_size: počet částic
        lb: dolní mez prohledávaného prostoru
        ub: horní mez prohledávaného prostoru

    return: poloha částic, rychlost částic, parametr pro omezení rychlosti částic
    """

    dimension = len(lb)
    v_max = 0.15 * (np.array(ub) - np.array(lb))

    particles = np.random.uniform(lb, ub, size=(swarm_size, dimension))
    velocities = np.zeros((swarm_size, dimension))

    return particles, velocities, v_max


def stat_tests(run_data: dict, alpha: float = 0.05) -> dict:
    """
    Testování statistických hypotéz

    Parametry:
        run_data: výsledky všech běhů algoritmů

    return: výsledky testování
    """

    algorithms = list(run_data.keys())
    data = list(run_data.values())

    stats_results = {
        'accuracy': {},
        'stability': {}
    }

    # Kruskalův-Wallisův test
    _, kw_p_value = stats.kruskal(*data)
    stats_results['accuracy']['kw_p_value'] = np.round(kw_p_value, 5)
    stats_results['accuracy']['has difference'] = kw_p_value < alpha

    # Neményiho metoda srovnávání
    # Porovnávání dvojic algoritmů v případě zamítnutí H0
    if stats_results['accuracy']['has difference']:
        accuracy_pairs = {}
        nemenyi_p_value = sp.posthoc_nemenyi(data)

        # Porovnávání párů
        for alg1, alg2 in combinations(algorithms, 2):
            i = algorithms.index(alg1)
            j = algorithms.index(alg2)
            p_value = nemenyi_p_value.iloc[i, j]
            is_significant = p_value < alpha

            med1, med2 = np.median(run_data[alg1]), np.median(run_data[alg2])

            if is_significant:  # Určíme přesnější algoritmus
                more_accurate = alg1 if med1 < med2 else alg2
            else:
                more_accurate = 'None'

            accuracy_pairs['{} & {}'.format(alg1, alg2)] = {
                'p_value': np.round(p_value, 5),
                'more_accurate': more_accurate
            }

        stats_results['accuracy']['pairs'] = accuracy_pairs

    # Leveneův test
    _, lev_p_value = stats.levene(*data, center='median')
    stats_results['stability']['lev_p_value'] = np.round(lev_p_value, 5)
    stats_results['stability']['has difference'] = lev_p_value < alpha

    # Porovnávání dvojic algoritmů v případě zamítnutí H0
    if stats_results['stability']['has difference']:
        num_comparations = len(list(combinations(algorithms, 2)))
        adj_alpha = alpha / num_comparations
        stability_pairs = {}

        for alg1, alg2 in combinations(algorithms, 2):

            _, pair_p_value = stats.levene(run_data[alg1], run_data[alg2], center='median')
            is_significant = pair_p_value < adj_alpha

            std1, std2 = np.std(run_data[alg1]), np.std(run_data[alg2])

            if is_significant:  # Určíme stabilnější algoritmus
                more_stable = alg1 if std1 < std2 else alg2
            else:
                more_stable = 'None'

            stability_pairs['{} & {}'.format(alg1, alg2)] = {
                'p_value': np.round(pair_p_value, 5),
                'more_stable': more_stable
            }

        stats_results['stability']['pairs'] = stability_pairs

    return stats_results


def convergence_curve(data: np.ndarray, name: str) -> None:
    """
    Vizualizace změny hodnoty fitness funkce
    v průběhu iterací

    Parametry:
        data: změna hodnot fitness
        name: název algoritmu

    """

    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.xlabel("Iterace")
    plt.ylabel("Hodnota Fitness")
    plt.title(name)
    plt.grid()
    plt.show()


def visual_curves(curves_data: dict, std_data: dict, all_results: dict, func_name: str) -> None:
    """
    Vizualizace několika grafů průměrné změny fitness

    Parametry:
        curves_data: data běhů algoritmů
        std_data: data o variabilitě výsledků
        all_results: výsledky běhů algoritmů
        func_name: název testovací funkce

    """

    plt.figure(figsize=(18, 6))

    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '--', '-.', ':']

    for i, (name, curve) in enumerate(curves_data.items()):
        avg_curve = curves_data[name]
        std_curve = std_data[name]

        ub = avg_curve + std_curve
        if func_name == 'function5':
            lb_clip = -6.551

        else:
            lb_clip = 1e-16

        lb = np.clip(avg_curve - std_curve, a_min=lb_clip, a_max=None)
        plt.plot(
            range(len(curve)),
            curve,
            marker=markers[i],
            linestyle=line_styles[i],
            linewidth=1.5,
            markersize=3,
            label=name.upper()
        )

        plt.fill_between(
            range(len(curve)),
            lb, ub,
            alpha=0.2
        )

    plt.xlabel('Iterace', fontsize=18)
    plt.ylabel('Fitness', fontsize=18)
    # plt.yscale('log')
    plt.title(f'testovací funkce {func_name}', fontsize=18)
    plt.xlim(0, 20)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid()
    plt.legend(fontsize=18)
    plt.show()

    # Vykreslení výledků běhů algoritmů
    for i, (name, fitness) in enumerate(all_results.items()):
        plt.plot(
            range(len(fitness)),
            fitness,
            marker=markers[i],
            linestyle=line_styles[i],
            linewidth=1.5,
            markersize=3,
            label=name.upper()
        )

    plt.title(f'testovací funkce {func_name}', fontsize=18)
    plt.ylabel('Fitness', fontsize=12)
    # plt.yscale('log')
    plt.xlabel('Pořadí spuštění algoritmu', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()


def swarm_motion(swarm_size: int, history: np.ndarray) -> None:
    """
    Vizualizace pohybu částic

    Parametry:
        swarm_size: počet částic
        history: poloha částic v jednotlivých iteracích

    """
    for i in range(swarm_size):
        trajectories = np.array([t[i] for t in history])
        plt.plot(trajectories[:, 0], trajectories[:, 1], alpha=0.6)
        plt.scatter(trajectories[0, 0], trajectories[0, 1], marker="o")

    plt.title("Pohyb částic")
    plt.grid(True)

    plt.show()


def swarm_animation(function: Callable[[np.ndarray], float], history: np.ndarray,
                    name: str) -> None:
    """
    Animace pohybu částic v jednotlivých iteracích

    Parametry:
        function: testovací funkce
        lb: dolní mez prohledávaného prostoru
        ub: horní mez prohledávaného prostoru
        history: poloha částic v jednotlivých iteracích
        name: název algoritmu

    """

    def update(frame: int) -> Axes:
        """
        Funkce vykreslující změnu polohy částic

        Parametr:
            frame: číslo snímku

        return: parametr pro vizualizaci
        """

        ax.clear()
        ax.contourf(X, Y, Z, levels=50, cmap='coolwarm')

        # Pozice částic
        ax.scatter(history[frame, :, 0], history[frame, :, 1], color='black', s=15, zorder=2)

        if frame > 0:

            for i in range(history.shape[1]):
                # Vizualizace změny pozic částic
                dx = history[frame, i, 0] - history[frame - 1, i, 0]
                dy = history[frame, i, 1] - history[frame - 1, i, 1]
                ax.arrow(history[frame - 1, i, 0], history[frame - 1, i, 1],
                         dx, dy,
                         head_width=0.3, head_length=0.3, fc='red', ec='purple', zorder=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Průběh algorimu {}'.format(name))

        return ax

    fig, ax = plt.subplots()

    lb, ub, _, _ = get_bounds(function, 2)
    x = np.linspace(lb[0] - 1, ub[0] + 1, 100)
    y = np.linspace(lb[1] - 1, ub[1] + 1, 100)
    X, Y = np.meshgrid(x, y)

    Z = function(np.stack([X, Y], axis=-1))

    ani = anim.FuncAnimation(fig, update, frames=len(history), interval=200)
    plt.colorbar(ax.contourf(X, Y, Z, levels=50, cmap='coolwarm'))

    plt.show()
