from modifications.sl_pso import sl_pso
from modifications.g_pso import g_pso
from modifications.pso import pso
from modifications.cso import cso
from benchmark.config import *


def run(function: Callable[[np.ndarray], float], dimension: int, test_params: test_parameters,
        alg_params: alg_parameters, n_runs: int) -> tuple[dict, dict, dict, dict, str]:
    """
    Spuštění všech algoritmů postupně

    Parametry:
        function: testovací funkce
        dimension: dimenze
        test_parameters: parametry testování
        alg_parameters: parametry jednotlivých algoritmů
        n_runs: počet běhů algoritmů

    return: sledované metriky, data pro statistické testování, data pro vizualizaci průměrných změn fitness,
    data pro vizualizaci variability změn fitness, název testovací funkce

    """

    if function.__name__ == 'function5' and dimension != 2:
        raise ValueError('funkce Peaks je definovaná pouze pro dimenzi 2')

    algorithms = {'pso': pso, 'cso': cso, 'sl_pso': sl_pso, 'g_pso': g_pso}

    stat_data = {alg: [] for alg in algorithms}
    average_cc = {}
    std_cc = {}
    results = {}

    # Určení globálního minima dané testovací funkce
    _, _, global_min_pos, glob_min_fit = get_bounds(function, dimension)
    success_threshold = 1e-5

    for alg_name, alg_function in algorithms.items():

        minimums = []
        positions = []
        iterations = []
        conv_curves = []

        # Postupné spuštění algoritmů a uložení výsledků
        for run_id in range(n_runs):

            g_best, gbest_fit, iterace, _, conv_data, _ = alg_function(function, dimension, test_params, alg_params)

            iterations.append(iterace)
            minimums.append(gbest_fit)
            positions.append(g_best)
            conv_curves.append(conv_data)

        stat_data[alg_name] = minimums  # Uložení dat pro statistické testování

        # Prodloužení grafů v případě předčasného zastavení algoritmu
        max_used_iterations = max(len(curve) for curve in conv_curves)
        longer_curves = []
        for curve in conv_curves:
            curve_list = list(curve)
            curve_length = len(curve_list)

            if curve_length < max_used_iterations:
                last_x = curve_list[-1]
                longer = [last_x] * (max_used_iterations - curve_length)
                curve_list.extend(longer)
            longer_curves.append(curve_list)

        conv_curves_matrix = np.array(longer_curves)

        avg_curve = np.mean(conv_curves_matrix, axis=0)
        std_curve = np.std(conv_curves_matrix, axis=0)

        average_cc[alg_name] = avg_curve
        std_cc[alg_name] = std_curve

        # Výpočet metriky success_rate
        success = sum(1 for x_fit in minimums if abs(glob_min_fit - x_fit) <= success_threshold)
        success_rate = (success / n_runs) * 100

        # Uložení výsledků všech běhů jednotlivého algoritmu
        results[alg_name] = {'mean': np.mean(minimums),
                             'median': np.median(minimums),
                             'best': np.min(minimums),
                             'mean distance': np.linalg.norm(np.mean(positions, axis=0) - global_min_pos),
                             'std': np.std(minimums),
                             'iterations': np.round(np.mean(iterations)),
                             'success rate': np.round(success_rate)
                             }

    return results, stat_data, average_cc, std_cc, function.__name__
