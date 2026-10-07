from benchmark.config import test_parameters, alg_parameters, stat_tests, visual_curves
from benchmark.test_functions import *
from benchmark.test_run import run

np.random.seed(15)

# Nastavení parametrů testování
test_params = test_parameters(
    swarm_size=50,
    max_iter=1000,
    tolerance=1e-5,
    patience=50
)

# Nastavení parametrů jednotlivých algoritmů
alg_params = alg_parameters(
    pso_params=[0.72, 1.108, 1.108],
    cso_params=0.27,
    slpso_params=0.2,
    gpso_params=[2.5, 0.5, 2, 4, 1e-5]
)

# Pro spuštění testů je nutné zadat dimenzi úlohy a vybrat testovací funkci {function1, function2, function3, function4}
dimension = 4
test_function = function4
all_metrics, all_data, curves_data, var_data, func_name = run(test_function, dimension, test_params, alg_params, n_runs=50)

# Zobrazení výsledků
for alg, metrics in all_metrics.items():
    print("\nAlgoritmus: {}".format(alg))
    for metric, value in metrics.items():
        print("  {}: {}".format(metric, value))


# Vizualizace průměrné změny fitness v iteracích
visual_curves(curves_data, var_data, all_data, func_name)


# Statistické testování
stat_result = stat_tests(all_data)
accuracy = stat_result['accuracy']
stability = stat_result['stability']

print('_' * 50)
print('\nKruskalův-Wallisův test\n')
print("Na základě p_value {} zamítame H0: {}".format(accuracy["kw_p_value"], accuracy["has difference"]))
if accuracy['has difference']:
    for pair, data in accuracy['pairs'].items():
        print("Dvojice {}, přesnějším algoritmem je {}, p_value {}".format(pair, data["more_accurate"], data["p_value"]))
print('_' * 50)

print('\nLeveneův test\n')
print("Na základě p_value {} zamítame H0: {}".format(stability["lev_p_value"], stability["has difference"]))
if stability['has difference']:
    for pair, data in stability['pairs'].items():
        print("Dvojice {}, stabilnějším algoritmem je {}, p_value {}".format(pair, data["more_stable"], data["p_value"]))
print('_' * 50)
