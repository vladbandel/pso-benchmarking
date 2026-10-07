from benchmark.config import *


def cso(function: Callable[[np.ndarray], float], dimension: int, test_params: test_parameters,
        alg_params: alg_parameters,
        cso_params=None) -> tuple[np.ndarray, float, int, np.ndarray, list, Callable]:
    """
    Implementace algoritmu Competitive Swarm Optimizer

    Parametry:
         function: účelová funkce
         dimension: dimenze
         test_params: parametry testování
         class_params: parametry všech algoritmů
         cso_params: parametry algoritmu Competitive Swarm Optimizer

    return: nalezený bod, hodnota v daném bodě, počet iterací, poloha částic pro vizualizaci
            změna hodnoty gbest pro vizualizaci, název testovací funkce
    """
    lb, ub, _, _ = get_bounds(function, dimension)

    if not cso_params:
        phi = alg_params.cso_params  # nastavení parametrů v případě použití benchmark.test_run
    else:
        phi = cso_params

    m = test_params.swarm_size

    if not m % 2 == 0:
        m += 1

    # Vytvoření hejna částic
    particles, velocities, v_max = swarm_generation(m, lb, ub)

    fitness = function(particles)
    gbest = np.copy(particles[np.argmin(fitness)])
    gbest_fit = np.min(fitness)

    # Uložení dat pro vizualizaci
    convergence_curve = []
    history = np.zeros((test_params.max_iter, m, len(lb)))

    # Parametry pro ukoncovací kritérium
    iterace = 0
    no_improve_count = 0
    prev_best_pos = np.copy(gbest)
    prev_best_fit = gbest_fit
    e = np.finfo(float).eps

    while iterace < test_params.max_iter:
        mean_pos = np.mean(particles, axis=0)

        # Vytvoření párů
        indicies = np.random.permutation(m)
        half = m // 2

        group_1 = indicies[:half]
        group_2 = indicies[half:]

        # Soutěž na základě fitness
        competition = fitness[group_1] <= fitness[group_2]

        winners = np.where(competition, group_1, group_2)
        losers = np.where(competition, group_2, group_1)

        r1 = np.random.rand(half, len(lb))
        r2 = np.random.rand(half, len(lb))
        r3 = np.random.rand(half, len(lb))

        # Aktualizace rychlostí částic
        velocities[losers] = r1 * velocities[losers] + r2 * (particles[winners] - particles[losers]) + phi * r3 * (
                mean_pos - particles[losers])

        velocities[losers] = np.clip(velocities[losers], -v_max, v_max)
        particles[losers] += velocities[losers]
        particles[losers] = np.clip(particles[losers], lb, ub)
        fitness[losers] = function(particles[losers])

        min_fit = np.argmin(fitness)

        # Aktualizace hodnoty gbest
        if fitness[min_fit] < gbest_fit:
            gbest_fit = fitness[min_fit]
            gbest = np.copy(particles[min_fit])

        # Aktualizace dat pro vizualizaci
        convergence_curve.append(gbest_fit)
        history[iterace] = particles
        iterace += 1

        # Ukoncovací kritérium
        fit_change = abs((gbest_fit - prev_best_fit)) / (abs(prev_best_fit) + e)
        pos_change = np.linalg.norm(gbest - prev_best_pos) / (np.linalg.norm(prev_best_pos) + e)

        if pos_change < test_params.tolerance and fit_change < test_params.tolerance:
            no_improve_count += 1

        else:
            no_improve_count = 0
            prev_best_fit = gbest_fit
            prev_best_pos = np.copy(gbest)

        if no_improve_count >= test_params.patience:
            break

    history = history[:iterace]
    return gbest, gbest_fit, iterace, history, convergence_curve, function


def main() -> None:  # Test algoritmu
    np.random.seed(10)
    test_params = test_parameters(

        swarm_size=10,
        max_iter=1000,
        tolerance=1e-5,
        patience=20
    )

    dimension = 2
    cso_params = 0.07
    test_function = function5

    x, fx, it, history, cc, func_name = cso(test_function, dimension, test_params, alg_params=None,
                                            cso_params=cso_params)
    print('Nalezené řešení: x={}, f(x)={}, počet iterací={}'.format(x, fx, it))

    if dimension == 2:
        swarm_animation(func_name, history, 'CSO')
        swarm_motion(test_params.swarm_size, history)
    convergence_curve(cc, 'CSO')


if __name__ == '__main__':
    main()
