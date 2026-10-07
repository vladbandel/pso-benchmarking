from benchmark.config import *


def sl_pso(function: Callable[[np.ndarray], float], dimension: int, test_params: test_parameters,
           alg_params: alg_parameters,
           slpso_params=None, M=100, alpha=0.5) -> tuple[np.ndarray, float, int, np.ndarray, list, Callable]:
    """
    Implementace algoritmu Social Learning Particle Swarm Optimizer

    Parametry:
         function: účelová funkce
         dimension: dimenze
         test_params: parametry testování
         class_params: parametry všech algoritmů
         slpso_params: parametry algoritmu Social Learning Particle Swarm Optimizer

    return: nalezený bod, hodnota v daném bodě, počet iterací, poloha částic pro vizualizaci
            změna hodnoty gbest pro vizualizaci, název testovací funkce
    """
    lb, ub, _, _ = get_bounds(function, dimension)

    if not slpso_params:
        phi = alg_params.slpso_params  # nastavení parametru v případě použití benchmark.test_run
    else:
        phi = slpso_params

    m = test_params.swarm_size

    # Vytvoření hejna částic
    particles, velocities, v_max = swarm_generation(test_params.swarm_size, lb, ub)

    fitness = function(particles)
    gbest = np.copy(particles[np.argmin(fitness)])
    gbest_fit = np.min(fitness)

    # Uložení dat pro vizualizaci
    convergence_curve = []
    history = np.zeros((test_params.max_iter, test_params.swarm_size, dimension))

    # Parametry pro ukoncovací kritérium
    iterace = 0
    no_improve_count = 0
    prev_best_pos = np.copy(gbest)
    prev_best_fit = gbest_fit
    e = np.finfo(float).eps

    while iterace < test_params.max_iter:

        # Uspořádání částic
        sorted_indices = np.argsort(fitness)[::-1]
        particles = particles[sorted_indices]
        velocities = velocities[sorted_indices]
        fitness = fitness[sorted_indices]

        mean_pos = np.mean(particles, axis=0)

        for i in range(test_params.swarm_size - 1):

            P_l = (1 - i / m) ** (alpha * np.log(np.ceil(dimension / M)))
            p_l = np.random.rand()

            if P_l > p_l:
                k_indices = np.random.randint(i + 1, m, size=dimension)
                dim_indices = np.arange(dimension)
                p_k = particles[k_indices, dim_indices]
                r1, r2, r3 = np.random.rand(3, dimension)

                # Aktualizace rychlostí částic
                velocities[i] = r1 * velocities[i] + r2 * (p_k - particles[i]) + r3 * phi * (mean_pos - particles[i])

                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
                fitness[i] = function(particles[i])

        # Aktualizace hodnoty gbest
        min_fit = np.argmin(fitness)
        if fitness[min_fit] < gbest_fit:
            gbest = np.copy(particles[min_fit])
            gbest_fit = fitness[min_fit]

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

    slpso_params = 0.2
    dimension = 2
    test_function = function5

    x, fx, it, history, cc, func_name = sl_pso(test_function, dimension, test_params, alg_params=None, slpso_params=slpso_params)

    print('Nalezené řešení: x={}, f(x)={}, počet iterací={}'.format(x, fx, it))

    # Vizualizace
    if dimension == 2:
        swarm_animation(func_name, history, 'SL-PSO')
        swarm_motion(test_params.swarm_size, history)

    convergence_curve(cc, 'SL-PSO')


if __name__ == '__main__':
    main()
