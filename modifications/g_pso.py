from benchmark.config import *


def g_pso(function: Callable[[np.ndarray], float], dimension: int, test_params: test_parameters,
          alg_params: alg_parameters,
          gpso_params=None) -> tuple[np.ndarray, float, int, np.ndarray, list, Callable]:
    """
    Implementace algoritmu Gregarious Particle Swarm Optimizer

    Parametry:
         function: účelová funkce
         dimension: dimenze
         test_params: parametry testování
         class_params: parametry všech algoritmů
         gpso_params: parametry algoritmu Gregarious Particle Swarm Optimizer

    return: nalezený bod, hodnota v daném bodě, počet iterací, poloha částic pro vizualizaci
            změna hodnoty gbest pro vizualizaci, testovací funkce
    """
    lb, ub, _, _ = get_bounds(function, dimension)

    if not gpso_params:
        gamma, delta, gamma_min, gamma_max, epsilon = alg_params.gpso_params  # nastavení parametrů v případě použití benchmark.test_run
    else:
        gamma, delta, gamma_min, gamma_max, epsilon = gpso_params

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
    gbest_fit_prev = gbest_fit
    e = np.finfo(float).eps

    while iterace < test_params.max_iter:

        start_gbest = gbest_fit

        for i in range(len(particles)):

            if np.linalg.norm(particles[i] - gbest) <= epsilon:
                # Reinicializace
                velocities[i] = np.random.uniform(-v_max, v_max, dimension)
            else:
                r1 = np.random.rand(len(lb))
                velocities[i] = gamma * r1 * (gbest - particles[i])

            velocities[i] = np.clip(velocities[i], -v_max, v_max)  # Omezení rychlostí
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], lb, ub)
            fitness[i] = function(particles[i])

            if fitness[i] < gbest_fit:
                gbest_fit = fitness[i]
                gbest = np.copy(particles[i])

        # Změna parametru gamma
        if gbest_fit < start_gbest:
            gamma = max(gamma - delta, gamma_min)
        else:
            gamma = min(gamma + delta, gamma_max)

        # Aktualizace dat pro vizualizaci
        convergence_curve.append(gbest_fit)
        history[iterace] = particles
        iterace += 1

        # Ukoncovací kritérium
        fit_change = abs((gbest_fit - gbest_fit_prev)) / (abs(gbest_fit_prev) + e)
        pos_change = np.linalg.norm(gbest - prev_best_pos) / (np.linalg.norm(prev_best_pos) + e)

        if pos_change < test_params.tolerance and fit_change < test_params.tolerance:
            no_improve_count += 1

        else:
            no_improve_count = 0
            gbest_fit_prev = gbest_fit
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
    # parametry = gamma, delta, gamma_min, gamma_max, epsilon
    gpso_params = [3, 0.5, 2, 4, 1e-5]
    test_function = function5

    x, fx, it, history, cc, func_name = g_pso(test_function, dimension, test_params, alg_params=None, gpso_params=gpso_params)

    print('Nalezené řešení: x={}, f(x)={}, počet iterací={}'.format(x, fx, it))

    # Vizualizace
    if dimension == 2:
        swarm_animation(func_name, history, 'G-PSO')
        swarm_motion(test_params.swarm_size, history)
    convergence_curve(cc, 'G-PSO')


if __name__ == '__main__':
    main()
