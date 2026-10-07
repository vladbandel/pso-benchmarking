from benchmark.config import *


def pso(function: Callable[[np.ndarray], float], dimension: int, test_params: test_parameters,
        alg_params: alg_parameters,
        pso_params=None) -> tuple[np.ndarray, float, int, np.ndarray, list, Callable]:
    """
    Implementace algoritmu Particle Swarm Optimization

    Parametry:
         function: účelová funkce
         dimension: dimenze
         test_params: parametry testování
         class_params: parametry všech algoritmů
         pso_params: parametry algoritmu Particle Swarm Optimization

    return: nalezený bod, hodnota v daném bodě, počet iterací, poloha částic pro vizualizaci
            změna hodnoty gbest pro vizualizaci, název testovací funkce

    """
    lb, ub, _, _ = get_bounds(function, dimension)

    if not pso_params:
        w, c1, c2 = alg_params.pso_params  # nastavení parametrů v případě použití benchmark.test_run
    else:
        w, c1, c2 = pso_params

    # Vytvoření hejna částic
    particles, velocities, v_max = swarm_generation(test_params.swarm_size, lb, ub)

    pbest = np.copy(particles)
    pbest_fit = function(particles)
    gbest = np.copy(pbest[np.argmin(pbest_fit)])
    gbest_fit = np.min(pbest_fit)

    # Uložení dat pro vizualizaci
    convergence_curve = []
    history = np.zeros((test_params.max_iter, test_params.swarm_size, len(lb)))

    # Parametry pro ukoncovací kritérium
    iterace = 0
    no_improve_count = 0
    prev_best_pos = np.copy(gbest)
    prev_best_fit = gbest_fit
    e = np.finfo(float).eps

    while iterace < test_params.max_iter:

        r1 = np.random.rand(test_params.swarm_size, len(lb))
        r2 = np.random.rand(test_params.swarm_size, len(lb))

        # Aktualizace rychlostí částic
        velocities = w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)

        velocities = np.clip(velocities, -v_max, v_max)
        particles += velocities
        particles = np.clip(particles, lb, ub)
        fitness_values = function(particles)

        lower_value = fitness_values < pbest_fit
        pbest[lower_value] = particles[lower_value]
        pbest_fit[lower_value] = fitness_values[lower_value]
        best_id = np.argmin(pbest_fit)

        # Aktualizace hodnoty gbest
        if pbest_fit[best_id] < gbest_fit:
            gbest = np.copy(pbest[best_id])
            gbest_fit = pbest_fit[best_id]

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

    # pso_params = [0.4802, 0.794, 0.4632]
    # pso_params = [1.1, 0.794, 0.4632]
    pso_params = [0.72, 1.108, 1.108]
    dimension = 2
    test_function = function4

    x, fx, it, history, cc, func_name = pso(test_function, dimension, test_params, alg_params=None, pso_params=pso_params)
    print('Nalezené řešení: x={}, f(x)={}, počet iterací={}'.format(x, fx, it))

    # Vizualizace
    if dimension == 2:
        swarm_animation(func_name, history, 'PSO')
        swarm_motion(test_params.swarm_size, history)

    convergence_curve(cc, 'PSO')


if __name__ == '__main__':
    main()
