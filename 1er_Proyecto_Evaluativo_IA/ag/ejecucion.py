

import time
import numpy as np
import pygad

from ag.configuraciones import ConfigAG
from fitness import convertir_cromosoma, calcular_fitness, contador_molinos


def _fitness_pygad(ga_instance, cromosoma, idx):
    return calcular_fitness(convertir_cromosoma(cromosoma))


def _ejecutar_corrida(config: ConfigAG, semilla: int) -> dict:
    np.random.seed(semilla)

    hist_mejor = []
    hist_promedio = []
    hist_std = []

    def on_generation(ga):
        fits = ga.last_generation_fitness
        mejor_actual = float(np.max(fits))

        # Historial monótono creciente
        if not hist_mejor:
            hist_mejor.append(mejor_actual)
        else:
            hist_mejor.append(max(hist_mejor[-1], mejor_actual))

        hist_promedio.append(float(np.mean(fits)))
        hist_std.append(float(np.std(fits)))

    kwargs = config.a_dict_pygad()
    kwargs["fitness_func"] = _fitness_pygad
    kwargs["on_generation"] = on_generation

    t0 = time.perf_counter()
    ga = pygad.GA(**kwargs)
    ga.run()
    tiempo = time.perf_counter() - t0

    solucion, fitness, _ = ga.best_solution()
    molinos = convertir_cromosoma(solucion)

    return {
        "semilla":           semilla,
        "mejor_fitness":     round(float(fitness), 6),
        "generaciones":      ga.generations_completed,
        "n_molinos":         contador_molinos(molinos),
        "tiempo_seg":        round(tiempo, 4),
        "historial_mejor":   hist_mejor,
        "historial_promedio": hist_promedio,
        "historial_std":     hist_std,
    }


def ejecutar_config(
    config: ConfigAG,
    n_corridas: int = 30,
    semilla_base: int = 42,
    verbose: bool = True,
) -> dict:
  
    semillas = [semilla_base + i for i in range(n_corridas)]
    corridas = []

    if verbose:
        sep = "=" * 50
        print(f"\n{sep}")
        print(f"  CONFIG: {config.nombre}")
        print(f"  {config.resumen()}")
        print(f"  Corridas: {n_corridas}")
        print(sep)

    for idx, semilla in enumerate(semillas, start=1):
        resultado = _ejecutar_corrida(config, semilla)

        if verbose:
            print(
                f"  [{idx:>2}/{n_corridas}] "
                f"fit={resultado['mejor_fitness']:.4f} MW  "
                f"gen={resultado['generaciones']:>4}  "
                f"t={resultado['tiempo_seg']:.2f}s"
            )

        corridas.append(resultado)

    return {
        "config":    config,
        "corridas":  corridas,
        "n_corridas": n_corridas,
    }


def ejecutar_suite(
    configs: list[ConfigAG],
    n_corridas: int = 30,
    semilla_base: int = 42,
    verbose: bool = True,
) -> list[dict]:
 
    return [
        ejecutar_config(c, n_corridas=n_corridas,
                        semilla_base=semilla_base, verbose=verbose)
        for c in configs
    ]
