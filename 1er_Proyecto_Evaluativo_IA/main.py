import pygad
import json
import numpy as np
from fitness import convertir_cromosoma, calcular_fitness, imprimir_grilla, contador_molinos
from graficos import generar_graficos

def fitness_pygad(ga_instance, cromosoma, idx):
    molinos = convertir_cromosoma(cromosoma)
    return calcular_fitness(molinos)

historial_fitness = []

def on_generation(ga_instance):
    mejor = ga_instance.best_solution()[1]
    historial_fitness.append(mejor)

ga = pygad.GA(
    # Población
    sol_per_pop=100,

    # Representación Cromosómica
    num_genes=50,
    gene_type=int,
    gene_space=range(0, 20),

    # Fitness
    fitness_func=fitness_pygad,

    # Selección
    parent_selection_type="tournament",
    K_tournament=3,
    keep_elitism=5,

    # Crossover
    crossover_type="two_points",
    num_parents_mating=50,

    # Mutación
    mutation_type="random",
    mutation_probability=0.05,

    # Criterios de parada
    num_generations=500,
    stop_criteria=["reach_53.25", "saturate_50"],

    # ✅ FIX: registrar el callback
    on_generation=on_generation,
)

ga.run()

with open("historial_fitness.json", "w") as f:
    json.dump(historial_fitness, f)

solucion, fitness, _ = ga.best_solution()
molinos = convertir_cromosoma(solucion)

print("==========================================")
print("MEJOR FITNESS")
print(f"Mejor fitness: {fitness:.2f} MW")
print("==========================================")

print("==========================================")
print("SOLUCIÓN")
imprimir_grilla(molinos)
print("==========================================")

print("==========================================")
print("CANTIDAD DE GENERACIONES")
print(f"Cantidad de molinos: {contador_molinos(molinos)}")
print(f"Generaciones hasta solución: {ga.generations_completed}")
print("==========================================")

## ---- ANÁLISIS ESTADÍSTICO ----
h = np.array(historial_fitness)
n = len(h)

ventana = min(50, n)

# ✅ FIX: el historial es ascendente, el óptimo está al final (h[-1])
mejor_fitness = h[-1]
idx_conv = int(np.argmax(h >= 0.99 * mejor_fitness)) + 1  # generación base-1

if n > 1:
    idx_salto = int(np.argmax(np.diff(h)) + 2)
else:
    idx_salto = 1

print("==========================================")
print("ANÁLISIS ESTADÍSTICO")
print(f"Fitness inicial (gen 1):                  {h[0]:.4f} MW")
print(f"Fitness final   (gen {n}):                {h[-1]:.4f} MW")
print(f"Mejora total:                             {h[-1] - h[0]:.4f} MW ({(h[-1]-h[0])/h[0]*100:.2f}%)")
print(f"Media  (últimas {ventana} gen):            {np.mean(h[-ventana:]):.4f} MW")
print("==========================================")
print("FITNESS POR GENERACIÓN")
for i, f_val in enumerate(historial_fitness, start=1):  # ✅ evitar shadowing de built-in `f`
    print(f"Gen {i:>4}: {f_val:.4f} MW")
print("==========================================")
print(f"Desv. estándar (últimas {ventana} gen):    {np.std(h[-ventana:]):.4f} MW")
print(f"Generación de mayor salto:                {idx_salto}")
print(f"Generación de convergencia (~99% óptimo): {idx_conv}")
print("==========================================")

## ---- GRÁFICOS ----
generar_graficos(historial=historial_fitness)