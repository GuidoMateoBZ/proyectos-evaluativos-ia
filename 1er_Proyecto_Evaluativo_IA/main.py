import pygad
import json
import numpy as np
from fitness import convertir_cromosoma, calcular_fitness, imprimir_grilla
from graficos import generar_graficos

def fitness_pygad(ga_instance, cromosoma, idx):
    molinos = convertir_cromosoma(cromosoma)
    return calcular_fitness(molinos)

historial_fitness = []

def on_generation(ga_instance):
    mejor = ga_instance.best_solution()[1]
    historial_fitness.append(mejor)

ga = pygad.GA(
    num_generations=500,
    sol_per_pop=100,
    num_parents_mating=50,
    num_genes=50,
    gene_type=int,
    gene_space=range(0, 20),
    fitness_func=fitness_pygad,
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="two_points",
    mutation_type="random",
    mutation_probability=0.05,
    stop_criteria=["reach_53.25", "saturate_50"],
    on_generation=on_generation,
    keep_elitism=5
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
print(f"Generaciones hasta solución: {ga.generations_completed}")
print("==========================================")

## ---- ANÁLISIS ESTADÍSTICO ----
h = np.array(historial_fitness)
n = len(h)

## ventana para estadísticas finales: últimas 50 gen o todas si fueron menos
ventana = min(50, n)

## generación de convergencia: primera vez que supera el 99% del mejor valor
idx_conv = np.argmax(h >= 0.99 * h[-1]) + 1  

## generación de mayor salto (solo tiene sentido si hubo más de 1 generación)
if n > 1:
    idx_salto = int(np.argmax(np.diff(h)) + 2)  ## +2: diff reduce índice en 1, y gen arranca en 1
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
for i, f in enumerate(historial_fitness, start=1):
    print(f"Gen {i:>4}: {f:.4f} MW")
print("==========================================")
print(f"Desv. estándar (últimas {ventana} gen):    {np.std(h[-ventana:]):.4f} MW")
print(f"Generación de mayor salto:                {idx_salto}")
print(f"Generación de convergencia (~99% óptimo): {idx_conv}")
print("==========================================")

## ---- GRÁFICOS ----
generar_graficos(historial=historial_fitness)