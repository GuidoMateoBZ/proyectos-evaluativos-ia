import pygad # Librería usada para AG
from fitness import convertir_cromosoma, calcular_fitness, imprimir_grilla, contador_molinos

# La convierto para usarlo con pygad
def fitness_pygad(ga_instance,cromosoma,idx):
    molinos = convertir_cromosoma(cromosoma)
    return calcular_fitness(molinos)

ga = pygad.GA(

    # Población
    sol_per_pop = 100, # Por cada generación son 100 individuos

    # Representación Cromosómica
    num_genes = 50, # Cromosoma de 50 numeros (25 molinos x 2 coordenadas)
    gene_type = int, # Coordenadas enteras
    gene_space = range(0, 19), # Cada coordenada de la grilla 20x20

    # Fitness
    fitness_func = fitness_pygad, # La funcion fitness
    
    # Selección
    parent_selection_type = "tournament", # Selección por torneo
    K_tournament = 3, # El torneo se elige entre 3 individuos
    keep_elitism = 5, # Los 5 mejores pasan a la siguiente generación (la elite)

    # Crossover
    crossover_type = "two_points", # Cruce de dos puntos
    num_parents_mating = 50, # 50 parejas se reproducen por generación

    # Mutación
    mutation_type = "random", # Mutación aleatoria
    mutation_probability = 0.05, # 5% de probabilidad de mutar cada gen
    
    # Criterios de parada
    num_generations = 500, # Máximo de generaciones
    stop_criteria = ["reach_53.25", "saturate_50"] # Para si hay 50 generaciones sin mejora
)

ga.run()

solucion, fitness, _ = ga.best_solution()
print("==========================================")
print("MEJOR FITNESS")

print(f"Mejor fitness: {fitness:.2f} MW")
print("==========================================")

print("==========================================")

print("SOLUCIÓN")

molinos = convertir_cromosoma(solucion)

imprimir_grilla(molinos)
print("==========================================")
print("==========================================")
print("CANTIDAD DE GENERACIONES")

print(f"Cantidad de molinos: {contador_molinos(molinos)}")
print(f"Generaciones hasta solución: {ga.generations_completed}")
print("==========================================")
