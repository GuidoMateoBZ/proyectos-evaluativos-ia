import pygad ##
from fitness import convertir_cromosoma, calcular_fitness, imprimir_grilla

##la convierto para usarlo con pygad
def fitness_pygad(ga_instance,cromosoma,idx):
    molinos = convertir_cromosoma(cromosoma)
    return calcular_fitness(molinos)

ga = pygad.GA(
    num_generations=500, ##maximo de generaciones
    sol_per_pop=100, ##por cada generacion son 100 individuos
    num_parents_mating=50, ##50 parejas se reproducen por genracion
    num_genes=50, ##cromosoma de 50 numeros (25 molinos x 2 corrdenadas)
    gene_type=int,##coordenadas enteras
    gene_space=range(0, 20), ##cada coordenada
    fitness_func=fitness_pygad, ##la funcion fitneess
    parent_selection_type="tournament", ##seleccion por torneo
    K_tournament=3, ##el torneo se elige entre 3 individuos
    crossover_type="two_points", ## cruce de dos puntos
    mutation_type="random", ##mutacion aleatoria
    mutation_probability=0.05, ## 5 % de probabildad de mutar cada gen
    stop_criteria=["reach_53.25", "saturate_50"], ## para si hay 50 generaciones sin mejora
    keep_elitism=5 ##los 5 mejores pasan a la siguiente generacion (la elite)
)

ga.run()

solucion, fitness, _ = ga.best_solution()

print(f"Mejor fitness: {fitness:.2f} MW")
molinos = convertir_cromosoma(solucion)

imprimir_grilla(molinos)
