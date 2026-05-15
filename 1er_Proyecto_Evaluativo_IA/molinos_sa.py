P_BASE = 2.13   
BETA = 0.08 

def contar_estela(idx, molinos):
    molino = molinos[idx]
    estela = 0
    for i, otro in enumerate(molinos):
        if i == idx:
            continue
        if otro == molino:
            estela += 1
            continue
        if (molino[0] == otro[0] and 0 < (molino[1] - otro[1]) <= 3):
            estela += 1
        elif (molino[1] == otro[1] and 0 < (otro[0] - molino[0]) <= 3):
            estela += 1
    return estela


def imprimir_grilla(molinos):
    grilla = [['.' for _ in range(20)] for _ in range(20)]
    
    for fila, col in molinos:
        grilla[fila][col] = 'M'
    
    for fila in grilla:
        print(' '.join(fila))


##calculo del fitness Z = ∑ PiBase . max(0, 1 - β · wake(i,j))

def calcular_fitness(molinos):
    energia_total = 0
    for i, molino in enumerate(molinos):
        estela = contar_estela(i, molinos)
        potencia = P_BASE * max(0, 1 - BETA * estela)
        energia_total += potencia
    return energia_total

def contador_molinos(molinos):
    set_molinos = set(molinos)
    return len(set_molinos)



import math
import random

def inicializar_tablero(num_molinos=25, tam_grilla=20):
    """
    Genera un vector de 50 coordenadas aleatorias (25 pares fila-columna)
    asegurando que no hayan dos molinos en la misma posición.
    """
    molinos_set = set()
    while len(molinos_set) < num_molinos:
        fila = random.randint(0, tam_grilla - 1)
        col = random.randint(0, tam_grilla - 1)
        molinos_set.add((fila, col))
    
    vector_coordenadas = []
    for fila, col in molinos_set:
        vector_coordenadas.extend([fila, col])
        
    return vector_coordenadas



def generar_vecino(tablero, tam_grilla=20):
    """
    Genera un vecino cambiando aleatoriamente un par de coordenadas (un molino)
    por una nueva ubicación vacía en la grilla.
    """
    vecino = tablero[:]
    
    # Guardamos pares actuales para validación
    pares_actuales = set()
    for i in range(0, len(vecino), 2):
        pares_actuales.add((vecino[i], vecino[i+1]))
        
    # Elegir un molino aleatorio a mover
    idx_molino = random.randint(0, (len(vecino) // 2) - 1)
    idx_coord = idx_molino * 2
    
    pos_actual = (vecino[idx_coord], vecino[idx_coord+1])
    pares_actuales.remove(pos_actual)
    
    # Generar nueva posición única
    while True:
        nueva_fila = random.randint(0, tam_grilla - 1)
        nueva_col = random.randint(0, tam_grilla - 1)
        nueva_pos = (nueva_fila, nueva_col)
        
        if nueva_pos not in pares_actuales:
            break
            
    # Asignar la nueva posición al vector
    vecino[idx_coord] = nueva_fila
    vecino[idx_coord+1] = nueva_col
    
    return vecino



def simulated_annealing(max_iteraciones):
    # Generar tablero inicial aleatorio (vector de 50 coordenadas)
    tablero_sa = inicializar_tablero(25, 20)
    
    # Función auxiliar para convertir el vector a lista de tuplas y calcular fitness
    def evaluar_fitness(vector):
        molinos_tuplas = [(vector[i], vector[i+1]) for i in range(0, len(vector), 2)]
        return calcular_fitness(molinos_tuplas)

    energia_actual = evaluar_fitness(tablero_sa)
    
    # Parámetros del SA
    temperatura = 5.0  # Temperatura inicial
    alfa = 0.99        # Factor de enfriamiento
    
    mejor_tablero = tablero_sa[:]
    mejor_energia = energia_actual
    
    # Máxima energía teórica = 25 * 2.13 = 53.25 MW
    ENERGIA_OPTIMA = 53.25
    
    for i in range(max_iteraciones):
        # Condición de parada: si se alcanza la energía óptima terminamos
        if mejor_energia >= ENERGIA_OPTIMA:
            return mejor_tablero, i
            
        vecino = generar_vecino(tablero_sa, 20)
        energia_vecino = evaluar_fitness(vecino)
        
        # Diferencia de energía (queremos MAXIMIZAR la energía)
        delta_e = energia_vecino - energia_actual
        
        if delta_e > 0:
            # Es MEJOR (más energía): Lo aceptamos siempre
            tablero_sa = vecino
            energia_actual = energia_vecino
            
            # Guardamos la mejor solución global encontrada
            if energia_actual > mejor_energia:
                mejor_energia = energia_actual
                mejor_tablero = tablero_sa[:]
        else:
            # Es PEOR o IGUAL (menos energía): usamos la Temperatura
            if temperatura > 0.0001: 
                # Probabilidad de Boltzmann
                probabilidad = math.exp(delta_e / temperatura)
                if random.random() < probabilidad:
                    # Lo aceptamos a pesar de ser peor
                    tablero_sa = vecino
                    energia_actual = energia_vecino
                    
        # Factor de enfriamiento
        temperatura = temperatura * alfa

    return mejor_tablero, max_iteraciones

if __name__ == "__main__":
    print("Iniciando Simulated Annealing...")
    max_iter = 1000
    mejor_vector, iteraciones = simulated_annealing(max_iter)
    
    # Convertir el vector 1D a lista de tuplas para calcular e imprimir
    molinos_tuplas = [(mejor_vector[i], mejor_vector[i+1]) for i in range(0, len(mejor_vector), 2)]
    
    mejor_fitness = calcular_fitness(molinos_tuplas)
    
    print("\n--- RESULTADO SIMULATED ANNEALING ---")
    print(f"Mejor fitness (energía): {mejor_fitness:.2f} MW")
    print(f"Iteraciones realizadas: {iteraciones}")
    print("\nGrilla resultante:")
    imprimir_grilla(molinos_tuplas)