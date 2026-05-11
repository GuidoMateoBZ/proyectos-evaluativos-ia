import numpy as np

P_BASE = 2.13   ## potencia base de cada molino en MW
BETA = 0.08     ## estela
cromosoma_prueba = [3,7, 15,2, 8,19, 4,11, 1,5,
                    0,0, 10,10, 17,3, 6,15, 9,3,
                    7,2, 12,8, 5,14, 18,1, 2,16,
                    13,4, 16,7, 11,9, 14,12, 3,18,
                    8,6, 19,0, 0,19, 6,6, 15,15]


## para convertir los cromosomas en coordenadas
def convertir_cromosoma(cromosoma):
    molinos = [(cromosoma[i], cromosoma[i+1]) for i in range(0, 50, 2)]
    return molinos

## dado un molino x me devuelve los y molinos que estane en el area de su estela
def contar_estela(molino, molinos):
    estela = 0
    for otro in molinos:
        ##chequeo qe no sea el mismo molino 
        if otro == molino:
            estela += 1
    ##este a oeste
        if (molino[0] == otro[0] and 0 < (molino[1] - otro[1]) <=3):
            estela +=1
        elif (molino[1] == otro[1] and 0 < (otro[0] - molino[0]) <= 3):
            estela +=1
    return estela

def imprimir_grilla(molinos):
    grilla = [['.' for _ in range(20)] for _ in range(20)]
    
    for fila, col in molinos:
        grilla[fila][col] = 'M'
    
    for fila in grilla:
        print(' '.join(fila))


##calculo del fitness (segun lo que entendi del docu) Z = ∑ PiBase . max(0, 1 - β · wake(i,j))

def calcular_fitness(molinos):
    energia_total = 0
    for molino in molinos: 
        estela = contar_estela(molino, molinos)
        potencia = P_BASE * max(0, 1 - BETA *estela)
        energia_total += potencia
    return energia_total

def contador_molinos(molinos):
    set_molinos = set(molinos)
    return len(set_molinos)


molinos = convertir_cromosoma(cromosoma_prueba)
