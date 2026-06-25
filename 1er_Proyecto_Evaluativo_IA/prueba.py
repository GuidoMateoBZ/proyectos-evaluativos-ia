P_BASE = 2.13
BETA = 0.08

# Grilla de prueba: 3 molinos en columna (fila 0, 1, 2) y uno aislado
molinos = [(0, 5), (1, 5), (2, 5), (10, 10),(14, 12),(19, 1)]

# ── versión original: cuenta los que YO afecto (downstream) ──────────────────
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

# ── versión corregida: cuenta los que ME afectan (upstream) ──────────────────
def contar_estela_corregida(idx, molinos):
    molino = molinos[idx]
    estela = 0
    for i, otro in enumerate(molinos):
        if i == idx:
            continue
        if otro == molino:
            estela += 1
            continue
        # otro está arriba de mí (me genera estela N→S)
        if (molino[1] == otro[1] and 0 < (molino[0] - otro[0]) <= 3):
            estela += 1
        # otro está a mi derecha (me genera estela E→O)
        elif (molino[0] == otro[0] and 0 < (otro[1] - molino[1]) <= 3):
            estela += 1
    return estela

# ── calcular fitness con cualquiera de las dos ────────────────────────────────
def calcular_fitness(molinos, fn_estela):
    total = 0
    for i, molino in enumerate(molinos):
        wake = fn_estela(i, molinos)
        potencia = P_BASE * max(0, 1 - BETA * wake)
        print(f"  Molino {i} {molino}: wake={wake}, potencia={potencia:.4f} MW")
        total += potencia
    return total

print("=== VERSIÓN ORIGINAL (cuenta a quién afecto) ===")
z1 = calcular_fitness(molinos, contar_estela)
print(f"  TOTAL: {z1:.4f} MW\n")

print("=== VERSIÓN CORREGIDA (cuenta quién me afecta) ===")
z2 = calcular_fitness(molinos, contar_estela_corregida)
print(f"  TOTAL: {z2:.4f} MW\n")

print(f"¿Totales iguales? {abs(z1 - z2) < 1e-9}")