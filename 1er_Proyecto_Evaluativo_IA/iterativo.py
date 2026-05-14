
import numpy as np
from ag.configuraciones import CONFIG_BASE
from ag.ejecucion       import ejecutar_config
from ag.graficos        import graficar_panel_config
from ag.exportacion     import imprimir_tabla


def experimentar(n_corridas: int = 30, semilla_base: int = 42) -> list[dict]:
 
    rc = ejecutar_config(CONFIG_BASE, n_corridas=n_corridas,
                         semilla_base=semilla_base, verbose=True)
    corridas = rc["corridas"]

    # Adaptador: agrega clave "corrida" (número 1-based) para compatibilidad
    for i, c in enumerate(corridas, start=1):
        c.setdefault("corrida", i)
        c.setdefault("poblacion", CONFIG_BASE.sol_per_pop)

    return corridas


def graficar_experimento(resultados: list[dict], guardar: bool = True) -> None:
 
    if resultados and "corridas" in resultados[0]:
        # Ya es un ResultadoConfig — pasar directo
        rc = resultados[0]
    else:
        # Lista de dicts de corrida — envolver en ResultadoConfig
        rc = {
            "config":    CONFIG_BASE,
            "corridas":  resultados,
            "n_corridas": len(resultados),
        }

    import matplotlib.pyplot as plt
    fig = graficar_panel_config(rc, guardar="exp_panel.png" if guardar else None)
    plt.show(block=True)
    plt.close(fig)


def imprimir_resumen(resultados: list[dict]) -> None:
 
    if not resultados:
        return

    if "corridas" in resultados[0]:
        suite = resultados
    else:
        suite = [{
            "config":    CONFIG_BASE,
            "corridas":  resultados,
            "n_corridas": len(resultados),
        }]

    imprimir_tabla(suite)


if __name__ == "__main__":
    print("=== Iniciando experimento ===")
    resultados = experimentar(n_corridas=30)
    imprimir_resumen(resultados)
    graficar_experimento(resultados)