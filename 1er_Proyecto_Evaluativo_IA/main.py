"""
Pipeline de experimentación comparativa — Algoritmos Genéticos para Wind Farm.

Uso:
    python main.py                    # suite completa (default)
    python main.py --modo base        # una sola config base, 30 corridas
    python main.py --modo poblacion   # compara tamaños de población
    python main.py --corridas 10      # reduce N corridas (pruebas rápidas)
    python main.py --cargar resultados_suite.json  # reanalizar sin re-ejecutar
"""

import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt

from ag.configuraciones import (
    CONFIG_BASE,
    CONFIGS_POBLACION,
    CONFIGS_MUTACION,
    CONFIGS_CRUCE,
    CONFIGS_SELECCION,
    CONFIGS_ELITISMO,
)
from ag.ejecucion   import ejecutar_config, ejecutar_suite
from ag.estadisticas import tabla_comparativa, matriz_heatmap
from ag.graficos    import (
    graficar_panel_config,
    graficar_convergencia_comparativa,
    graficar_boxplot_fitness,
    graficar_boxplot_generaciones,
    graficar_barras_comparativas,
    graficar_heatmap,
)
from ag.exportacion import (
    guardar_json,
    cargar_json,
    guardar_csv_corridas,
    guardar_csv_resumen,
    imprimir_tabla,
)

matplotlib.use("Agg")   # sin ventanas interactivas; guardar directo a PNG


SUITES_DISPONIBLES = {
    "base":      [CONFIG_BASE],
    "poblacion": CONFIGS_POBLACION,
    "mutacion":  CONFIGS_MUTACION,
    "cruce":     CONFIGS_CRUCE,
    "seleccion": CONFIGS_SELECCION,
    "elitismo":  CONFIGS_ELITISMO,
}


def _parsear_args():
    parser = argparse.ArgumentParser(
        description="Framework de experimentación comparativa para AG Wind Farm"
    )
    parser.add_argument(
        "--modo", default="base",
        choices=list(SUITES_DISPONIBLES.keys()),
        help="Grupo de configuraciones a comparar (default: base)",
    )
    parser.add_argument(
        "--corridas", type=int, default=30,
        help="Número de corridas independientes por configuración (default: 30)",
    )
    parser.add_argument(
        "--semilla", type=int, default=42,
        help="Semilla base para la secuencia de corridas (default: 42)",
    )
    parser.add_argument(
        "--cargar", type=str, default=None,
        metavar="ARCHIVO.json",
        help="Reusar resultados previamente guardados (omite ejecución)",
    )
    parser.add_argument(
        "--salida", type=str, default="resultados",
        help="Directorio de salida para figuras y CSV (default: resultados/)",
    )
    return parser.parse_args()


def _pipeline(resultados_suite: list[dict], directorio: str, modo: str) -> None:
    """Analiza, visualiza y exporta una suite de resultados."""
    from pathlib import Path
    import os

    outdir = Path(directorio) / modo
    outdir.mkdir(parents=True, exist_ok=True)

    # Tabla en consola
    imprimir_tabla(resultados_suite)

    # Exportar datos
    guardar_json(resultados_suite,            outdir / "suite.json")
    guardar_csv_corridas(resultados_suite,    outdir / "corridas.csv")
    guardar_csv_resumen(resultados_suite,     outdir / "resumen.csv")

    # Figuras por configuración individual
    for rc in resultados_suite:
        nombre = rc["config"].nombre
        fig = graficar_panel_config(rc)
        fig.savefig(outdir / f"panel_{nombre}.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[OK] Panel guardado -> '{outdir / ('panel_' + nombre + '.png')}'")

    # Figuras comparativas (solo si hay más de una config)
    if len(resultados_suite) > 1:
        titulo_modo = f"Suite: {modo.upper()}"

        fig = graficar_convergencia_comparativa(
            resultados_suite,
            titulo=f"Convergencia Comparativa — {titulo_modo}",
        )
        fig.savefig(outdir / "comp_convergencia.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        fig = graficar_boxplot_fitness(
            resultados_suite,
            titulo=f"Fitness Final — {titulo_modo}",
        )
        fig.savefig(outdir / "comp_boxplot_fitness.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        fig = graficar_boxplot_generaciones(
            resultados_suite,
            titulo=f"Generaciones hasta Convergencia — {titulo_modo}",
        )
        fig.savefig(outdir / "comp_boxplot_generaciones.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        fig = graficar_barras_comparativas(
            resultados_suite,
            titulo=f"Fitness Medio ± Std — {titulo_modo}",
        )
        fig.savefig(outdir / "comp_barras_fitness.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        fig = graficar_barras_comparativas(
            resultados_suite,
            metrica="gen_media",
            error_metrica="gen_std",
            titulo=f"Generaciones Medias ± Std — {titulo_modo}",
            ylabel="Generaciones medias",
        )
        fig.savefig(outdir / "comp_barras_generaciones.png", dpi=150,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"\n[OK] Pipeline completado. Resultados en: '{outdir.resolve()}'")


def main():
    args = _parsear_args()

    if args.cargar:
        print(f"[INFO] Cargando resultados desde '{args.cargar}' ...")
        resultados_suite = cargar_json(args.cargar)
        modo = "cargado"
    else:
        configs = SUITES_DISPONIBLES[args.modo]
        print(f"[INFO] Modo: {args.modo}  |  Configs: {len(configs)}  |  Corridas: {args.corridas}")
        resultados_suite = ejecutar_suite(
            configs,
            n_corridas=args.corridas,
            semilla_base=args.semilla,
        )
        modo = args.modo

    _pipeline(resultados_suite, args.salida, modo)


if __name__ == "__main__":
    main()