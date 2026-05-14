

import json
import csv
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from ag.estadisticas import metricas_corridas, tabla_comparativa


# ---------------------------------------------------------------------------
# Serialización JSON
# ---------------------------------------------------------------------------

def _config_a_dict(config) -> dict:
    return {
        "nombre":                config.nombre,
        "sol_per_pop":           config.sol_per_pop,
        "num_generations":       config.num_generations,
        "parent_selection_type": config.parent_selection_type,
        "K_tournament":          config.K_tournament,
        "num_parents_mating":    config.num_parents_mating,
        "crossover_type":        config.crossover_type,
        "mutation_type":         config.mutation_type,
        "mutation_probability":  config.mutation_probability,
        "keep_elitism":          config.keep_elitism,
        "stop_criteria":         list(config.stop_criteria),
        "num_genes":             config.num_genes,
    }


def guardar_json(
    resultados_suite: list[dict],
    ruta: str | Path = "resultados_suite.json",
) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "n_configs":  len(resultados_suite),
        "suite": [
            {
                "config":    _config_a_dict(rc["config"]),
                "n_corridas": rc["n_corridas"],
                "corridas":  rc["corridas"],
            }
            for rc in resultados_suite
        ],
    }

    with ruta.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] JSON guardado -> '{ruta}'")
    return ruta


def cargar_json(ruta: str | Path) -> list[dict]:
 
    from ag.configuraciones import ConfigAG

    ruta = Path(ruta)
    with ruta.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    suite = []
    for bloque in payload["suite"]:
        cfg_dict = bloque["config"]
        config = ConfigAG(
            nombre=cfg_dict["nombre"],
            sol_per_pop=cfg_dict["sol_per_pop"],
            num_generations=cfg_dict["num_generations"],
            parent_selection_type=cfg_dict["parent_selection_type"],
            K_tournament=cfg_dict["K_tournament"],
            num_parents_mating=cfg_dict["num_parents_mating"],
            crossover_type=cfg_dict["crossover_type"],
            mutation_type=cfg_dict["mutation_type"],
            mutation_probability=cfg_dict["mutation_probability"],
            keep_elitism=cfg_dict["keep_elitism"],
            stop_criteria=cfg_dict["stop_criteria"],
            num_genes=cfg_dict["num_genes"],
        )
        suite.append({
            "config":    config,
            "corridas":  bloque["corridas"],
            "n_corridas": bloque["n_corridas"],
        })

    print(f"[OK] JSON cargado <- '{ruta}'  ({len(suite)} configs)")
    return suite


# ---------------------------------------------------------------------------
# Exportación CSV
# ---------------------------------------------------------------------------

def guardar_csv_corridas(
    resultados_suite: list[dict],
    ruta: str | Path = "corridas.csv",
) -> Path:
    """Tabla plana con una fila por corrida (sin historial)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    filas = []
    for rc in resultados_suite:
        nombre = rc["config"].nombre
        for c in rc["corridas"]:
            filas.append({
                "config":        nombre,
                "semilla":       c["semilla"],
                "mejor_fitness": c["mejor_fitness"],
                "generaciones":  c["generaciones"],
                "n_molinos":     c["n_molinos"],
                "tiempo_seg":    c["tiempo_seg"],
            })

    df = pd.DataFrame(filas)
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"[OK] CSV corridas guardado -> '{ruta}'  ({len(df)} filas)")
    return ruta


def guardar_csv_resumen(
    resultados_suite: list[dict],
    ruta: str | Path = "resumen_estadistico.csv",
) -> Path:
    """Tabla de métricas estadísticas, una fila por configuración."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df = tabla_comparativa(resultados_suite)
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"[OK] CSV resumen guardado -> '{ruta}'  ({len(df)} configs)")
    return ruta


# ---------------------------------------------------------------------------
# Guardado de figuras matplotlib
# ---------------------------------------------------------------------------

def guardar_figuras(
    figs: dict[str, "plt.Figure"],
    directorio: str | Path = "figuras",
    dpi: int = 150,
) -> list[Path]:
    import matplotlib.pyplot as plt

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    rutas = []
    for nombre, fig in figs.items():
        ruta = directorio / (nombre if nombre.endswith(".png") else nombre + ".png")
        fig.savefig(ruta, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[OK] Figura guardada -> '{ruta}'")
        rutas.append(ruta)

    return rutas


# ---------------------------------------------------------------------------
# Tabla ASCII al terminal
# ---------------------------------------------------------------------------

def imprimir_tabla(resultados_suite: list[dict]) -> None:
    df = tabla_comparativa(resultados_suite)

    cols = [
        ("rank",             "Rank",      4,  "d"),
        ("nombre",           "Config",    20, "s"),
        ("fit_media",        "Fit. Media",11, ".4f"),
        ("fit_std",          "Fit. Std",  10, ".4f"),
        ("fit_max",          "Fit. Max",  10, ".4f"),
        ("gen_media",        "Gen. Med.", 10, ".1f"),
        ("gen_std",          "Gen. Std",  9,  ".1f"),
        ("tasa_convergencia","T. Conv.",  9,  ".0%"),
        ("tiempo_media",     "t_med(s)",  8,  ".2f"),
    ]

    header = "  ".join(f"{lbl:>{w}}" for _, lbl, w, _ in cols)
    sep    = "  ".join("-" * w for _, _, w, _ in cols)

    print()
    print("=" * len(sep))
    print("TABLA COMPARATIVA DE CONFIGURACIONES")
    print("=" * len(sep))
    print(header)
    print(sep)

    for _, row in df.iterrows():
        partes = []
        for col, _, w, fmt in cols:
            val = row[col]
            if fmt == "s":
                partes.append(f"{str(val):>{w}}")
            elif fmt == "d":
                partes.append(f"{int(val):>{w}d}")
            elif fmt == ".0%":
                partes.append(f"{float(val):>{w}.0%}")
            else:
                partes.append(f"{float(val):>{w}{fmt}}")
        print("  ".join(partes))

    print("=" * len(sep))
    print()
