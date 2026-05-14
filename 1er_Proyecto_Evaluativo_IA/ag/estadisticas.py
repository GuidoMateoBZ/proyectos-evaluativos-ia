

import numpy as np
import pandas as pd


def metricas_corridas(resultado_config: dict) -> dict:

    corridas = resultado_config["corridas"]
    config = resultado_config["config"]

    fits = np.array([c["mejor_fitness"] for c in corridas])
    gens = np.array([c["generaciones"] for c in corridas])
    tiempos = np.array([c["tiempo_seg"] for c in corridas])

    # Corridas que alcanzaron el criterio reach_* (no saturaron)
    gen_max = config.num_generations
    convergidas = np.sum(gens < gen_max)

    return {
        "nombre":           config.nombre,
        "n_corridas":       len(corridas),
        # Fitness
        "fit_media":        round(float(np.mean(fits)), 6),
        "fit_mediana":      round(float(np.median(fits)), 6),
        "fit_std":          round(float(np.std(fits, ddof=1)), 6),
        "fit_min":          round(float(np.min(fits)), 6),
        "fit_max":          round(float(np.max(fits)), 6),
        "fit_p25":          round(float(np.percentile(fits, 25)), 6),
        "fit_p75":          round(float(np.percentile(fits, 75)), 6),
        # Generaciones
        "gen_media":        round(float(np.mean(gens)), 2),
        "gen_mediana":      round(float(np.median(gens)), 1),
        "gen_std":          round(float(np.std(gens, ddof=1)), 2),
        "gen_min":          int(np.min(gens)),
        "gen_max":          int(np.max(gens)),
        # Tiempo
        "tiempo_media":     round(float(np.mean(tiempos)), 4),
        "tiempo_total":     round(float(np.sum(tiempos)), 2),
        # Convergencia
        "tasa_convergencia": round(float(convergidas / len(corridas)), 4),
    }


def tabla_comparativa(resultados_suite: list[dict]) -> pd.DataFrame:
  
    filas = [metricas_corridas(rc) for rc in resultados_suite]
    df = pd.DataFrame(filas)
    df = df.sort_values("fit_media", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def matriz_heatmap(
    resultados_suite: list[dict],
    param_x: str,
    param_y: str,
    metrica: str = "fit_media",
) -> pd.DataFrame:

    registros = []
    for rc in resultados_suite:
        m = metricas_corridas(rc)
        cfg = rc["config"]
        registros.append({
            param_x: getattr(cfg, param_x),
            param_y: getattr(cfg, param_y),
            metrica: m[metrica],
        })

    df = pd.DataFrame(registros)
    return df.pivot(index=param_y, columns=param_x, values=metrica)


def curvas_promedio(resultado_config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  
    corridas = resultado_config["corridas"]
    historiales = [c["historial_mejor"] for c in corridas]
    max_gen = max(len(h) for h in historiales)

    mat = np.full((len(corridas), max_gen), np.nan)
    for i, h in enumerate(historiales):
        mat[i, : len(h)] = h

    media = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    gens = np.arange(1, max_gen + 1)
    return media, std, gens
