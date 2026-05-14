

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ag.estadisticas import curvas_promedio, metricas_corridas

# ---------------------------------------------------------------------------
# Tema global
# ---------------------------------------------------------------------------

BG     = "#0d1117"
AX_BG  = "#161b22"
GRID   = "#21262d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

PALETTE = [
    "#58a6ff", "#3fb950", "#e3b341", "#f78166",
    "#d2a8ff", "#79c0ff", "#56d364", "#ffa657",
    "#ff7b72", "#a5d6ff",
]

BEST_COLOR  = "#3fb950"
MEAN_COLOR  = "#e3b341"
BAND_ALPHA  = 0.15


def _aplicar_tema():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    AX_BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "legend.facecolor":  AX_BG,
        "legend.edgecolor":  GRID,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })


def _estilo_ax(ax, title: str, fontsize: int = 13):
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)


# ---------------------------------------------------------------------------
# 1. Curvas de convergencia — una configuración
# ---------------------------------------------------------------------------

def graficar_convergencia_config(
    resultado_config: dict,
    guardar: str | None = None,
) -> plt.Figure:
  
    _aplicar_tema()

    corridas = resultado_config["corridas"]
    config   = resultado_config["config"]
    n = len(corridas)

    fits = [c["mejor_fitness"] for c in corridas]
    idx_mejor = int(np.argmax(fits))

    media, std, gens = curvas_promedio(resultado_config)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.11)

    _estilo_ax(ax, f"Convergencia — {config.nombre}  ({n} corridas independientes)")

    cmap = plt.cm.get_cmap("cool", n)

    # Banda ±std
    ax.fill_between(gens, media - std, media + std,
                    alpha=BAND_ALPHA, color=MEAN_COLOR, zorder=1,
                    label="Banda ±1 std")

    # Corridas individuales
    for i, c in enumerate(corridas):
        if i == idx_mejor:
            continue
        h = c["historial_mejor"]
        ax.plot(range(1, len(h) + 1), h,
                color=cmap(i), linewidth=0.85, alpha=0.45, zorder=2)

    # Curva media
    ax.plot(gens, media,
            color=MEAN_COLOR, linewidth=1.8, linestyle="--", alpha=0.9,
            zorder=3, label=f"Media ({n} corridas)")

    # Mejor corrida
    mejor_c = corridas[idx_mejor]
    h_best  = mejor_c["historial_mejor"]
    ax.plot(range(1, len(h_best) + 1), h_best,
            color=BEST_COLOR, linewidth=2.5, alpha=1.0, zorder=4,
            label=f"Mejor corrida (seed={mejor_c['semilla']}, "
                  f"{mejor_c['mejor_fitness']:.4f} MW)")

    ax.axhline(mejor_c["mejor_fitness"],
               color=BEST_COLOR, linestyle=":", linewidth=1.1, alpha=0.5)

    ax.set_xlabel("Generación", fontsize=11, labelpad=6)
    ax.set_ylabel("Fitness  (MW)", fontsize=11, labelpad=6)
    ax.set_xlim(1, len(gens))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.3)

    m = metricas_corridas(resultado_config)
    fig.suptitle(
        f"{config.resumen()}  |  "
        f"fit_media={m['fit_media']:.4f}  std={m['fit_std']:.4f}  "
        f"gen_media={m['gen_media']:.1f}",
        color=TEXT, fontsize=10,
    )

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Figura guardada -> '{guardar}'")

    return fig


def graficar_convergencia_comparativa(
    resultados_suite: list[dict],
    titulo: str = "Comparación de Convergencia",
    guardar: str | None = None,
) -> plt.Figure:
  
    _aplicar_tema()

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.10)
    _estilo_ax(ax, titulo)

    for i, rc in enumerate(resultados_suite):
        color = PALETTE[i % len(PALETTE)]
        media, std, gens = curvas_promedio(rc)
        nombre = rc["config"].nombre

        ax.fill_between(gens, media - std, media + std,
                        alpha=0.12, color=color, zorder=1)
        ax.plot(gens, media,
                color=color, linewidth=2.0, label=nombre, zorder=2)

    ax.set_xlabel("Generación", fontsize=11, labelpad=6)
    ax.set_ylabel("Fitness medio (MW)", fontsize=11, labelpad=6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, framealpha=0.3)

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Figura guardada -> '{guardar}'")

    return fig


# 3. Boxplot comparativo de fitness final

def graficar_boxplot_fitness(
    resultados_suite: list[dict],
    titulo: str = "Distribución de Fitness Final por Configuración",
    guardar: str | None = None,
) -> plt.Figure:
    
    _aplicar_tema()

    nombres = [rc["config"].nombre for rc in resultados_suite]
    datos   = [[c["mejor_fitness"] for c in rc["corridas"]]
               for rc in resultados_suite]
    n_configs = len(resultados_suite)

    # Ordenar por mediana descendente
    orden = sorted(range(n_configs),
                   key=lambda i: np.median(datos[i]), reverse=True)
    nombres = [nombres[i] for i in orden]
    datos   = [datos[i] for i in orden]

    alto = max(5, n_configs * 0.7)
    fig, ax = plt.subplots(figsize=(12, alto), facecolor=BG)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.10)
    _estilo_ax(ax, titulo)

    bp = ax.boxplot(
        datos,
        vert=False,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        showfliers=True,
    )

    for i, (box, whisker_pair, cap_pair) in enumerate(
        zip(bp["boxes"],
            zip(bp["whiskers"][::2], bp["whiskers"][1::2]),
            zip(bp["caps"][::2], bp["caps"][1::2]))
    ):
        c = PALETTE[i % len(PALETTE)]
        box.set(facecolor=c + "33", edgecolor=c, linewidth=1.8)
        for w in whisker_pair:
            w.set(color=MUTED, linewidth=1.3, linestyle="--")
        for cap in cap_pair:
            cap.set(color=MUTED, linewidth=1.8)

    for med in bp["medians"]:
        med.set(color=BEST_COLOR, linewidth=2.5)

    for mean in bp["means"]:
        mean.set(marker="D", markerfacecolor=MEAN_COLOR,
                 markeredgecolor="white", markersize=7)

    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="#f78166",
                  markeredgecolor="white", markersize=5, alpha=0.8)

    ax.set_yticklabels(nombres, fontsize=9)
    ax.set_xlabel("Fitness  (MW)", fontsize=11, labelpad=6)

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Figura guardada -> '{guardar}'")

    return fig


# ---------------------------------------------------------------------------
# 4. Boxplot de generaciones hasta convergencia
# ---------------------------------------------------------------------------

def graficar_boxplot_generaciones(
    resultados_suite: list[dict],
    titulo: str = "Generaciones hasta Convergencia por Configuración",
    guardar: str | None = None,
) -> plt.Figure:
 
    _aplicar_tema()

    nombres = [rc["config"].nombre for rc in resultados_suite]
    datos   = [[c["generaciones"] for c in rc["corridas"]]
               for rc in resultados_suite]
    n_configs = len(resultados_suite)

    # Ordenar por mediana ascendente (menos generaciones = más rápido)
    orden = sorted(range(n_configs),
                   key=lambda i: np.median(datos[i]))
    nombres = [nombres[i] for i in orden]
    datos   = [datos[i] for i in orden]

    alto = max(5, n_configs * 0.7)
    fig, ax = plt.subplots(figsize=(12, alto), facecolor=BG)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.10)
    _estilo_ax(ax, titulo)

    bp = ax.boxplot(datos, vert=False, patch_artist=True,
                    widths=0.55, showmeans=True, showfliers=True)

    for i, box in enumerate(bp["boxes"]):
        c = PALETTE[i % len(PALETTE)]
        box.set(facecolor=c + "33", edgecolor=c, linewidth=1.8)
    for med in bp["medians"]:
        med.set(color=BEST_COLOR, linewidth=2.5)
    for mean in bp["means"]:
        mean.set(marker="D", markerfacecolor=MEAN_COLOR,
                 markeredgecolor="white", markersize=7)
    for w in bp["whiskers"]:
        w.set(color=MUTED, linewidth=1.3, linestyle="--")
    for cap in bp["caps"]:
        cap.set(color=MUTED, linewidth=1.8)
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="#f78166",
                  markeredgecolor="white", markersize=5, alpha=0.8)

    ax.set_yticklabels(nombres, fontsize=9)
    ax.set_xlabel("Generaciones", fontsize=11, labelpad=6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Figura guardada -> '{guardar}'")

    return fig


# 5. Gráfico de barras — métricas resumidas (media ± std)

def graficar_barras_comparativas(
    resultados_suite: list[dict],
    metrica: str = "fit_media",
    error_metrica: str = "fit_std",
    titulo: str | None = None,
    ylabel: str = "Fitness medio (MW)",
    guardar: str | None = None,
) -> plt.Figure:
  
    _aplicar_tema()

    from ag.estadisticas import metricas_corridas

    metricas = [metricas_corridas(rc) for rc in resultados_suite]
    nombres  = [m["nombre"] for m in metricas]
    valores  = [m[metrica] for m in metricas]
    errores  = [m[error_metrica] for m in metricas]

    orden = sorted(range(len(valores)), key=lambda i: valores[i], reverse=True)
    nombres = [nombres[i] for i in orden]
    valores = [valores[i] for i in orden]
    errores = [errores[i] for i in orden]

    colores = [PALETTE[i % len(PALETTE)] for i in range(len(nombres))]

    fig, ax = plt.subplots(figsize=(max(8, len(nombres) * 1.4), 6), facecolor=BG)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.18)
    _estilo_ax(ax, titulo or f"Comparación — {metrica}")

    bars = ax.bar(nombres, valores, color=[c + "aa" for c in colores],
                  edgecolor=colores, linewidth=1.5, zorder=2)
    ax.errorbar(nombres, valores, yerr=errores,
                fmt="none", color=MUTED, capsize=5, capthick=1.5,
                elinewidth=1.5, zorder=3)

    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errores) * 0.05,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8, color=TEXT, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=11, labelpad=6)
    ax.set_xticklabels(nombres, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, max(v + e for v, e in zip(valores, errores)) * 1.12)

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Figura guardada -> '{guardar}'")

    return fig


# 6. Heatmap

def graficar_heatmap(
    df_pivot: "pd.DataFrame",
    titulo: str = "Heatmap de Métricas",
    label_colorbar: str = "Fitness medio (MW)",
    guardar: str | None = None,
) -> plt.Figure:
  
    import pandas as pd  # noqa: F401 — necesario para type hint en runtime
    _aplicar_tema()

    data = df_pivot.values.astype(float)
    rows = [str(r) for r in df_pivot.index]
    cols = [str(c) for c in df_pivot.columns]

    fig, ax = plt.subplots(
        figsize=(max(6, len(cols) * 1.5), max(4, len(rows) * 1.2)),
        facecolor=BG,
    )
    fig.subplots_adjust(left=0.18, right=0.92, top=0.88, bottom=0.18)
    _estilo_ax(ax, titulo)

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(label_colorbar, color=TEXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(rows, fontsize=9)

    vmin, vmax = np.nanmin(data), np.nanmax(data)
    mid = (vmin + vmax) / 2
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = data[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if val < mid else "black"
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")

    ax.set_xlabel(df_pivot.columns.name or "Columnas", fontsize=10, labelpad=6)
    ax.set_ylabel(df_pivot.index.name or "Filas", fontsize=10, labelpad=6)

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Figura guardada -> '{guardar}'")

    return fig


# 7. Panel único: una configuración (convergencia + boxplot generaciones)

def graficar_panel_config(
    resultado_config: dict,
    guardar: str | None = None,
) -> plt.Figure:

    _aplicar_tema()

    config  = resultado_config["config"]
    corridas = resultado_config["corridas"]
    n = len(corridas)

    fits = [c["mejor_fitness"] for c in corridas]
    gens = [c["generaciones"] for c in corridas]
    idx_mejor = int(np.argmax(fits))

    media, std, eje_gens = curvas_promedio(resultado_config)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.11,
                        wspace=0.28)

    # --- Convergencia ---
    _estilo_ax(ax1, f"Convergencia — {config.nombre}")
    cmap = plt.cm.get_cmap("cool", n)

    ax1.fill_between(eje_gens, media - std, media + std,
                     alpha=BAND_ALPHA, color=MEAN_COLOR, zorder=1)

    for i, c in enumerate(corridas):
        if i == idx_mejor:
            continue
        h = c["historial_mejor"]
        ax1.plot(range(1, len(h) + 1), h,
                 color=cmap(i), linewidth=0.8, alpha=0.45, zorder=2)

    mejor_c = corridas[idx_mejor]
    h_best  = mejor_c["historial_mejor"]
    ax1.plot(range(1, len(h_best) + 1), h_best,
             color=BEST_COLOR, linewidth=2.5, zorder=4,
             label=f"Mejor (seed={mejor_c['semilla']}, {mejor_c['mejor_fitness']:.4f} MW)")
    ax1.plot(eje_gens, media,
             color=MEAN_COLOR, linewidth=1.8, linestyle="--", alpha=0.85,
             zorder=3, label=f"Media ({n} corridas)")
    ax1.axhline(mejor_c["mejor_fitness"],
                color=BEST_COLOR, linestyle=":", linewidth=1.0, alpha=0.5)

    ax1.set_xlabel("Generación", fontsize=10, labelpad=5)
    ax1.set_ylabel("Fitness  (MW)", fontsize=10, labelpad=5)
    ax1.set_xlim(1, len(eje_gens))
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.3)

    # --- Boxplot de generaciones ---
    _estilo_ax(ax2, "Generaciones hasta Convergencia")

    bp = ax2.boxplot(
        gens, vert=True, patch_artist=True, widths=0.45,
        showmeans=True, showfliers=True,
        boxprops=dict(facecolor=PALETTE[0] + "44", edgecolor=PALETTE[0], linewidth=1.8),
        medianprops=dict(color=BEST_COLOR, linewidth=2.5),
        meanprops=dict(marker="D", markerfacecolor=MEAN_COLOR,
                       markeredgecolor="white", markersize=8),
        whiskerprops=dict(color=MUTED, linewidth=1.3, linestyle="--"),
        capprops=dict(color=MUTED, linewidth=1.8),
        flierprops=dict(marker="o", markerfacecolor="#f78166",
                        markeredgecolor="white", markersize=5, alpha=0.8),
    )

    jitter = np.random.uniform(-0.15, 0.15, size=n)
    idx_rapido = int(np.argmin(gens))
    colores_pts = [BEST_COLOR if i == idx_rapido else PALETTE[0] for i in range(n)]
    sizes_pts   = [110 if i == idx_rapido else 45 for i in range(n)]
    ax2.scatter(np.ones(n) + jitter, gens,
                color=colores_pts, s=sizes_pts,
                edgecolors="white", linewidths=0.6, zorder=5, alpha=0.85)

    ax2.set_ylabel("Generaciones", fontsize=10, labelpad=5)
    ax2.set_xticks([1])
    ax2.set_xticklabels([f"{n} corridas"], fontsize=9)
    ax2.set_xlim(0.5, 1.9)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.grid(False, axis="x")

    m = metricas_corridas(resultado_config)
    fig.suptitle(
        f"{config.resumen()}  |  "
        f"fit_media={m['fit_media']:.4f} ± {m['fit_std']:.4f} MW  "
        f"·  gen_media={m['gen_media']:.1f} ± {m['gen_std']:.1f}  "
        f"·  tasa_conv={m['tasa_convergencia']:.0%}",
        color=TEXT, fontsize=10, fontweight="bold",
    )

    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Panel guardado -> '{guardar}'")

    return fig
