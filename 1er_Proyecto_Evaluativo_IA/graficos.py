
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Tema oscuro global
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "text.color":        "#e6edf3",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "lines.linewidth":   2,
})

# Paleta de colores
C_BLUE   = "#58a6ff"   # curva principal
C_GREEN  = "#3fb950"   # optimo / mediana
C_RED    = "#f78166"   # mayor salto / outliers
C_YELLOW = "#e3b341"   # convergencia / media
C_GRAY   = "#30363d"   # caja del boxplot
C_LIGHT  = "#c9d1d9"   # texto secundario


# ---------------------------------------------------------------------------
# Calculo de metricas
# ---------------------------------------------------------------------------
def _metricas(h: np.ndarray, ventana: int = 50) -> dict:
    n = len(h)
    v = min(ventana, n)
    idx_conv  = int(np.argmax(h >= 0.99 * h[-1])) + 1
    idx_salto = int(np.argmax(np.diff(h)) + 2) if n > 1 else 1
    return {
        "n":          n,
        "v":          v,
        "h_ini":      h[0],
        "h_fin":      h[-1],
        "mejora":     h[-1] - h[0],
        "mejora_pct": (h[-1] - h[0]) / h[0] * 100 if h[0] != 0 else 0,
        "media":      float(np.mean(h[-v:])),
        "std":        float(np.std(h[-v:])),
        "minimo":     float(np.min(h)),
        "maximo":     float(np.max(h)),
        "mediana":    float(np.median(h)),
        "q1":         float(np.percentile(h, 25)),
        "q3":         float(np.percentile(h, 75)),
        "idx_conv":   idx_conv,
        "idx_salto":  idx_salto,
    }


# ---------------------------------------------------------------------------
# FIGURA 1 – Gráfica de Convergencia
# ---------------------------------------------------------------------------
def _fig_convergencia(h: np.ndarray, m: dict, guardar: bool, salida: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0d1117")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.11)

    generaciones = np.arange(1, m["n"] + 1)

    # Area rellena bajo la curva
    ax.fill_between(generaciones, h, h.min() * 0.9995,
                    alpha=0.12, color=C_BLUE)

    # Curva principal
    ax.plot(generaciones, h,
            color=C_BLUE, linewidth=2.5, zorder=3,
            label="Mejor fitness por generacion")

    # Suavizado (solo si hay generaciones suficientes para que tenga sentido)
    if m["n"] > 10:
        w = max(3, m["n"] // 10)
        suav = np.convolve(h, np.ones(w) / w, mode="same")
        ax.plot(generaciones, suav,
                color=C_YELLOW, linewidth=1.5, alpha=0.65,
                linestyle="-", label="Tendencia (media movil)")

    # Linea horizontal: fitness optimo final
    ax.axhline(m["h_fin"], color=C_GREEN, linestyle="-.",
               linewidth=1.3, alpha=0.8,
               label=f"Optimo final: {m['h_fin']:.4f} MW")

    # Linea vertical: generacion de mayor salto
    ax.axvline(m["idx_salto"], color=C_RED, linestyle="--",
               linewidth=1.5, alpha=0.85,
               label=f"Mayor salto (gen {m['idx_salto']})")

    # Linea vertical: convergencia 99%
    ax.axvline(m["idx_conv"], color=C_YELLOW, linestyle=":",
               linewidth=1.8, alpha=0.9,
               label=f"Convergencia 99 % (gen {m['idx_conv']})")

    # Punto destacado en el ultimo valor
    ax.scatter([generaciones[-1]], [h[-1]],
               s=100, color=C_GREEN, edgecolors="white",
               linewidths=1.2, zorder=5)

    # Anotacion del valor final
    ax.annotate(
        f" {h[-1]:.4f} MW",
        xy=(generaciones[-1], h[-1]),
        xytext=(generaciones[-1] - max(1, m["n"] * 0.15), h[-1]),
        color=C_GREEN, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.1),
        va="center",
    )

    ax.set_title("Grafica de Convergencia  –  Evolucion del Mejor Fitness por Generacion",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Generacion", fontsize=11, labelpad=6)
    ax.set_ylabel("Fitness  (MW)", fontsize=11, labelpad=6)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(1, m["n"])
    ax.legend(loc="lower right", fontsize=9, framealpha=0.3,
              ncol=1, borderpad=0.6)

    if guardar:
        fig.savefig(salida, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Figura 1 guardada -> '{salida}'")


# ---------------------------------------------------------------------------
# FIGURA 2 – Box Plot + Tabla estadistica
# ---------------------------------------------------------------------------
def _fig_estadisticas(h: np.ndarray, m: dict, guardar: bool, salida: str) -> None:
    fig = plt.figure(figsize=(14, 6), facecolor="#0d1117")
    fig.suptitle("Analisis Estadistico del Fitness  –  Algoritmo Genetico",
                 fontsize=14, fontweight="bold", color="#e6edf3", y=0.97)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.06, right=0.97,
        top=0.88, bottom=0.14,
        wspace=0.38,
        width_ratios=[1.2, 1],
    )

    ax_box   = fig.add_subplot(gs[0, 0])
    ax_tabla = fig.add_subplot(gs[0, 1])

    # ── Box Plot ──────────────────────────────────────────────────────────────
    ax_box.boxplot(
        h,
        vert=False,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="D", markerfacecolor=C_YELLOW,
                       markeredgecolor="white", markersize=8),
        flierprops=dict(marker="x", markeredgecolor=C_RED,
                        markersize=6, alpha=0.8),
        medianprops=dict(color=C_GREEN, linewidth=2.8),
        whiskerprops=dict(color=C_LIGHT, linewidth=1.5, linestyle="--"),
        capprops=dict(color=C_LIGHT, linewidth=2),
        boxprops=dict(facecolor=C_GRAY, color=C_BLUE, linewidth=2),
    )

    # Puntos individuales (strip) sobre el boxplot
    jitter = np.random.uniform(-0.18, 0.18, size=len(h))
    ax_box.scatter(h, np.ones(len(h)) + jitter,
                   alpha=0.4, color=C_BLUE, s=22, zorder=2)

    # Etiquetas de estadisticos clave — posicion escalonada para evitar solapamiento
    estadisticos = [
        (m["minimo"],  "Min",  f"{m['minimo']:.3f}",  C_RED,    0.44),
        (m["q1"],      "Q1",   f"{m['q1']:.3f}",      C_LIGHT,  0.34),
        (m["mediana"], "Med",  f"{m['mediana']:.3f}",  C_GREEN,  0.44),
        (m["q3"],      "Q3",   f"{m['q3']:.3f}",      C_LIGHT,  0.34),
        (m["maximo"],  "Max",  f"{m['maximo']:.3f}",   C_BLUE,   0.44),
    ]
    for val, nombre, numero, color, y_txt in estadisticos:
        ax_box.annotate(
            f"{nombre}\n{numero}",
            xy=(val, 0.75),          # punto en el borde inferior de la caja
            xytext=(val, y_txt),     # posicion del texto (alterno arriba/abajo)
            ha="center", va="top",
            fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9, alpha=0.55),
        )

    ax_box.set_title("Box Plot  –  Distribucion Estadistica",
                     fontsize=13, fontweight="bold", pad=10)
    ax_box.set_xlabel("Fitness  (MW)", fontsize=11, labelpad=6)
    ax_box.set_yticks([])
    ax_box.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax_box.set_ylim(0.25, 1.65)  # espacio extra para etiquetas

    # ── Tabla de metricas ─────────────────────────────────────────────────────
    ax_tabla.axis("off")

    if m["std"] < 0.01:
        interpretacion = "Alta estabilidad convergente"
    elif m["std"] < 0.05:
        interpretacion = "Convergencia moderada"
    else:
        interpretacion = "Alta dispersion"

    filas = [
        ("Fitness inicial (gen 1)",          f"{m['h_ini']:.4f} MW"),
        ("Fitness final   (gen " + str(m["n"]) + ")", f"{m['h_fin']:.4f} MW"),
        ("Mejora total",                     f"{m['mejora']:.4f} MW  ({m['mejora_pct']:.2f} %)"),
        ("",                                 ""),
        ("Media (ult. " + str(m["v"]) + " gen)", f"{m['media']:.4f} MW"),
        ("Desv. estandar",                   f"{m['std']:.4f} MW"),
        ("",                                 ""),
        ("Minimo",                           f"{m['minimo']:.4f} MW"),
        ("Mediana",                          f"{m['mediana']:.4f} MW"),
        ("Maximo",                           f"{m['maximo']:.4f} MW"),
        ("",                                 ""),
        ("Gen. de mayor salto",              str(m["idx_salto"])),
        ("Gen. de convergencia (~99 %)",     str(m["idx_conv"])),
        ("",                                 ""),
        ("Interpretacion",                   interpretacion),
    ]

    # Cabeceras
    ax_tabla.text(0.02, 0.98, "Metrica",
                  transform=ax_tabla.transAxes, fontsize=10,
                  fontweight="bold", color=C_BLUE, va="top")
    ax_tabla.text(0.72, 0.98, "Valor",
                  transform=ax_tabla.transAxes, fontsize=10,
                  fontweight="bold", color=C_BLUE, va="top", ha="center")

    # Linea separadora bajo cabecera
    ax_tabla.plot([0.0, 1.0], [0.95, 0.95],
                  transform=ax_tabla.transAxes,
                  color="#30363d", linewidth=1)

    row_h  = 0.054
    y_start = 0.93

    for i, (etiqueta, valor) in enumerate(filas):
        y = y_start - i * row_h

        # Fondo alternado (saltar filas vacias)
        if etiqueta:
            fondo = "#1c2128" if i % 2 == 0 else "#161b22"
            ax_tabla.add_patch(
                plt.Rectangle((-0.01, y - row_h * 0.35), 1.02, row_h * 0.95,
                               transform=ax_tabla.transAxes,
                               color=fondo, zorder=0, clip_on=False)
            )
            ax_tabla.text(0.02, y, etiqueta,
                          transform=ax_tabla.transAxes,
                          fontsize=9, color=C_LIGHT, va="center")
            ax_tabla.text(0.98, y, valor,
                          transform=ax_tabla.transAxes,
                          fontsize=9, color=C_GREEN, va="center",
                          ha="right", fontweight="bold")

    ax_tabla.set_title("Resumen Estadistico",
                       fontsize=13, fontweight="bold", pad=10)

    if guardar:
        fig.savefig(salida, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Figura 2 guardada -> '{salida}'")


# ---------------------------------------------------------------------------
# Funcion publica principal
# ---------------------------------------------------------------------------
def generar_graficos(
    historial: list | None = None,
    archivo_json: str = "historial_fitness.json",
    guardar: bool = True,
    salida_conv:  str = "fig1_convergencia.png",
    salida_stats: str = "fig2_estadisticas.png",
) -> None:
   
    # Cargar datos
    if historial is None:
        ruta = Path(archivo_json)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontro '{archivo_json}'.\n"
                "Corra main.py primero para generar el historial."
            )
        with open(ruta) as f:
            historial = json.load(f)

    h = np.array(historial, dtype=float)
    if len(h) == 0:
        raise ValueError("El historial de fitness esta vacio.")

    m = _metricas(h)

    _fig_convergencia(h, m, guardar, salida_conv)
    _fig_estadisticas(h, m, guardar, salida_stats)

    plt.show()


# ---------------------------------------------------------------------------
# Entry point standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generar_graficos()