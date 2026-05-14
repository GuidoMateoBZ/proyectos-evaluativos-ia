import pygad
import numpy as np
import matplotlib.pyplot as plt
from fitness import convertir_cromosoma, calcular_fitness, contador_molinos


def _fitness_pygad(ga_instance, cromosoma, idx):
    return calcular_fitness(convertir_cromosoma(cromosoma))

def _construir_ga(on_generation_cb,poblacion):
    return pygad.GA(
        sol_per_pop          = poblacion,
        num_genes            = 50,
        gene_type            = int,
        gene_space           = range(0, 20),
        fitness_func         = _fitness_pygad,
        parent_selection_type= "tournament",
        K_tournament         = 3,
        keep_elitism         = 5,
        crossover_type       = "two_points",
        num_parents_mating   = 50,
        mutation_type        = "random",
        mutation_probability = 0.05,
        num_generations      = 500,
        stop_criteria        = ["reach_53.25", "saturate_50"],
        on_generation        = on_generation_cb,
    )


def experimentar(n_corridas: int = 10) -> list[dict]:

    resultados = []
    poblacion=0
    for i in range(1, n_corridas + 1):
        print(f"\n{'='*44}")
        print(f"  CORRIDA {i}/{n_corridas}")
        print(f"{'='*44}")

        hist_mejor    = []
        hist_promedio = []
        hist_std      = []
        poblacion+=50
        print(poblacion)
        def on_generation(ga):
            fits = ga.last_generation_fitness
            hist_mejor.append(float(np.max(fits)))
            hist_promedio.append(float(np.mean(fits)))
            hist_std.append(float(np.std(fits)))

        ga = _construir_ga(on_generation,poblacion)
        ga.run()

        solucion, fitness, _ = ga.best_solution()
        molinos = convertir_cromosoma(solucion)

        resultado = {
            "corrida"          : i,
            "mejor_fitness"    : round(float(fitness), 4),
            "generaciones"     : ga.generations_completed,
            "n_molinos"        : contador_molinos(molinos),
            "historial_mejor"  : hist_mejor,
            "historial_promedio": hist_promedio,
            "historial_std"    : hist_std,
            "poblacion":poblacion
        }
        resultados.append(resultado)

        print(f"  Fitness:     {fitness:.4f} MW")
        print(f"  Generaciones:{ga.generations_completed}")
        print(f"  Molinos:     {contador_molinos(molinos)}")

    return resultados


def graficar_experimento(resultados: list[dict], guardar: bool = True):
   
    n         = len(resultados)
    fs        = [r["mejor_fitness"] for r in resultados]
    idx_mejor = int(np.argmax(fs))
    mejor_r   = resultados[idx_mejor]

    BG        = "#0d1117"
    AX_BG     = "#161b22"
    C_GRID    = "#21262d"
    C_TEXT    = "#e6edf3"
    C_MUTED   = "#8b949e"
    C_BEST    = "#3fb950"   # corrida ganadora
    C_MEAN    = "#e3b341"   # media / banda
    C_BOX     = "#58a6ff"   # caja boxplot
    C_MED     = "#3fb950"   # mediana boxplot
    C_OUT     = "#f78166"   # outliers

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   AX_BG,
        "axes.edgecolor":   C_GRID,
        "axes.labelcolor":  C_TEXT,
        "axes.titlecolor":  C_TEXT,
        "xtick.color":      C_MUTED,
        "ytick.color":      C_MUTED,
        "text.color":       C_TEXT,
        "grid.color":       C_GRID,
        "legend.facecolor": AX_BG,
        "legend.edgecolor": C_GRID,
        "font.size":        10,
    })

    def _base_ax(ax, title):
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GRID)

    # Convergencia
    fig1, ax1 = plt.subplots(figsize=(13, 6), facecolor=BG)
    fig1.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.11)
    _base_ax(ax1, f"Curvas de Convergencia — {n} Corridas  (Mejor Fitness por Generación)")

    cmap = plt.cm.get_cmap("cool", n)

    # Calcular media de fitness por generación
    max_gen = max(len(r["historial_mejor"]) for r in resultados)
    mat = np.full((n, max_gen), np.nan)
    for i, r in enumerate(resultados):
        hist = r["historial_mejor"]
        mat[i, :len(hist)] = hist
    media_gen = np.nanmean(mat, axis=0)

    # Banda de dispersión (±1 std)
    std_gen = np.nanstd(mat, axis=0)
    gens_media = np.arange(1, max_gen + 1)
    ax1.fill_between(gens_media,
                     media_gen - std_gen, media_gen + std_gen,
                     alpha=0.15, color=C_MEAN, zorder=1)

    # Todas las corridas
    for i, r in enumerate(resultados):
        hist = r["historial_mejor"]
        gens = range(1, len(hist) + 1)
        if i == idx_mejor:
            continue  # la mejor la dibujamos al final para que quede encima
        ax1.plot(gens, hist,
                 color=cmap(i), linewidth=0.9, alpha=0.50, zorder=2)

    # Corrida ganadora
    hist_best = resultados[idx_mejor]["historial_mejor"]
    ax1.plot(range(1, len(hist_best) + 1), hist_best,
             color=C_BEST, linewidth=2.5, alpha=1.0, zorder=4,
             label=f"★ Corrida {mejor_r['corrida']}  ({mejor_r['mejor_fitness']:.4f} MW)")

    # Curva de media global
    ax1.plot(gens_media, media_gen,
             color=C_MEAN, linewidth=1.8, linestyle="--", alpha=0.85, zorder=3,
             label=f"Media ({n} corridas)")

    # Línea horizontal: mejor fitness obtenido
    ax1.axhline(mejor_r["mejor_fitness"],
                color=C_BEST, linestyle=":", linewidth=1.2, alpha=0.6)

    ax1.set_xlabel("Generación", fontsize=11, labelpad=6)
    ax1.set_ylabel("Fitness  (MW)", fontsize=11, labelpad=6)
    ax1.set_xlim(1, max_gen)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.3)

    fig1.suptitle(
        f"Experimento: {n} corridas independientes  ·  "
        f"Mejor: {mejor_r['mejor_fitness']:.4f} MW  (corrida {mejor_r['corrida']}, "
        f"gen {mejor_r['generaciones']})",
        color=C_TEXT, fontsize=11, fontweight="bold",
    )

    if guardar:
        fig1.savefig("exp_convergencia.png", dpi=150, bbox_inches="tight",
                     facecolor=fig1.get_facecolor())
        print("[OK] Figura 1 guardada -> 'exp_convergencia.png'")

    plt.show(block=True)
    plt.close(fig1)

    # Box Plot de generaciones hasta convergencia
    gs_vals = [r["generaciones"] for r in resultados]
    idx_mas_rapido = int(np.argmin(gs_vals))
    rapido_r = resultados[idx_mas_rapido]

    fig2, ax2 = plt.subplots(figsize=(9, 6), facecolor=BG)
    fig2.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
    _base_ax(ax2, f"Generaciones hasta Convergencia — {n} Corridas")

    bp = ax2.boxplot(
        gs_vals,
        vert=True,
        patch_artist=True,
        widths=0.45,
        showmeans=True,
        showfliers=True,
        boxprops=dict(facecolor=C_BOX + "55", color=C_BOX, linewidth=2),
        medianprops=dict(color=C_MED, linewidth=2.5),
        meanprops=dict(marker="D", markerfacecolor=C_MEAN,
                       markeredgecolor="white", markersize=8),
        whiskerprops=dict(color=C_MUTED, linewidth=1.5, linestyle="--"),
        capprops=dict(color=C_MUTED, linewidth=2),
        flierprops=dict(marker="o", markerfacecolor=C_OUT,
                        markeredgecolor="white", markersize=6, alpha=0.8),
    )

    # Strip de puntos individuales 
    jitter = np.random.uniform(-0.15, 0.15, size=n)
    colores_strip = [C_OUT if i == idx_mas_rapido else C_BOX for i in range(n)]
    sizes_strip   = [120   if i == idx_mas_rapido else 50    for i in range(n)]
    ax2.scatter(np.ones(n) + jitter, gs_vals,
                color=colores_strip, s=sizes_strip,
                edgecolors="white", linewidths=0.7,
                zorder=5, alpha=0.85)

    # Anotar la corrida más rápida
    ax2.annotate(
        f"  Corrida {rapido_r['corrida']}\n   {rapido_r['generaciones']} gen",
        xy=(1 + jitter[idx_mas_rapido], rapido_r["generaciones"]),
        xytext=(1.28, rapido_r["generaciones"]),
        color=C_OUT, fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.1),
        va="center",
    )

    # Estadísticos clave en el eje Y
    rng = max(gs_vals) - min(gs_vals) if max(gs_vals) != min(gs_vals) else 1
    estadisticos = [
        (np.min(gs_vals),                "Mín",   C_OUT),
        (np.percentile(gs_vals, 25),     "Q1",    C_MUTED),
        (float(np.median(gs_vals)),      "Med",   C_MED),
        (float(np.mean(gs_vals)),        "Media", C_MEAN),
        (np.percentile(gs_vals, 75),     "Q3",    C_MUTED),
        (np.max(gs_vals),                "Máx",   C_BEST),
    ]
    seen_y = []
    for val, lbl, color in estadisticos:
        y_txt = val
        for prev in seen_y:
            if abs(y_txt - prev) < rng * 0.04 + 1e-9:
                y_txt = prev + rng * 0.06
        seen_y.append(y_txt)
        ax2.axhline(val, color=color, linewidth=0.8, linestyle=":", alpha=0.5)
        ax2.text(1.52, val, f"{lbl}: {val:.1f}",
                 color=color, fontsize=8.5, va="center", fontweight="bold")

    ax2.set_ylabel("Generaciones", fontsize=11, labelpad=6)
    ax2.set_xticks([1])
    ax2.set_xticklabels([f"{n} corridas"], fontsize=10)
    ax2.set_xlim(0.55, 1.95)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.grid(False, axis="x")

    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leyenda = [
        Patch(facecolor=C_BOX + "55", edgecolor=C_BOX, label="Rango intercuartílico (IQR)"),
        Line2D([0], [0], color=C_MED, linewidth=2.5,
               label=f"Mediana: {np.median(gs_vals):.0f} gen"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_MEAN,
               markeredgecolor="white", markersize=8,
               label=f"Media: {np.mean(gs_vals):.1f} gen"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_OUT,
               markeredgecolor="white", markersize=8, label="Corrida más rápida"),
    ]
    ax2.legend(handles=leyenda, loc="upper right", fontsize=8.5,
               framealpha=0.3, borderpad=0.7)

    fig2.suptitle(
        f"Box Plot — Generaciones hasta convergencia ({n} corridas)  "
        f"·  std={np.std(gs_vals):.1f} gen",
        color=C_TEXT, fontsize=11, fontweight="bold",
    )

    if guardar:
        fig2.savefig("exp_boxplot.png", dpi=150, bbox_inches="tight",
                     facecolor=fig2.get_facecolor())
        print("[OK] Figura 2 guardada -> 'exp_boxplot.png'")

    plt.show(block=True)
    plt.close(fig2)




def imprimir_resumen(resultados: list[dict]):
    fs = [r["mejor_fitness"] for r in resultados]
    gs = [r["generaciones"]  for r in resultados]
    idx_mejor = int(np.argmax(fs))

    print("\n" + "="*52)
    print("RESUMEN DEL EXPERIMENTO")
    print("="*52)
    print(f"{'Corrida':>8} {'Fitness (MW)':>14} {'Generaciones':>13} {'Molinos':>8}")
    print("-"*52)
    for r in resultados:
        marca = " " if resultados.index(r) == idx_mejor else ""
        print(f"{r['corrida']:>8} {r['mejor_fitness']:>14.4f} "
              f"{r['generaciones']:>13} {r['n_molinos']:>8}{marca}")
    print("="*52)
    print(f"{'Promedio':>8} {np.mean(fs):>14.4f} {np.mean(gs):>13.1f}")
    print(f"{'Std':>8} {np.std(fs):>14.4f} {np.std(gs):>13.1f}")
    print(f"{'Máximo':>8} {np.max(fs):>14.4f} {np.max(gs):>13}")
    print(f"{'Mínimo':>8} {np.min(fs):>14.4f} {np.min(gs):>13}")
    print("="*52)