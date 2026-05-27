"""
Medición de rendimiento empírico — Juego de la Vida de Conway
Tarea 1 - Programación Paralela

Mide el tiempo promedio por iteración para distintos tamaños de grilla
y genera gráficas de complejidad (lineal y log-log).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from game_of_life import GameOfLife


# ── Configuración del experimento ────────────────────────────────────────────

GRID_SIZES   = [32, 64, 128, 256, 512, 1024]   # lados de la grilla
WARMUP_STEPS = 5      # pasos de calentamiento (no se miden)
MEASURE_STEPS = 20    # pasos que sí se miden para promediar
REPETITIONS  = 3      # repetir todo el experimento N veces y tomar la mediana


# ── Función de benchmarking ──────────────────────────────────────────────────

def benchmark_size(size: int) -> dict:
    """
    Mide el tiempo promedio por iteración para una grilla (size x size).

    Hace REPETITIONS experimentos completos y toma la mediana
    para reducir el ruido del sistema operativo.

    Returns:
        dict con: size, n_cells, avg_ms, std_ms, memory_mb
    """
    n_cells = size * size
    times_per_iter = []

    for _ in range(REPETITIONS):
        gol = GameOfLife(size, size)

        # Calentamiento: dejar que NumPy compile/cachee sus rutas internas
        gol.run(WARMUP_STEPS)

        # Medición real
        start = time.perf_counter()
        gol.run(MEASURE_STEPS)
        elapsed = time.perf_counter() - start

        times_per_iter.append((elapsed / MEASURE_STEPS) * 1000)  # ms por iter

    avg_ms = float(np.median(times_per_iter))
    std_ms = float(np.std(times_per_iter))

    # Memoria: la grilla es uint8 (1 byte por celda)
    # más el array de vecinos temporal (también NxN uint8)
    memory_mb = (n_cells * 2) / (1024 ** 2)

    return {
        "size":      size,
        "n_cells":   n_cells,
        "avg_ms":    avg_ms,
        "std_ms":    std_ms,
        "memory_mb": memory_mb,
    }


def run_benchmark() -> list[dict]:
    """Ejecuta el benchmark para todos los tamaños definidos."""
    results = []
    print(f"{'Grilla':>10}  {'Celdas':>10}  {'ms/iter':>10}  "
          f"{'std (ms)':>10}  {'Memoria':>10}")
    print("-" * 58)

    for size in GRID_SIZES:
        r = benchmark_size(size)
        results.append(r)
        print(f"{size:>4}x{size:<4}  {r['n_cells']:>10,}  "
              f"{r['avg_ms']:>10.4f}  {r['std_ms']:>10.4f}  "
              f"{r['memory_mb']:>8.3f} MB")

    return results


# ── Curvas teóricas de referencia ────────────────────────────────────────────

def theoretical_curves(n_values: np.ndarray, measured: list[dict]) -> dict:
    """
    Genera curvas O(n), O(n log n), O(n²) escaladas para que pasen
    por el primer punto medido (así son comparables visualmente).
    """
    n0  = measured[0]["n_cells"]
    t0  = measured[0]["avg_ms"]

    curves = {
        "O(n)":       t0 * (n_values / n0),
        "O(n log n)": t0 * (n_values * np.log2(n_values)) /
                           (n0 * np.log2(n0)),
        "O(n²)":      t0 * (n_values / n0) ** 2,
    }
    return curves


# ── Gráficas ─────────────────────────────────────────────────────────────────

DARK_BG  = "#1a1a2e"
GRID_COL = "#2e2e4e"
COLORS   = {
    "data":       "#39ff14",   # verde neón (datos reales)
    "O(n)":       "#ff6b6b",   # rojo suave
    "O(n log n)": "#ffd93d",   # amarillo
    "O(n²)":      "#6bcbff",   # azul claro
}


def _style_ax(ax, title, xlabel, ylabel):
    """Aplica el estilo oscuro a un eje."""
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=10)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, linestyle="--", linewidth=0.6)


def plot_linear(results: list[dict], save_path: str):
    """
    Gráfica lineal: tiempo (ms/iter) vs número de celdas.
    Incluye barras de error (desviación estándar) y curvas teóricas.
    """
    n   = np.array([r["n_cells"] for r in results])
    t   = np.array([r["avg_ms"]  for r in results])
    err = np.array([r["std_ms"]  for r in results])

    n_smooth = np.linspace(n[0], n[-1], 400)
    curves   = theoretical_curves(n_smooth, results)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "Tiempo por iteración vs Tamaño de entrada",
              "Número de celdas  (filas × columnas)",
              "Tiempo promedio por iteración (ms)")

    # Curvas teóricas
    for name, vals in curves.items():
        ax.plot(n_smooth, vals, "--", color=COLORS[name],
                linewidth=1.4, alpha=0.75, label=name)

    # Datos reales con barras de error
    ax.errorbar(n, t, yerr=err, fmt="o-",
                color=COLORS["data"], linewidth=2, markersize=7,
                capsize=4, capthick=1.5, elinewidth=1.2,
                label="Medido", zorder=5)

    # Etiquetas de tamaño en cada punto
    labels = [f"{r['size']}×{r['size']}" for r in results]
    for xi, yi, lbl in zip(n, t, labels):
        ax.annotate(lbl, (xi, yi),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="white")

    ax.legend(facecolor="#2e2e4e", edgecolor=GRID_COL,
              labelcolor="white", fontsize=9)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Guardado: {save_path}")
    plt.close()


def plot_loglog(results: list[dict], save_path: str):
    """
    Gráfica log-log: permite ver la pendiente real de complejidad.
    En escala log-log, O(n^k) aparece como una línea recta de pendiente k.
    """
    n   = np.array([r["n_cells"] for r in results], dtype=float)
    t   = np.array([r["avg_ms"]  for r in results], dtype=float)
    err = np.array([r["std_ms"]  for r in results], dtype=float)

    n_smooth = np.logspace(np.log10(n[0]), np.log10(n[-1]), 300)
    curves   = theoretical_curves(n_smooth, results)

    # Ajuste de regresión log-log para estimar la pendiente real
    log_n = np.log10(n)
    log_t = np.log10(t)
    slope, intercept = np.polyfit(log_n, log_t, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax,
              f"Escala Log-Log  (pendiente empírica ≈ {slope:.2f})",
              "Número de celdas  (log)",
              "Tiempo por iteración en ms  (log)")

    # Curvas teóricas
    for name, vals in curves.items():
        ax.loglog(n_smooth, vals, "--", color=COLORS[name],
                  linewidth=1.4, alpha=0.75, label=name)

    # Datos reales
    ax.errorbar(n, t, yerr=err, fmt="o",
                color=COLORS["data"], markersize=8,
                capsize=4, capthick=1.5, elinewidth=1.2,
                label=f"Medido  (slope≈{slope:.2f})", zorder=5)

    # Línea de regresión
    t_fit = 10 ** (intercept + slope * np.log10(n_smooth))
    ax.loglog(n_smooth, t_fit, "-", color=COLORS["data"],
              linewidth=1.5, alpha=0.5, label="_nolegend_")

    # Etiquetas en cada punto
    labels = [f"{r['size']}²" for r in results]
    for xi, yi, lbl in zip(n, t, labels):
        ax.annotate(lbl, (xi, yi),
                    textcoords="offset points", xytext=(6, 2),
                    fontsize=8, color="white")

    ax.legend(facecolor="#2e2e4e", edgecolor=GRID_COL,
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Guardado: {save_path}")
    plt.close()


def plot_memory(results: list[dict], save_path: str):
    """
    Gráfica de uso de memoria estimado vs tamaño de grilla.
    """
    sizes = [r["size"]      for r in results]
    mbs   = [r["memory_mb"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    _style_ax(ax, "Uso de memoria estimado vs Tamaño de grilla",
              "Lado de la grilla (N)", "Memoria (MB)")

    ax.bar(range(len(sizes)), mbs, color="#a78bfa", edgecolor=GRID_COL,
           linewidth=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}×{s}" for s in sizes], color="#aaaaaa")

    for i, (mb, r) in enumerate(zip(mbs, results)):
        ax.text(i, mb + 0.02, f"{mb:.2f} MB\n({r['n_cells']:,} celdas)",
                ha="center", va="bottom", fontsize=8, color="white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Guardado: {save_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  BENCHMARK — Juego de la Vida de Conway")
    print(f"  {MEASURE_STEPS} pasos medidos · {REPETITIONS} repeticiones · mediana")
    print("=" * 58 + "\n")

    results = run_benchmark()

    print("\nGenerando gráficas...")
    plot_linear(results, "outputs/benchmark_linear.png")
    plot_loglog(results, "outputs/benchmark_loglog.png")
    plot_memory(results, "outputs/benchmark_memory.png")

    # Resumen de la pendiente empírica
    n = np.array([r["n_cells"] for r in results], dtype=float)
    t = np.array([r["avg_ms"]  for r in results], dtype=float)
    slope, _ = np.polyfit(np.log10(n), np.log10(t), 1)

    print(f"\n{'='*58}")
    print(f"  Pendiente empírica log-log: {slope:.3f}")
    if slope < 1.15:
        verdict = "≈ O(n)  — escala linealmente ✓"
    elif slope < 1.6:
        verdict = "≈ O(n log n)"
    else:
        verdict = "≈ O(n²)  — escala cuadráticamente"
    print(f"  Complejidad estimada: {verdict}")
    print("=" * 58)
