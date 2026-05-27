"""
Visualización del Juego de la Vida de Conway
Genera imágenes PNG de cada patrón clásico en distintos momentos.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from game_of_life import GameOfLife, PATTERNS


# Colores: muerta=negro, viva=verde lima
CMAP = ListedColormap(["#0d0d0d", "#39ff14"])


def _draw_grid(ax, grid, title, generation):
    """Dibuja una sola grilla en un eje de matplotlib."""
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{title}\nGen {generation}", fontsize=9,
                 color="white", pad=4)
    ax.axis("off")


def visualize_pattern(pattern_name: str, steps_list: list, grid_size=20,
                      save_path=None):
    """
    Muestra la evolución de un patrón en varios momentos (snapshots).

    Args:
        pattern_name : nombre del patrón en PATTERNS
        steps_list   : lista de generaciones a capturar, ej [0, 4, 8, 16]
        grid_size    : tamaño de la grilla cuadrada
        save_path    : ruta para guardar la imagen (None = mostrar en pantalla)
    """
    pattern = PATTERNS[pattern_name]

    # Tablero limpio con el patrón centrado
    empty = np.zeros((grid_size, grid_size), dtype=np.uint8)
    gol = GameOfLife(grid_size, grid_size, initial_state=empty)
    r = grid_size // 2 - pattern.shape[0] // 2
    c = grid_size // 2 - pattern.shape[1] // 2
    gol.set_pattern(pattern, row=r, col=c)

    n = len(steps_list)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    fig.patch.set_facecolor("#1a1a2e")

    if n == 1:
        axes = [axes]

    current_gen = 0
    for ax, target_gen in zip(axes, steps_list):
        steps_needed = target_gen - current_gen
        if steps_needed > 0:
            gol.run(steps_needed)
            current_gen = target_gen
        _draw_grid(ax, gol.get_state(), pattern_name.capitalize(), current_gen)

    fig.suptitle(f"Patrón: {pattern_name.upper()}  ({grid_size}×{grid_size})",
                 fontsize=13, color="white", y=1.02, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Guardado: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_all_patterns(save_path="patterns_showcase.png"):
    """
    Panel resumen: todos los patrones clásicos lado a lado en 4 momentos.
    """
    patterns_cfg = {
        "glider":  [0, 4, 8, 16],
        "blinker": [0, 1, 2, 3],
        "toad":    [0, 1, 2, 3],
        "block":   [0, 5, 10, 20],
        "beacon":  [0, 1, 2, 3],
    }

    n_patterns = len(patterns_cfg)
    n_snaps = 4

    fig = plt.figure(figsize=(3 * n_snaps, 3.4 * n_patterns))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(n_patterns, n_snaps, figure=fig,
                           hspace=0.55, wspace=0.08)

    for row_idx, (pname, steps_list) in enumerate(patterns_cfg.items()):
        pattern = PATTERNS[pname]
        size = 20
        empty = np.zeros((size, size), dtype=np.uint8)
        gol = GameOfLife(size, size, initial_state=empty)
        r = size // 2 - pattern.shape[0] // 2
        c = size // 2 - pattern.shape[1] // 2
        gol.set_pattern(pattern, row=r, col=c)

        current_gen = 0
        for col_idx, target_gen in enumerate(steps_list):
            steps_needed = target_gen - current_gen
            if steps_needed > 0:
                gol.run(steps_needed)
                current_gen = target_gen

            ax = fig.add_subplot(gs[row_idx, col_idx])
            _draw_grid(ax, gol.get_state(),
                       pname.capitalize() if col_idx == 0 else "",
                       current_gen)

            # Etiqueta del patrón a la izquierda del primer snapshot
            if col_idx == 0:
                ax.set_ylabel(pname.upper(), color="white",
                              fontsize=11, fontweight="bold",
                              rotation=90, labelpad=6)
                ax.yaxis.set_label_position("left")

    fig.suptitle("Patrones Clásicos — Juego de la Vida de Conway",
                 fontsize=15, color="white", y=1.01, fontweight="bold")

    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Panel guardado: {save_path}")
    plt.close()


def visualize_random_evolution(size=64, total_steps=50,
                               snapshots=None,
                               save_path="random_evolution.png"):
    """
    Muestra la evolución de un tablero aleatorio en varios momentos.

    Args:
        size      : tamaño de la grilla (size x size)
        total_steps: generaciones totales a simular
        snapshots : lista de generaciones a capturar
        save_path : ruta de salida
    """
    if snapshots is None:
        snapshots = [0, 10, 20, 30, 40, 50]

    gol = GameOfLife(size, size)   # estado aleatorio por defecto

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    fig.patch.set_facecolor("#1a1a2e")

    current_gen = 0
    for ax, target_gen in zip(axes, snapshots):
        steps_needed = target_gen - current_gen
        if steps_needed > 0:
            gol.run(steps_needed)
            current_gen = target_gen
        _draw_grid(ax, gol.get_state(), f"{size}×{size}", current_gen)

    fig.suptitle(f"Evolución aleatoria — {size}×{size}",
                 fontsize=13, color="white", y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Guardado: {save_path}")
    plt.close()


# ── Ejecución directa ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generando visualizaciones...\n")

    # 1. Panel con todos los patrones clásicos
    visualize_all_patterns(save_path="outputs/patterns_showcase.png")

    # 2. Evolución aleatoria en 64x64
    visualize_random_evolution(
        size=64, snapshots=[0, 10, 25, 50],
        save_path="outputs/random_evolution_64.png"
    )

    # 3. Glider individual a mayor detalle
    visualize_pattern(
        "glider",
        steps_list=[0, 4, 8, 12, 16, 20],
        grid_size=30,
        save_path="outputs/glider_detail.png"
    )

    print("\nTodas las imágenes generadas correctamente.")
