"""
Juego de la Vida de Conway
Tarea 1 - Programación Paralela
Autor: Susana Herrera
"""

import numpy as np
import time


class GameOfLife:
    """
    Implementación del Juego de la Vida de Conway usando NumPy.

    El tablero es una matriz 2D de enteros donde:
        1 = celda viva
        0 = celda muerta

    Los bordes son "envolventes" (toroidal): el lado derecho conecta
    con el izquierdo, el superior con el inferior.
    """

    def __init__(self, rows: int, cols: int, initial_state=None):
        """
        Inicializa el tablero.

        Args:
            rows: número de filas
            cols: número de columnas
            initial_state: matriz numpy (rows x cols) con 0s y 1s.
                           Si es None, se genera un estado aleatorio.
        """
        self.rows = rows
        self.cols = cols
        self.generation = 0

        if initial_state is not None:
            assert initial_state.shape == (rows, cols), \
                f"initial_state debe tener forma ({rows}, {cols})"
            self.grid = initial_state.astype(np.uint8)
        else:
            # Estado aleatorio: ~30% de celdas vivas
            self.grid = np.random.choice(
                [0, 1], size=(rows, cols), p=[0.7, 0.3]
            ).astype(np.uint8)

    def _count_neighbors(self) -> np.ndarray:
        """
        Cuenta los vecinos vivos de cada celda usando np.roll.

        np.roll desplaza toda la grilla en una dirección (como si fuera
        un cilindro toroidal), permitiendo sumar los 8 vecinos sin loops.

        Returns:
            Matriz (rows x cols) con el conteo de vecinos vivos por celda.
        """
        g = self.grid
        neighbors = (
            np.roll(g, -1, axis=0) +           # arriba
            np.roll(g,  1, axis=0) +           # abajo
            np.roll(g, -1, axis=1) +           # izquierda
            np.roll(g,  1, axis=1) +           # derecha
            np.roll(np.roll(g, -1, axis=0), -1, axis=1) +  # diagonal ↖
            np.roll(np.roll(g, -1, axis=0),  1, axis=1) +  # diagonal ↗
            np.roll(np.roll(g,  1, axis=0), -1, axis=1) +  # diagonal ↙
            np.roll(np.roll(g,  1, axis=0),  1, axis=1)    # diagonal ↘
        )
        return neighbors

    def step(self):
        """
        Avanza UNA generación aplicando las 4 reglas de Conway.

        Reglas:
            1. Superpoblación: viva con >3 vecinos → muere
            2. Soledad:        viva con <2 vecinos → muere
            3. Supervivencia:  viva con 2 o 3 vecinos → vive
            4. Reproducción:   muerta con exactamente 3 vecinos → vive
        """
        neighbors = self._count_neighbors()

        # Aplicamos todas las reglas de una vez con operaciones booleanas
        # (mucho más rápido que un loop celda por celda)
        new_grid = (
            # Celdas que sobreviven (vivas con 2 o 3 vecinos)
            (self.grid == 1) & ((neighbors == 2) | (neighbors == 3)) |
            # Celdas que nacen (muertas con exactamente 3 vecinos)
            (self.grid == 0) & (neighbors == 3)
        ).astype(np.uint8)

        self.grid = new_grid
        self.generation += 1

    def run(self, steps: int) -> float:
        """
        Ejecuta múltiples generaciones y mide el tiempo total.

        Args:
            steps: número de generaciones a simular

        Returns:
            Tiempo total de ejecución en segundos.
        """
        start = time.perf_counter()
        for _ in range(steps):
            self.step()
        elapsed = time.perf_counter() - start
        return elapsed

    def get_state(self) -> np.ndarray:
        """
        Devuelve una copia del estado actual del tablero.

        Returns:
            Matriz numpy (rows x cols) con el estado actual.
        """
        return self.grid.copy()

    def set_pattern(self, pattern: np.ndarray, row: int, col: int):
        """
        Inserta un patrón en una posición específica del tablero.

        Args:
            pattern: matriz pequeña con el patrón (ej: Glider)
            row: fila donde colocar la esquina superior izquierda
            col: columna donde colocar la esquina superior izquierda
        """
        pr, pc = pattern.shape
        self.grid[row:row+pr, col:col+pc] = pattern

    def reset(self, initial_state=None):
        """Reinicia el tablero a un estado inicial."""
        self.generation = 0
        if initial_state is not None:
            self.grid = initial_state.astype(np.uint8)
        else:
            self.grid = np.random.choice(
                [0, 1], size=(self.rows, self.cols), p=[0.7, 0.3]
            ).astype(np.uint8)

    def __repr__(self):
        alive = int(self.grid.sum())
        return (f"GameOfLife({self.rows}x{self.cols}, "
                f"gen={self.generation}, vivas={alive})")


# ── Patrones clásicos ────────────────────────────────────────────────────────

PATTERNS = {
    "glider": np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8),

    "blinker": np.array([
        [1, 1, 1]
    ], dtype=np.uint8),

    "toad": np.array([
        [0, 1, 1, 1],
        [1, 1, 1, 0]
    ], dtype=np.uint8),

    "block": np.array([
        [1, 1],
        [1, 1]
    ], dtype=np.uint8),

    "beacon": np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ], dtype=np.uint8),
}


# ── Prueba rápida ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Prueba básica ===")

    # 1. Estado aleatorio
    gol = GameOfLife(32, 32)
    print(f"Inicial:  {gol}")
    elapsed = gol.run(10)
    print(f"Después de 10 gens: {gol}")
    print(f"Tiempo: {elapsed*1000:.3f} ms")

    # 2. Con el patrón Glider en tablero limpio
    print("\n=== Prueba con Glider ===")
    empty = np.zeros((20, 20), dtype=np.uint8)
    gol2 = GameOfLife(20, 20, initial_state=empty)
    gol2.set_pattern(PATTERNS["glider"], row=1, col=1)
    print(f"Inicial: {gol2}")
    gol2.run(4)
    print(f"Tras 4 gens: {gol2}")
    print("OK - La clase funciona correctamente")
