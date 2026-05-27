# Tarea 1 — Juego de la Vida de Conway
**Susana Herrera | LEAD University | Programación Paralela y Distribuida**

---

## Estructura de la carpeta

```
GofL_SusanaHerrera/
│
├── game_of_life.py        # Clase GameOfLife (núcleo del algoritmo)
├── visualizer.py          # Generación de imágenes por patrones
├── benchmark.py           # Medición de rendimiento y gráficas
│
└── outputs/               # Imágenes generadas (se crean al correr)
    ├── patterns_showcase.png
    ├── glider_detail.png
    ├── random_evolution_64.png
    ├── benchmark_linear.png
    ├── benchmark_loglog.png
    └── benchmark_memory.png
```

---

## Requisitos

- Python 3.10 o superior
- Las siguientes librerías:

```bash
pip install numpy matplotlib
```

---

## Cómo ejecutar en Visual Studio Code

### Paso 1 — Abrir la carpeta
Abre VS Code y selecciona **File → Open Folder** y elige `GofL_SusanaHerrera`.

### Paso 2 — Instalar dependencias
Abre la terminal integrada con **Ctrl + `** (acento grave) y ejecuta:

```bash
pip install numpy matplotlib
```

### Paso 3 — Correr la simulación base
Para verificar que la clase funciona correctamente:

```bash
python game_of_life.py
```

Salida esperada:
```
=== Prueba básica ===
Inicial:  GameOfLife(32x32, gen=0, vivas=...)
Después de 10 gens: GameOfLife(32x32, gen=10, vivas=...)
Tiempo: ... ms

=== Prueba con Glider ===
Inicial: GameOfLife(20x20, gen=0, vivas=5)
Tras 4 gens: GameOfLife(20x20, gen=4, vivas=5)
OK - La clase funciona correctamente
```

### Paso 4 — Generar las visualizaciones
```bash
python visualizer.py
```

Crea 3 imágenes en la carpeta `outputs/`:
- `patterns_showcase.png` — los 5 patrones clásicos en 4 generaciones
- `glider_detail.png` — evolución del Glider en 6 momentos
- `random_evolution_64.png` — tablero aleatorio 64×64 estabilizándose

### Paso 5 — Correr el benchmark de rendimiento
```bash
python benchmark.py
```

Mide el tiempo por iteración para grillas de 32×32 hasta 1024×1024 y genera 3 gráficas en `outputs/`:
- `benchmark_linear.png` — tiempo vs celdas con curvas teóricas
- `benchmark_loglog.png` — escala log-log con pendiente empírica
- `benchmark_memory.png` — memoria estimada por tamaño

> ⚠️ El benchmark tarda aproximadamente 30–60 segundos por la grilla 1024×1024.

---

## Uso desde código (API de la clase)

```python
from game_of_life import GameOfLife, PATTERNS
import numpy as np

# Estado aleatorio
gol = GameOfLife(64, 64)
gol.run(10)
estado = gol.get_state()   # numpy array (64, 64)

# Estado inicial manual
tablero = np.zeros((20, 20), dtype=np.uint8)
gol2 = GameOfLife(20, 20, initial_state=tablero)
gol2.set_pattern(PATTERNS["glider"], row=2, col=2)
gol2.step()                # avanza 1 generación
print(gol2)                # GameOfLife(20x20, gen=1, vivas=5)
```

---

## Patrones disponibles

| Nombre    | Tipo               | Descripción                            |
|-----------|--------------------|----------------------------------------|
| `glider`  | Nave espacial      | Se desplaza diagonalmente              |
| `blinker` | Oscilador (periodo 2) | Alterna horizontal ↔ vertical       |
| `toad`    | Oscilador (periodo 2) | Forma cambiante cada generación     |
| `block`   | Estructura estática   | No cambia nunca                     |
| `beacon`  | Oscilador (periodo 2) | Parpadea en las esquinas            |

---

## Reglas de Conway implementadas

| Regla          | Condición                                      |
|----------------|------------------------------------------------|
| Superpoblación | Celda viva con más de 3 vecinos vivos → muere  |
| Soledad        | Celda viva con menos de 2 vecinos vivos → muere|
| Supervivencia  | Celda viva con 2 o 3 vecinos vivos → sobrevive |
| Reproducción   | Celda muerta con exactamente 3 vecinos → nace  |

---

## Resultados de rendimiento (resumen)

| Grilla     | ms/iteración | Memoria   |
|------------|-------------|-----------|
| 32×32      | ~0.15 ms    | ~0.002 MB |
| 256×256    | ~0.25 ms    | ~0.125 MB |
| 512×512    | ~0.68 ms    | ~0.500 MB |
| 1024×1024  | ~2.99 ms    | ~2.000 MB |

**Complejidad empírica:** pendiente log-log ≈ 0.41 → mejor que O(n) gracias al paralelismo SIMD interno de NumPy y al aprovechamiento de la caché del CPU.
