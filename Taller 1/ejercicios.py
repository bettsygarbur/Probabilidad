"""
EJERCICIO A — DENSIDAD CONJUNTA CONTINUA

Enunciado (según la foto):
La función de densidad conjunta f(x,y) viene dada por:
    f(x,y) = (2/5) * (2x + 3y)    para 0 <= x <= 1 y 0 <= y <= 1
            0                     en otro caso

1) Verificar que f(x,y) es una f.d.p. (i.e. >=0 y la integral en el soporte = 1).
2) Calcular P( (X,Y) en R ) donde R = { 0 < x < 1/2; 1/4 < y < 1/2 } (según lo que se entiende de la foto).
3) Graficar la densidad y sombrear la región R.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from matplotlib.patches import Rectangle
import pandas as pd

# Parámetros
def f_xy(x, y):
    # f(x,y) = (2/5)*(2x + 3y) dentro del cuadrado [0,1]x[0,1]
    x = np.asarray(x)
    y = np.asarray(y)
    val = (2.0/5.0) * (2.0*x + 3.0*y)
    # fuera del soporte -> 0 (pero las integradoras solo llamarán dentro del soporte)
    return val

# 1) Verificación: integral sobre [0,1]x[0,1]
# Usamos integración doble con scipy
integral_total, err_total = integrate.dblquad(
    lambda yy, xx: (2.0/5.0)*(2.0*xx + 3.0*yy),  # integrand: note order yy, xx esperada por dblquad
    0.0, 1.0,  # limites en x
    lambda x: 0.0, lambda x: 1.0  # limites en y (constantes)
)
print("=== EJERCICIO A ===")
print(f"Integral total de f(x,y) sobre [0,1]^2 = {integral_total:.8f} (error estimado {err_total:.1e})")
# deberia aproximarse a 1.0

# 2) Probabilidad en la región R: 0 < x < 1/2, 1/4 < y < 1/2
x0, x1 = 0.0, 0.5
y0, y1 = 0.25, 0.5

prob_R, err_R = integrate.dblquad(
    lambda yy, xx: (2.0/5.0)*(2.0*xx + 3.0*yy),
    x0, x1,
    lambda x: y0, lambda x: y1
)
print(f"P( 0 < X < 1/2 , 1/4 < Y < 1/2 ) = {prob_R:.8f} (err {err_R:.1e})")

# 3) Gráfica de la densidad y sombreado de la región R
# Malla para graficar
nx, ny = 120, 120
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
Z = f_xy(X, Y)

plt.figure(figsize=(8, 6))
# mapa de calor
plt.contourf(X, Y, Z, levels=30, cmap='viridis')
plt.colorbar(label='f(x,y)')
plt.title('Densidad conjunta f(x,y) = (2/5)(2x+3y) en [0,1]^2')
plt.xlabel('x')
plt.ylabel('y')

# sombrear región R
rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.2,
                 edgecolor='red', facecolor='none', linestyle='--', label='Región R')
plt.gca().add_patch(rect)
# relleno semitransparente sobre la region R usando patch
plt.fill_between([x0, x1, x1, x0], [y0, y0, y1, y1], alpha=0.15, color='red', edgecolor=None)

plt.legend()
plt.show()


# ---------------------------
# EJERCICIO B — SELECCIÓN DISCRETA
# ---------------------------
"""
Enunciado (según la foto):
Se seleccionan al azar 2 estudiantes de un salón que contiene:
 - 3 estudiantes de Sistemas
 - 2 estudiantes de Electrónica
 - 3 estudiantes de Industrial

Sea X = número de estudiantes de Sistemas en la muestra de 2 (sin orden).
Sea Y = número de estudiantes de Electrónica en la muestra de 2.

1) Hallar la función de probabilidad conjunta P(X = i, Y = j) para todos los i, j posibles.
"""

from math import comb

print("\n=== EJERCICIO B ===")
# datos
n_sis = 3
n_elec = 2
n_ind = 3
n_total = n_sis + n_elec + n_ind
n_choose = 2
total_ways = comb(n_total, n_choose)

# generamos la pmf P(X=i, Y=j) para i,j >=0 y i+j <= 2
pmf = {}
for i in range(0, n_choose+1):           # i = sistemas en la muestra
    for j in range(0, n_choose+1-i):     # j = electronica; i+j<=2
        k = n_choose - i - j            # k = industriales
        # Ways: elegir i de sistemas, j de electronica, k de industrial
        ways = 0
        if i <= n_sis and j <= n_elec and k <= n_ind:
            ways = comb(n_sis, i) * comb(n_elec, j) * comb(n_ind, k)
        p = ways / total_ways
        pmf[(i, j)] = p

# Mostrar la tabla de probabilidades (ordenada)
df_rows = []
for (i, j), p in sorted(pmf.items()):
    df_rows.append({'X (sistemas)': i, 'Y (electrónica)': j, 'P(X=i, Y=j)': p})
df = pd.DataFrame(df_rows)
print("Tabla PMF P(X=i, Y=j):")
print(df.to_string(index=False))

# Verificar suma = 1
sum_p = sum(pmf.values())
print(f"Suma de todas las probabilidades = {sum_p:.8f} (debe ser 1.0)")

# Mostrar matriz/heatmap de PMF para visualización
# Construimos matriz 3x3 (pero solo i+j<=2 tienen probabilidad)
max_i = 2
max_j = 2
mat = np.zeros((max_i+1, max_j+1))
for (i, j), p in pmf.items():
    mat[i, j] = p

plt.figure(figsize=(6,5))
plt.imshow(mat, origin='lower', interpolation='none', cmap='plasma', extent=[-0.5, 2.5, -0.5, 2.5])
plt.colorbar(label='P(X=i, Y=j)')
plt.xticks([0,1,2])
plt.yticks([0,1,2])
plt.xlabel('Y = # Electrónica')
plt.ylabel('X = # Sistemas')
plt.title('PMF conjunta P(X=i, Y=j) (selección 2 estudiantes)')
# Anotar probabilidades en cada celda
for i in range(max_i+1):
    for j in range(max_j+1):
        plt.text(j, i, f"{mat[i,j]:.3f}", ha='center', va='center', color='white', fontsize=12)

plt.show()

# Información adicional: listar casos con sus combinaciones
print("\nCasos con probabilidad positiva y su combinatoria (numerador/28):")
for (i, j), p in sorted(pmf.items()):
    if p > 0:
        k = n_choose - i - j
        ways = comb(n_sis, i) * comb(n_elec, j) * comb(n_ind, k)
        print(f"X={i}, Y={j} -> formas = {ways}  -> P = {ways}/{total_ways} = {p:.5f}")
