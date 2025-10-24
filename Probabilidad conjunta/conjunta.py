"""
EJERCICIO A — DISTRIBUCIÓN CONJUNTA CONTINUA

Sea la función de densidad conjunta:

    f(x,y) = (2/5) * (2x + 3y),   para 0 <= x <= 1 y 0 <= y <= 1
             0,                   en otro caso.

1. Verificar que f(x,y) es una función de densidad de probabilidad (∬f(x,y)dxdy = 1)
2. Calcular las marginales f_X(x) y f_Y(y)
3. Calcular la probabilidad P(0.2 < X < 0.6, 0.3 < Y < 0.8)
4. Graficar f(x,y), f_X(x), f_Y(y)

EJERCICIO B — DISTRIBUCIÓN CONJUNTA DISCRETA

Un grupo de 8 estudiantes tiene:
  - 3 de sistemas
  - 2 de electrónica
  - 3 de industrial

Se eligen 2 estudiantes al azar sin reemplazo.
Defina:
    X = número de estudiantes de sistemas seleccionados
    Y = número de estudiantes de electrónica seleccionados

1. Hallar la función de probabilidad conjunta P(X=i, Y=j)
2. Calcular las marginales P_X(i) y P_Y(j)
3. Graficar la tabla conjunta en un heatmap.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy import integrate

# ==============================================
#   EJERCICIO A — DISTRIBUCIÓN CONJUNTA CONTINUA
# ==============================================

def f_xy(x, y):
    """Función de densidad conjunta."""
    if 0 <= x <= 1 and 0 <= y <= 1:
        return (2/5) * (2*x + 3*y)
    return 0

# 1️⃣ Verificar que es válida
integral_total, _ = integrate.dblquad(lambda yy, xx: (2/5)*(2*xx + 3*yy), 0, 1, lambda x: 0, lambda x: 1)
print("=== EJERCICIO A (CONTINUA) ===")
print(f"Integral total de f(x,y) = {integral_total:.6f} ✅ (debe ser 1.0)")

# 2️⃣ Marginales
# f_X(x) = ∫ f(x,y) dy de 0 a 1
# f_Y(y) = ∫ f(x,y) dx de 0 a 1
fX = lambda x: integrate.quad(lambda y: f_xy(x, y), 0, 1)[0]
fY = lambda y: integrate.quad(lambda x: f_xy(x, y), 0, 1)[0]

# Evaluar marginales en puntos
x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(0, 1, 100)
fX_vals = np.array([fX(x) for x in x_vals])
fY_vals = np.array([fY(y) for y in y_vals])

print("\nMarginal f_X(x) y f_Y(y) calculadas correctamente.")

# 3️⃣ Calcular una probabilidad en una región
p_region, _ = integrate.dblquad(lambda yy, xx: (2/5)*(2*xx + 3*yy),
                                0.2, 0.6, lambda x: 0.3, lambda x: 0.8)
print(f"\nP(0.2 < X < 0.6, 0.3 < Y < 0.8) = {p_region:.6f}")

# 4️⃣ Gráficas
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
X, Y = np.meshgrid(x_vals, y_vals)
Z = (2/5) * (2*X + 3*Y)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title("Densidad conjunta f(x,y)")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("f(x,y)")

ax2 = fig.add_subplot(1,2,2)
ax2.plot(x_vals, fX_vals, label="f_X(x)", color='blue')
ax2.plot(y_vals, fY_vals, label="f_Y(y)", color='orange')
ax2.set_title("Func. marginales f_X(x) y f_Y(y)")
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================
#   EJERCICIO B — DISTRIBUCIÓN CONJUNTA DISCRETA
# ==============================================
print("\n=== EJERCICIO B (DISCRETA) ===")
n_sis, n_elec, n_ind = 3, 2, 3
n_total = n_sis + n_elec + n_ind
n_sample = 2
total_ways = comb(n_total, n_sample)

# P(X=i, Y=j)
pmf = {}
for i in range(0, n_sample+1):
    for j in range(0, n_sample+1-i):
        k = n_sample - i - j
        if i <= n_sis and j <= n_elec and k <= n_ind:
            ways = comb(n_sis, i) * comb(n_elec, j) * comb(n_ind, k)
            pmf[(i,j)] = ways / total_ways

# Mostrar tabla
print("P(X=i, Y=j):")
for (i,j), p in sorted(pmf.items()):
    print(f"X={i}, Y={j} -> P={p:.5f}")

# 1️⃣ Verificar suma = 1
print(f"Suma total = {sum(pmf.values()):.5f} ✅")

# 2️⃣ Marginales
P_X = {}
P_Y = {}
for (i,j), p in pmf.items():
    P_X[i] = P_X.get(i,0) + p
    P_Y[j] = P_Y.get(j,0) + p

print("\nMarginal P_X(i):", P_X)
print("Marginal P_Y(j):", P_Y)

# 3️⃣ Heatmap de la probabilidad conjunta
max_i = 2
max_j = 2
mat = np.zeros((max_i+1, max_j+1))
for (i,j), p in pmf.items():
    mat[i,j] = p

plt.figure(figsize=(6,5))
plt.imshow(mat, origin='lower', cmap='plasma', interpolation='none', extent=[-0.5,2.5,-0.5,2.5])
plt.colorbar(label='P(X=i, Y=j)')
plt.xticks([0,1,2]); plt.yticks([0,1,2])
plt.xlabel("Y = electrónica")
plt.ylabel("X = sistemas")
plt.title("Distribución conjunta discreta P(X=i,Y=j)")

# Anotar valores
for i in range(max_i+1):
    for j in range(max_j+1):
        plt.text(j, i, f"{mat[i,j]:.3f}", ha='center', va='center', color='white', fontsize=12)
plt.show()
