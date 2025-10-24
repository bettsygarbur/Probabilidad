"""
PROBABILIDAD ACUMULATIVA (CDF) - Dos ejercicios

Ejercicio A (continua):
f(x,y) = (2/5)*(2x + 3y)   para 0 <= x <=1, 0 <= y <=1 (0 en otro caso)

Queremos F(a,b) = P(X <= a, Y <= b).
Analíticamente:
  F(a,b) = ∫_{x=0}^a ∫_{y=0}^b (2/5)(2x+3y) dy dx
         = (2/5) * [ b * a^2 + (3/2) * b^2 * a ]
         = (2/5) * b * a^2 + (3/5) * b^2 * a

Ejercicio B (discreta):
Pmf conjunta P(X=i, Y=j) calculada por combinaciones para seleccionar 2 de:
  sistemas = 3, electronica = 2, industrial = 3
X = # sistemas en la muestra de 2
Y = # electronica en la muestra de 2
CDF discreta: F(i,j) = sum_{u<=i, v<=j} P(X=u, Y=v)
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D plot)

# -----------------------
# EJERCICIO A: CONTINUA
# -----------------------
def F_continuo(a, b):
    """CDF analítica F(a,b) para 0<=a<=1, 0<=b<=1.
       Si a or b fuera del rango, se trunca.
    """
    # Truncar a [0,1]
    a_clamped = max(0.0, min(1.0, a))
    b_clamped = max(0.0, min(1.0, b))
    # Formula derivada:
    term1 = (2.0/5.0) * b_clamped * (a_clamped ** 2)      # (2/5) * b * a^2
    term2 = (3.0/5.0) * (b_clamped ** 2) * a_clamped    # (3/5) * b^2 * a
    return term1 + term2

# Pruebas puntuales
print("=== EJERCICIO A (CDF analítica) ===")
samples = [(0.5, 0.5), (1.0, 1.0), (0.25, 0.4), (0.0, 1.0)]
for a, b in samples:
    print(f"F({a:.2f}, {b:.2f}) = {F_continuo(a,b):.6f}")

# Visualizar la superficie de la CDF en la malla [0,1]^2
na = 60
nb = 60
a_vals = np.linspace(0, 1, na)
b_vals = np.linspace(0, 1, nb)
A, B = np.meshgrid(a_vals, b_vals)
Fvals = F_continuo(A, B)

fig = plt.figure(figsize=(10, 4))

# 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(A, B, Fvals, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.set_title('CDF continua F(a,b)')
ax1.set_xlabel('a (X ≤ a)')
ax1.set_ylabel('b (Y ≤ b)')
ax1.set_zlabel('F(a,b)')

# Contour
ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contourf(A, B, Fvals, levels=25, cmap='viridis')
plt.colorbar(cs, ax=ax2, label='F(a,b)')
ax2.set_title('Contornos de la CDF continua')
ax2.set_xlabel('a (X ≤ a)')
ax2.set_ylabel('b (Y ≤ b)')

plt.tight_layout()
plt.show()

# Ejemplo: calcular P(X <= 0.5, Y <= 0.4)
pa = 0.5; pb = 0.4
print(f"\nEjemplo: P(X <= {pa}, Y <= {pb}) = {F_continuo(pa, pb):.6f}")

# -----------------------
# EJERCICIO B: DISCRETA
# -----------------------
print("\n=== EJERCICIO B (CDF discreta) ===")
n_sis = 3
n_elec = 2
n_ind = 3
n_total = n_sis + n_elec + n_ind
sample = 2
total_ways = comb(n_total, sample)

# pmf conjunta (como antes)
pmf = {}
for i in range(0, sample+1):
    for j in range(0, sample+1-i):
        k = sample - i - j
        ways = 0
        if i <= n_sis and j <= n_elec and k <= n_ind:
            ways = comb(n_sis, i) * comb(n_elec, j) * comb(n_ind, k)
        pmf[(i, j)] = ways / total_ways

# Construir CDF discreta F(i,j) = sum_{u<=i, v<=j} pmf(u,v)
max_i = sample
max_j = sample
Fdisc = np.zeros((max_i+1, max_j+1))
for i in range(max_i+1):
    for j in range(max_j+1):
        s = 0.0
        for u in range(0, i+1):
            for v in range(0, j+1):
                s += pmf.get((u, v), 0.0)
        Fdisc[i, j] = s

# Mostrar tabla PMF y CDF
rows = []
for i in range(max_i+1):
    for j in range(max_j+1):
        rows.append({'i (X≤?)': i, 'j (Y≤?)': j, 'PMF sum upto (i,j) CDF': Fdisc[i,j], 'PMF at (i,j)': pmf.get((i,j), 0.0)})
df = pd.DataFrame(rows)
print(df.pivot(index='i (X≤?)', columns='j (Y≤?)', values='CDF').fillna(''))

# Mostrar matriz CDF y heatmap
print("\nMatriz CDF discreta (F(i,j)):")
print(Fdisc)

plt.figure(figsize=(6,5))
plt.imshow(Fdisc, origin='lower', interpolation='none', cmap='coolwarm', extent=[-0.5, max_j+0.5, -0.5, max_i+0.5])
plt.colorbar(label='F(i,j)')
plt.xticks(range(max_j+1))
plt.yticks(range(max_i+1))
plt.xlabel('j = Y (eléctrónica) ≤ j')
plt.ylabel('i = X (sistemas) ≤ i')
plt.title('CDF discreta F(i,j) = P(X ≤ i, Y ≤ j)')
# Anotar valores
for i in range(max_i+1):
