"""
EJERCICIO DE DISTRIBUCIÓN DE POISSON

Enunciado:
Un centro de atención recibe en promedio 5 llamadas por hora.
Asumiendo que el número de llamadas sigue una distribución de Poisson,
calcular:

1. La probabilidad de que se reciban exactamente 7 llamadas en una hora.
2. La probabilidad acumulada de recibir hasta 7 llamadas (P(X ≤ 7)).
3. La probabilidad de recibir más de 7 llamadas (P(X > 7)).
4. Representar gráficamente la función de probabilidad y la función acumulada.

Fórmula teórica de la distribución de Poisson:
    P(X = k) = (e^(-λ) * λ^k) / k!

Donde:
    λ (lambda) = número promedio de eventos por intervalo (media)
    k = número de eventos observados
    e = número de Euler (≈ 2.71828)
"""

# =======================
# 📚 Importar librerías
# =======================
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# =======================
# ⚙️ Parámetros del problema
# =======================
lmbda = 5   # promedio (λ)
k = 7       # número de llamadas específicas

# =======================
# 🧮 Cálculos de probabilidades
# =======================
# 1. Probabilidad exacta (P(X = k))
p_exacta = poisson.pmf(k, lmbda)

# 2. Probabilidad acumulada (P(X ≤ k))
p_acumulada = poisson.cdf(k, lmbda)

# 3. Probabilidad complementaria (P(X > k))
p_mayor = 1 - p_acumulada

# =======================
# 🖨️ Mostrar resultados
# =======================
print(f"λ (promedio): {lmbda}")
print(f"k (valor específico): {k}")
print(f"P(X = {k})  = {p_exacta:.5f}")
print(f"P(X ≤ {k})  = {p_acumulada:.5f}")
print(f"P(X > {k})  = {p_mayor:.5f}")

# =======================
# 📊 Gráfica de la distribución
# =======================
# Rango de valores posibles (0 a 15 llamadas)
x = np.arange(0, 16)
y_pmf = poisson.pmf(x, lmbda)  # Distribución de probabilidad
y_cdf = poisson.cdf(x, lmbda)  # Distribución acumulada

# Crear figura con dos gráficos
plt.figure(figsize=(10, 5))

# ---- Gráfico 1: Función de probabilidad ----
plt.subplot(1, 2, 1)
plt.bar(x, y_pmf, color='skyblue', edgecolor='black')
plt.title('Distribución de Poisson (Función de Probabilidad)')
plt.xlabel('Número de llamadas (k)')
plt.ylabel('P(X = k)')
plt.grid(alpha=0.3)

# Marcar el valor k de interés
plt.axvline(k, color='red', linestyle='--', label=f'k = {k}')
plt.legend()

# ---- Gráfico 2: Función acumulada ----
plt.subplot(1, 2, 2)
plt.step(x, y_cdf, where='mid', color='green')
plt.title('Función Acumulada de Probabilidad (CDF)')
plt.xlabel('Número de llamadas (k)')
plt.ylabel('P(X ≤ k)')
plt.grid(alpha=0.3)
plt.axvline(k, color='red', linestyle='--', label=f'k = {k}')
plt.legend()

# Mostrar ambas gráficas
plt.tight_layout()
plt.show()
