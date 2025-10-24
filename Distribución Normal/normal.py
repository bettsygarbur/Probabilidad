"""
EJERCICIO DE DISTRIBUCIÓN NORMAL

Enunciado:
La duración de vida de una batería (en horas) sigue una distribución normal
con una media (μ) de 400 horas y una desviación estándar (σ) de 10 horas.

Calcular:

1. La probabilidad de que una batería dure menos de 390 horas (P(X < 390)).
2. La probabilidad de que dure entre 395 y 410 horas (P(395 < X < 410)).
3. La probabilidad de que dure más de 420 horas (P(X > 420)).
4. Graficar la función de densidad con las áreas sombreadas correspondientes.

Fórmula teórica de la distribución normal:
    f(x) = (1 / (σ * sqrt(2π))) * e^(-0.5 * ((x - μ)/σ)^2)

Donde:
    μ = media
    σ = desviación estándar
    x = valor de la variable aleatoria
"""

# =======================
# 📚 Importar librerías
# =======================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =======================
# ⚙️ Parámetros del problema
# =======================
mu = 400   # media
sigma = 10 # desviación estándar

# =======================
# 🧮 Cálculos de probabilidades
# =======================
# 1. P(X < 390)
p_menor_390 = norm.cdf(390, mu, sigma)

# 2. P(395 < X < 410)
p_entre_395_410 = norm.cdf(410, mu, sigma) - norm.cdf(395, mu, sigma)

# 3. P(X > 420)
p_mayor_420 = 1 - norm.cdf(420, mu, sigma)

# =======================
# 🖨️ Mostrar resultados
# =======================
print(f"Media (μ): {mu}")
print(f"Desviación estándar (σ): {sigma}\n")

print(f"P(X < 390)       = {p_menor_390:.5f}")
print(f"P(395 < X < 410) = {p_entre_395_410:.5f}")
print(f"P(X > 420)       = {p_mayor_420:.5f}")

# =======================
# 📊 Graficar la distribución
# =======================
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)  # rango de valores
y = norm.pdf(x, mu, sigma)  # función de densidad

plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', label='Distribución Normal')

# ---- Sombrear áreas de interés ----

# 1️⃣ P(X < 390)
x_fill = np.linspace(mu - 4*sigma, 390, 200)
plt.fill_between(x_fill, norm.pdf(x_fill, mu, sigma), color='red', alpha=0.3, label='P(X < 390)')

# 2️⃣ P(395 < X < 410)
x_fill = np.linspace(395, 410, 200)
plt.fill_between(x_fill, norm.pdf(x_fill, mu, sigma), color='orange', alpha=0.3, label='P(395 < X < 410)')

# 3️⃣ P(X > 420)
x_fill = np.linspace(420, mu + 4*sigma, 200)
plt.fill_between(x_fill, norm.pdf(x_fill, mu, sigma), color='green', alpha=0.3, label='P(X > 420)')

# ---- Decoración del gráfico ----
plt.title("Distribución Normal de Duración de Baterías")
plt.xlabel("Duración (horas)")
plt.ylabel("Densidad de probabilidad")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
