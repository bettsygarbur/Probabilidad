import numpy as np
from scipy import stats

# Parámetros del problema
media_muestral = 328.5
desviacion_muestral = 3.5
n = 20
media_hipotetizada = 330
alfa = 0.01

# 1. Simulación de los datos
# Aseguramos que la media y s sean las del ejemplo.
np.random.seed(42) 
datos_simulados = np.random.normal(loc=media_muestral, scale=desviacion_muestral, size=n)
# Ajuste para que coincida exactamente con la media y s del problema (ddof=1 para muestral)
datos_simulados = (datos_simulados - np.mean(datos_simulados)) / np.std(datos_simulados, ddof=1) * desviacion_muestral + media_muestral


# 2. Ejecutar la Prueba t (Mención del Procedimiento Python)
t_digital, p_valor_digital = stats.ttest_1samp(
    a=datos_simulados, 
    popmean=media_hipotetizada, 
    alternative='less' # Usamos 'less' para la prueba unilateral (cola izquierda)
)

# 3. Imprimir Resultados
print(f"Estadístico t manual: -1.917")
print(f"Estadístico t digital (Python): {t_digital:.3f}")
print("-" * 40)
print(f"P-valor (Python): {p_valor_digital:.4f}")
print(f"Nivel de Significancia (alfa): {alfa}")

# 4. Decisión Digital
if p_valor_digital <= alfa:
    print("\nDecisión Digital: SE RECHAZA H₀ (p-valor ≤ α)")
else:
    print("\nDecisión Digital: NO SE RECHAZA H₀ (p-valor > α)")
