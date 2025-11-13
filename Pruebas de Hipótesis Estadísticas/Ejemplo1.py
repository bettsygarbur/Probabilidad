import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ============================================================
# PASO 1: DEFINIR LOS DATOS
# ============================================================
# Crear arrays con los datos de presión arterial (mmHg)
grupo_tratamiento = np.array([118, 122, 115, 120, 119, 117, 121, 118, 116, 120])
grupo_control = np.array([128, 132, 130, 125, 131, 129, 128, 127, 133, 130])

print("="*70)
print("PRUEBA t DE MUESTRAS INDEPENDIENTES - MEDICAMENTO PARA PRESIÓN ARTERIAL")
print("="*70)
print("\nDatos del Grupo de Tratamiento (mmHg):", grupo_tratamiento)
print("Datos del Grupo Control (mmHg):", grupo_control)

# ============================================================
# PASO 2: CALCULAR ESTADÍSTICAS DESCRIPTIVAS
# ============================================================
# Calcular media, desviación estándar y tamaño muestral para cada grupo
media_trat = np.mean(grupo_tratamiento)
std_trat = np.std(grupo_tratamiento, ddof=1)  # ddof=1 para muestra
n_trat = len(grupo_tratamiento)

media_control = np.mean(grupo_control)
std_control = np.std(grupo_control, ddof=1)
n_control = len(grupo_control)

# Crear un DataFrame con las estadísticas
estadisticas = pd.DataFrame({
    'Grupo': ['Tratamiento', 'Control'],
    'n': [n_trat, n_control],
    'Media': [media_trat, media_control],
    'Desv. Estándar': [std_trat, std_control],
    'Error Estándar': [std_trat/np.sqrt(n_trat), std_control/np.sqrt(n_control)]
})

print("\n" + "-"*70)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("-"*70)
print(estadisticas.to_string(index=False))

# ============================================================
# PASO 3: REALIZAR LA PRUEBA t DE MUESTRAS INDEPENDIENTES
# ============================================================
# La función ttest_ind de scipy realiza:
# - Calcula el estadístico t
# - Calcula el valor p (probabilidad)
# - Por defecto asume varianzas iguales (equal_var=True)

# Realizar la prueba t bilateral (two-sided)
t_statistic, p_value = stats.ttest_ind(grupo_tratamiento, grupo_control, 
                                        equal_var=True)

print("\n" + "-"*70)
print("RESULTADOS DE LA PRUEBA t")
print("-"*70)
print(f"Estadístico t: {t_statistic:.4f}")
print(f"Valor p (bilateral): {p_value:.6f}")
print(f"Nivel de significancia (α): 0.05")

# ============================================================
# PASO 4: CALCULAR INTERVALO DE CONFIANZA 95%
# ============================================================
# El intervalo de confianza para la diferencia de medias
diferencia_medias = media_trat - media_control
gl = n_trat + n_control - 2  # Grados de libertad

# Varianza combinada
varianza_combinada = ((n_trat - 1) * std_trat**2 + 
                       (n_control - 1) * std_control**2) / gl

# Error estándar de la diferencia
se_diferencia = np.sqrt(varianza_combinada * (1/n_trat + 1/n_control))

# Valor crítico t para IC 95%
t_critico = stats.t.ppf(0.975, gl)

# Límites del intervalo de confianza
ic_inferior = diferencia_medias - t_critico * se_diferencia
ic_superior = diferencia_medias + t_critico * se_diferencia

print(f"\nGrados de libertad: {gl}")
print(f"Diferencia de medias: {diferencia_medias:.2f} mmHg")
print(f"IC 95% para la diferencia: [{ic_inferior:.2f}, {ic_superior:.2f}]")

# ============================================================
# PASO 5: TAMAÑO DEL EFECTO (d de Cohen)
# ============================================================
# Mide la magnitud práctica de la diferencia entre grupos
d_cohen = (media_trat - media_control) / np.sqrt(varianza_combinada)

print(f"\nTamaño del efecto (d de Cohen): {d_cohen:.4f}")
if abs(d_cohen) < 0.2:
    efecto = "Pequeño"
elif abs(d_cohen) < 0.5:
    efecto = "Pequeño a Medio"
elif abs(d_cohen) < 0.8:
    efecto = "Medio"
else:
    efecto = "Grande"
print(f"Interpretación: Efecto {efecto}")

# ============================================================
# PASO 6: DECISIÓN ESTADÍSTICA
# ============================================================
print("\n" + "="*70)
print("CONCLUSIONES")
print("="*70)

alpha = 0.05
if p_value < alpha:
    print(f"\n✓ Dado que p-value ({p_value:.6f}) < α ({alpha}),")
    print("  RECHAZAMOS la hipótesis nula (H₀).")
    print("\n✓ Conclusión: Existe evidencia estadísticamente significativa")
    print("  de que la presión arterial media del grupo de tratamiento")
    print("  DIFIERE significativamente del grupo control.")
    print(f"\n✓ El medicamento tiene un efecto {efecto.lower()} sobre la")
    print("  reducción de la presión arterial.")
else:
    print(f"\n✗ Dado que p-value ({p_value:.6f}) ≥ α ({alpha}),")
    print("  NO RECHAZAMOS la hipótesis nula (H₀).")
    print("\n✗ Conclusión: No existe evidencia estadísticamente significativa")
    print("  de diferencia entre los grupos.")

# ============================================================
# PASO 7: VISUALIZACIÓN DE LOS RESULTADOS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: Boxplot de ambos grupos
ax1 = axes[0]
data_plot = [grupo_tratamiento, grupo_control]
bp = ax1.boxplot(data_plot, labels=['Tratamiento', 'Control'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax1.set_ylabel('Presión Arterial Sistólica (mmHg)', fontsize=11)
ax1.set_title('Comparación de Presión Arterial\nTratamiento vs Control', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# Gráfico 2: Histogramas superpuestos
ax2 = axes[1]
ax2.hist(grupo_tratamiento, bins=5, alpha=0.6, label='Tratamiento', color='blue')
ax2.hist(grupo_control, bins=5, alpha=0.6, label='Control', color='red')
ax2.axvline(media_trat, color='blue', linestyle='--', linewidth=2, label=f'Media Trat: {media_trat:.1f}')
ax2.axvline(media_control, color='red', linestyle='--', linewidth=2, label=f'Media Control: {media_control:.1f}')
ax2.set_xlabel('Presión Arterial Sistólica (mmHg)', fontsize=11)
ax2.set_ylabel('Frecuencia', fontsize=11)
ax2.set_title('Distribución de Presión Arterial', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
