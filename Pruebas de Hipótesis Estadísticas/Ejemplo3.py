"""
ejercicio_ttest_statsmodels.py
Ejercicio: comparación de medias (dos muestras independientes) con statsmodels.
Autor: Jhon (ejemplo completo)
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans, ttest_ind
import math
# -----------------------------
# Datos
# -----------------------------
A = np.array([1020, 980, 1005, 995, 1010, 1000, 990, 1008, 1002, 1015, 995, 1007])
B = np.array([990, 975, 980, 970, 985, 960, 978, 982, 968, 974, 976, 971])
n1 = len(A)
n2 = len(B)
# -----------------------------
# Estadísticas descriptivas (numpy)
# -----------------------------
mean1 = A.mean()
mean2 = B.mean()
var1 = A.var(ddof=1)
var2 = B.var(ddof=1)
print("=== Estadísticas descriptivas ===")
print(f"n1={n1}, mean1={mean1:.4f}, s1^2={var1:.6f}")
print(f"n2={n2}, mean2={mean2:.4f}, s2^2={var2:.6f}\n")
# -----------------------------
# SOLUCIÓN DIGITAL con statsmodels
# -----------------------------
# 1) Prueba t clásica asumiendo varianzas iguales (student)
# Usamos CompareMeans a partir de DescrStatsW
ds1 = DescrStatsW(A)  # wrapper con pesos; facilita cálculos
ds2 = DescrStatsW(B)
cm = CompareMeans(ds1, ds2)
# ttest_ind devuelve (tstat, pvalue, df)
tstat, pvalue, df = cm.ttest_ind(usevar='pooled', alternative='two-sided')
diff = mean1 - mean2
# Intervalo de confianza para la diferencia (95%)
ci_low, ci_upp = cm.tconfint_diff(alpha=0.05, usevar='pooled')
# Cohen's d (efecto estandarizado) usando pooled sd
sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
pooled_sd = math.sqrt(sp2)
cohen_d = diff / pooled_sd
print("=== Resultado statsmodels (prueba t, pooled) ===")
print(f"t-statistic = {tstat:.6f}")
print(f"p-value (two-sided) = {pvalue:.8f}")
print(f"degrees of freedom (df) = {df}")
print(f"Differencia de medias (mean1 - mean2) = {diff:.4f}")
print(f"IC 95% para la diferencia = [{ci_low:.4f}, {ci_upp:.4f}]")
print(f"Cohen's d = {cohen_d:.4f}\n")
# -----------------------------
# Comparación con cálculo "manual" usando scipy
# -----------------------------
# (esto repite los pasos manuales para ver la coincidencia)
sp2_manual = sp2
se_manual = math.sqrt(sp2_manual * (1/n1 + 1/n2))
t_manual = diff / se_manual
p_manual = 2 * stats.t.sf(abs(t_manual), df)
tcrit = stats.t.ppf(0.975, df)
ci_manual = (diff - tcrit * se_manual, diff + tcrit * se_manual)
print("=== Cálculo manual replicado (scipy + fórmulas) ===")
print(f"sp^2 (pooled) = {sp2_manual:.6f}")
print(f"SE = {se_manual:.6f}")
print(f"t (manual) = {t_manual:.6f}")
print(f"p (manual, two-sided) = {p_manual:.8f}")
print(f"IC manual 95% = [{ci_manual[0]:.6f}, {ci_manual[1]:.6f}]\n")
# -----------------------------
# Decisión y conclusión
# -----------------------------
alpha = 0.05
print("=== DECISIÓN ===")
if pvalue < alpha:
    print(f"p-value = {pvalue:.8f} < {alpha} => Rechazamos H0: las medias difieren significativamente.")
else:
    print(f"p-value = {pvalue:.8f} >= {alpha} => No rechazamos H0: no hay evidencia de diferencia.")

# -----------------------------
# Notas sobre funciones usadas (explicación de la herramienta Python)
# -----------------------------
print("\n=== EXPLICACIÓN de funciones y pasos en Python ===")
print("- numpy: cálculo de medias y varianzas (A.mean(), A.var(ddof=1))")
print("- statsmodels.stats.weightstats.DescrStatsW: envuelve arrays para facilitar cálculos (weights/summary stats).")
print("- statsmodels.stats.weightstats.CompareMeans: permite comparar dos DescrStatsW y ejecutar t-tests y confints.")
print("    • cm.ttest_ind(usevar='pooled', alternative='two-sided') -> (tstat, pvalue, df)")
print("    • cm.tconfint_diff(alpha=0.05, usevar='pooled') -> (ci_low, ci_high)")
print("- scipy.stats.t.sf: cola superior de t para obtener p-valor manual si lo deseas.")
print("- Las funciones de statsmodels devuelven exactamente el t, p y el IC que usamos en el análisis.\n")
# -----------------------------
# FIN
# -----------------------------
