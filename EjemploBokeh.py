# Importamos librerías necesarias
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import numpy as np

# -------------------------------
# 1. Generación de los datos
# -------------------------------
# Simulamos 1000 datos siguiendo una distribución normal con media=0 y desviación=1
datos = np.random.normal(loc=0, scale=1, size=1000)

# Construimos el histograma
hist, edges = np.histogram(datos, bins=30, density=True)

# -------------------------------
# 2. Modelo teórico (curva normal estándar)
# -------------------------------
x = np.linspace(-4, 4, 1000)  # valores en el eje X
y = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)  # función de densidad normal

# -------------------------------
# 3. Preparar fuentes de datos
# -------------------------------
source_hist = ColumnDataSource(data=dict(
    left=edges[:-1],
    right=edges[1:],
    top=hist
))

source_curve = ColumnDataSource(data=dict(
    x=x,
    y=y
))

# -------------------------------
# 4. Crear la figura
# -------------------------------
p = figure(title="Histograma vs Distribución Normal",
           x_axis_label="Valores",
           y_axis_label="Densidad",
           background_fill_color="#f5f5f5")

# -------------------------------
# 5. Añadir el histograma
# -------------------------------
# Usamos vbar (barras verticales)
p.vbar(x=(edges[:-1] + edges[1:]) / 2,  # centro de cada barra
       top=hist,
       width=0.1,
       fill_color="navy",
       fill_alpha=0.6,
       line_color="white",
       legend_label="Datos simulados")

# -------------------------------
# 6. Añadir la curva normal teórica
# -------------------------------
p.line("x", "y", source=source_curve,
       line_width=3,
       color="red",
       legend_label="Normal(0,1)")

# -------------------------------
# 7. Personalización
# -------------------------------
p.legend.location = "top_left"
p.legend.background_fill_alpha = 0.3

# -------------------------------
# 8. Mostrar el resultado
# -------------------------------
show(p)
