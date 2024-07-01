import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
# Cargar el archivo CSV
data = pd.read_csv('datosenCSV.csv')

# Convertir la columna de fecha a tipo datetime especificando el formato correcto
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Crear nuevas características
data['mes'] = data['Order Date'].dt.month
data['dia_semana'] = data['Order Date'].dt.dayofweek
data['dia'] = data['Order Date'].dt.day


# Manejar valores faltantes (si los hay)
data.fillna(0, inplace=True)

# Agrupar las ventas por fecha si es necesario
ventas_por_dia = data.groupby('Order Date').agg({'Sales': 'sum', 'mes': 'first', 'dia_semana': 'first', 'dia': 'first'}).reset_index()

# Crear características adicionales basadas en datos anteriores
ventas_por_dia = ventas_por_dia.drop_duplicates(subset=['Order Date', 'Sales'])
ventas_por_dia['ventas_anteriores'] = ventas_por_dia['Sales'].shift(1).fillna(0)
# Definir características y etiquetas
X = ventas_por_dia[['mes', 'dia_semana', 'dia', 'ventas_anteriores']]
y = ventas_por_dia['Sales']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Crear y entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular el coeficiente de determinación R^2
r2 = r2_score(y_test, y_pred)
print(f'Coeficiente de determinación R^2: {r2}')

# Visualizar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Real')
plt.plot(y_pred, label='Predicho')
plt.legend()
plt.show()

# Supongamos que queremos predecir el próximo día
ultima_fila = ventas_por_dia.iloc[-1]
nueva_fila = ultima_fila.copy()
nueva_fila['Order Date'] = nueva_fila['Order Date'] + pd.Timedelta(days=1)
nueva_fila['mes'] = nueva_fila['Order Date'].month
nueva_fila['dia_semana'] = nueva_fila['Order Date'].dayofweek
nueva_fila['dia'] = nueva_fila['Order Date'].day
nueva_fila['ventas_anteriores'] = ultima_fila['Sales']

# Predecir la venta del siguiente periodo
X_nueva = nueva_fila[['mes', 'dia_semana', 'dia', 'ventas_anteriores']].values.reshape(1, -1)
prediccion = model.predict(X_nueva)
print(f'Predicción de ventas para el próximo día: {prediccion[0]}')





