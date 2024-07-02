import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from matplotlib import pyplot as plt

# Cargamos el archivo CSV que se encuentra en el paquete data
dataframe = pd.read_csv('../data/datos-ventas-predict.csv')
# Convertimos los datos a numéricos ya que tuvimos problemas para que se leyesen
dataframe["año 2011"] = pd.to_numeric(dataframe["año 2011"], errors="coerce")
dataframe["año2012"] = pd.to_numeric(dataframe["año2012"], errors="coerce")
dataframe["año 2013"] = pd.to_numeric(dataframe["año 2013"], errors="coerce")
dataframe["año 2014"] = pd.to_numeric(dataframe["año 2014"], errors="coerce")
#Hacemos un print de el tipo de dato de cada columna y lo mostramos por pantalla




print(dataframe.dtypes)
print(dataframe)


# Eliminamos filas sin valores
dataframe = dataframe.dropna()
# Dividimos los datos en entrenamiento (75%) y prueba (25%)
train_data = dataframe.sample(frac=0.75, random_state=0)
test_data = dataframe.drop(train_data.index)
train_label = train_data.pop("año 2014")
test_label = test_data.pop("año 2014")
# Entrenamos el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(train_data, train_label)
# Predecimos para los datos de prueba
prediction = modelo.predict(test_data)
# Predicción para las 53 semanas del año 2014
all_data = dataframe.drop(columns=["año 2014"])
full_prediction = modelo.predict(all_data)
print(full_prediction)

#Calculamos el error porcentual medio y el error cuadratico
mape = mean_absolute_percentage_error(test_label, prediction)
print(f"MAPE: {mape * 100:.2f}%")
r2 = r2_score(y_true=test_label, y_pred=prediction)
print(f"R^2 Score: {r2}")

# Realizamos graficos los datos reales y predichos
plt.figure(figsize=(12, 6))
plt.plot(dataframe.index, dataframe["año 2011"], label="Ventas Año 2011", color= "red", marker="x")
plt.plot(dataframe.index, dataframe["año2012"], label="Ventas Año 2012", color= "blue", marker="o")
plt.plot(dataframe.index, dataframe["año 2013"], label="Ventas Año 2013", color= "green", marker="x")
plt.plot(dataframe.index, dataframe["año 2014"], label="Ventas Año 2014", color= "grey", marker="o")
plt.plot(dataframe.index, full_prediction, label="Predicción Año 2014", color= "orange", linestyle="--", marker="x")
plt.xlabel("Semana")
plt.ylabel("Ventas")
plt.title("Ventas Semanales 2011-2014 y Predicción para 2014")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(test_label.values, label="Valores reales", color= "blue", marker="o")
plt.plot(prediction, label="Valores predichos", color= "red", marker="x")
plt.xlabel("Índice de muestra")
plt.ylabel("Ventas")
plt.title("Real vs Predicciones Regresion Lineal")
plt.legend()
plt.show()


