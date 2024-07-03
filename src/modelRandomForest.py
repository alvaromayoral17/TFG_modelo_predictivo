import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt

# Cargamos el archivo CSV
dataframe = pd.read_csv('../data/datos-ventas-predict.csv')

dataframe["año 2011"] = pd.to_numeric(dataframe["año 2011"], errors="coerce")
dataframe["año2012"] = pd.to_numeric(dataframe["año2012"], errors="coerce")
dataframe["año 2013"] = pd.to_numeric(dataframe["año 2013"], errors="coerce")
dataframe["año 2014"] = pd.to_numeric(dataframe["año 2014"], errors="coerce")

print(dataframe.dtypes)
print(dataframe)
#Establecemos la opcion para adoptar el comportamiento futuro
pd.set_option('future.no_silent_downcasting', True)
#Asignamos las variables independientes a X y la variable dependiente a Y
#x = dataframe.drop("Sales", axis=1)
#y = dataframe["Sales"]
dataframe = dataframe.dropna()
# Construimos el modelo de Random Forest
train_data = dataframe.sample(frac=0.75, random_state=0)
test_data = dataframe.drop(train_data.index)
train_label = train_data.pop("año 2014")
test_label = test_data.pop("año 2014")
modelo = RandomForestRegressor(random_state=0)
#Entrenamos el modelo y realizamos las predicciones
modelo.fit(train_data, train_label)
prediction = modelo.predict(test_data)
#Evaluamos el modelo
mape = mean_absolute_percentage_error(y_true=test_label, y_pred=prediction)
r2 = r2_score(y_true=test_label, y_pred=prediction)
print(f"MAPE: {mape * 100:.2f}%")
print(f"R^2 Score: {r2}")
# Predicción para las 53 semanas del año 2014
all_data = dataframe.drop(columns=["año 2014"])
full_prediction = modelo.predict(all_data)

# Realizamos graficas de los datos reales y predichos
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
plt.title("Real vs Predicciones algoritmo RandomForest")
plt.legend()
plt.show()
