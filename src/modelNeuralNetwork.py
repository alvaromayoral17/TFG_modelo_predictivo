import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt

# Cargamos el archivo CSV
dataframe = pd.read_csv('../data/datos-ventas-predict.csv')

dataframe["año 2011"] = pd.to_numeric(dataframe["año 2011"], errors="coerce")
dataframe["año2012"] = pd.to_numeric(dataframe["año2012"], errors="coerce")
dataframe["año 2013"] = pd.to_numeric(dataframe["año 2013"], errors="coerce")
dataframe["año 2014"] = pd.to_numeric(dataframe["año 2014"], errors="coerce")

print(dataframe.dtypes)
print(dataframe)

# Establecemos la opcion para adoptar el comportamiento futuro
pd.set_option('future.no_silent_downcasting', True)

# Eliminar filas con valores NaN
dataframe = dataframe.dropna()

# Dividir los datos en conjuntos de entrenamiento y prueba
train_data = dataframe.sample(frac=0.75, random_state=0)
test_data = dataframe.drop(train_data.index)

# Separar la columna objetivo
train_label = train_data.pop("año 2014")
test_label = test_data.pop("año 2014")

# Construir el modelo de redes neuronales
model = Sequential()
model.add(Dense(64, input_dim=train_data.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(train_data, train_label, epochs=100, batch_size=10, verbose=1)

# Realizar predicciones
prediction = model.predict(test_data)

# Evaluar el modelo
mape = mean_absolute_percentage_error(test_label, prediction)
r2 = r2_score(test_label, prediction)
print(f"MAPE: {mape * 100:.2f}%")
print(f"R^2 Score: {r2}")

plt.plot(test_label, label="Valores reales", color= "blue", marker="o")
plt.plot(prediction, label="Valores predecidos", color= "red", marker="x")
plt.xlabel("reales")
plt.ylabel("pred")
plt.title("Real vs Predicciones")
plt.show()
