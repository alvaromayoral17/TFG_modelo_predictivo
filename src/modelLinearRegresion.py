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


train_data = dataframe.sample(frac=0.75, random_state=0)
test_data = dataframe.drop(train_data.index)

train_label = train_data.pop("año 2014")
test_label = test_data.pop("año 2014")
modelo = LinearRegression()
"""print(train_data)
print(train_label)
print(test_data)
print(test_label)
"""
modelo.fit(train_data, train_label)
prediction = modelo.predict(test_data)
print(prediction)
"""
error = np.sqrt(mean_squared_error(test_label, prediction))
print("Error porcentual : %f " % (error*100))
"""
mape = mean_absolute_percentage_error(test_label, prediction)
print(f"MAPE: {mape * 100:.2f}%")

r2 = r2_score(y_true=test_label, y_pred=prediction)
print(f"R^2 Score: {r2}")

plt.plot(prediction, label="Valores reales", color= "blue", marker="o")
plt.plot(test_label, label="Valores predecidos", color= "red", marker="x")
plt.xlabel("reales")
plt.ylabel("pred")
plt.title("Real vs Predicciones")
plt.show()