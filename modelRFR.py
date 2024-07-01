import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt

# Cargamos el archivo CSV
dataframe = pd.read_csv('TFG_dataset_CSV1.csv')

#Pasamos cuatro atributos a otros formatos de tipo de datos
dataframe["Order Date"] = pd.to_datetime(dataframe["Order Date"])
dataframe["Ship Date"] = pd.to_datetime(dataframe["Ship Date"])
dataframe["Sales"] = pd.to_numeric(dataframe["Sales"], errors="coerce")
dataframe["Profit"] = pd.to_numeric(dataframe["Profit"], errors="coerce")
#Quitamos del dataset atributos que no son necesarios para el modelo predictivo
dataframe = dataframe.drop(columns=["Order ID", "Year", "Customer ID", "Customer Name", "Country", "Postal Code", "Product ID", "Product Name", "City", "State", "Order Date", "Ship Date"])

#Establecemos la opcion para adoptar el comportamiento futuro
pd.set_option('future.no_silent_downcasting', True)
#Pasamos los valores de los siguientes atributos a valores numericos para su analisis predicticvo
values_shipmode = {"Ship Mode": {"First Class": 0, "Second Class": 1, "Same Day": 2, "Standard Class": 3}}
dataframe.replace(values_shipmode, inplace=True)
values_segment = {"Segment": {"Consumer": 0, "Corporate": 1, "Home Office": 2}}
dataframe.replace(values_segment, inplace=True)
values_region = {"Region": {"Central": 0, "South": 1, "East": 2, "West": 3}}
dataframe.replace(values_region, inplace=True)
values_category = {"Category": {"Furniture": 0, "Office Supplies": 1, "Technology": 2}}
dataframe.replace(values_category, inplace=True)
values_subcategory = {"Sub-Category": {"Accessories": 0, "Appliances": 1, "Art": 2, "Binders": 3, "Bookcases": 4, "Chairs": 5, "Copiers": 6, "Envelopes": 7, "Fasteners": 8, "Furnishings": 9, "Labels": 10, "Machines": 11, "Paper": 12, "Phones": 13, "Storage": 14, "Supplies": 15, "Tables": 16}}
dataframe.replace(values_subcategory, inplace=True)

#Asignamos las variables independientes a X y la variable dependiente a Y
#x = dataframe.drop("Sales", axis=1)
#y = dataframe["Sales"]
dataframe = dataframe.dropna()


train_data = dataframe.sample(frac=0.8, random_state=0)
test_data = dataframe.drop(train_data.index)

train_label = train_data.pop("Sales")
test_label = test_data.pop("Sales")
modelo = RandomForestRegressor(random_state=0)

modelo.fit(train_data, train_label)
prediction = modelo.predict(test_data)

#Evaluacion del modelo
mape = mean_absolute_percentage_error(y_true=test_label, y_pred=prediction)
r2 = r2_score(y_true=test_label, y_pred=prediction)
print(f"MAPE: {mape * 100:.2f}%")
print(f"R^2 Score: {r2}")


plt.scatter(x = dataframe["Profit"], y = dataframe["Sales"])
plt.xlabel("Profit")
plt.ylabel("Sales")
plt.title("Sales vs Profit")
plt.show()

"""
plt.scatter(test_label, prediction, color='blue', label='Actual vs Predicted')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Sales Prediction')
plt.legend()
plt.show()
"""


