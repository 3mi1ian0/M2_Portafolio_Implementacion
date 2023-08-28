#====================================================================================
# Importación de bibliotecas:
# Se importan las bibliotecas necesarias: numpy para operaciones numéricas, 
# pandas para manejo de datos en forma de DataFrame y matplotlib.pyplot para graficación.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#====================================================================================
# Carga del dataset:
# Se proporciona la ruta del archivo CSV y se definen los nombres de las columnas. 
# Luego, se lee el CSV omitiendo la primera fila (encabezado) y 
# se almacena en un DataFrame llamado df.

# Carga el dataset desde un archivo CSV
csv_path = 'C:/Users/emime/Documents/IA/Portafolio1/Housing.csv' # ESTA LINEA DEBE CAMBIARSE PARA LA RUTA DONDE SE DESCARGUE
columns = ["price", "area", "bedrooms", "bathrooms", "stories", "mainroad",
           "guestroom", "basement", "hotwaterheating", "airconditioning",
           "parking", "prefarea", "furnishingstatus"]
# Omitir la primera fila que contiene el encabezado
df = pd.read_csv(csv_path, skiprows=[0], names=columns)


#====================================================================================
# Selección y limpieza de datos:
# Se seleccionan las columnas relevantes para el análisis y se eliminan las filas con 
# valores nulos del DataFrame.

selected_columns = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
df = df[selected_columns]
df = df.dropna()


#====================================================================================
# Conversión a tipos numéricos:
# Se convierten las columnas numéricas en tipos numéricos utilizando la función
# pd.to_numeric().

numeric_columns = ["area", "bedrooms", "bathrooms", "stories", "parking"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)


#====================================================================================
# Creación de matrices de características y etiquetas
# Se crea la matriz X con las características seleccionadas 
# (área, dormitorios, baños, etc.) y la matriz y con las etiquetas (precios).

# Convierte los datos en matrices numpy
X = df.iloc[:, 1:].to_numpy()  # Características del dataset
y = df["price"].to_numpy()      # Etiquetas (precios) del dataset


#====================================================================================
# Escalado de características:
# Se define la función scaling() que escala las características para que tengan una 
# media de cero y estén en el rango [0, 1]. Luego, se aplican estas transformaciones 
# a X y se almacenan en X_scaled.

# Escala las características para ayudar en la convergencia del descenso de gradiente
def scaling(samples):
    return (samples - np.mean(samples, axis=0)) / np.max(samples, axis=0)

X_scaled = scaling(X)  # Características escaladas


#====================================================================================
# Función h(x):
# Se define la función h(params, sample) que calcula la predicción h(x) utilizando 
# los parámetros del modelo y las características escaladas.

def h(params, sample):
    return np.dot(params, sample)


#====================================================================================
# Función de mean square error (error cuadrático medio):
# Se define la función mean_squared_error(params, samples, y) que calcula el error 
# cuadrático medio entre las predicciones y las etiquetas reales.

# Función para calcular el error cuadrático medio
def mse(params, samples, y):
    errors = np.square(np.dot(samples, params) - y)
    return np.mean(errors)


#====================================================================================
# Implementación del descenso de gradiente:
# Se define la función gradient_descent() que realiza el descenso de gradiente para
# ajustar los parámetros del modelo. Se calcula el gradiente y se actualizan los
# parámetros en cada iteración.

def grad_des(params, samples, y, learning_rate, epochs):
    errors = []
    for _ in range(epochs):
        errors.append(mse(params, samples, y))
        gradient = np.dot(samples.T, np.dot(samples, params) - y) / len(samples)
        params -= learning_rate * gradient
    return params, errors


#====================================================================================
# Entrenamiento del modelo:
# Se definen los parámetros iniciales (initial_params), la tasa de aprendizaje 
# (learning_rate) y el número de épocas (epochs). Luego, se entrena el modelo 
# utilizando la función gradient_descent() y se obtienen los 
# parámetros finales (final_params) y el historial de errores (error_history).

# Parámetros iniciales y hiperparámetros
initial_params = np.zeros(X_scaled.shape[1])
learning_rate = 0.01
epochs = 1000

# Entrena el modelo
final_params, error_history = grad_des(initial_params, X_scaled, y, learning_rate, epochs)


#====================================================================================
# Graficación de errores:
# Se grafica el historial de errores en función del número de épocas para visualizar
# cómo se reduce el error cuadrático medio durante el entrenamiento.

# Grafica la reducción del error a lo largo de las épocas
plt.plot(error_history)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Error Reduction over Epochs')
plt.show()


#====================================================================================
# Predicción:
# Se realiza una predicción utilizando las características del ejemplo proporcionado
# (sample_to_predict). Se escalan las características y se utiliza la función 
# h() para obtener la predicción. Luego, se imprime el precio predicho.

# Realiza predicciones
sample_to_predict = np.array([2250, 3, 2, 2, 1, 2])  # Características del ejemplo a predecir
sample_to_predict_scaled = scaling(sample_to_predict)
predicted_price = h(final_params, sample_to_predict_scaled)
print(f"Predicted Price: ${predicted_price:.2f}")
