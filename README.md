# Modelo Predictivo para Análisis de Ventas

## Descripción

Este proyecto de Trabajo de Fin de Grado (TFG) se centra en el desarrollo de un cuadro de mando interactivo y un modelo predictivo para analizar y predecir datos de ventas.
El cuadro de mando proporciona visualizaciones y KPIs clave para diferentes roles dentro de la empresa (CEO, CLO, CMO), mientras que el modelo predictivo ayuda a anticipar tendencias y tomar decisiones informadas.

## Contenidos del Proyecto

- `src/`: Código fuente del proyecto.
- `data/`: Conjuntos de datos utilizados.
- `CuadroMandoIntegral.pbix`: Archivo referente al Cuadro de Mando.

## Requisitos Previos

- Python 3.8 o superior
- Power BI o Tableau (dependiendo de la herramienta utilizada)
- Bibliotecas de Python: pandas, numpy, scikit-learn, matplotlib, seaborn, etc.

## Instalación

1. Clonar el repositorio:
    ```bash
    git clone https://github.com/alvaromayoral17/TFG_modelo_predictivo.git
    ```

2. Navegar al directorio del proyecto:
    ```bash
    cd TFG_predict/
    ```

3. Crear un entorno virtual:
    ```bash
    python -m venv venv
    ```

4. Activar el entorno virtual:
    - En Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - En macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

5. Instalar las dependencias:
    ```bash
    pip install pandas
    pip install numpy
    pip install scikit-learn
    pip install matplotlib
    pip install tensorflow
    
    ```

## Uso

### Visualizar el Cuadro de Mando Integral

Para la visualización del cuadro de mando se debe descargar localmente el archivo y requiere tener instalado Power BI Desktop.

### Ejecutar el Modelo Predictivo

1. Navegar al directorio del modelo predictivo:
    ```bash
    cd src/modelLinearRegresion.py
    cd src/modelNeuralNetwork.py
    cd src/modelRandomForest.py
    ```

2. Ejecutar el script de entrenamiento del modelo:
    ```bash
    python modelLinearRegresion.py
    python modelNeuralNetwork.py
    python modelRandomForest.py
    ```

## Estructura del Código

- `data/`: Archivo relacionado con el conjunto de datos
- `src/`: Archivos relacionados con el modelo predictivo.
    - `modelLinearRegresion.py`: Script para entrenar el modelo predictivo con algoritmo Regresion Lineal.
    - `modelNeuralNetwork.py`: Script para entrenar el modelo predictivo con algoritmo Redes Neuronales.
    -  `modelRandomForest.py`: Script para entrenar el modelo predictivo con algoritmo Random Forest Regression.

## Contribuir

Para contribuir a este proyecto, por favor sigue estos pasos:
1. Realiza un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva_funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Envía un pull request.

## Autor

- [alvaromayoral17](mailto:a.mayoral5@usp.ceu.es)

## Licencia

Este proyecto está bajo Licencia CEU USP.

