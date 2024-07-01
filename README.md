# Modelo Predictivo para Análisis de Ventas

## Descripción

Este proyecto de Trabajo de Fin de Grado (TFG) se centra en el desarrollo de un cuadro de mando interactivo y un modelo predictivo para analizar y predecir datos de ventas.
El cuadro de mando proporciona visualizaciones y KPIs clave para diferentes roles dentro de la empresa (CEO, CFO, CMO), mientras que el modelo predictivo ayuda a anticipar tendencias y tomar decisiones informadas.

## Contenidos del Proyecto

- `src/`: Código fuente del proyecto.
- `data/`: Conjuntos de datos utilizados.

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
    cd TFG_modelo_predictivo
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
    pip install -r requirements.txt
    ```

## Uso

### Ejecutar el Modelo Predictivo

1. Navegar al directorio del modelo predictivo:
    ```bash
    cd src/modeloLR.py
    cd src/modelRFR.py
    ```

2. Ejecutar el script de entrenamiento del modelo:
    ```bash
    python modeloLR.py
    python modeloRFR.py
    ```

## Estructura del Código

- `main.py`: Archivo principal para ejecutar el proyecto.
- `dashboard/`: Archivos relacionados con el cuadro de mando.
- `predictive_model/`: Archivos relacionados con el modelo predictivo.
    - `train_model.py`: Script para entrenar el modelo predictivo.
    - `predict.py`: Script para realizar predicciones utilizando el modelo entrenado.
- `data/`: Conjuntos de datos utilizados en el proyecto.
- `notebooks/`: Jupyter notebooks para exploración de datos y desarrollo del modelo.

## Contribuir

Para contribuir a este proyecto, por favor sigue estos pasos:
1. Realiza un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva_funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir nueva funcionalidad'`).
4. Envía un pull request.

## Autor

- [Tu Nombre](mailto:tu_email@example.com)

## Licencia

Este proyecto está bajo Licencia.

