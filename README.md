- [Versión de python](#versión-de-python)
- [Modo de uso](#modo-de-uso)
  - [Procesado](#procesado)
  - [Entrenamiento](#entrenamiento)
  - [Test](#test)
  - [Uso básico](#uso-básico)
  - [Opciones extra](#opciones-extra)
  - [Ejemplo preparado](#ejemplo-preparado)
- [Explicación detallada de config.json](#explicación-detallada-de-configjson)
  - [Explicación de cada diccionario + campo](#explicación-de-cada-diccionario--campo)
    - [General](#general)
    - [Procesado](#procesado-1)
    - [Train](#train)
    - [Test](#test-1)
- [Hiperparámetros de cada algoritmo](#hiperparámetros-de-cada-algoritmo)
  - [KNN](#knn)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [Categorical Naive Bayes](#categorical-naive-bayes)
---

# Versión de python
Python 3.12.13

---

# Modo de uso
Hemos dividido el proyecto en tres fases, preprocesado, entrenamiento y test.
## Procesado
Este script procesa los datos de dos secciones de un dataset a la vez, de esta forma nos aseguramos que tanto el train_dev como el test se tratan igual.

## Entrenamiento
Script que entrena uno o varios modelos distintos a la vez. Desde el archivo de configuración se puede seleccionar que modelos se van a entrenar, basta con poner true el campo con el nombre del algoritmo.

## Test
Utiliza diferentes métricas para probar cada modelo y los compara. También guarda en un json los resultados de los tests para poder consultarlos con facilidad.

## Uso básico
 La forma de ejecutar los distintos scripts es exáctamente la misma, una vez modificado el json se pueden ejecutar en este orden:
```bash
python3 process.py
python3 train.py
python3 test.py
```

## Opciones extra
Para poder utilizar distintos archivos de configuración, todos los scripts tienen una opción `-c` o `--config` para especificar que archivo de configuración utilizar. Además, test.py tiene una opción adicional que permite especificar donde guardar las métricas de los modelos `-m` o `--metricas`:
```bash
python3 process.py -c "config_pruebas.json"
python3 train.py --config "config_pruebas.json"
python3 test.py -c "config_pruebas.json" --metricas "modelos/metricasKNN.json
```
Ambas opciones tienen valores por defecto; config.json y modelos/metricas_modelos.json, por lo que no es obligatorio usarlas.

## Ejemplo preparado
Hemos separado un dataset (brainstroke de la práctica de dataiku) y creado un archivo de configuración de ejemplo para poder probar los scripts sin tener que configurar nada:
```bash
python3 process.py -c "config_ejemplo.json"
python3 train.py -c "config_ejemplo.json"
python3 test.py -c "config_ejemplo.json" -m "modelos/metricas_ejemplo.json"
```
---


# Explicación detallada de config.json
Cada script de la plantilla utiliza config.json como archivo de configuración común, para facilitar cargar la configuración sin tener que parsear los datos que necesitan otros scripts se ha dividido en cuatro diccionarios. Todos los scripts necesitan los datos de general y el script train.py necesita también el diccionario train, por lo que en train.py se carga solo general + train.

## Explicación de cada diccionario + campo
### General
```json
"general": {
        "random_state": 42,
        "column": "columna_a_predecir",
        "data": {
            "train_dev": "traindev.csv",
            "test": "test.csv",
            "train_dev_output": "traindev_procesado.csv",
            "test_output": "test_procesado.csv"
        }
    }
```
- general: Este diccionario contiene información que usan todos los scripts.
- random_state: La semilla de números aleatorios, asegura que podamos reproducir los resultados.
- column: La columna que queremos predecir.
- data: Los directorios de entrada y salida de los datos.

### Procesado
```json
"procesado": {
        "text_process": "tf_idf",
        "sampling": "oversampling",
        "imputacion_numerico": "mean",
        "imputacion_categorico": "mode",
        "drop_features": []
    }
```
- procesado: Configuración especifica que necesita el script process.py.
- text_process: Que estrategia se usa en el procesado de texto, valores posibles; tf_idf, bow.
- sampling: Si se realiza over o undersampling o no, valores posibles; oversampling, undersampling, null.
- imputacion_numerico: La estrategia de imputación a seguir en datos numéricos, puede valer: mean, median o mode.
- imputacion_categorico: La estrategia de imputación a seguir en datos categóricos, puede valer: mean, median o mode.
- drop_features: Que columnas eliminar del dataset, los valores son un array con el nombre de las columnas a eliminar.

### Train
```json
    "train": {
        "dev":0.25,
        "cpu": -1,
        "scoring": "f1_macro",
        "modelos": [
            {
                "knn":true,
                "modelo_output":"modelos/knn_BestModel.pickle",
                "parametros": {
                    "n_neighbors": [1,2,3,4,5,6,7,8,9,10],
		            "weights": ["uniform","distance"],
		            "p": [1,2]
                }
            },
            {
                "random_forest":false,
                "modelo_output":"modelos/random_forest_BestModel.pickle",
                "parametros": {
                    "n_estimators": [10,50,100],
                    "criterion": ["gini","entropy"],
                    "max_depth": [null,5,10,15,20],
                    "min_samples_split": [2,5],
                    "min_samples_leaf": [1,2,4],
                    "bootstrap": [true,false]
                }
            },
            {
                "decision_tree":true,
                "modelo_output":"modelos/decision_tree_BestModel.pickle",
                "parametros": {
                    "criterion": ["gini","entropy"],
                    "max_depth": [null,5,10,15,20],
                    "min_samples_split": [2,5,10],
                    "min_samples_leaf": [1,2,4]
                }
            },
            {
                "categorical_nb": true,
                "modelo_output": "modelos/categorical_nb_BestModel.pickle",
                "parametros": {
                    "preprocessor__num__n_bins": [3, 5],
                    "preprocessor__num__strategy": ["uniform", "quantile"],
                    "clf__alpha": [1e-9, 0.1, 0.5, 1.0, 2.0],
                    "clf__min_categories": [100]
                }
            },
        ]
    }
```

- train: Configuración que necesita train.py.
- dev: El porcentaje del dataset que corresponde al dev.
- cpu: Los nucleos que puede utilizar el script al entrenar a los modelos, -1 significa que no hay restricciones y puede usar todos.
- scoring: La métrica que usara el barrido de hiperparámetros, estas pueden ser; f1, f1_macro, f1_weighted y f1_micro.
- modelos: Array con la configuración de cada modelo.
- modelo_output: La ruta donde se almacenará el mejor modelo.
- "knn": Si el campo con el nombre del modelo es true se entrenará este modelo, si es false se ignora. Se puede entrenar cualquier combinación de modelos. En este ejemplo se entrenan todos los modelos menos random forest.
- parametros: Los hiperparámetros de cada modelo. (Al ser tantos tienen una sección aparte)

### Test
```json
    "test": {
        "metricas": "modelos/metricas.csv",
        "modelos": [
            "modelos/knn_BestModel.pkl",
            "modelos/random_forest_BestModel.pkl",
            "modelos/decision_tree_BestModel.pkl",
            "modelos/categorical_nb_BestModel.pkl"
        ]
    }
```
- test: Configuración específica de test.py.
- metricas: Ruta donde se almacenan las metricas de los modelos.
- modelos: Array con las rutas de los modelos a evaluar.

---

# Hiperparámetros de cada algoritmo
## KNN
KNN tiene 3 hiperparámetros:
- n_neighbours: Los K-ésimos vecinos más cercanos, es un valor entero.
- weights: Tiene dos valores; "uniform" y "distance". Uniform significa que no usamos pesos y distance que el peso se calcula en base a la distancia entre vecinos.
- p: Es el tipo de cálculo que utilizamos para la distancia p=1 distancia manhattan y p=2 distancia euclideana.

## Decision Tree
Decision Tree tiene 4 hiperparámetros:
- criterion: La función que se utiliza para medir la calidad de las bifurcaciones, tiene tres valores posibles; "gini", "entropy" y "log_loss".
- max_depth: La profundidad máxima del árbol. Es un número entero o None, si es None se expanden los nodos hasta que las hojas son puras o contienen menos muestras que el min_samples_split. En json el objeto None de python se puede guardar como null, sin comillas, "None" se interpreta como un string y None como un error. 
- min_samples_split: El mínimo de muestras para poder seguir bifurcando. Es un número entero.
- min_samples_leaf: El mínimo de muestras por hoja. Es un número entero.

## Random Forest
Random Forest ajusta varios decision trees por lo que comparte parametros con decision tree y añade dos nuevos:
- n_estimators: La cantidad de árboles de decisión a ajustar, es un número entero.
- bootstrap: Si se usa o no bootstraping, es decir, si todos los árboles se entrenan con un subset distinto o no. Puede valer true o false.

## Categorical Naive Bayes 

Este modelo requiere que **todas** las características sean categóricas (discretas). Para lograrlo, el código emplea un `Pipeline` de Scikit-Learn que primero discretiza (agrupa en "cajas") las variables numéricas continuas mediante `KBinsDiscretizer`, y luego pasa los datos resultantes al modelo `CategoricalNB`.
Tiene 4 hiperparámetros principales (los prefijos indican a qué parte del Pipeline pertenecen):
- preprocessor__num__n_bins: El número de intervalos o "cajas" en los que se dividirán las variables continuas. Es un número entero (ej. 3, 5). Si hay características con valores muy constantes y no puede generar suficientes cajas fusiona los que son iguales, y el script lanza warnings avisando sobre esto.
- preprocessor__num__strategy: La estrategia usada para calcular los anchos de esas cajas. Puede valer "uniform" (todas las cajas tienen el mismo ancho matemático) o "quantile" (todas las cajas contendrán la misma cantidad de muestras).
- clf__alpha: Parámetro de suavizado aditivo (Laplace/Lidstone). Es un float que se suma a los recuentos para evitar que una categoría no vista arrastre toda la probabilidad a 0. Puede valer 1e-9, 0.1, 1.0, etc.
- clf__min_categories: Número mínimo de categorías esperadas. Lo fijamos en un valor genérico alto (ej. 100) para prevenir errores (IndexError) durante la validación cruzada si en un pliegue de validación aparece una categoría "nueva" que no estaba en el pliegue de entrenamiento.