# Modo de uso:


## Preprocesado:
...


## Entrenamiento:
...


## Test:
...

---


# Explicación detallada de config.json:
Cada script de la plantilla utiliza config.json como archivo de configuración común, para facilitar cargar la configuración sin tener que parsear los datos que necesitan otros scripts se ha dividido en cuatro diccionarios. Todos los scripts necesitan los datos de general y el script train.py necesita también el diccionario train, por lo que en train.py se carga solo general + train.

## Explicación de cada diccionario + campo:
### General:
´´´json
"general":{
        "random_state": 42,
        "column": "columna_a_predecir",
        "data": {
            "train_dev": "traindev.csv",
            "test": "test.csv",
            "train_dev_output": "traindev_procesado.csv",
            "test_output": "test_procesado.csv"
        }
    }
´´´
- general: Este diccionario contiene información que usan todos los scripts.
- random_state: La semilla de números aleatorios, asegura que podamos reproducir los resultados.
- column: La columna que queremos predecir.
- data: Los directorios de entrada y salida de los datos.

### Procesado:
´´´json
"procesado": {
        "text_process": "tf_idf",
        "sampling": "oversampling",
        "drop_features": []
    }
´´´
- procesado: Configuración especifica que necesita el script process.py.
- text_process: Que estrategia se usa en el procesado de texto, valores posibles; tf_idf, bow.
- sampling: Si se realiza over o undersampling o no, valores posibles; oversampling, undersampling, "".
- drop_features: Que columnas eliminar del dataset, los valores son un array con el nombre de las columnas a eliminar.

### Train:
´´´json
    "train": {
        "dev":0.25,
        "cpu": -1,
        "modelo_output": "modelos/knnBestModel",
        "modelos": [
            {
                "modelo": "knn",
                "ejecutar": true,
                "parametros": {
                    "n_neighbors": [1,2,3,4,5,6,7,8,9,10]
                }
            },
            {
                "modelo": "rf",
                "ejecutar": false,
                "parametros": {
                    "n_estimators": [10,50,100],
                    "max_depth": [0,5,10,15,20],
                    "min_samples_split": [2,5],
                    "min_samples_leaf": [1,2,4],
                    "bootstrap": ["True","False"]
                }
            },
            {
                "modelo": "dt",
                "ejecutar": false,
                "parametros": {
                    "criterion": "entropy",
                    "max_depth": [0,5,10,15,20],
                    "min_samples_split": [2,5,10],
                    "min_samples_leaf": [1,2,4]
                }
            },
            {
                "modelo": "naive_bayes",
                "ejecutar": false,
                "parametros": {
                    "alpha": [0.01,0.1,0.5,1,10]
                }
            }
        ]
    }
```
- train: Configuración que necesita train.py.
- dev: El porcentaje del dataset que corresponde al dev.
- cpu: Los nucleos que puede utilizar el script al entrenar a los modelos, -1 significa que no hay restricciones y puede usar todos.
- modelo_output: La ruta donde se almacenará el mejor modelo.
- modelos: Array con la configuración de cada modelo.
- modelo + ejecutar: modelo simplemente muestra el nombre del modelo que queremos ejecutar, si ejecutar == true entonces solo cargamos ese modelo, y el resto se ignoran. Es decir, **elegimos que modelo queremos usar cambiando ejecutar para que sea true**. Puede haber varios modelos en true a la vez por si se quiere hacer pruebas.
- parametros: Los hiperparámetros de cada modelo.

### Test:
´´´json
    "test": {
        "modelo": "modelos/knnBestModel.pickle"
    }
´´´
- test: Configuración específica de test.py.
- modelo: La ruta del modelo a evaluar.