import csv
import numpy as np
import argparse
import sys
import pickle

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from funciones import loadConfig, load_data

def loadModel(model_output: str) -> object: # He cambiado obj por object
    """
    Función que carga un modelo con pickle.
    Parámetros:
        - file: La ruta del archivo con el modelo.
    Return:
        - model: El modelo cargado.
    Errores:
        - Muestra por la terminal un error si el archivo no existe o si surge otro error.
    """

    try:
        with open(model_output, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None  # Lo he cambiado, ya que de esta manera si no hemos entrenado para todos los tipos de algoritmos, el programa va a seguir en vez de pararse.


def comparar_metricas(metricas: dict) -> None:
    """
    Compara todas las métricas numéricas de los modelos evaluados
    y muestra por terminal el mejor modelo para cada una de ellas.

    Parámetros:
        - metricas: Diccionario con los resultados detallados de cada modelo.
    """
    metricas_evaluar = ["f1_micro", "f1_macro", "f1_weighted"]
    mejores_modelos = {}

    print("\n--------------------------------------------------")
    print("COMPARATIVA DE MODELOS")
    print("--------------------------------------------------")

    for metrica in metricas_evaluar:
        mejor_modelo = None
        mejor_puntuacion = -1.0

        print(f"\nResultados para la métrica: {metrica.upper()}")

        for nombre_modelo, scores in metricas.items():
            puntuacion_actual = scores.get(metrica, 0)

            if puntuacion_actual == 0:
                continue

            print(f"  - {nombre_modelo}: {puntuacion_actual:.4f}")

            if puntuacion_actual > mejor_puntuacion:
                mejor_puntuacion = puntuacion_actual
                mejor_modelo = nombre_modelo

        if mejor_modelo:
            mejores_modelos[metrica] = {"modelo": mejor_modelo, "puntuacion": mejor_puntuacion}
            print(f"  > MEJOR MODELO: {mejor_modelo} ({mejor_puntuacion:.4f})")

    print("\n--------------------------------------------------")
    print("RESULTADOS")
    print("--------------------------------------------------")
    if mejores_modelos:
        for metrica, datos in mejores_modelos.items():
            print(f"{metrica.upper()}: {datos['modelo']} ({datos['puntuacion']:.4f})")

def guardar_metricas_csv(metricas: dict, ruta_archivo: str) -> None:
    """
    Almacena el diccionario completo de métricas en un archivo CSV.

    Parámetros:
        - metricas: Diccionario con los resultados de cada modelo.
        - ruta_archivo: Ruta y nombre del archivo de salida.
    """
    fieldnames = ['modelo', 'f1_micro', 'f1_macro', 'f1_weighted']
    
    try:
        with open(ruta_archivo, 'w', newline='', encoding='utf-8') as archivo:
            writer = csv.DictWriter(archivo, fieldnames=fieldnames)
            writer.writeheader() # Escribimos la cabecera
            
            # Recorremos el diccionario para extraer los datos
            for nombre_modelo, scores in metricas.items():
                writer.writerow({
                    'modelo': nombre_modelo,
                    'f1_micro': scores.get('f1_micro', 0),
                    'f1_macro': scores.get('f1_macro', 0),
                    'f1_weighted': scores.get('f1_weighted', 0)
                })
                
        print(f"Métricas almacenadas correctamente en: {ruta_archivo}")
    except Exception as e:
        print(f"Error al intentar guardar las métricas en formato CSV: {e}")

if __name__ == '__main__':
    # Argumentos de la terminal (config.json)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="El directorio donde se encuentra el archivo de configuración.", default="config.json")
    parser.add_argument("-m", "--metricas", type=str, help="La ruta donde se almacenarán las métricas de los modelos",
                        default="modelos/metricas_modelos.json")
    args = parser.parse_args()

    config = loadConfig(args.config, "test")

    # Separamos el dataset
    data = load_data(config["test_output"])  # Cargamos el dataset del test completo
    y_true = data[config["column"]].values  # Cargamos los valores a predecir
    y_true = LabelEncoder().fit_transform(y_true) # Pasamos datos categoricos a numericos.
    data = data.drop(
        columns=[config["column"]])  # Separamos los valores a predecir del dataset para poder hacer predicciones.

    metricas = {}  # Diccionario vacío que contendrá las métricas de todos los modelos.

    # Evaluamos cada modelo y guardamos sus métricas.
    for ruta_modelo in config["modelos"]:
        # Cargamos el modelo
        print(f"Cargando el modelo en {ruta_modelo}...")
        model = loadModel(ruta_modelo)

        # En caso de que no hayamos realizado el entrenamiento y por consiguiente la generación del mejor modelo del algoritmo que no hayamos querido.
        if model is None:
            print(f"Saltando la evaluación para {ruta_modelo} porque no existe el archivo.\n")
            continue

        nombre_modelo = str(model.estimator).split("Classifier")[0].split("(")[0]  # KNeighborsClassifier() -> KNeighbors, DecisionTreeClassifier(random_state=42) -> DecisionTree,...
        if nombre_modelo == "Pipeline": nombre_modelo = "CategoricalNB" # Parche guarro para que no llame Pipeline al nb.
        print(f"Modelo {nombre_modelo} cargado correctamente.")

        try:
            predicciones = model.predict(data.values)
            data["Predicción"] = predicciones  # Guardamos las predicciones en una nueva columna.
            data.to_csv(f"{config["test_output"].replace('.csv', '')}_prediccion_{nombre_modelo}.csv", index=False)
            data = data.drop(columns="Predicción")
            print(
                f"Predicciones del modelo {nombre_modelo} guardadas en:{config["test_output"].replace('.csv', '')}_prediccion_{nombre_modelo}.csv")

            # Calcular métricas
            metricas[nombre_modelo] = {}
            metricas[nombre_modelo]["classification_report"] = classification_report(y_true, predicciones)
            metricas[nombre_modelo]["confusion_matrix"] = confusion_matrix(y_true, predicciones)
            metricas[nombre_modelo]["f1_micro"] = f1_score(y_true, predicciones, average='micro')
            metricas[nombre_modelo]["f1_macro"] = f1_score(y_true, predicciones, average='macro')
            metricas[nombre_modelo]["f1_weighted"] = f1_score(y_true, predicciones, average='weighted')
            print(f"""  
Test del modelo {nombre_modelo}:

    Informe de clasificación: 
    {metricas[nombre_modelo]["classification_report"]}

    Matriz de confusión:
    {metricas[nombre_modelo]["confusion_matrix"]}

    F1:
        F1-score micro:
        {metricas[nombre_modelo]["f1_micro"]}

        F1-score macro:
        {metricas[nombre_modelo]["f1_macro"]}

        F1-score weighted:
        {metricas[nombre_modelo]["f1_weighted"]}
                        """)
        except Exception as e:
            print(e)
            sys.exit(1)

    comparar_metricas(metricas)
    if config["metricas"]==None: guardar_metricas_csv(metricas, args.metricas)
    else: guardar_metricas_csv(metricas, config["metricas"])
    sys.exit(0)
