import json
import csv
import argparse
import pickle
import pandas as pd

# Importaciones de Scikit-Learn necesarias para el entrenamiento
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ==========================================
# FUNCIONES DE CONFIGURACIÓN Y CARGA
# ==========================================

def loadConfig(file: str) -> dict:
    """
    Función que carga el archivo .json de configuración.
    Para train.py, extraemos 'general' y filtramos los 'modelos' activos de la sección 'train'.
    """
    with open(file, 'r', encoding='utf-8') as f:
        config_completa = json.load(f)

    config = {}

    # 1. Extraer la sección 'general' (rutas de datos, columna objetivo, etc.)
    general = config_completa.get("general", {})
    for key, value in general.items():
        if key == "data" and isinstance(value, dict):
            for data_key, data_value in value.items():
                config[data_key] = data_value
        else:
            config[key] = value

    # 2. Extraer la sección 'train' y filtrar solo los modelos que estén en 'true'
    train = config_completa.get("train", {})
    for key, value in train.items():
        if key == "modelos" and isinstance(value, list):
            modelos_activos = []
            for modelo in value:
                es_activo = any(v is True for k, v in modelo.items() if k != "parametros")
                if es_activo:
                    modelos_activos.append(modelo)
            config["modelos"] = modelos_activos
        else:
            config[key] = value

    return config


def load_data(file: str, encoding='utf-8') -> pd.DataFrame:
    """Carga el dataset CSV preprocesado en un DataFrame, ignorando columnas vacías."""
    try:
        data = pd.read_csv(file, encoding=encoding)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        return data
    except UnicodeDecodeError:
        data = pd.read_csv(file, encoding='latin1')
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        return data


# ==========================================
# BLOQUE DE ENTRENAMIENTO Y UTILIDADES
# ==========================================

def divide_data():
    """
    Divide el dataset ya preprocesado en conjuntos de Entrenamiento (Train) y Validación (Dev).
    Aplica LabelEncoding a la variable a predecir para que el modelo la entienda.
    """
    global data  # Accedemos a la variable global del dataset ya procesado en el __main__
    global config  # Accedemos a la configuración global para no hardcodear valores

    # 1. Separar características (X) de la variable objetivo a predecir (y)
    x = data.drop(columns=[config["column"]])
    y = data[config["column"]]

    # 2. Codificar la variable objetivo (ej. pasa 'Stroke'/'No Stroke' a 1 y 0)
    y = LabelEncoder().fit_transform(y)

    # 3. Dividir los datos basándonos en el porcentaje 'dev' del JSON (ej. 0.25)
    # stratify=y asegura que ambos conjuntos tengan el mismo porcentaje de casos positivos/negativos
    x_train, x_dev, y_train, y_dev = train_test_split(
        x, y,
        test_size=config["dev"],
        stratify=y,
        random_state=config.get("random_state", 42)
    )

    # 4. Convertir los datos a arrays de NumPy (evita warnings de Scikit-Learn por nombres de columnas)
    x_train = x_train.values
    x_dev = x_dev.values

    return x_train, x_dev, y_train, y_dev


def save_model(model_output: str, model):
    """
    Guarda en disco el mejor modelo encontrado (.pkl) y un registro con
    los resultados de todas las combinaciones de hiperparámetros probadas (.csv).
    """
    try:
        # Extraemos el nombre base sin la extensión para guardar tanto .pkl como .csv
        base_path = model_output.rsplit('.', 1)[0]

        # 1. Guardar el modelo ejecutable con la librería pickle
        with open(f"{base_path}.pkl", "wb") as file:
            pickle.dump(model, file)
            print(f"Modelo guardado exitosamente en: {base_path}.pkl")

        # 2. Generar el reporte CSV con las figuras de mérito extraídas de GridSearchCV
        with open(f"{base_path}.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Params", "Score"])
            for params, score in zip(model.cv_results_['params'], model.cv_results_["mean_test_score"]):
                writer.writerow([params, score])
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")


def knn(model_output: str, parametros: dict):
    """
    Lógica principal de entrenamiento para el algoritmo K-Nearest Neighbors.
    """
    # 1. Obtener los datos listos para entrenar
    x_train, x_dev, y_train, y_dev = divide_data()

    # 2. Configurar la búsqueda exhaustiva de hiperparámetros (GridSearchCV)
    # n_jobs utiliza los núcleos de CPU indicados en el JSON (-1 = todos)
    model = GridSearchCV(KNeighborsClassifier(), parametros, n_jobs=config.get("cpu", -1))

    # 3. Iniciar el entrenamiento (ajuste)
    model.fit(x_train, y_train)

    # 4. Guardar los resultados
    save_model(model_output, model)


# Funciones reservadas para futuros algoritmos (Se implementarán igual que KNN)
def decision_tree(model_output: str, parametros: dict):
    pass


def random_forest(model_output: str, parametros: dict):
    pass


def naive_bayes(model_output: str, parametros: dict):
    pass


# ==========================================
# BLOQUE PRINCIPAL
# ==========================================

if __name__ == '__main__':
    # 1. Configuración de la lectura del archivo .json desde la terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Archivo de configuración.", default="config.json")
    args = parser.parse_args()

    # 2. Cargar el diccionario de configuración
    config = loadConfig(args.config)

    # 3. Cargar el dataset YA PROCESADO por el script process.py
    # Se utiliza la ruta 'train_dev_output' del JSON
    output_file = config.get("train_dev_output", "datasets/brain_stroke_procesado.csv")
    print(f"Cargando dataset preprocesado desde: {output_file}")
    data = load_data(output_file)

    # 4. Ciclo de Entrenamiento: Iteramos los modelos activos en el JSON
    for modelo in config.get("modelos", []):
        if "knn" in modelo:
            print("Entrenando modelo KNN...")
            knn(modelo["modelo_output"], modelo["parametros"])
        elif "decision_tree" in modelo:
            print("Entrenando modelo Decision Tree...")
            decision_tree(modelo["modelo_output"], modelo["parametros"])
        elif "random_forest" in modelo:
            print("Entrenando modelo Random Forest...")
            random_forest(modelo["modelo_output"], modelo["parametros"])
        elif "naive_bayes" in modelo:
            print("Entrenando modelo Naive Bayes...")
            naive_bayes(modelo["modelo_output"], modelo["parametros"])