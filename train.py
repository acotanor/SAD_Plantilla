import csv
import argparse
import pickle
import sys

# Importaciones de Scikit-Learn necesarias para el entrenamiento
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin # <-- NUEVO: Para crear el Wrapper

# --- NUEVAS IMPORTACIONES PARA NAIVE BAYES ---
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# --- https://pypi.org/project/mixed-naive-bayes/ (mixed-naive-bayes proporcionado por Aitziber, especificado en la tarea 4)
from mixed_naive_bayes import MixedNB

# Funciones definidas en funciones.py
from funciones import loadConfig, load_data

# ==========================================
# BLOQUE DE ENTRENAMIENTO Y UTILIDADES
# ==========================================

# CLASE WRAPPER PARA MIXED NAIVE BAYES
class ScikitMixedNB(BaseEstimator, ClassifierMixin):
    """
    Envoltorio ultra-ligero para conectar mixed_naive_bayes con GridSearchCV.
    Utiliza np.errstate para manejar de forma segura y localizada el defecto de
    underflow (0.0 / 0.0) de la librería externa, sin usar filtros globales.
    """

    def __init__(self, alpha=0.5, categorical_features=None, max_categories=None):
        self.alpha = alpha
        self.categorical_features = categorical_features
        self.max_categories = max_categories
        self.model_ = None

    def fit(self, X, y):
        import numpy as np
        self.classes_ = np.unique(y)

        self.model_ = MixedNB(
            alpha=self.alpha,
            categorical_features=self.categorical_features,
            max_categories=self.max_categories
        )

        # Aislamos el error matemático solo durante el ajuste
        with np.errstate(divide='ignore', invalid='ignore'):
            self.model_.fit(X, y)
        return self

    def predict(self, X):
        import numpy as np
        # Aislamos el error matemático de 0/0 solo durante la predicción
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.model_.predict(X)

    def predict_proba(self, X):
        import numpy as np
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.model_.predict_proba(X)

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
    model = GridSearchCV(KNeighborsClassifier(), parametros, n_jobs=config.get("cpu", -1), scoring=config["scoring"])

    # 3. Iniciar el entrenamiento (ajuste)
    model.fit(x_train, y_train)

    # 4. Guardar los resultados
    save_model(model_output, model)

def decision_tree(model_output: str, parametros: dict):
    """
    Lógica principal de entrenamiento para el algoritmo Decision Tree.
    """
    # 1. Obtener los datos listos para entrenar
    x_train, x_dev, y_train, y_dev = divide_data()

    # 2. Barrido de hiperparámetros
    model = GridSearchCV(DecisionTreeClassifier(random_state=config["random_state"]), parametros, n_jobs=config["cpu"],
                         scoring=config["scoring"])

    # 3. Iniciar el entrenamiento (ajuste)
    model.fit(x_train, y_train)

    # 4. Guardar los resultados
    save_model(model_output, model)

def random_forest(model_output: str, parametros: dict):
    """
    Lógica principal de entrenamiento para el algoritmo Random Forest.
    """
    # 1. Obtener los datos listos para entrenar
    x_train, x_dev, y_train, y_dev = divide_data()

    # 2. Barrido de hiperparámetros
    model = GridSearchCV(RandomForestClassifier(random_state=config["random_state"]), parametros, n_jobs=config["cpu"],
                         scoring=config["scoring"])

    # 3. Iniciar el entrenamiento (ajuste)
    model.fit(x_train, y_train)

    # 4. Guardar los resultados
    save_model(model_output, model)


def categorical_nb(model_output: str, parametros: dict):
    """
    Lógica principal de entrenamiento para Categorical Naive Bayes.

    Según el enunciado, para usar CategoricalNB necesitamos que TODOS los datos sean discretos/categóricos.
    Como nuestro dataset (procesado en process.py) tiene algunas variables numéricas continuas (floats),
    tenemos que discretizarlas primero (agruparlas en "cajas" o "bins") usando KBinsDiscretizer.
    """
    # 1. Obtenemos los datos listos para entrenar en formato NumPy array.
    x_train, x_dev, y_train, y_dev = divide_data()

    # 2. Identificar qué columnas son continuas y cuáles son categóricas.
    # Miramos el DataFrame 'data' original (cargado globalmente) para leer sus tipos (dtypes).
    x_df = data.drop(columns=[config["column"]])

    # Guardamos la posición (índice) de las columnas que ya son categóricas (enteros o texto).
    cat_cols_idx = [i for i, dtype in enumerate(x_df.dtypes) if dtype in ['int64', 'int32', 'object']]

    # Guardamos la posición (índice) de las columnas numéricas continuas (floats).
    num_cols_idx = [i for i, dtype in enumerate(x_df.dtypes) if dtype in ['float64', 'float32']]

    # 3. ColumnTransformer: El "director de tráfico" de las columnas.
    # Le decimos: "Aplica KBinsDiscretizer SOLO a las columnas numéricas continuas.
    # A las categóricas, déjalas pasar tal cual ('passthrough') para no estropearlas."
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', KBinsDiscretizer(encode='ordinal'), num_cols_idx),
            ('cat', 'passthrough', cat_cols_idx)
        ])

    # 4. Pipeline: La tubería que une el preprocesado y el modelo de clasificación.
    # Esto es VITAL para evitar la "Fuga de Datos" (Data Leakage) al usar GridSearchCV.
    # Asegura que los rangos de las cajas de discretización se calculen estrictamente con los
    # datos de entrenamiento de cada pliegue interno, sin hacer trampa mirando los de validación.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', CategoricalNB())
    ])

    # 5. Búsqueda exhaustiva de hiperparámetros (GridSearchCV).
    # OJO: Como usamos un Pipeline, el GridSearchCV busca combinaciones para toda la tubería a la vez
    # (por eso en el config.json los parámetros se llaman 'preprocessor__num__n_bins' o 'clf__alpha').
    model = GridSearchCV(pipeline, parametros, n_jobs=config.get("cpu", -1), scoring=config["scoring"])

    # 6. Entrenamiento y guardado de resultados.
    model.fit(x_train, y_train)
    save_model(model_output, model)

def mixed_nb(model_output: str, parametros: dict):
    """
    Lógica principal de entrenamiento para Mixed Naive Bayes.

    A diferencia de CategoricalNB, MixedNB maneja inteligentemente datos mixtos por dentro.
    Para las variables continuas asume una curva Gaussiana y para las discretas una Categórica.
    Nuestro único trabajo es chivarle explícitamente cuáles son las columnas categóricas.
    """
    # 1. Obtenemos los datos en formato NumPy array.
    x_train, x_dev, y_train, y_dev = divide_data()

    # 2. Localizar la posición (índices) de las columnas categóricas.
    # Repetimos el proceso de leer el DataFrame original para separar tipos de datos.
    x_df = data.drop(columns=[config["column"]])
    cat_cols_idx = [i for i, dtype in enumerate(x_df.dtypes) if dtype in ['int64', 'int32', 'object']]

    # 3. Inicializar el modelo MixedNB.
    # Calculamos cuántas categorías únicas tiene cada columna categórica.
    # Así evitamos el error de validación cruzada si falta alguna en un pliegue.

    # (Usamos nuestro envoltorio compatible (ScikitMixedNB))
    max_categories = [int(x_df.iloc[:, i].max()) + 1 for i in cat_cols_idx]

    clf = ScikitMixedNB(categorical_features=cat_cols_idx, max_categories=max_categories)

    # 4. Búsqueda exhaustiva de hiperparámetros y entrenamiento.
    # Aquí no usamos Pipeline porque el modelo ya hace la separación internamente.
    model = GridSearchCV(clf, parametros, n_jobs=config.get("cpu", -1), scoring=config["scoring"])
    model.fit(x_train, y_train)

    # 5. Guardado del mejor modelo y reporte de métricas.
    save_model(model_output, model)


# ==========================================
# BLOQUE PRINCIPAL
# ==========================================

if __name__ == '__main__':
    # 1. Configuración de la lectura del archivo .json desde la terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Archivo de configuración.", default="config.json")
    args = parser.parse_args()

    # 2. Cargar el diccionario de configuración
    config = loadConfig(args.config, "train")

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
            print("Modelo KNN entrenado con éxito.")
        elif "decision_tree" in modelo:
            print("Entrenando modelo Decision Tree...")
            decision_tree(modelo["modelo_output"], modelo["parametros"])
            print("Modelo Decision Tree entrenado con éxito.")
        elif "random_forest" in modelo:
            print("Entrenando modelo Random Forest...")
            random_forest(modelo["modelo_output"], modelo["parametros"])
            print("Modelo Random Forest entrenado con éxito.")
        elif "categorical_nb" in modelo:
            print("Entrenando modelo Categorical Naive Bayes con Discretización...")
            categorical_nb(modelo["modelo_output"], modelo["parametros"])
            print("Modelo CategoricalNB entrenado con éxito.")
        elif "mixed_nb" in modelo:
            print("Entrenando modelo Mixed Naive Bayes...")
            mixed_nb(modelo["modelo_output"], modelo["parametros"])
            print("Modelo MixedNB entrenado con éxito.")
    sys.exit(0)