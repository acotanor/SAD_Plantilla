import json
import csv
import argparse
import pickle
import pandas as pd

# Importaciones de Scikit-Learn para modelos, métricas y preprocesamiento
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Importaciones de Imblearn para técnicas de balanceo de clases
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Importaciones de NLTK para procesamiento de lenguaje natural (texto libre)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#    ____                                    _             _        ____        _
#   |  _ \ _ __ ___   ___ ___  ___  __ _  __| | ___     __| | ___  |  _ \  __ _| |_ ___  ___
#   | |_) | '__/ _ \ / __/ _ \/ __|/ _` |/ _` |/ _ \   / _` |/ _ \ | | | |/ _` | __/ _ \/ __|
#   |  __/| | | (_) | (_|  __/\__ \ (_| | (_| | (_) | | (_| |  __/ | |_| | (_| | || (_) \__ \
#   |_|   |_|  \___/ \___\___||___/\__,_|\__,_|\___/   \__,_|\___| |____/ \__,_|\__\___/|___/


def loadConfig(file: str) -> dict:
    """
    Función que carga el archivo .json de configuración.
    Se encarga de leer las secciones 'general', 'procesado' y 'train',
    y las unifica en un solo diccionario plano más fácil de manejar.
    """
    with open(file, 'r', encoding='utf-8') as f:
        config_completa = json.load(f)

    config = {}

    # 1. Extraer la sección 'general' (rutas de datos, variable objetivo, etc.)
    general = config_completa.get("general", {})
    for key, value in general.items():
        if key == "data" and isinstance(value, dict):
            # Aplanamos el diccionario interno 'data' para que sus claves pasen al nivel principal
            for data_key, data_value in value.items():
                config[data_key] = data_value
        else:
            config[key] = value

    # 2. Extraer la sección 'procesado' (sampling, text_process, drop_features)
    procesado = config_completa.get("procesado", {})
    for key, value in procesado.items():
        config[key] = value

    # 3. Extraer la sección 'train' y filtrar para quedarnos solo con los modelos activos (true)
    train = config_completa.get("train", {})
    for key, value in train.items():
        if key == "modelos" and isinstance(value, list):
            modelos_activos = []
            # Iteramos sobre la lista de modelos para comprobar cuál tiene el valor 'true'
            for modelo in value:
                es_activo = any(v is True for k, v in modelo.items() if k != "parametros")
                if es_activo:
                    modelos_activos.append(modelo)
            config["modelos"] = modelos_activos  # Guardamos solo los que vamos a entrenar
        else:
            config[key] = value

    return config


def load_data(file: str, encoding='utf-8') -> pd.DataFrame:
    """
    Carga el dataset CSV en un DataFrame de Pandas.
    Además, gestiona posibles errores de codificación de caracteres e ignora
    las típicas columnas vacías 'Unnamed: 0' que se generan al guardar CSVs.
    """
    try:
        data = pd.read_csv(file, encoding=encoding)
        # Filtramos columnas que empiecen por "Unnamed"
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        return data
    except UnicodeDecodeError:
        # Fallback de seguridad si el archivo fue guardado con otra codificación en Windows
        print(f"Error decodificando {encoding}. Intentando con 'latin1'.")
        data = pd.read_csv(file, encoding='latin1')
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        return data


# ==========================================
# BLOQUE DE PREPROCESAMIENTO DINÁMICO
# Este bloque depende al 100% de lo que dictamine el config.json
# ==========================================

def drop_features_func(data: pd.DataFrame, drop_list: list) -> pd.DataFrame:
    """Elimina las columnas indicadas explícitamente en el JSON (drop_features)."""
    if not drop_list:
        return data  # Si la lista está vacía, devuelve el dataset intacto

    # Comprobamos si nos piden borrar una columna que no existe en el dataset
    missing_features = [f for f in drop_list if f not in data.columns]
    if missing_features:
        print(f"Advertencia: Estas columnas no existen y no se eliminarán: {missing_features}")

    # Eliminamos las columnas; errors='ignore' evita crasheos si la columna no existe
    data.drop(columns=drop_list, inplace=True, errors='ignore')
    print(f"Columnas eliminadas: {drop_list}")
    return data


def select_features(data: pd.DataFrame, target_col: str):
    """
    Clasifica automáticamente todas las columnas del dataset en tres grupos:
    Numéricas, Categóricas y de Texto. Protege la variable objetivo (target_col).
    """
    # 1. Detectar numéricas (enteros o decimales)
    numerical_feature = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 2. Detectar categóricas (cadenas de texto u objetos)
    categorical_feature = data.select_dtypes(include=['object', 'string']).columns.tolist()

    # 3. Separar las categóricas normales de las que son texto largo (NLP)
    # Consideramos texto libre si la media de longitud supera los 30 caracteres
    text_feature = [col for col in categorical_feature if data[col].str.len().mean() > 30]

    # Quitamos el texto libre de la lista de categóricas normales
    categorical_feature = [col for col in categorical_feature if col not in text_feature]

    # MUY IMPORTANTE: Sacar la variable a predecir de estas listas para no alterarla
    if target_col in numerical_feature: numerical_feature.remove(target_col)
    if target_col in categorical_feature: categorical_feature.remove(target_col)
    if target_col in text_feature: text_feature.remove(target_col)

    return numerical_feature, text_feature, categorical_feature


def process_missing_values(data: pd.DataFrame, num_feat: list, cat_feat: list) -> pd.DataFrame:
    """Imputa (rellena) los valores nulos/vacíos en el dataset."""
    # Para numéricas: rellenamos con la Media (mean) de la columna
    for feature in num_feat:
        data[feature] = data[feature].fillna(data[feature].mean())

    # Para categóricas: rellenamos con la Moda (mode), es decir, el valor más repetido
    for feature in cat_feat:
        if not data[feature].mode().empty:
            data[feature] = data[feature].fillna(data[feature].mode()[0])

    return data


def cat2num(data: pd.DataFrame, cat_feat: list) -> pd.DataFrame:
    """Convierte las categorías de texto (ej. 'Hombre', 'Mujer') en números (ej. 0, 1)."""
    le = LabelEncoder()
    for feature in cat_feat:
        # Se fuerza la conversión a string por si hubiese algún dato corrupto
        data[feature] = le.fit_transform(data[feature].astype(str))
    return data


def process_text_func(data: pd.DataFrame, text_feat: list, text_process: str) -> pd.DataFrame:
    """Aplica procesamiento de lenguaje natural (NLP) a las columnas de texto libre."""
    if not text_feat or not text_process:
        return data  # Salimos si no hay columnas de texto o no se pidió en el JSON

    # Cargamos palabras vacías en inglés (the, is, at...) y el reductor a raíz (stemmer)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for feature in text_feat:
        # FASE 1: Limpieza del texto
        # Pasar a minúsculas y separar en tokens (palabras individuales)
        data[feature] = data[feature].str.lower().apply(word_tokenize)
        # Eliminar stopwords y dejar solo la raíz léxica de cada palabra
        data[feature] = data[feature].apply(lambda x: [stemmer.stem(word) for word in x if word not in stop_words])
        # Volver a unir las palabras en una sola frase limpia
        data[feature] = data[feature].apply(lambda x: ' '.join(x))

        # FASE 2: Vectorización (Convertir el texto limpio a una matriz numérica)
        if text_process == "tf_idf":
            vectorizer = TfidfVectorizer()
        elif text_process == "bow":
            vectorizer = CountVectorizer()  # BoW es simplemente contar la frecuencia
        else:
            continue

        # Transformamos el texto y lo convertimos en un DataFrame temporal
        matrix = vectorizer.fit_transform(data[feature])
        df_text = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Unimos la matriz de texto al dataset original y borramos la columna de texto vieja
        data = pd.concat([data.reset_index(drop=True), df_text.reset_index(drop=True)], axis=1)
        data.drop(columns=[feature], inplace=True)
        print(f"Texto de '{feature}' procesado con {text_process}")

    return data


def over_under_sampling_func(data: pd.DataFrame, target_col: str, sampling: str, random_state: int) -> pd.DataFrame:
    """Equilibra el dataset si una de las clases a predecir es muy minoritaria."""
    if not sampling or sampling not in ["oversampling", "undersampling"]:
        return data

    # Separamos temporalmente las features (X) de la variable objetivo (y)
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Elegimos el algoritmo según el JSON
    if sampling == "undersampling":
        print("Realizando undersampling...")
        sampler = RandomUnderSampler(random_state=random_state)
    elif sampling == "oversampling":
        print("Realizando oversampling...")
        sampler = RandomOverSampler(random_state=random_state)

    # Aplicamos el balanceo y volvemos a juntar (X) e (y) en un solo DataFrame
    X_res, y_res = sampler.fit_resample(X, y)
    data_resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)

    return data_resampled


def preprocesar_datos(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Función orquestadora. Llama una por una a todas las funciones de limpieza
    siguiendo estrictamente la configuración cargada del JSON.
    """
    print("Iniciando preprocesamiento...")
    target_col = config["column"]

    # 1. Eliminar features inútiles
    data = drop_features_func(data, config.get("drop_features", []))

    # 2. Identificar qué tipo de variable es cada columna restante
    num_feat, text_feat, cat_feat = select_features(data, target_col)

    # 3. Limpiar valores nulos
    data = process_missing_values(data, num_feat, cat_feat)

    # 4. Pasar texto estructurado a números (Label Encoding)
    data = cat2num(data, cat_feat)

    # 5. Aplicar NLP al texto no estructurado (TF-IDF / BoW)
    data = process_text_func(data, text_feat, config.get("text_process", ""))

    # 6. Normalizar variables numéricas para que todas estén entre 0 y 1
    # Esto es crucial para modelos basados en distancias como KNN
    if num_feat:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[num_feat] = scaler.fit_transform(data[num_feat])
        print("Características numéricas reescaladas.")

    # 7. Balancear las clases a predecir (Oversampling / Undersampling)
    data = over_under_sampling_func(data, target_col, config.get("sampling", ""), config.get("random_state", 42))

    print("Preprocesamiento finalizado.")
    return data


# ==========================================
# BLOQUE DE ENTRENAMIENTO Y UTILIDADES
# ==========================================

def divide_data():
    """
    Divide el dataset ya preprocesado en dos conjuntos: Train (entrenamiento) y Dev (validación).
    Además, aplica LabelEncoding a la variable a predecir (y).
    """
    global data  # Accedemos a la variable global generada en el bloque __main__
    x = data.drop(columns=[config["column"]])
    y = data[config["column"]]

    # Convertimos la etiqueta a predecir a valores numéricos
    y = LabelEncoder().fit_transform(y)

    # stratify=y asegura que la proporción de clases se mantenga igual en Train y en Dev
    x_train, x_dev, y_train, y_dev = train_test_split(
        x, y, test_size=config["dev"], stratify=y, random_state=config["random_state"]
    )

    # Pasamos a arrays de numpy para evitar problemas con los nombres de las columnas en Sklearn
    x_train = x_train.values
    x_dev = x_dev.values

    return x_train, x_dev, y_train, y_dev


def save_model(model_output: str, model):
    """
    Guarda en el disco duro el mejor modelo encontrado (.pkl) y genera un
    reporte en CSV con los resultados de todas las combinaciones de hiperparámetros.
    """
    try:
        # Se usa rsplit para quitar la extensión que venga en el JSON (.pickle o .pkl) y asegurar un estándar
        with open(f"{model_output.rsplit('.', 1)[0]}.pkl", "wb") as file:
            pickle.dump(model, file)
            print(f"Modelo guardado en: {model_output.rsplit('.', 1)[0]}.pkl")

        # Generación del archivo de reporte CSV con métricas (cv_results_)
        with open(f"{model_output.rsplit('.', 1)[0]}.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Params", "Score"])
            for params, score in zip(model.cv_results_['params'], model.cv_results_["mean_test_score"]):
                writer.writerow([params, score])
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

def knn(model_output: str, parametros: dict):
    """
    Lógica de entrenamiento para K-Nearest Neighbors (KNN).
    Utiliza GridSearchCV para automatizar el barrido de hiperparámetros.
    """
    x_train, x_dev, y_train, y_dev = divide_data()

    # GridSearchCV prueba TODAS las combinaciones de hiperparámetros posibles definidas en el JSON
    model = GridSearchCV(KNeighborsClassifier(), parametros, n_jobs=config["cpu"])
    model.fit(x_train, y_train)

    # Guardamos el mejor modelo
    save_model(model_output, model)

def decision_tree(model_output: str, parametros: dict):
    pass


def random_forest(model_output: str, parametros: dict):
    pass


def naive_bayes(model_output: str, parametros: dict):
    pass


# ==========================================
# BLOQUE PRINCIPAL (PUNTO DE ENTRADA)
# ==========================================

if __name__ == '__main__':
    # 1. Descarga silenciosa de recursos lingüísticos de NLTK para no saturar la terminal
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

    # 2. Configuración de parámetros por línea de comandos (Argparse)
    parser = argparse.ArgumentParser()
    # Permitimos indicar por terminal qué JSON cargar (-c config_pruebas.json).
    # Si no se indica nada, coge "config.json" por defecto.
    parser.add_argument("-c", "--config", type=str, help="Directorio del archivo de configuración.",
                        default="config.json")
    args = parser.parse_args()

    # 3. Cargar el diccionario de configuración
    config = loadConfig(args.config)

    # 4. Cargar el dataset en bruto especificado en 'train_dev'
    data = load_data(config["train_dev"])

    # 5. Ejecutar limpieza y preparación de los datos
    data = preprocesar_datos(data, config)

    # 6. Guardar los datos ya procesados en disco
    data.to_csv(config["train_dev_output"], index=False)
    print(f"Datos limpios guardados en: {config['train_dev_output']}")

    # 7. Ciclo de Entrenamiento: Iteramos la lista de "modelos" del JSON
    # Solo los modelos que tengan "true" habrán sobrevivido a la función loadConfig()
    for modelo in config["modelos"]:
        if "knn" in modelo:
            knn(modelo["modelo_output"], modelo["parametros"])
        elif "decision_tree" in modelo:
            decision_tree(modelo["modelo_output"], modelo["parametros"])
        elif "random_forest" in modelo:
            random_forest(modelo["modelo_output"], modelo["parametros"])
        elif "naive_bayes" in modelo:
            naive_bayes(modelo["modelo_output"], modelo["parametros"])