import json
import csv
import argparse
import pickle
import sys
import numpy as np
import pandas as pd

# Sklearn e Imblearn (Añadidos de la plantilla)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# NLTK para procesamiento de texto
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def loadConfig(file: str) -> dict:
    """
    Función que carga el .json de configuración.
    Genera una configuración específica: general + procesado + train.
    """
    with open(file, 'r', encoding='utf-8') as f:
        config_completa = json.load(f)

    config = {}

    # 1. Extraer 'general'
    general = config_completa.get("general", {})
    for key, value in general.items():
        if key == "data" and isinstance(value, dict):
            for data_key, data_value in value.items():
                config[data_key] = data_value
        else:
            config[key] = value

    # 2. Extraer 'procesado' (NUEVO: Necesario para leer text_process, sampling, drop_features)
    procesado = config_completa.get("procesado", {})
    for key, value in procesado.items():
        config[key] = value

    # 3. Extraer 'train' y filtrar modelos
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
    try:
        data = pd.read_csv(file, encoding=encoding)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        return data
    except UnicodeDecodeError:
        print(f"Error decodificando {encoding}. Intentando con 'latin1'.")
        data = pd.read_csv(file, encoding='latin1')
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        return data


# ==========================================
# BLOQUE DE PREPROCESAMIENTO DINÁMICO
# ==========================================

def drop_features_func(data: pd.DataFrame, drop_list: list) -> pd.DataFrame:
    """Elimina las columnas especificadas en el JSON."""
    if not drop_list:
        return data
    missing_features = [f for f in drop_list if f not in data.columns]
    if missing_features:
        print(f"Advertencia: Estas columnas no existen y no se eliminarán: {missing_features}")
    data.drop(columns=drop_list, inplace=True, errors='ignore')
    print(f"Columnas eliminadas: {drop_list}")
    return data


def select_features(data: pd.DataFrame, target_col: str):
    """Clasifica dinámicamente las columnas, excluyendo la columna objetivo."""
    numerical_feature = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_feature = data.select_dtypes(include=['object', 'string']).columns.tolist()
    text_feature = [col for col in categorical_feature if data[col].str.len().mean() > 30]
    categorical_feature = [col for col in categorical_feature if col not in text_feature]

    # Asegurarnos de que la columna a predecir no se preprocese como una feature normal
    if target_col in numerical_feature: numerical_feature.remove(target_col)
    if target_col in categorical_feature: categorical_feature.remove(target_col)
    if target_col in text_feature: text_feature.remove(target_col)

    return numerical_feature, text_feature, categorical_feature


def process_missing_values(data: pd.DataFrame, num_feat: list, cat_feat: list) -> pd.DataFrame:
    for feature in num_feat:
        data[feature] = data[feature].fillna(data[feature].mean())
    for feature in cat_feat:
        if not data[feature].mode().empty:
            data[feature] = data[feature].fillna(data[feature].mode()[0])
    return data


def cat2num(data: pd.DataFrame, cat_feat: list) -> pd.DataFrame:
    le = LabelEncoder()
    for feature in cat_feat:
        data[feature] = le.fit_transform(data[feature].astype(str))
    return data


def process_text_func(data: pd.DataFrame, text_feat: list, text_process: str) -> pd.DataFrame:
    if not text_feat or not text_process:
        return data

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for feature in text_feat:
        # Simplificación básica
        data[feature] = data[feature].str.lower().apply(word_tokenize)
        data[feature] = data[feature].apply(lambda x: [stemmer.stem(word) for word in x if word not in stop_words])
        data[feature] = data[feature].apply(lambda x: ' '.join(x))

        # Vectorización
        if text_process == "tf_idf":
            vectorizer = TfidfVectorizer()
        elif text_process == "bow":
            vectorizer = CountVectorizer()
        else:
            continue

        matrix = vectorizer.fit_transform(data[feature])
        df_text = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
        data = pd.concat([data.reset_index(drop=True), df_text.reset_index(drop=True)], axis=1)
        data.drop(columns=[feature], inplace=True)
        print(f"Texto de '{feature}' procesado con {text_process}")
    return data


def over_under_sampling_func(data: pd.DataFrame, target_col: str, sampling: str, random_state: int) -> pd.DataFrame:
    if not sampling or sampling not in ["oversampling", "undersampling"]:
        return data

    X = data.drop(columns=[target_col])
    y = data[target_col]

    if sampling == "undersampling":
        print("Realizando undersampling...")
        sampler = RandomUnderSampler(random_state=random_state)
    elif sampling == "oversampling":
        print("Realizando oversampling...")
        sampler = RandomOverSampler(random_state=random_state)

    X_res, y_res = sampler.fit_resample(X, y)
    data_resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)
    return data_resampled


def preprocesar_datos(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Pipeline centralizado de preprocesamiento, 100% dirigido por el JSON."""
    print("Iniciando preprocesamiento...")
    target_col = config["column"]

    # 1. Eliminar features indicadas en el JSON
    data = drop_features_func(data, config.get("drop_features", []))

    # 2. Seleccionar tipos de datos
    num_feat, text_feat, cat_feat = select_features(data, target_col)

    # 3. Imputar nulos
    data = process_missing_values(data, num_feat, cat_feat)

    # 4. Transformar categóricas
    data = cat2num(data, cat_feat)

    # 5. Procesar texto (TF-IDF o BoW según JSON)
    data = process_text_func(data, text_feat, config.get("text_process", ""))

    # 6. Escalar numéricas
    if num_feat:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[num_feat] = scaler.fit_transform(data[num_feat])
        print("Características numéricas reescaladas.")

    # 7. Sampling (Oversampling/Undersampling según JSON)
    data = over_under_sampling_func(data, target_col, config.get("sampling", ""), config.get("random_state", 42))

    print("Preprocesamiento finalizado.")
    return data


# ==========================================
# ENTRENAMIENTO Y UTILIDADES
# ==========================================

def divide_data():
    global data
    x = data.drop(columns=[config["column"]])
    y = data[config["column"]]

    y = LabelEncoder().fit_transform(y)
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=config["dev"], stratify=y,
                                                      random_state=config["random_state"])

    x_train = x_train.values
    x_dev = x_dev.values

    return x_train, x_dev, y_train, y_dev


def save_model(model_output: str, model):
    try:
        with open(f"{model_output.rsplit('.', 1)[0]}.pkl", "wb") as file:
            pickle.dump(model, file)
            print(f"Modelo guardado en: {model_output.rsplit('.', 1)[0]}.pkl")

        with open(f"{model_output.rsplit('.', 1)[0]}.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Params", "Score"])
            for params, score in zip(model.cv_results_['params'], model.cv_results_["mean_test_score"]):
                writer.writerow([params, score])
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")


def knn(model_output: str, parametros: dict):
    x_train, x_dev, y_train, y_dev = divide_data()
    model = GridSearchCV(KNeighborsClassifier(), parametros, n_jobs=config["cpu"])
    model.fit(x_train, y_train)
    save_model(model_output, model)


# Resto de funciones (decision_tree, random_forest, naive_bayes) permanecen igual...
def decision_tree(model_output: str, parametros: dict):
    pass


def random_forest(model_output: str, parametros: dict):
    pass


def naive_bayes(model_output: str, parametros: dict):
    pass


if __name__ == '__main__':
    # Descargar recursos de NLTK de forma silenciosa si no existen
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Directorio del archivo de configuración.",
                        default="config.json")
    args = parser.parse_args()

    # 1. Cargar Configuración Total
    config = loadConfig(args.config)

    # 2. Cargar Datos
    data = load_data(config["train_dev"])

    # 3. Ejecutar Pipeline de Preprocesamiento guiado por el JSON
    data = preprocesar_datos(data, config)

    # Guardar los datos procesados (opcional, pero útil para depurar)
    data.to_csv(config["train_dev_output"], index=False)
    print(f"Datos limpios guardados en: {config['train_dev_output']}")

    # 4. Entrenar modelos activos
    for modelo in config["modelos"]:
        if "knn" in modelo:
            knn(modelo["modelo_output"], modelo["parametros"])
        elif "decision_tree" in modelo:
            decision_tree(modelo["modelo_output"], modelo["parametros"])
        elif "random_forest" in modelo:
            random_forest(modelo["modelo_output"], modelo["parametros"])
        elif "naive_bayes" in modelo:
            naive_bayes(modelo["modelo_output"], modelo["parametros"])