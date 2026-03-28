import os
import argparse
import pandas as pd

# Importaciones de Scikit-Learn para transformaciones y vectorización
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Importaciones de Imblearn para el balanceo de clases
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Importaciones de NLTK para el procesamiento de lenguaje natural (NLP)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from funciones import loadConfig,load_data

# ==========================================
# FUNCIONES MODULARES DE PREPROCESAMIENTO
# Todas reciben tanto Train como Test para aprender de Train y aplicar a ambos.
# ==========================================

def drop_features_func(df_train: pd.DataFrame, df_test: pd.DataFrame, drop_list: list):
    """
    Elimina las columnas especificadas en la lista 'drop_features' del JSON.
    Se eliminan de ambos conjuntos (Train y Test) para mantener la misma estructura.
    """
    if drop_list:
        df_train.drop(columns=drop_list, inplace=True, errors='ignore')
        # Verificamos que df_test no esté vacío antes de intentar borrar
        if not df_test.empty:
            df_test.drop(columns=drop_list, inplace=True, errors='ignore')
        print(f"Columnas eliminadas: {drop_list}")
    return df_train, df_test


def select_features(df_train: pd.DataFrame, target_col: str):
    """
    Identifica dinámicamente qué tipo de dato contiene cada columna.
    Esta identificación se hace SOLO mirando el df_train.
    Separa en: numéricas, de texto libre (NLP) y categóricas normales.
    """
    num_feat = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_feat = df_train.select_dtypes(include=['object', 'string']).columns.tolist()

    # 1. Detectamos texto libre PRIMERO (solo sobre las columnas que realmente son texto/strings)
    text_feat = [col for col in cat_feat if df_train[col].str.len().mean() > 30]

    # 2. El resto de variables tipo object/string se consideran categóricas normales
    cat_feat = [col for col in cat_feat if col not in text_feat]

    # 3. Movemos las variables binarias (0 y 1) a categóricas
    for col in num_feat[:]:
        if df_train[col].nunique() <= 2:
            num_feat.remove(col)
            cat_feat.append(col)


    # PROTECCIÓN CRÍTICA: Quitamos la columna objetivo de estas listas para evitar
    # que se escale, se vectorice o se modifique accidentalmente durante el preprocesado.
    if target_col in num_feat: num_feat.remove(target_col)
    if target_col in cat_feat: cat_feat.remove(target_col)
    if target_col in text_feat: text_feat.remove(target_col)

    return num_feat, text_feat, cat_feat


def process_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, num_feat: list, cat_feat: list, imp_num: str,
                           imp_cat: str):
    """
    Imputa (rellena) los valores nulos (NaN) basándose en las estrategias del JSON.
    Para evitar el Data Leakage, los cálculos se hacen EXCLUSIVAMENTE
    en el df_train, y ese valor se inyecta en Train y Test.
    """

    def obtener_valor_imputacion(df, columna, estrategia):
        # print(estrategia)
        if estrategia == "mean":
            return df[columna].mean()
        elif estrategia == "median":
            return df[columna].median()
        elif estrategia == "mode" and not df[columna].mode().empty:
            return df[columna].mode()[0]
        return None  # Por si la columna está completamente vacía

    # Variables Numéricas
    for feature in num_feat:
        val = obtener_valor_imputacion(df_train, feature, imp_num)
        if val is not None:
            df_train[feature] = df_train[feature].fillna(val)
            if not df_test.empty and feature in df_test.columns:
                df_test[feature] = df_test[feature].fillna(val)

    # Variables Categóricas
    for feature in cat_feat:
        try:
            val = obtener_valor_imputacion(df_train, feature, imp_cat)
        except TypeError:
            # Control de errores: Si piden media/mediana de un string, Pandas fallará.
            print(f"Advertencia: No se puede aplicar '{imp_cat}' a la variable de texto '{feature}'. Se usará 'mode'.")
            val = obtener_valor_imputacion(df_train, feature, "mode")

        if val is not None:
            df_train[feature] = df_train[feature].fillna(val)
            if not df_test.empty and feature in df_test.columns:
                df_test[feature] = df_test[feature].fillna(val)

    return df_train, df_test


def cat2num(df_train: pd.DataFrame, df_test: pd.DataFrame, cat_feat: list):
    """
    Convierte categorías de texto a números enteros (Label Encoding).
    El diccionario de categorías (ej. 'Hombre'->0, 'Mujer'->1) se aprende en Train.
    """
    for feature in cat_feat:
        le = LabelEncoder()
        # FIT (Aprender las categorías) y TRANSFORM (Aplicarlas) en Train
        df_train[feature] = le.fit_transform(df_train[feature].astype(str))

        # Solo TRANSFORM en Test usando el diccionario aprendido en Train
        if not df_test.empty and feature in df_test.columns:
            try:
                df_test[feature] = le.transform(df_test[feature].astype(str))
            except Exception as e:
                # Si en el Test aparece una categoría nueva que no existía en el Train,
                # LabelEncoder fallará. Imprimimos el aviso para depurar.
                print(f"Advertencia en LabelEncoding para test en columna {feature}: {e}")

    return df_train, df_test


def process_text_func(df_train: pd.DataFrame, df_test: pd.DataFrame, text_feat: list, text_process: str):
    """
    Procesamiento de Lenguaje Natural (NLP).
    Aplica TF-IDF o Bag of Words (BoW) según lo indique el JSON.
    El vocabulario (las palabras conocidas) se extrae SOLO del Train.
    """
    if not text_feat or text_process not in ["tf_idf", "bow"]:
        return df_train, df_test

    # Cargamos palabras vacías (stopwords) y el reductor a raíz léxica (stemmer)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Función interna auxiliar para limpiar el texto paso a paso
    def clean_text(text_series):
        # 1. Minúsculas y separar en tokens (palabras sueltas)
        s = text_series.str.lower().apply(word_tokenize)
        # 2. Eliminar stopwords y aplicar stemming
        s = s.apply(lambda x: [stemmer.stem(word) for word in x if word not in stop_words])
        # 3. Volver a unir las palabras limpias en un solo string
        return s.apply(lambda x: ' '.join(x))

    for feature in text_feat:
        # Limpieza básica en ambos datasets
        df_train[feature] = clean_text(df_train[feature])
        if not df_test.empty and feature in df_test.columns:
            df_test[feature] = clean_text(df_test[feature])

        # Instanciar el vectorizador correcto según el config.json
        vectorizer = TfidfVectorizer() if text_process == "tf_idf" else CountVectorizer()

        # FIT (Aprender vocabulario) y TRANSFORM en Train
        matrix_train = vectorizer.fit_transform(df_train[feature])
        df_text_train = pd.DataFrame(matrix_train.toarray(), columns=vectorizer.get_feature_names_out())

        # Concatenar las nuevas columnas vectorizadas y eliminar la columna de texto original
        df_train = pd.concat([df_train.reset_index(drop=True), df_text_train.reset_index(drop=True)], axis=1)
        df_train.drop(columns=[feature], inplace=True)

        # Solo TRANSFORM en Test (las palabras nuevas en test que no estén en train se ignoran)
        if not df_test.empty and feature in df_test.columns:
            matrix_test = vectorizer.transform(df_test[feature])
            df_text_test = pd.DataFrame(matrix_test.toarray(), columns=vectorizer.get_feature_names_out())
            df_test = pd.concat([df_test.reset_index(drop=True), df_text_test.reset_index(drop=True)], axis=1)
            df_test.drop(columns=[feature], inplace=True)

        print(f"Texto de '{feature}' procesado con {text_process}")

    return df_train, df_test


def reescaler(df_train: pd.DataFrame, df_test: pd.DataFrame, num_feat: list):
    """
    Escala las variables numéricas a un rango entre 0 y 1.
    El valor Mínimo y Máximo de cada columna se aprende SOLO del Train.
    """
    if num_feat:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # FIT (Calcular Min/Max) y TRANSFORM en Train
        df_train[num_feat] = scaler.fit_transform(df_train[num_feat])

        # Solo TRANSFORM en Test (Se escalan usando los Min/Max encontrados en el Train)
        if not df_test.empty:
            df_test[num_feat] = scaler.transform(df_test[num_feat])
        print("Características numéricas reescaladas.")

    return df_train, df_test


def over_under_sampling(df_train: pd.DataFrame, target_col: str, sampling: str, random_state: int):
    """
    Aplica balanceo de clases (Oversampling o Undersampling) si hay desbalanceo.
    MUY IMPORTANTE: Se aplica EXCLUSIVAMENTE al df_train. El df_test debe
    mantener su distribución real original para evaluar el modelo de forma justa.
    """
    if sampling in ["oversampling", "undersampling"]:
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]

        if sampling == "undersampling":
            print("Realizando undersampling solo en TRAIN...")
            sampler = RandomUnderSampler(random_state=random_state)
        else:
            print("Realizando oversampling solo en TRAIN...")
            sampler = RandomOverSampler(random_state=random_state)

        # Generar nuevas muestras artificiales o eliminar existentes
        X_res, y_res = sampler.fit_resample(X_train, y_train)

        # Reconstruir el DataFrame de Train con los datos balanceados
        df_train = pd.concat([pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=target_col)], axis=1)

    return df_train


# ==========================================
# ORQUESTADOR DEL PIPELINE
# ==========================================

def preprocesar_datos(config: dict):
    """
    Función maestra que encadena y ejecuta en orden todas las funciones
    modulares de preprocesamiento, pasando ambos DataFrames por la tubería.
    """
    print("Iniciando preprocesamiento modular...")

    # 1. Cargar datos en bruto desde el disco
    df_train = load_data(config["train_dev"])

    # Comprobamos 3 cosas: que la clave "test" exista en el JSON, que no esté vacía,
    # y lo más importante: que el archivo exista FÍSICAMENTE en el disco (os.path.exists).
    ruta_test = config.get("test", "")
    tiene_test = ruta_test != "" and os.path.exists(ruta_test)

    if tiene_test:
        df_test = load_data(ruta_test)
    else:
        print("Aviso: No se encontró archivo de test físico o no se especificó en el JSON. Se procesará solo el Train.")
        df_test = pd.DataFrame()  # Creamos un DataFrame vacío para que el resto del código no falle

    target_col = config["column"]

    # 2. Ejecutar pipeline secuencial (Train y Test fluyen juntos)
    df_train, df_test = drop_features_func(df_train, df_test, config.get("drop_features", []))

    num_feat, text_feat, cat_feat = select_features(df_train, target_col)

    # Extraer las estrategias de imputación del config (con valores por defecto seguros)
    imp_num = config.get("imputacion_numerico", "mean")
    imp_cat = config.get("imputacion_categorico", "mode")
    df_train, df_test = process_missing_values(df_train, df_test, num_feat, cat_feat, imp_num, imp_cat)

    df_train, df_test = cat2num(df_train, df_test, cat_feat)

    df_train, df_test = process_text_func(df_train, df_test, text_feat, config.get("text_process", ""))

    df_train, df_test = reescaler(df_train, df_test, num_feat)

    # 3. Balanceo de clases (Atención: SOLO fluye el Train en este paso)
    df_train = over_under_sampling(df_train, target_col, config.get("sampling", ""), config.get("random_state", 42))

    # 4. Guardar los resultados procesados listos para que train.py y test.py los consuman
    df_train.to_csv(config["train_dev_output"], index=False)
    print(f"Datos de TRAIN limpios guardados en: {config['train_dev_output']}")

    if tiene_test:
        df_test.to_csv(config["test_output"], index=False)
        print(f"Datos de TEST limpios guardados en: {config['test_output']}")


if __name__ == '__main__':
    # Descargar recursos de NLTK de forma silenciosa para no ensuciar la terminal
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Configurar la lectura del archivo .json desde la terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Archivo de configuración.", default="config.json")
    args = parser.parse_args()

    # Cargar el diccionario y disparar la orquestación
    config = loadConfig(args.config,"procesado")
    preprocesar_datos(config)