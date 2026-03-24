import os
import sys
import json
import argparse

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Funciones auxiliares sin "global data"

def select_features(X):
    """Separa las características en numéricas, de texto y categóricas."""
    try:
        numerical_feature = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_feature = X.select_dtypes(include='object').columns.tolist()

        # Detectar columnas con texto largo
        text_feature = [col for col in categorical_feature if X[col].str.len().mean() > 30]
        # Excluir texto de las categóricas
        categorical_feature = [col for col in categorical_feature if col not in text_feature]

        print("Características numéricas identificadas:", numerical_feature)
        print("Características categóricas identificadas:", categorical_feature)
        print("Características de texto identificadas:", text_feature)

        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(f"Error al separar los datos: {e}")
        sys.exit(1)


def drop_features(X, features_to_drop):
    """Elimina las columnas indicadas en la configuración."""
    try:
        missing_features = [f for f in features_to_drop if f not in X.columns]
        if missing_features:
            print(f"Advertencia: Estas columnas no existen y no se eliminarán: {missing_features}")

        X = X.drop(columns=features_to_drop, errors='ignore')
        print(f"Columnas eliminadas: {features_to_drop}")
        return X
    except Exception as e:
        print(f"Error al eliminar columnas: {e}")
        sys.exit(1)


def process_missing_values(X, numerical_feature, categorical_feature):
    """Rellena nulos con media (numéricas) y moda (categóricas)."""
    for feature in numerical_feature:
        X[feature] = X[feature].fillna(X[feature].mean())
    for feature in categorical_feature:
        X[feature] = X[feature].fillna(X[feature].mode()[0])
    return X


def cat2num(X, categorical_feature):
    """Codifica variables categóricas empleando LabelEncoder."""
    le = LabelEncoder()
    for feature in categorical_feature:
        X[feature] = le.fit_transform(X[feature].astype(str))
    return X


def reescaler(X, numerical_feature):
    """Reescala características numéricas usando MinMaxScaler."""
    if not numerical_feature:
        print("No se encontraron características numéricas para reescalar.")
        return X

    scaler = MinMaxScaler(feature_range=(0, 1))
    X[numerical_feature] = scaler.fit_transform(X[numerical_feature])
    return X


def simplify_text(X, text_feature):
    """Simplifica texto (minúsculas, tokens, stopwords, stemming)."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for feature in text_feature:
        X[feature] = X[feature].astype(str).str.lower()
        X[feature] = X[feature].apply(word_tokenize)
        X[feature] = X[feature].apply(lambda x: [stemmer.stem(word) for word in x if word not in stop_words])
        X[feature] = X[feature].apply(lambda x: ' '.join(x))
    return X


def process_text(X, text_feature, text_process):
    """Aplica TF-IDF o BoW a las columnas de texto."""
    if not text_feature:
        return X

    for feature in text_feature:
        if text_process == "tf_idf":
            vectorizer = TfidfVectorizer()
            print(f"Procesando '{feature}' usando TF-IDF...")
        elif text_process == "bow":
            vectorizer = CountVectorizer()
            print(f"Procesando '{feature}' usando Bag of Words...")
        else:
            print(f"No se procesará el texto (estrategia '{text_process}' no reconocida).")
            continue

        matrix = vectorizer.fit_transform(X[feature])
        df_text = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=X.index)

        # Concatenar y eliminar la columna original
        X = pd.concat([X, df_text], axis=1)
        X = X.drop(columns=[feature])

    return X


def over_under_sampling(X, y, sampling_strategy, random_state):
    """Aplica técnicas de balanceo de clases."""
    if sampling_strategy == "undersampling":
        print("Realizando undersampling...")
        sampler = RandomUnderSampler(random_state=random_state)
    elif sampling_strategy == "oversampling":
        print("Realizando oversampling...")
        sampler = RandomOverSampler(random_state=random_state)
    else:
        print(f"Sampling '{sampling_strategy}' no reconocido o vacío. No se aplicará balanceo.")
        return X, y

    X_res, y_res = sampler.fit_resample(X, y)
    print(f"Balanceo completado. Nuevas dimensiones de X: {X_res.shape}")
    return X_res, y_res


# --- PIPELINE PRINCIPAL ---

def preprocesar_datos(data, config):
    """
    Pipeline principal de preprocesamiento orquestado por config.json.
    """
    print("\n--- Iniciando Preprocesamiento ---")

    # 1. Extraer configuraciones del JSON
    target_col = config['general']['column']
    random_state = config['general']['random_state']
    drop_cols = config['procesado']['drop_features']
    text_process_strategy = config['procesado']['text_process']
    sampling_strategy = config['procesado']['sampling']

    # 2. Separar Target (y) de Features (X) temprano para no alterar la variable objetivo
    if target_col not in data.columns:
        print(f"Error: La columna objetivo '{target_col}' no se encuentra en el dataset.")
        sys.exit(1)

    y = data[target_col]
    X = data.drop(columns=[target_col])

    # 3. Eliminar columnas explícitas en config
    if drop_cols:
        X = drop_features(X, drop_cols)

    # Limpiar posibles columnas residuales (como Unnamed)
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]

    # 4. Clasificar tipos de variables
    num_feat, text_feat, cat_feat = select_features(X)

    # 5. Tratamiento de nulos
    X = process_missing_values(X, num_feat, cat_feat)

    # 6. Codificación Categórica
    if cat_feat:
        X = cat2num(X, cat_feat)

    # 7. Procesamiento de Texto
    if text_feat:
        X = simplify_text(X, text_feat)
        X = process_text(X, text_feat, text_process_strategy)

    # 8. Escalado de Numéricas
    if num_feat:
        print("Reescalando características numéricas...")
        X = reescaler(X, num_feat)

    # 9. Sampling (balanceo de clases)
    X, y = over_under_sampling(X, y, sampling_strategy, random_state)

    print("--- Preprocesamiento finalizado ---")
    return X, y


def load_config(file_path):
    """Carga el archivo de configuración JSON en un diccionario."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de configuración en {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: El archivo {file_path} no tiene un formato JSON válido.")
        sys.exit(1)


if __name__ == '__main__':
    # 1. Configurar los argumentos de línea de comandos (Requisito de la práctica)
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento de Modelos de ML.")
    parser.add_argument("-c", "--config", type=str, help="Directorio del archivo de configuración JSON.",
                        default="config.json")
    args = parser.parse_args()

    # 2. Cargar la configuración general
    print(f"Cargando configuración desde: {args.config}...")
    config = load_config(args.config)

    # 3. Cargar los datos sin hardcodear el nombre del archivo
    train_data_path = config['general']['data']['train_dev']
    print(f"Cargando datos desde: {train_data_path}...")

    try:
        data = pd.read_csv(train_data_path)
    except Exception as e:
        print(f"Error fatal al cargar los datos: {e}")
        sys.exit(1)

    # 4. Llamar a nuestro pipeline robusto de preprocesado
    X, y = preprocesar_datos(data, config)

    # 5. Guardar el dataset procesado (opcional pero muy útil para debugear)
    output_path = config['general']['data'].get('train_dev_output', 'traindev_procesado.csv')
    df_procesado = pd.concat([X, y], axis=1)
    df_procesado.to_csv(output_path, index=False)
    print(f"Dataset preprocesado guardado en: {output_path}")

    # 6. Dividir en Train y Dev (validación) leyendo el % desde el JSON
    dev_size = config['train']['dev']
    random_state = config['general']['random_state']

    print(f"\nDividiendo los datos en Train y Dev (Dev size: {dev_size * 100}%)...")
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=dev_size, random_state=random_state)
    print(f"Dimensiones -> X_train: {X_train.shape}, X_dev: {X_dev.shape}")

    # 7. Bucle de Entrenamiento Dinámico
    print("\n--- Iniciando Bucle de Entrenamiento ---")
    modelos_config = config['train']['modelos']

    for modelo_info in modelos_config:
        # Verificamos si la clave del modelo está a "true" en el JSON
        if modelo_info.get('knn', False):
            print("\n>> Configurando y entrenando KNN...")
            params_knn = modelo_info['parametros']
            ruta_salida = modelo_info['modelo_output']

            # Corrección preventiva: scikit-learn espera "weights" no "weight" (como aparece en tu config.json) TODO
            if 'weight' in params_knn:
                params_knn['weights'] = params_knn.pop('weight')

            # Definimos las múltiples figuras de mérito que pide el enunciado
            scoring_metrics = {
                'Accuracy': 'accuracy',
                'Precision': 'precision_macro',
                'Recall': 'recall_macro',
                'F_score': 'f1_macro'
            }

            knn = KNeighborsClassifier()

            # refit='F_score' indica que, tras probar todo, re-entrenará el mejor modelo basándose en el F-score
            cpu_cores = config['train'].get('cpu', -1)
            grid_search_knn = GridSearchCV(
                estimator=knn,
                param_grid=params_knn,
                cv=5,
                scoring=scoring_metrics,
                refit='F_score',
                n_jobs=cpu_cores
            )

            print("Ejecutando barrido de hiperparámetros. Esto puede tardar...")
            grid_search_knn.fit(X_train, y_train)

            # --- Extraer y guardar métricas en CSV (Requisito del PDF) ---
            resultados_cv = pd.DataFrame(grid_search_knn.cv_results_)

            # Filtramos solo las columnas de interés para el CSV
            cols_to_save = ['params', 'mean_test_Accuracy', 'mean_test_Precision', 'mean_test_Recall',
                            'mean_test_F_score', 'rank_test_F_score']
            csv_knn = resultados_cv[cols_to_save].sort_values(by='rank_test_F_score')

            # Renombramos para ajustarnos a la tabla del PDF
            csv_knn = csv_knn.rename(columns={
                'params': 'Combinación',
                'mean_test_Accuracy': 'Accuracy',
                'mean_test_Precision': 'Precisión',
                'mean_test_Recall': 'Recall',
                'mean_test_F_score': 'F_score(Macro)'
            })

            nombre_csv_resultados = "knn_resultados_barrido.csv"
            csv_knn.to_csv(nombre_csv_resultados, index=False)
            print(f"✅ Tabla de figuras de mérito guardada en: {nombre_csv_resultados}")

            # --- Mejor modelo ---
            best_knn = grid_search_knn.best_estimator_
            print(f"🏆 Mejores hiperparámetros encontrados: {grid_search_knn.best_params_}")

            # Evaluar el mejor modelo en el conjunto Dev
            print("\nEvaluando el mejor KNN en el conjunto Dev (Validación):")
            y_pred_dev = best_knn.predict(X_dev)

            # Generamos un reporte completo para verlo por consola
            print(classification_report(y_dev, y_pred_dev))

            # --- Guardar el modelo con pickle ---
            # Asegurarse de que el directorio existe antes de guardar (por si la ruta es "modelos/knn_BestModel.pickle")
            os.makedirs(os.path.dirname(ruta_salida) or '.', exist_ok=True)

            with open(ruta_salida, 'wb') as f:
                pickle.dump(best_knn, f)
            print(f"💾 Mejor modelo guardado correctamente empleando Pickle en: {ruta_salida}")


        elif modelo_info.get('random_forest', False):
            print("\n>> Configurando y entrenando Random Forest...")
            params_rf = modelo_info['parametros'].copy()
            ruta_salida = modelo_info['modelo_output']

            # Corrección preventiva: scikit-learn espera booleanos (True/False) en bootstrap,
            # pero tu JSON los tiene como strings ("True"/"False"). Los convertimos al vuelo:

            if 'bootstrap' in params_rf:
                params_rf['bootstrap'] = [True if str(b).lower() == 'true' else False for b in params_rf['bootstrap']]

            scoring_metrics = {
                'Accuracy': 'accuracy',
                'Precision': 'precision_macro',
                'Recall': 'recall_macro',
                'F_score': 'f1_macro'
            }

            # Añadimos random_state para que tus resultados sean reproducibles
            rf = RandomForestClassifier(random_state=random_state)
            cpu_cores = config['train'].get('cpu', -1)

            grid_search_rf = GridSearchCV(
                estimator=rf,
                param_grid=params_rf,
                cv=5,
                scoring=scoring_metrics,
                refit='F_score',
                n_jobs=cpu_cores
            )

            print("Ejecutando barrido de hiperparámetros. Esto puede tardar bastante...")
            grid_search_rf.fit(X_train, y_train)

            # --- Extraer y guardar métricas en CSV ---
            resultados_cv = pd.DataFrame(grid_search_rf.cv_results_)
            cols_to_save = ['params', 'mean_test_Accuracy', 'mean_test_Precision', 'mean_test_Recall',
                            'mean_test_F_score', 'rank_test_F_score']
            csv_rf = resultados_cv[cols_to_save].sort_values(by='rank_test_F_score')

            csv_rf = csv_rf.rename(columns={
                'params': 'Combinación',
                'mean_test_Accuracy': 'Accuracy',
                'mean_test_Precision': 'Precisión',
                'mean_test_Recall': 'Recall',
                'mean_test_F_score': 'F_score(Macro)'
            })

            nombre_csv = "random_forest_resultados_barrido.csv"
            csv_rf.to_csv(nombre_csv, index=False)
            print(f"✅ Tabla de figuras de mérito guardada en: {nombre_csv}")

            # --- Mejor modelo ---
            best_rf = grid_search_rf.best_estimator_
            print(f"🏆 Mejores hiperparámetros encontrados: {grid_search_rf.best_params_}")

            print("\nEvaluando el mejor Random Forest en el conjunto Dev (Validación):")
            y_pred_dev = best_rf.predict(X_dev)
            print(classification_report(y_dev, y_pred_dev))

            os.makedirs(os.path.dirname(ruta_salida) or '.', exist_ok=True)
            with open(ruta_salida, 'wb') as f:
                pickle.dump(best_rf, f)
            print(f"💾 Mejor modelo guardado correctamente empleando Pickle en: {ruta_salida}")

        elif modelo_info.get('decision_tree', False):
            print("\n>> Configurando y entrenando Decision Tree...")
            params_dt = modelo_info['parametros']
            ruta_salida = modelo_info['modelo_output']

            # Nota: El PDF menciona max_depth = 3, 6, 9 para Decision Tree, pero tu JSON usa [0, 5, 10, 15, 20].
            # Al pasarlo dinámicamente con param_grid=params_dt, respetamos el JSON y evitamos "hardcodearlo".

            scoring_metrics = {
                'Accuracy': 'accuracy',
                'Precision': 'precision_macro',
                'Recall': 'recall_macro',
                'F_score': 'f1_macro'
            }

            dt = DecisionTreeClassifier(random_state=random_state)
            cpu_cores = config['train'].get('cpu', -1)
            grid_search_dt = GridSearchCV(
                estimator=dt,
                param_grid=params_dt,
                cv=5,
                scoring=scoring_metrics,
                refit='F_score',
                n_jobs=cpu_cores
            )

            print("Ejecutando barrido de hiperparámetros...")
            grid_search_dt.fit(X_train, y_train)

            # --- Extraer y guardar métricas en CSV ---
            resultados_cv = pd.DataFrame(grid_search_dt.cv_results_)
            cols_to_save = ['params', 'mean_test_Accuracy', 'mean_test_Precision', 'mean_test_Recall',
                            'mean_test_F_score', 'rank_test_F_score']
            csv_dt = resultados_cv[cols_to_save].sort_values(by='rank_test_F_score')

            csv_dt = csv_dt.rename(columns={
                'params': 'Combinación',
                'mean_test_Accuracy': 'Accuracy',
                'mean_test_Precision': 'Precisión',
                'mean_test_Recall': 'Recall',
                'mean_test_F_score': 'F_score(Macro)'
            })

            nombre_csv = "decision_tree_resultados_barrido.csv"
            csv_dt.to_csv(nombre_csv, index=False)
            print(f"✅ Tabla de figuras de mérito guardada en: {nombre_csv}")

            # --- Mejor modelo ---
            best_dt = grid_search_dt.best_estimator_
            print(f"🏆 Mejores hiperparámetros encontrados: {grid_search_dt.best_params_}")

            print("\nEvaluando el mejor Decision Tree en el conjunto Dev (Validación):")
            y_pred_dev = best_dt.predict(X_dev)
            print(classification_report(y_dev, y_pred_dev))

            os.makedirs(os.path.dirname(ruta_salida) or '.', exist_ok=True)
            with open(ruta_salida, 'wb') as f:
                pickle.dump(best_dt, f)
            print(f"💾 Mejor modelo guardado correctamente empleando Pickle en: {ruta_salida}")

        elif modelo_info.get('naive_bayes', False):
            print("\n>> Configurando y entrenando Naive Bayes...")
            params_nb = modelo_info['parametros']
            ruta_salida = modelo_info['modelo_output']
            # TODO: Lógica de Naive Bayes...

    print("\n--- Pipeline de entrenamiento finalizado ---")