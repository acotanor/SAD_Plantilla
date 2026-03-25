import sys
import json
import pandas as pd

def loadConfig(file: str, group: str) -> dict:
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

    # 2. Extraer la sección específica:
    if group == "procesado":
        procesado = config_completa.get("procesado", {})
        for key, value in procesado.items():
            config[key] = value

    elif group == "train":
        # Extraer la sección 'train' y filtrar solo los modelos que estén en 'true'
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

    elif group == "test":
        # Extraer la sección 'test'
        test = config_completa.get("test", {})  # Diccionario "test".
        for key, value in test.items():
            config[key] = value

    else:
        print(f"{group} no es una categoría válida, elige entre: procesado, train y test.")
        sys.exit(0)
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