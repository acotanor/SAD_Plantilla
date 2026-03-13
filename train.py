import json
import argparse

def loadConfig(file: str) -> dict():
    # Cargar el archivo JSON
    with open(file, 'r', encoding='utf-8') as f:
        config_completa = json.load(f)                  # Diccionario con toda la configuración (incluyendo campos que no necesitamos)

    config = {}                                         # Diccionario que tendrá la configuración específica de general + train.
    
    # Extraer y aplanar la sección 'general'.
    general = config_completa.get("general", {})        # Diccionario "general".
    for key, value in general.items():
        if key == "data" and isinstance(value, dict):
            # Aplanar los campos dentro de 'data'.
            for data_key, data_value in value.items():
                config[data_key] = data_value
        else:
            config[key] = value
            
    # Extraer la sección 'train' y filtrar modelos
    train = config_completa.get("train", {})            # Diccionario "train".
    for key, value in train.items():
        if key == "modelos" and isinstance(value, list):
            modelos_activos = []                        # Array con cada modelo que queremos cargar.
            for modelo in value:
                # Comprobamos si el modelo está marcado como True
                # Buscamos cualquier clave cuyo valor sea estrictamente True (ignorando 'parametros')
                es_activo = any(v is True for k, v in modelo.items() if k != "parametros")
                if es_activo:
                    modelos_activos.append(modelo)
            config["modelos"] = modelos_activos         # Guardamos un array con un diccionario de cada modelo.
        else:
            config[key] = value
            
    return config

    
if __name__ == '__main__':
    # Argumentos de la terminal (config.json)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",type=str, help="El directorio donde se encuentra el archivo de configuración.", default="config.json")
    args = parser.parse_args()

    config = loadConfig(args.config)
    print (config)