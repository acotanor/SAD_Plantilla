import json
import argparse

def loadConfig(file: str) -> dict():
    """
    Función que carga el .json de configuración.
    Genera una configuración específica, general + train para poder reutilizar el json.
    Parámetros:
        - file: La ruta del archivo de configuración, config.json por defecto.
    Return:
        - config: Un diccionario con la configuración específica del script.
    """
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

def save_model(model_output: str, model: obj):
    """
    Función que guarda el modelo (.pkl) y los resultados del barrido de hiperparámetros (.csv).
    Parámetros:
    - model_output: La ruta donde se guardará el modelo y el barrido de hiperparámetros.
    - model: El modelo que queremos guardar.
    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.
    """
    try:
        # Guardamos el modelo .pkl.
        with open(f"{model_output.rsplit('.',1)[0]}.pkl","wb") as file: # Quitamos la extensión y la reañadimos por si acaso la añadimos sin querer al escribir el json.
            pickle.dump(model,file)
            print(f"Modelo guardado en: {model_output.rsplit('.',1)[0]}.pkl")
        
        # Guardamos el registro de los resultados del barrido de hiperparámetros .csv.
        with open(f"{model_output.rsplit('.',1)[0]}.csv","wb") as file:
            writer = csv.writer(file)
            writer.writerow(["Params","Score"])     # Campos del csv.
            for params, score in zip(model.cv_results_['params'], model.cv_results_["mean_test_score"]):
                writer.writerow([params,score])     # Array de parametros, nota.
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")



def knn(model_output: str):
    """
    Función que implementa un algoritmo kNN.
    Hace un barrido de hiperparámetros para encontrar la combinación óptima.
    """
    pass

def decision_tree(model_output:str):
    """
    Función que implementa un algoritmo decision tree.
    Hace un barrido de hiperparámetros para encontrar la combinación óptima.
    """
    pass

def random_forest(model_output:str):
    """
    Función que implementa un algoritmo random forest.
    Hace un barrido de hiperparámetros para encontrar la combinación óptima.
    """
    pass

def naive_bayes(model_output:str):
    """
    Función que implementa un algoritmo naive bayes.
    Hace un barrido de hiperparámetros para encontrar la combinación óptima.
    """
    pass

if __name__ == '__main__':
    # Argumentos de la terminal (config.json)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",type=str, help="El directorio donde se encuentra el archivo de configuración.", default="config.json")
    args = parser.parse_args()

    config = loadConfig(args.config)
    print (config)