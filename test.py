import json
import argparse
import sys
import pickle

from funciones import loadConfig,load_data

def loadModel(model_output: str) -> obj:
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
        with open(model_output,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        sys.exit(1)

    
if __name__ == '__main__':
    # Argumentos de la terminal (config.json)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",type=str, help="El directorio donde se encuentra el archivo de configuración.", default="config.json")
    args = parser.parse_args()

    config = loadConfig(args.config,"test")

    # Separamos el dataset
    data = load_data(config["test_output"]) # Cargamos el dataset del test completo
    y_true = data[config["column"]].values # Cargamos los valores a predecir
    data.drop(columns=[config["column"]]) # Separamos los valores a predecir del dataset para poder hacer predicciones.

    # Evaluamos cada modelo
    for modelo in config["modelos"]:
        # Cargamos el modelo
        print(f"Cargando el modelo {modelo}...")
        model = loadModel(config['modelo'])
        try:
            test(model, y_true)
            print(f"Test del modelo {modelo} realizado con éxito.")
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
