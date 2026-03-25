import pandas as pd
from sklearn.model_selection import train_test_split
from funciones import loadConfig
import argparse

# Argumentos de la terminal (config.json)
parser = argparse.ArgumentParser()
parser.add_argument("-c","--config",type=str, help="El directorio donde se encuentra el archivo de configuración.", default="config.json")
parser.add_argument("-s","--split",type=int, help="El porcentaje del tamaño que tendrá el test", default=20)
args = parser.parse_args()

config = loadConfig(args.config,"procesado")

df = pd.read_csv(config["train_dev"]) # Cargamos el train_dev aunque solo haya un archivo.

# Dividimos el dataset:
df_train, df_test = train_test_split(df, test_size=args.split/100,random_state=42, stratify=df[config["column"]])

# Guardamos el resultado:
df_train.to_csv(config["train_dev"].replace('.csv','')+"_train_dev.csv",index=False)
df_test.to_csv(config["train_dev"].replace('.csv','')+"_test.csv",index=False)