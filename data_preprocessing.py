import pandas as pd
import unicodedata
import math

def preprocess_data(cars_train_raw):
    df1 = cars_train_raw.copy()

    # Rename Columns
    df1.columns = [unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('utf-8') for col in df1.columns]

    # Fillout Na
    df1 = df1.drop(['veiculo_alienado'], axis=1)
    df1['num_fotos'] = df1['num_fotos'].apply(lambda x: 0 if math.isnan(x) else x)
    df1['dono_aceita_troca'] = df1['dono_aceita_troca'].apply(lambda x: 0 if pd.isna(x) else 1)
    df1['veiculo_unico_dono'] = df1['veiculo_unico_dono'].apply(lambda x: 0 if pd.isna(x) else 1)
    df1['revisoes_concessionaria'] = df1['revisoes_concessionaria'].apply(lambda x: 0 if pd.isna(x) else 1)
    df1['ipva_pago'] = df1['ipva_pago'].apply(lambda x: 0 if pd.isna(x) else 1)
    df1['veiculo_licenciado'] = df1['veiculo_licenciado'].apply(lambda x: 0 if pd.isna(x) else 1)
    df1['garantia_de_fabrica'] = df1['garantia_de_fabrica'].apply(lambda x: 0 if pd.isna(x) else 1)
    df1['revisoes_dentro_agenda'] = df1['revisoes_dentro_agenda'].apply(lambda x: 0 if pd.isna(x) else 1)

    # Change Types
    df1['num_fotos'] = df1['num_fotos'].astype('int64')
    df1['ano_modelo'] = df1['ano_modelo'].astype('int64')

    return df1

def feature_engineering(df1):
    df2 = df1.copy()
    df2['blindado'] = df2['blindado'].apply(lambda x: 0 if x == 'N' else 1)
    df2 = df2.drop(['elegivel_revisao'], axis=1)
    df2['entrega_delivery'] = df2['entrega_delivery'].apply(lambda x: 0 if x == False else 1)
    df2['troca'] = df2['troca'].apply(lambda x: 0 if x == False else 1)

    return df2

def main():
    # Load dataset
    cars_train_raw = pd.read_csv('data/cars_train.csv', encoding='utf-16', delimiter='\t')

    # Preprocess data
    df1 = preprocess_data(cars_train_raw)

    # Feature engineering
    df3 = feature_engineering(df1)

    return df3

if __name__ == "__main__":
    df3 = main()
