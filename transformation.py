import pickle
import numpy as np
import pandas as pd

class Transformation( object ):
    def __init__( self ):
        self.home_path = '/Users/Luan/repos/lighthouse/'
        self.mms_num_portas = pickle.load( open( self.home_path + 'features/num_portas_scaler.pkl', 'rb'))
        self.le_cambio = pickle.load( open( self.home_path + 'features/cambio_scaler.pkl', 'rb'))
        self.le_cidade_vendedor = pickle.load( open( self.home_path + 'features/cidade_vendedor_scaler.pkl', 'rb'))
        self.le_estado_vendedor = pickle.load( open( self.home_path + 'features/estado_vendedor_scaler.pkl', 'rb'))
        self.le_anunciante = pickle.load( open( self.home_path + 'features/anunciante_scaler.pkl', 'rb'))
        self.le_ano_de_fabricacao = pickle.load( open( self.home_path + 'features/ano_de_fabricacao_scaler.pkl', 'rb'))
        self.le_ano_modelo = pickle.load( open( self.home_path + 'features/ano_modelo_scaler.pkl', 'rb'))
        self.mean_ano_modelo = pickle.load( open( self.home_path + 'features/mean_ano_modelo.pkl', 'rb'))
        self.target_encode_marca = pickle.load( open( self.home_path + 'features/target_encode_marca_scaler.pkl','rb'))
        self.target_encode_modelo = pickle.load( open( self.home_path + 'features/target_encode_modelo_scaler.pkl', 'rb'))
        self.target_encode_versao = pickle.load( open( self.home_path + 'features/target_encode_versao_scaler.pkl', 'rb'))
        
    def data_transform( self, df_pre ):
        # num_portas
        df_pre['num_portas'] = self.mms_num_portas.transform( df_pre[['num_portas']].values )

        # 'cambio' Label Encoder
        df_pre['cambio'] = self.le_cambio.transform( df_pre['cambio'] )

        # 'tipo' Ordinal Encoding
        tipo_dict = {'Hatchback':1, 'Picape':2, 'Utilitário esportivo':3, 'Sedã':4, 'Cupê':5, 'Perua/SW':6, 'Minivan':7, 'Conversível':8}
        df_pre['tipo'] = df_pre['tipo'].map( tipo_dict )

        # 'cor' Ordinal Encoding
        cor_dict = {'Branco': 1 , 'Preto': 2, 'Prata': 3, 'Cinza': 4, 'Verde': 5, 'Vermelho': 6, 'Dourado':7, 'Azul':8}
        df_pre['cor'] = df_pre['cor'].map( cor_dict )

        # 'cidade_vendedor' Label Encoder
        #df_pre['cidade_vendedor'] = self.le_cidade_vendedor.transform( df_pre['cidade_vendedor'] )
        for i, cidade in enumerate(df_pre['cidade_vendedor']):
            if cidade not in self.le_cidade_vendedor.classes_:
                # Se a cidade não foi vista durante o treinamento, atribua um valor especial
                df_pre.at[i, 'cidade_vendedor'] = -1
            else:
                # Se a cidade foi vista durante o treinamento, faça a transformação normal com o LabelEncoder
                df_pre.at[i, 'cidade_vendedor'] = self.le_cidade_vendedor.transform([cidade])[0]

        # 'estado_vendedor' Label Encoder
        df_pre['estado_vendedor'] = self.le_estado_vendedor.transform( df_pre['estado_vendedor'] )

        # 'anunciante' Label Encoder
        df_pre['anunciante'] = self.le_anunciante.transform( df_pre['anunciante'] )

        # 'tipo_vendedor' One-Hot Encoding
        df_pre = pd.get_dummies( df_pre, prefix=['tipo_vendedor'], columns=['tipo_vendedor'] )

        # marca - Target Encoding
        df_pre.loc[:, 'marca'] = df_pre['marca'].map( self.target_encode_marca )

        # modelo - Target Encoding
        df_pre.loc[:, 'modelo'] = df_pre['modelo'].map( self.target_encode_modelo )

        # versao - Target Encoding
        df_pre.loc[:, 'versao'] = df_pre['versao'].map( self.target_encode_versao )

        # 'ano_de_fabricacao' Label Encoder
        df_pre['ano_de_fabricacao'] = self.le_ano_de_fabricacao.transform( df_pre['ano_de_fabricacao'] )

        # 'ano_modelo' Label Encoder
        #df_pre['ano_modelo'] = self.le_ano_modelo.transform( df_pre['ano_modelo'] )
        for i, ano in enumerate(df_pre['ano_modelo']):
            if ano not in self.le_ano_modelo.classes_:
                # Se o ano não foi vista durante o treinamento, atribua o valor médio
                df_pre.at[i, 'ano_modelo'] = self.mean_ano_modelo
            else:
                # Se o ano foi vista durante o treinamento, faça a transformação normal com o LabelEncoder
                df_pre.at[i, 'ano_modelo'] = self.le_ano_modelo.transform([ano])[0]

        df_pre = df_pre.fillna( 0 )
        
        return df_pre