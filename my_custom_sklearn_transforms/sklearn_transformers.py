from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

#Calculamos a média de exatas
df_data_3['MEDIA_EXATAS'] = (df_data_3['NOTA_MF'] + df_data_3['NOTA_GO']) / 2
return data.drop(labels=self.MEDIA_EXATAS, axis='MEDIA_EXATAS')


#Calculamos a média de humanas
df_data_3['MEDIA_HUMANAS'] = (df_data_3['NOTA_EM'] + df_data_3['NOTA_DE']) / 2
return data.drop(labels=self.MEDIA_HUMANAS, axis='MEDIA_HUMANAS'
