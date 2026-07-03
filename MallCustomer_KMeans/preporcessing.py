import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler


DIR = 'dataset'
def preprocess():
    file_path = os.path.join(DIR, os.listdir(DIR)[0])
    data_df =  pd.read_csv(file_path)
    print(data_df.head())
    print('==============dataset is read===============')
    data_df.drop(['CustomerID'], axis=1, inplace=True)
    data_df_encoded = pd.get_dummies(data_df, columns=['Genre'])
    print(data_df_encoded.head())
    print('==============dataset is encoded===============')
    X = data_df_encoded[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]
    #data_df_encoded.dropna(inplace=True)
    data_df_encoded.fillna(data_df_encoded.mean(), inplace=True)
    print('==============dataset is cleaned===============')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print('==============features are scaled===============')
    pca = PCA(n_components=3)
    X = pca.fit_transform(X)
    print('==============features are reduced===============')
    return X
    