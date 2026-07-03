import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def plotData(data: pd.DataFrame, x, y, plotType = ''):
    if plotType != '':
        data.plot(kind=plotType, x=x, y=y)
        data['Humidity'].plot(kind='hist')
        plt.show()
    else:
        data.plot()
        plt.show()
def readCleanData(filename):
    df = pd.read_csv(filename)
    #print(df.head())
    #print(df.tail())
    print(df.shape)
    print(df.describe())
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    original_df = df.copy()
    encoder = LabelEncoder()
    sc = StandardScaler()
    for column in ['Cloud Cover', 'Season', 'Location', 'Weather Type']:
        labels = encoder.fit_transform(df[column])
        df.drop(column, axis=1, inplace=True)
        df[column] = labels
    #df = sc.fit_transform(df)
    #print(df)
    correlation_matrix = df.corr()
    print(correlation_matrix)
    #return df.to_dict(orient='index')
    return df
def preprocessData(filename):
    data = readCleanData(filename)
    plotData(data, 'Humidity', 'Temperature', 'scatter')
    
#print(pd.__version__)
preprocessData('dataset/weather_classification_data.csv')
