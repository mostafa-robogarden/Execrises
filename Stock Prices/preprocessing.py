import pandas as pd
import quandl

def preprocessData():
    df = quandl.get('WIKI/GOOGL')
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['HL_PERCENT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
    df['PERCENT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
    df = df[['Adj. Close', 'HL_PERCENT', 'PERCENT_CHANGE', 'Adj. Volume']]
    
preprocessData()