import pandas as pd
from prophet import Prophet

def transform_data(data):
    """Transform retrieve data from API"""
    df = pd.DataFrame(data)
    df.index = df.index.date
    df['datetime'] = pd.to_datetime(df.index)
    df = df.drop(['High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Capital Gains'], axis=1)
    df.columns = ['y', 'ds']
    train = df[df['ds'] < '2023-01-01']
    test = df[df['ds'] >= '2023-01-01']
    model = Prophet()
    model = model.fit(train)
    

    # return the model
    return model

