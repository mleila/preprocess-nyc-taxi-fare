import pandas as pd

traintypes = {'fare_amount': 'float32',
          'pickup_datetime': 'str',
          'pickup_longitude': 'float32',
          'pickup_latitude': 'float32',
          'dropoff_longitude': 'float32',
          'dropoff_latitude': 'float32',
          'passenger_count': 'uint8'}


def read_training_data(pth, chunksize=5000000):
    cols = list(traintypes.keys())
    df_list = []
    for df_chunk in pd.read_csv(pth, usecols=cols, dtype=traintypes, chunksize=chunksize):
        df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
        df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
        df_list.append(df_chunk)
    return pd.concat(df_list)
