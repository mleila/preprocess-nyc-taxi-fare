import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from .distance_functions import haversine, compute_distance


nyc_bb = [-74.2589, 40.4774, -73.7004, 40.9176]


def filter_df(df, bb=nyc_bb):
    lon1, lat1, lon2, lat2 = bb
    df = df[
        (df['passenger_count'] > 1) &\
        (df['Manhatten'] != 0) &\
        (df['fare_amount'] > 2.5) & (df['fare_amount'] < 500) &\
        (df['pickup_latitude'] >= lat1) & (df['pickup_latitude'] <= lat2) &\
        (df['dropoff_latitude'] >= lat1) & (df['dropoff_latitude'] <= lat2) &\
        (df['pickup_longitude'] >= lon1) & (df['pickup_longitude'] <= lon2) &\
        (df['dropoff_longitude'] >= lon1) & (df['dropoff_longitude'] <= lon2)]
    return df


class Haversiner(TransformerMixin):

    def fit(self, df):
        return self

    def transform(self, df):
        df['haversine'] = df.apply(compute_distance, args=[haversine], axis=1)
        return df


class RemoveBadData(TransformerMixin):

    def fit(self, df):
        return self

    def transform(self, df):
        return filter_df(df)


class AbsDiff(TransformerMixin):

    def fit(self, df):
        return self

    def transform(self, df):
        df['abs_diff_lat'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])
        df['abs_diff_lon'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
        df['Manhatten'] = df['abs_diff_lat'] + df['abs_diff_lon']
        return df


class AddDateTime(TransformerMixin):

    def fit(self, df):
        return self

    def transform(self, df):
        df['year'] = df.pickup_datetime.dt.year
        df['month'] = df.pickup_datetime.dt.month
        df['week_day'] = df.pickup_datetime.dt.weekday
        df['month_day'] = df.pickup_datetime.dt.day
        df['hour'] = df.pickup_datetime.dt.hour
        df = pd.concat([df, pd.get_dummies(df['hour'], prefix='hr')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['year'], prefix='yr')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['month'], prefix='month')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['week_day'], prefix='dow')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['month_day'], prefix='dom')], axis=1)
        df.drop(['pickup_datetime', 'hour', 'year', 'month', 'week_day', 'month_day'], axis=1, inplace=True)
        return df


class DFStandardScaler(TransformerMixin):
    # StandardScaler but for pandas DataFrames

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)


class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, df):
        return self

    def transform(self, X):
        return X[self.cols]


class DFFeatureUnion(TransformerMixin):

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        for _, transformer in self.transformers:
            t.fit(X, y)
        return self

    def transform(self, X):
        dfs = [t.transform(X) for _, t in self.transformers]
        return pd.concat(dfs, axis=1)


def build_preprocessing_pipeline():
    pipeline =  Pipeline([
            ('abs_diff', AbsDiff()),
            ('filter', RemoveBadData()),
            ('haversin', Haversiner()),
            ('datetime', AddDateTime()),
            ('StandardScaler', DFStandardScaler())
      ])
    return pipeline
