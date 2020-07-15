import pandas as pd
from . import fbprophet
from . import engines

class Prophet(fbprophet.Prophet):
    def __init__(self, *args, engine=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_df = None
        self.fit_kwargs = None
        self.keys = None
        self.engine = engines.make_engine(engine)

    def _load_stan_backend(self, stan_backend):
        # Disable loading stan backend
        # Hyperprophet does not use stan locally, but run it on the remote server
        pass

    def fit(self, df, **kwargs):
        self.history_dates = pd.to_datetime(df['ds'].unique()).sort_values()
        self.keys = df['key'].unique()
        self.fit_df = df
        self.fit_kwargs = kwargs
        return self

    def predict(self, df=None):
        options = {}
        return self.engine.forecast(self.fit_df, df, options)

    def make_future_dataframe(self, periods, freq='D', include_history=True):
        keys = pd.DataFrame({"key": self.keys})
        df = super().make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        # cross product
        return keys.assign(_merge_col=1).merge(df.assign(_merge_col=1)).drop("_merge_col", axis=1)