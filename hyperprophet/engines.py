"""
hyperprophet.engines
~~~~~~~~~~~~~~~~~~~~

Compute engines for Prophet.
"""
import pandas as pd

ENGINES = {}
def make_engine(engine=None):
    if isinstance(engine, BaseEngine):
        return engine
    if isinstance(engine, str):
        if engine not in ENGINES:
            raise ValueError(f"Invaid Engine: {engine!r}")
        return ENGINES[engine]

    return ENGINES['default']

def register_engine(name, engine):
    """Registers a new engine.

    To change the default engine, call this function with name='default'.
    """
    # if not isinstance(engine, BaseEngine):
    #     raise ValueError("The engine must be an instance of BaseEngine.")
    ENGINES[name] = engine

class BaseEngine:
    def forecast(self, df_fit, df_predict, options):
        raise NotImplementedError()

class ZeroEngine:
    """Forecast zero for all values.

    Used for testing.
    """
    def forecast(self, df_fit, df_predict, options):
        df = df_predict.copy()
        columns = [
            'trend',
            'yhat_lower',
            'yhat_upper',
            'trend_lower',
            'trend_upper',
            'additive_terms',
            'additive_terms_lower',
            'additive_terms_upper',
            'multiplicative_terms',
            'multiplicative_terms_lower',
            'multiplicative_terms_upper',
            'yhat'
        ]
        for c in columns:
            df[c] = 0.0
        return df

class LocalEngine(BaseEngine):
    """Forecast locally using Prophet.
    """
    def forecast(self, df_fit, df_predict, options):
        keys = df_fit['key'].unique()
        df_fit_parts = dict(iter(df_fit.groupby('key')))
        df_predict_parts = dict(iter(df_predict.groupby('key')))

        # Does df_predict have any keys that are not part of df_fit
        missing_keys = {k for k in df_predict_parts if k not in df_fit_parts}
        if missing_keys:
            raise ValueError("Can't forecast for a key that is not part of the dataframe given to fit")

        dfs = [self.forecast_one_series(k, df_fit_parts[k], df_predict_parts[k], options) for k in df_predict_parts]
        return pd.concat(dfs)

    def forecast_one_series(self, key, df_fit, df_predict, options):
        # TODO: use the options
        from fbprophet import Prophet

        df_fit = df_fit.drop('key', axis=1)
        df_predict = df_predict.drop('key', axis=1)

        m = Prophet()
        m.fit(df_fit)
        forecast = m.predict(df_predict)

        # Add key as the first column
        columns = ['key'] + list(forecast.columns)
        forecast['key'] = key
        return forecast[columns]


register_engine('zero', ZeroEngine())
register_engine('local', LocalEngine())
register_engine('default', LocalEngine())
