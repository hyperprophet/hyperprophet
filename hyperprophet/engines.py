"""
hyperprophet.engines
~~~~~~~~~~~~~~~~~~~~

Compute engines for Prophet.
"""

ENGINES = {}
def make_engine(engine=None):
    if isinstance(engine, BaseEngine):
        return engine
    if isinstance(engine, str):
        if engine not in ENGINES:
            raise ValueError(f"Invaid Engine: {engine!r}")
        return ENGINES[engine]()

    # default
    return ZeroEngine()


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

ENGINES["zero"]= ZeroEngine
