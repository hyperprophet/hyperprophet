import pandas as pd
from . import fbprophet
from . import engines
from typing import Dict, Any

class Prophet(fbprophet.Prophet):
    def __init__(self, *args, engine=None, **kwargs):
        kwargs.setdefault("yearly_seasonality", False)
        kwargs.setdefault("weekly_seasonality", False)
        kwargs.setdefault("daily_seasonality", False)

        super().__init__(*args, **kwargs)
        self.fit_df = None
        self.fit_kwargs = None
        self.keys = None
        self.engine = engines.make_engine(engine)

    def validate_inputs(self):
        super().validate_inputs()

        if self.yearly_seasonality == 'auto':
            raise ValueError("Hyperprophet doesn't support yearly_seasonality=auto")
        if self.weekly_seasonality == 'auto':
            raise ValueError("Hyperprophet doesn't support weekly_seasonality=auto")
        if self.daily_seasonality == 'auto':
            raise ValueError("Hyperprophet doesn't support daily_seasonality=auto")

    def _get_options(self) -> Dict[str, Any]:
        """Returns the options/parameters pased to the Prophet.

        These will be passed to the engine to initialize Prophet
        on the remote server with these options.
        """
        # TODO: handle holidays as well
        return {
            "growth": self.growth,
            "changepoints": self.changepoints and list(self.changepoints.astype('str')),
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "mcmc_samples": self.mcmc_samples,
            "interval_width": self.interval_width,
            "uncertainty_samples": self.uncertainty_samples,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "seasonality_mode": self.seasonality_mode,
            "seasonality_prior_scale": self.seasonality_prior_scale,

            "seasonalities": self.seasonalities,
            "extra_regressors": self.extra_regressors
        }

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
        options = self._get_options()
        return self.engine.forecast(self.fit_df, df, options)

    def make_future_dataframe(self, periods, freq='D', include_history=True):
        keys = pd.DataFrame({"key": self.keys})
        df = super().make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        # cross product
        return keys.assign(_merge_col=1).merge(df.assign(_merge_col=1)).drop("_merge_col", axis=1)