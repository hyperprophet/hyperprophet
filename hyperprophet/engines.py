"""
hyperprophet.engines
~~~~~~~~~~~~~~~~~~~~

Compute engines for Prophet.
"""
import pandas as pd
import requests
import os
import tempfile
import zipfile
import time

ENGINES = {}
def make_engine(engine=None):
    if isinstance(engine, BaseEngine):
        return engine
    if isinstance(engine, str):
        if engine not in ENGINES:
            raise ValueError(f"Invaid Engine: {engine!r}")
        return ENGINES[engine]()

    return ENGINES['default']()

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

        seasonalities = options.pop('seasonalities', {})
        extra_regressors = options.pop('extra_regressors', {})

        m = Prophet(**options)
        m.seasonalities = seasonalities
        m.extra_regressors = extra_regressors
        m.fit(df_fit)
        forecast = m.predict(df_predict)

        # Add key as the first column
        columns = ['key'] + list(forecast.columns)
        forecast['key'] = key
        return forecast[columns]

DEFAULT_ENDPOINT_URL = "https://api.hyperprophet.com"
DEFAULT_API_TOKEN = None

def setup(api_token, endpoint_url=None):
    """Setup the default credentials for the hyperprophet service.

    These credentials will be used when api_token is not explictly
    specfied when creating HyperprophetEngine or Prophet instance.

    Parameters
    ----------
    api_token:
        API token to access the hyperprophet service

    endpiont_url:
        Optional endpoint URL to specify when woring against a different server.

    Example
    -------

    from hyperprophet import setup
    setup(api_token="0ef097a52c0942............e8a7d6")
    """
    global DEFAULT_API_TOKEN, DEFAULT_ENDPOINT_URL
    DEFAULT_API_TOKEN = api_token

    if endpoint_url is not None:
        DEFAULT_ENDPOINT_URL = endpoint_url

class HyperprophetEngine(BaseEngine):
    """Engine to run forecast on the hyperprophet cloud.
    """
    def __init__(self, api_token=None, endpoint_url=None):
        self.api_token = api_token or DEFAULT_ENDPOINT_URL
        if self.api_token is None:
            raise ValueError("Please provide api_token. You can also call the setup function to set it globally.")

        self.endpoint_url = endpoint_url or DEFAULT_ENDPOINT_URL
        self.endpoint_url = self.endpoint_url.rstrip("/")

    def request(self, method, path, json=None, **kwargs):
        headers = {"Authorization": "Bearer " + self.api_token}
        url = self.endpoint_url + path
        return requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
            **kwargs)

    def forecast(self, df_fit, df_predict, options):
        job = Job.create(self, options)
        job.upload_files(df_fit, df_predict)
        job.start()
        job.wait()
        return job.read_results_df()

class Job:
    def __init__(self, engine, id, status, data_upload_url=None, results_url=None, progress=0.0):
        self.engine = engine
        self.id = id
        self.status = status
        self.data_upload_url = data_upload_url
        self.results_url = results_url
        self.progress = progress

    def start(self):
        """Starts the job execution.

        This must be called after uploading the files.
        """
        url = "/jobs.start"
        payload = {
            "id": self.id
        }
        response = self.engine.request("POST", url, json=payload)
        if response.status_code != 200:
            raise EngineError("Failed to upload the job payload. ({} - {})".format(response.status_code, response.text[:100]))

        d = response.json()
        if d['ok'] is False:
            raise EngineError("Failed to upload the job payload. ({})".format(d['error']))

        self._update(d['job'])

    def _update(self, data):
        """Updates the job with fresh details from the API.
        """
        self.status = data['status']
        self.progress = data['progress']

    def _refresh(self):
        response = self.engine.request("GET", "/jobs.info", params={"id": self.id})
        d = response.json()
        if d['ok'] is False:
            raise EngineError("Failed to get jon status. ({})".format(d['error']))
        self._update(d['job'])

    def read_results_df(self):
        response = self.engine.request("GET", "/jobs.result", params={"id": self.id})
        if response.status_code != 200:
            raise EngineError("Failed to read the results. ({} - {})".format(response.status_code, response.text[:100]))
        d = response.json()
        if d['ok'] is False:
            raise EngineError("Failed to read the results. ({})".format(d['error']))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.parq")
            self._download(d['download_url'], path)
            return pd.read_parquet(path)

    def _download(self, url, path):
        response = requests.get(url)
        response.raise_for_status()
        with open(path, 'wb') as f:
            CHUNK_SIZE = 1024*1024 # 1MB
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)

    def upload_files(self, df_train, df_predict):
        """Uploads the required files to the job.
        """
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.parq")
            df_train.to_parquet(train_path)

            predict_path = os.path.join(tmp, "predict.parq")
            df_predict.to_parquet(predict_path)

            zip_path = os.path.join(tmp, "payload.zip")
            with zipfile.ZipFile(zip_path, "w") as z:
                z.write(train_path, "train.parq")
                z.write(predict_path, "predict.parq")

            headers = {
                "content-type": "application/zip"
            }
            response = requests.put(
                self.data_upload_url,
                data=open(zip_path, 'rb'),
                headers=headers)
            if response.status_code != 200:
                raise EngineError("Failed to upload the job payload. ({} - {})".format(response.status_code, response.text[:100]))

    @classmethod
    def create(cls, engine, options):
        payload = {
            "options": options
        }

        response = engine.request("POST", "/jobs.create", json=payload)
        if response.status_code != 200:
            raise EngineError("Failed to create a new job. ({} - {})".format(response.status_code, response.text[:100]))

        d = response.json()
        if not d['ok']:
            raise EngineError("Failed to create a new job. ({})".format(d['error']))

        job = d['job']

        return Job(
            engine=engine,
            id=job['id'],
            status=job['status'],
            data_upload_url=job['data_upload_url'])

    def wait(self):
        while self.status not in ['SUCCESS', 'FAILED', 'ABORTED']:
            self._refresh()
            print("[JOB {}] status={} progress={}".format(self.id, self.status, self.progress))
            time.sleep(5)

class EngineError(Exception):
    pass

register_engine('zero', ZeroEngine)
register_engine('local', LocalEngine)
register_engine('remote', HyperprophetEngine)
register_engine('default', HyperprophetEngine)