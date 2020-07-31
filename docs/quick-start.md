# Quick Start

HyperProphet API is very close the `Prophet` API. It supports almost all features of Prophet while extending it to compute multiple forecasts in a single function call.

The input HyperProphet is always a dataframe with three columns: `key`, `ds` and `y`. The `ds` and `y` columns are the time and measurement to forecast, just like in `Prophet`. The key column uniquely identifies each time series.

As an example, let’s look at a time series of the log daily page views for a [thousand Wikipedia pages][dataset].

First we’ll import the data:

```
import pandas as pd
import hyperprophet as hp

url = "https://raw.githubusercontent.com/hyperprophet/wikipedia-pageviews-2020/master/wikipedia_pageviews.csv"
df = pd.read_csv(url)
df.head()
```

```
     article        date   pageviews
0  Main_Page  2019-01-01  18056372.0
1  Main_Page  2019-01-02  18216021.0
2  Main_Page  2019-01-03  18019006.0
3  Main_Page  2019-01-04  16561555.0
4  Main_Page  2019-01-05  17738812.0
```

We need to rename the columns before we can forecast:

```
df = df.rename(columns={"article": "key", "date": "ds", "pageviews": "y"})
```

The rest of the flow is exactly similar to [Prophet][prophet-quickstart].

```
# step1: fit the model

m = hp.Prophet()
m.fit(df)
```

Next we need to create the dataframe with future dates.

```
future = m.make_future_dataframe(periods=30, include_history=False)
future.tail()
```

The training dataframe has data until 2020-06-30. The future dataframe will have all those dates and additional 30 days.

```
                                        key         ds
576995  2020_coronavirus_pandemic_in_Kerala 2020-07-26
576996  2020_coronavirus_pandemic_in_Kerala 2020-07-27
576997  2020_coronavirus_pandemic_in_Kerala 2020-07-28
576998  2020_coronavirus_pandemic_in_Kerala 2020-07-29
576999  2020_coronavirus_pandemic_in_Kerala 2020-07-30
```

The next step is to `predict` the value for for each row in future.

```
forecast = m.predict(future)
forecast[['key', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

```
                                       key          ds          yhat       yhat_lower     yhat_upper
22975	2020_coronavirus_pandemic_in_Kerala	2020-07-26	-7926.387844	-13812.629571	-2086.144717
22976	2020_coronavirus_pandemic_in_Kerala	2020-07-27	-8119.845628	-14736.411976	-2665.455641
22977	2020_coronavirus_pandemic_in_Kerala	2020-07-28	-8313.303413	-13771.766192	-2455.529794
22978	2020_coronavirus_pandemic_in_Kerala	2020-07-29	-8506.761197	-14693.574632	-3042.119390
22979	2020_coronavirus_pandemic_in_Kerala	2020-07-30	-8700.218982	-14599.509330	-2615.481860
```

The forecast of all the 1000 timeseies would be complete in couple of minutes.

!!! Warning
    While the forecast will include all the rows in the predict dataframe, the order of rows
    may be different.


[dataset]: https://github.com/hyperprophet/wikipedia-pageviews-2020

## Seasonality

Unlike Prophet, HyperProphet doens't enable any seasonalities by default. They must be enabled explicitly.

For example, the following enables yearly and daily seasonality, but disabled weekly seasonlity.

```
m = hp.Prophet(
    yearly_seasonality=True,
    weekly_seasonlity=False,
    daily_seasonality=True
)
```

### Specifying Custom Seasonalities

Custom seasonalities can be specified just like Prophet.

```
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
```