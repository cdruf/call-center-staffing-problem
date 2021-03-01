import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn import linear_model

from src import input_file_path, data_folder_path
from src.util import datetime_to_columns, calculate_mae, poly5

# %%
""" 
# Load data
"""
# df = pd.read_excel(input_file_path, sheet_name='Sheet1')
df = pd.read_csv(data_folder_path / "all.csv")
df['Call_Start'] = pd.to_datetime(df['Call_Start'])
datetime_to_columns(df, 'Call_Start')

df['WaitTime vs QueuedTime'] = df['WaitTime'] - df['QueuedTime']

# WaitTime ~ RingTime + QueuedTime


print(df.head())
print(df.tail())

dt = df['Call_Start']
n_days = (dt.max() - dt.min()).days
n_weeks = n_days / 7

avg_calls_week = len(dt) / n_weeks
print(f'Average number of calls per week = {avg_calls_week}')
avg_calls_day = len(dt) / n_days
print(f'Average number of calls per day = {avg_calls_day}')
avg_calls_hour = avg_calls_day / 24
print(f'Average number of calls per hour = {avg_calls_hour}')

# %%
"""
# Demand by week 
"""


def plot_demand(df, ax, title='', xlabel=''):
    ax.plot(df.index, df, linestyle='--', marker='o', )
    plt.xticks(rotation=90)
    if title != '':
        plt.title(title)
    if xlabel != '':
        plt.xlabel(xlabel)
    plt.ylabel('No. incoming calls')
    plt.tight_layout()


df_by_week = df.groupby('year_week').size()
x = np.arange(len(df_by_week))
y = df_by_week.to_numpy()
params, _ = curve_fit(poly5, x, y)
print(f"MAE = {calculate_mae(x, y, poly5, params)}")

fig, ax = plt.subplots()
plot_demand(df_by_week, ax, title='Demand by weeks', xlabel='Year, week')
plt.plot(x, poly5(x, *params), color='red')
plt.tight_layout()
plt.show()

# %%
"""
# Demand by day
"""
df_by_day = df.groupby('date').size()
lm = linear_model.LinearRegression()
x = np.arange(len(df_by_day))
y = df_by_day.to_numpy()
lm.fit(x.reshape(-1, 1), y.reshape(-1, 1))
params, _ = curve_fit(poly5, x, y)
print(f"MAE = {calculate_mae(x, y, poly5, params)}")

fig, ax = plt.subplots()
df_by_day.index = df_by_day.index.astype(str)  # avoids that matplotlib uses the date, which leads to a shift on x-axis
plot_demand(df_by_day, ax, title='Demand by day', xlabel='Date')
my_ticks = [tick for idx, tick in enumerate(ax.get_xticks()) if idx % 10 == 0]
plt.xticks(my_ticks)
plt.plot(x, lm.predict(x.reshape(-1, 1)), color='grey', linewidth=2)
plt.plot(x, poly5(x, *params), color='red')
plt.tight_layout()
plt.show()

# %%
"""
# Demand by hour
"""
df_by_hour = df.groupby(['date', 'hour_of_day']).size().reset_index()
df_by_hour.index = df_by_hour['date'].astype(str) + ' ' + df_by_hour['hour_of_day'].astype(str).map(
    lambda x: x.rjust(2, ' '))
df_by_hour.drop(columns=['date', 'hour_of_day'], inplace=True)
df_by_hour.head()
fig, ax = plt.subplots()
plot_demand(df_by_hour, ax, title='Demand by hour', xlabel='Date, hour')
my_ticks = [tick for idx, tick in enumerate(ax.get_xticks()) if idx % 100 == 0]
plt.xticks(my_ticks)
plt.tight_layout()
plt.show()

# %%
"""
# Yearly seasonality - not enough data!
"""

# %%
"""
# Weakly seasonality on daily granularity
"""
counts = dt.dt.dayofweek.value_counts().sort_index()  # Monday=0
counts = counts / n_weeks
fig, ax = plt.subplots()
ax.bar(counts.index, counts)
ax.set_xticks(np.arange(7))
ax.set_xticklabels(('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'))
plt.axhline(avg_calls_day, color='grey')
plt.title('Weekly seasonality')
plt.xlabel('Weekday')
plt.ylabel('No. incoming calls')
plt.tight_layout()
plt.show()

# %%
"""
# Daily seasonality on hourly granularity
"""
counts = dt.dt.hour.value_counts().sort_index()
counts = counts / n_days
fig, ax = plt.subplots()
ax.bar(counts.index, counts)
plt.axhline(avg_calls_hour, color='grey')
plt.title('Daily seasonality')
plt.xlabel('Hour of day')
plt.ylabel('No. incoming calls')
plt.tight_layout()
plt.show()

# %%
"""
# Weakly seasonality on hourly granularity
"""
df['weekday_hour'] = df['weekday'].astype(str) + "_" + df['hour'].astype(str).map(lambda x: x.rjust(2, ' '))
counts = df['weekday_hour'].value_counts().sort_index()
counts = counts / n_weeks
fig, ax = plt.subplots()
ax.bar(counts.index, counts)
plt.axhline(avg_calls_hour, color='grey')
plt.title('Weekly seasonality - hourly precision')
plt.xlabel('Weekday_hour')
plt.ylabel('No. incoming calls')
plt.xticks(rotation=90)
my_ticks = [tick for idx, tick in enumerate(ax.get_xticks()) if idx % 4 == 0]
ax.set_xticks(my_ticks)
plt.tight_layout()
plt.show()

# %%
"""
# Demand forecast 

The time series goes only back one year.  
From the conversations I know there is a strong seasonal pattern over the year, as well as over the week.
The seasonal pattern is visible in the data. 

=> 

* Fit a curve to the demand points per calendar week, as there is not enough data to average out the noise
* Use the last year fitted values as forecast for the volume per week
* Use weekly seasonality to distribute each weeks volume to hour intervals. 

Start at week 13 = Monday 29th of March

"""

"""
## Smooth demand by week
"""

df_by_week = df.groupby(['year', 'week']).size().reset_index()
df_by_week['idx'] = df_by_week.index
df_by_week.set_index(['year', 'week'], inplace=True)
idx_by_year_week = df_by_week['idx'].to_dict()

df_by_week.head()
x = df_by_week['idx'].to_numpy()
y = df_by_week[0].to_numpy()
params, _ = curve_fit(poly5, x, y)


def get_forecast_for_week(year, week):
    idx = idx_by_year_week[year - 1, week]
    return poly5(idx, *params)


get_forecast_for_week(2021, 13)  # ~ 9900
# %%


forecast_for_weeks = pd.DataFrame({'year': 2021,
                                   'week': np.arange(start=13, stop=53),
                                   'n_calls': [get_forecast_for_week(2021, w) for w in range(13, 53)]})

forecast_for_weeks.head()
forecast_for_weeks.to_csv(data_folder_path / "forecast_for_weeks.csv")

# %%

"""
## Weekly distribution of calls 
"""

df_dist = df.groupby(['weekday', 'hour']).size().to_frame(name='total_n_calls')
df_dist['n_calls'] = df_dist['total_n_calls'] / n_weeks
df_dist['share_calls'] = df_dist['total_n_calls'] / df_dist['total_n_calls'].sum()

# Visual sanity checks
df_dist.reset_index(inplace=True)
fig, ax = plt.subplots()
ax.plot(df_dist['weekday'].astype(str) + " " + df_dist['hour'].astype(str),
        df_dist['n_calls'], linestyle='--', marker='o')
plt.axhline(avg_calls_hour)
plt.show()

# %%

fig, ax = plt.subplots()
ax.plot(df_dist['weekday'].astype(str) + " " + df_dist['hour'].astype(str),
        df_dist['share_calls'], linestyle='--', marker='o')
plt.title('Weekly distribution of calls')
plt.xticks([tick for idx, tick in enumerate(ax.get_xticks()) if idx % 10 == 0])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Sanity checks
print(f"Avg call per week = {avg_calls_week:.0f} ~ {df_dist['n_calls'].sum():.0f} = sum calls per hour forecast")
print(f"Sum of shares = {df_dist['share_calls'].sum()}")

# %%

"""
## Forecast 
"""
hours = pd.date_range(start='2021-03-29', end='2021-12-31 23:59', freq='H')
forecast = pd.DataFrame({'hours': hours}, index=hours)
datetime_to_columns(forecast, 'hours')
forecast.head()

size = len(forecast)
forecast = pd.merge(left=forecast, right=forecast_for_weeks, on=['year', 'week']).set_index('hours').sort_index()
assert size == len(forecast)
forecast.rename(columns={'n_calls': 'n_calls_in_week'}, inplace=True)

forecast = pd.merge(left=forecast, right=df_dist[['weekday', 'hour', 'share_calls']],
                    on=['weekday', 'hour'])
assert size >= len(forecast)  # hours without any calls are not in df_dist
forecast['est_n_calls'] = forecast['n_calls_in_week'] * forecast['share_calls']

forecast.head()

forecast.to_csv(data_folder_path / 'call_forecast.csv', index=False)

# %%

# Plot week 13 ff
fig, ax = plt.subplots()
fc = forecast.loc[forecast['week'].isin(range(13, 17)), :]
fc = fc.sort_values(['year', 'week'])
x = fc['week'].astype(str) + " " + fc['weekday'].astype(str) + " " + fc['hour'].astype(str)
ax.plot(x, fc['est_n_calls'], linestyle='--', marker='o')
plt.axhline(avg_calls_hour)
plt.title('Inbound call volume forecast - 2021, calendar weeks 13 - 16')
plt.xticks([tick for idx, tick in enumerate(ax.get_xticks()) if idx % 10 == 0])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

fc.head()
fc.tail()
