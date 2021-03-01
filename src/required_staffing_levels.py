import pandas as pd
import matplotlib.pyplot as plt

from src import data_folder_path

forecast = pd.read_csv(data_folder_path / 'call_forecast.csv')
forecast.iloc[0, :]
forecast.loc[:, 'est_n_calls']
forecast.sort_values(['date', 'hour'], inplace=True)
forecast['avg_inter_arrival_secs'] = 1.0 / (forecast['est_n_calls'] / 60 / 60)

# %%
# same as in other file to check
fig, ax = plt.subplots()
fc = forecast.loc[forecast['week'].isin(range(13, 17)), :]
fc = fc.sort_values(['date', 'hour'])
x = fc['week'].astype(str) + " " + fc['weekday'].astype(str) + " " + fc['hour'].astype(str)
ax.plot(x, fc['est_n_calls'], linestyle='--', marker='o')

# %%


sims = pd.read_csv(data_folder_path / 'sim_results.csv')
sims.head()
sims.iloc[0, :]
inter_arrival_col = 'arrival_rate'  # should be named differently
sims.loc[:, inter_arrival_col]


def get_best_n_servers(est_n_calls_per_hour):
    if est_n_calls_per_hour < 0.0001:
        return 0

    inter_arrival_secs = round(1.0 / (est_n_calls_per_hour / 60 / 60))
    max_ = sims.loc[:, inter_arrival_col].max()
    if inter_arrival_secs > max_:
        return 0

    min_ = sims.loc[:, inter_arrival_col].min()
    if inter_arrival_secs < min_:
        print(f'Warning: very large c calls per hour {est_n_calls_per_hour}')
        return sims.loc[:, 'n_servers_smoothed'].max()

    ar = round(inter_arrival_secs)
    return sims.loc[sims['arrival_rate'] == ar, 'n_servers_smoothed'].iloc[0]


for i in range(0, 100):
    print(get_best_n_servers(i))

# %%
forecast['est_n_calls'].map(get_best_n_servers)
forecast['best_n_servers'] = forecast['est_n_calls'].map(get_best_n_servers)
forecast.head()
forecast.iloc[0, :]

forecast.to_csv(data_folder_path / 'call_forecast_extended.csv', index=False)

# %%


fig, ax = plt.subplots()
period = forecast.loc[('2021-03-29' <= forecast['date'].astype(str)) &
                      (forecast['date'].astype(str) < '2021-04-26')].copy()
period.sort_values(['date', 'hour'], inplace=True)
x = period['date'].astype(str) + " " + period['hour'].astype(str).map(lambda x: x.rjust(2, ' '))
y = period['best_n_servers']
ax.plot(x, y)
plt.title("Best staffing levels for 2021 week 13 - 16")
plt.xlabel('Date')
plt.ylabel('Number of servers')
plt.xticks([tick for idx, tick in enumerate(ax.get_xticks()) if idx % 10 == 0])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

