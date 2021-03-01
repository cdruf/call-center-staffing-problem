import matplotlib.pyplot as plt
import pandas as pd

from src import data_folder_path

# %%

"""
# Determine the inter-arrival time during the peak time on 2020-03-31 for the baseline.
"""
df = pd.read_csv(data_folder_path / "all.csv")
df = df.loc[df['Exit_Reason'].isin(['Abandoned', 'AgentAnswered'])]  # ignore redirects
dt = pd.to_datetime(df['Call_Start'])
dt = dt.loc[dt.dt.date.astype(str) == '2020-03-31'].copy()

dt.hist()
plt.title("Histogram of incoming calls on 2020-03-31")
plt.tight_layout()

for i in range(12, 18):
    peak_time = dt.loc[(i <= dt.dt.hour) & (dt.dt.hour < i + 1)].copy()
    inter_arrival_time = 60 * 60 / len(peak_time)
    print(f"Inter arrival time from {i} to {i + 1} = {inter_arrival_time}")
# 17.142857142857142
