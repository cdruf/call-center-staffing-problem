from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data_folder_path
from src.step_function import StepFunction
from src.util import show_histogram

# %%

"""
# Load answered data set
"""
df_answered = pd.read_csv(data_folder_path / 'answered.csv')
df_answered['Call_Start'] = pd.to_datetime(df_answered['Call_Start'])
df_answered['agent_answer_time'] = pd.to_datetime(df_answered['agent_answer_time'])
df_answered['agent_finish_time'] = pd.to_datetime(df_answered['agent_finish_time'])

print(df_answered.head())

# %%
"""
# Agents
"""
names = df_answered['Party_Name'].unique()
print(f"Number of agents = {len(names)}")

grp = df_answered.groupby(['Party_Name']).agg({'Call_Start': ['min', 'max', 'count']})
grp = grp['Call_Start']
grp['span'] = grp['max'] - grp['min']
print(f"Agents' avg. length of stay = {grp['span'].mean()}")
show_histogram(grp['span'].dt.days, "Agents length of stay in days")

# %%
"""
# PLot calls
"""


def plot_calls_by_agent(start='2000-01-01', end='2030-12-31'):
    print('Create plot')
    data = df_answered.loc[(start <= df_answered['Call_Start']) &
                           (df_answered['Call_Start'] < end)].copy()
    data['date'] = data['Call_Start'].dt.date
    fig, ax = plt.subplots()
    for idx, agent in enumerate(names):
        dfa = data.loc[data['Party_Name'] == agent, :]
        x = dfa['Call_Start']
        y = np.repeat(idx, len(dfa))  # each agent get his lane in the plot
        ax.scatter(x, y, s=1, marker="o", label=agent)
    plt.grid(True)
    for d in data['date']:
        plt.axvline(datetime(d.year, d.month, d.day, 8, 0, 0), linewidth=0.1)
        plt.show()


plot_calls_by_agent()
plot_calls_by_agent(start='2020-03-29', end='2020-05-01')
