import matplotlib.pyplot as plt
import pandas as pd

from src import data_folder_path
from src.step_function import StepFunction

# %%

df_answered = pd.read_csv(data_folder_path / 'answered.csv')
df_answered['Call_Start'] = pd.to_datetime(df_answered['Call_Start'])
df_answered['agent_answer_time'] = pd.to_datetime(df_answered['agent_answer_time'])
df_answered['agent_finish_time'] = pd.to_datetime(df_answered['agent_finish_time'])

# %%
"""
# Simultaneous calls
"""
print(df_answered.columns)
data = df_answered.loc[df_answered['Call_Start'].dt.date.astype(str) == '2020-03-31', :].copy()
data.dtypes

data['agent_finish_time'].isna().sum()  # ?? die muessen weg
data['agent_finish_time'] = data.apply(
    lambda row: row['agent_answer_time'] if pd.isnull(row['agent_finish_time']) else row['agent_finish_time'], axis=1)

starts = pd.DataFrame({'t': data['agent_answer_time'], 'what': 'answer'})
ends = pd.DataFrame({'t': data['agent_finish_time'], 'what': 'finish'})
ends.dropna(inplace=True)

ts = starts.append(ends).sort_values(by='t')
ts['timestamp'] = ts['t'].astype(int) / 1000 / 1000 / 1000  # sec
ts

fkt = StepFunction(xs=[ts.iloc[0, 2]], ys=[])
fx = 0
for idx, row in ts.iterrows():
    if row['what'] == 'answer':
        fx += 1
    else:
        fx -= 1
    x = row['timestamp']
    fkt.append_step(x, fx, x)

fkt.plot()
plt.title("No. simultaneous calls - 2020-03-31")
plt.tight_layout()
