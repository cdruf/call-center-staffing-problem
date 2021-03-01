import matplotlib.pyplot as plt
import pandas as pd

from src import data_folder_path
from src.util import drop_rows_with_extreme_values

# %%

"""
# Determine average work time per year to determine hourly employee cost.
"""
df = pd.read_csv(data_folder_path / "answered.csv")
df['Call_Start'] = pd.to_datetime(df['Call_Start'])
df['date'] = df['Call_Start'].dt.date
df.columns

# Group by agent and determine their length of stay, remove agents with very short LOS
grp = df.groupby('Party_Name').agg({'Call_Start': ['min', 'max']})
grp['span'] = grp[('Call_Start', 'max')] - grp[('Call_Start', 'min')]
# grp = grp.loc[grp['span'] > pd.to_timedelta(365, unit='d')]
grp['years'] = grp['span'].dt.days / 365
grp = grp.loc[grp['years'] > 0.5].copy()
grp.head()
len(grp)

# Group by agent and day to determine the worked hours per day
grp2 = df.groupby(['Party_Name', 'date']).agg({'Call_Start': ['min', 'max']})
grp2['shift_length'] = (grp2[('Call_Start', 'max')] - grp2[('Call_Start', 'min')]) / pd.Timedelta(hours=1)
grp2.head()

# Group the previous group by agent to determine the total worked hours
grp3 = grp2.groupby('Party_Name').sum()
grp3.columns
grp3 = grp3.rename(columns={'shift_length': 'total_worked_hours'}, level=0)
grp3.head()
len(grp3)

# Join the first group with the 3rd
joined = pd.merge(grp, grp3, on='Party_Name')
len(joined)
joined.head()
joined.columns
joined['worked_hours_per_year'] = joined[('total_worked_hours', '')] / joined[('years', '')]
joined['cost_per_hour'] = 65000 / joined['worked_hours_per_year']
joined.head()

# %%

drop_rows_with_extreme_values(joined, 'cost_per_hour', q=0.95)
joined['cost_per_hour'].hist()  # looks bad
plt.title("Cost per agent hour (assuming 65 k / year)")

avg_cost_per_agent_hour = 65000 / (joined[('total_worked_hours', '')].sum() / joined[('years', '')].sum())
print(f"Average cost per agent hour = {avg_cost_per_agent_hour}")

# Average cost per agent hour = 46.62088498620127
