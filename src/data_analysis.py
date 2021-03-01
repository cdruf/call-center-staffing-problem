import pandas as pd
import pandas_profiling as pp

from src import input_file_path, data_folder_path
from src.util import clean_dataset, clean_dataset_safe

# %%
# df = pd.read_excel(input_file_path, sheet_name='Sheet1')
# df.to_csv(data_folder_path / 'all.csv', index=False)

df = pd.read_csv(data_folder_path / 'all.csv')
df['Call_Start'] = pd.to_datetime(df['Call_Start'])
print(df.head())
print(df.columns)

print(f'No. records = {df.shape[0]}')
print(f"Date range from {df['Call_Start'].min()} to {df['Call_Start'].max()}")
# => February not complete


# %%

df = clean_dataset_safe(df)

df['WaitTime vs QueuedTime'] = df['WaitTime'] - df['QueuedTime']
df['AgentTime'] = df['TalkTime'] + df['HoldTime'] + df['WrapTime']
# Is the agent busy during hold time?
# * Could be a redirect to another agent in the same department
# * Could be that the agent puts the customer in the waiting line in order to do sth.

df['agent_answer_time'] = df['Call_Start'] + pd.to_timedelta(df['WaitTime'], unit='s')
df['call_end'] = (df['agent_answer_time']
                  + pd.to_timedelta(df['TalkTime'], unit='s')
                  + pd.to_timedelta(df['HoldTime'], unit='s'))
df['agent_finish_time'] = df['call_end'] + pd.to_timedelta(df['WrapTime'], unit='s')


# %%
"""
# Subsets
"""
df_answered = df.loc[df['Exit_Reason'] == 'AgentAnswered', :]
df_reneged = df.loc[df['Exit_Reason'] == 'Abandoned', :]
df_reneged = df_reneged.drop(columns=['Party_Name', 'TalkTime', 'HoldTime', 'WrapTime'])
df_redirected = df.loc[df['Exit_Reason'] == 'Redirected', :]

df_answered.to_csv(data_folder_path / 'answered.csv', index=False)
df_reneged.to_csv(data_folder_path / 'reneged.csv', index=False)
df_redirected.to_csv(data_folder_path / 'redirected.csv', index=False)

# %%
"""
# Profile
"""
pp.ProfileReport(df).to_file(data_folder_path / 'profile.html')

pp.ProfileReport(df_answered).to_file(data_folder_path / 'profile_answered.html')
pp.ProfileReport(df_reneged).to_file(data_folder_path / 'profile_abandoned.html')
pp.ProfileReport(df_redirected).to_file(data_folder_path / 'profile_redirected.html')

# %%
"""
# Profile cleaned data
"""

df_cleaned = clean_dataset(df)
profile = pp.ProfileReport(df_cleaned)
profile.to_file(data_folder_path / 'profile_all_cleaned.html')

# %%

df_answered_cleaned = clean_dataset(df_answered)
profile = pp.ProfileReport(df_answered_cleaned)
profile.to_file(data_folder_path / 'profile_answered_cleaned.html')
