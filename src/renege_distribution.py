import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data_folder_path

# %%

df = pd.read_csv(data_folder_path / 'all.csv')
df_reneged = pd.read_csv(data_folder_path / 'reneged.csv')

print(df.head())
df
df['WaitTime'].describe()
df['WaitTime'].quantile(0.99)
df = df.loc[df['WaitTime'] <= 276, :]

fig, ax = plt.subplots()
ax.hist(df['WaitTime'])
plt.show()

# %%


df_reneged = df_reneged.loc[df_reneged['WaitTime'] <= df_reneged['WaitTime'].quantile(0.99), :]
fig, ax = plt.subplots()
ax.hist(df_reneged['WaitTime'])
plt.show()


# %%

def exponenial(x, scale):
    lambda_ = 1.0 / scale
    return lambda_ * np.exp(-lambda_ * x)


x = np.arange(100, step=0.1)
plt.plot(x, exponenial(x, 2))
plt.show()
