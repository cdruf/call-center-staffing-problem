import multiprocessing
import warnings
from math import floor, ceil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from src import data_folder_path
from src.util import clean_dataset
from src.util import distributions

# %%
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)


def load_data():
    df = pd.read_csv(data_folder_path / 'answered.csv')
    df = clean_dataset(df)
    series = df['AgentTime']
    series = series.dropna()  # from TalkTime
    print(series.head())
    return series


def get_center_points(arr):
    return (arr + np.roll(arr, -1))[:-1] / 2.0


class FitResult:

    def __init__(self, distribution, params, xs, pdf, sse):
        """

        Parameters
        ----------
        distribution : ?
            The distribution.
        params : ?
            Distribution parameters
        xs : ?
            x-values for PDF
        pdf : ?
            Probability density function values to the x-values
        sse : float
            Sum of squared errors.
        """
        self.distribution = distribution
        self.params = params
        self.pdf = pdf
        self.sse = sse

    def __str__(self):
        return f"name={self.distribution.name}, params={self.params}, sse={self.sse}"


def fit_distribution(series, distribution, return_container=None):
    print(f"Fit distribution {distribution.name}")
    y, x = np.histogram(series, bins=200, density=True)
    x = get_center_points(x)
    try:
        with warnings.catch_warnings():  # Ignore warnings from data that can't be fit
            warnings.filterwarnings('ignore')

            # Fit distribution to data
            params = distribution.fit(series)

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and sum of squared errors with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))

            f = FitResult(distribution, params, x, pdf, sse)
            if return_container is not None:
                return_container.put(f)
            return f

    except Exception:
        pass


def fit_distribution_with_timeout(series, distribution):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=fit_distribution, args=(series, distribution, queue))
    p.start()

    # Wait for 30 seconds or until process finishes
    p.join(30)

    if p.is_alive():  # thread is still active
        print("timeout => stop fitting")
        p.terminate()  # Terminate - may not work if process is stuck for good
        if p.is_alive():
            p.kill()  # works for sure, no chance for process to finish nicely however
        p.join()

    if queue.qsize() > 0:
        return queue.get()
    return None


def fit_distributions(series):
    results = []
    for i, distribution in enumerate(distributions):
        ret = fit_distribution_with_timeout(series, distribution)
        if ret is not None:
            results.append(ret)
    print(f"Fitted {len(results)} distributions")
    print("Sort ...")
    list.sort(results, key=lambda r: r.sse)
    print(' > '.join([d.name for d in results]))
    return results


def plot_first_n(results, n=3):
    n = min(n, len(results))
    print(f"Plot {n} distributions")
    if n == 1:
        fig, ax = plt.subplots()
        axs = (ax,)
    else:
        fig, axs = plt.subplots(n, 1)

    mini = floor(min(series))
    maxi = ceil(max(series))

    for i, r in enumerate(results[:n]):
        axs[i].hist(series, bins=100, density=True)
        axs[i].title.set_text(r.distribution.name)
        x = np.linspace(mini, maxi, num=(maxi - mini) * 10)
        arg = r.params[:-2]
        loc = r.params[-2]
        scale = r.params[-1]
        y = r.distribution.pdf(x, loc=loc, scale=scale, *arg)
        axs[i].plot(x, y, 'r-', lw=5, alpha=0.6, label=r.distribution.name)

    plt.show()
    plt.tight_layout()


# %%


if __name__ == '__main__':
    series = load_data()
    results = fit_distributions(series)
    plot_first_n(results)

# %%

series = load_data()
result = fit_distribution(series, st.beta)
print(result)
# name=beta,
# params=(1.2396816025688038, 13.148071086271802, 0.9922421985825189, 3716.3614369650018), sse=2.699348948778481e-06

a = 1.2396816025688038
b = 13.148071086271802
loc = 0.9922421985825189
scale = 3716.3614369650018

fig, ax = plt.subplots(1, 1)
ax.hist(series, density=True, bins=50)
x = np.linspace(st.beta.ppf(0.01, a, b, loc, scale), st.beta.ppf(0.99, a, b, loc, scale), 100)
ax.plot(x, st.beta.pdf(x, a, b, loc, scale), 'r-', lw=5, alpha=0.6, label='beta pdf')
plt.title('Agent time')
plt.xlabel('Agent busy (secs)')
plt.ylabel('Density')
plt.tight_layout()

# %%
random_pop = st.beta.rvs(a, b, loc, scale, size=1000)
pd.Series(random_pop).hist(density=True)
