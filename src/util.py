import matplotlib.pyplot as plt
import scipy.stats as st


# %

def datetime_to_columns(df, dt_col):
    df['date'] = df[dt_col].dt.date
    df['year'] = df[dt_col].dt.year
    df['week'] = df[dt_col].dt.isocalendar().week
    df['year_week'] = df['year'].astype(str) + " W" + df['week'].astype(str).map(lambda x: x.rjust(2, ' '))
    df['month'] = df[dt_col].dt.month
    df['year_month'] = df['year'].astype(str) + " M" + df['month'].astype(str).map(lambda x: x.rjust(2, ' '))
    df['day'] = df[dt_col].dt.day
    df['weekday'] = df[dt_col].dt.dayofweek  # Monday=0
    df['weekday_name'] = df[dt_col].dt.day_name().str.slice(0, 3)
    df['hour'] = df[dt_col].dt.hour


# %%
"""
# Clean
"""


def drop_rows_with_extreme_values(df, col, q=0.99):
    cutoff = df[col].quantile(q)
    n_rows = (df[col] >= cutoff).sum()
    print(f'{col} is truncated at {q}% quantile {cutoff} removing {n_rows} of {len(df)} rows')
    df.drop(df[(df[col] >= cutoff)].index, inplace=True)


def replace_missing(df, col, value=0):
    n_missing = df["HoldTime"].isnull().sum()
    print(f'{col}: replace {n_missing} missing values with {value}.')
    df[col] = df[col].fillna(0)


def clean_dataset_safe(df):
    df_clean = df.copy()
    replace_missing(df_clean, 'HoldTime', 0)
    replace_missing(df_clean, 'WrapTime', 0)
    return df_clean


def clean_dataset(df):
    df_clean = df.copy()
    drop_rows_with_extreme_values(df_clean, 'QueuedTime')
    drop_rows_with_extreme_values(df_clean, 'TalkTime')
    replace_missing(df_clean, 'HoldTime', 0)
    drop_rows_with_extreme_values(df_clean, 'HoldTime')
    replace_missing(df_clean, 'WrapTime', 0)
    drop_rows_with_extreme_values(df_clean, 'WrapTime')
    drop_rows_with_extreme_values(df_clean, 'WaitTime')
    return df_clean


# %%
"""
# Curve fitting
"""


def poly5(x, a, b, c, d, e, f):
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f


def calculate_mae(x, y, func, params):
    fitted = func(x, *params)
    return abs(fitted - y).sum() / len(x)


# %%
"""
# List of distributions
"""

distributions = [st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi,
                 st.chi2, st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib,
                 st.exponpow, st.f, st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto,
                 st.gennorm, st.genexpon, st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic,
                 st.gilbrat, st.gompertz, st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm,
                 st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu,
                 st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l, st.levy_stable, st.logistic, st.loggamma,
                 st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2, st.ncf, st.nct,
                 st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
                 st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
                 st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max,
                 st.wrapcauchy]

# %%
"""
# Plotting 
"""


def show_histogram(series, title, bins=10):
    fig, ax = plt.subplots()
    ax.hist(series, bins=bins)
    ax.set_title(title)
    plt.show()
