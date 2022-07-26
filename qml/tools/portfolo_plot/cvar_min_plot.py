import plotly.express as px
import xarray as xr
from plotly.graph_objs import Figure

from qml.tools.monte_carlo.brownian_motion import START_VALUE
from qml.tools.portfolio_optimize.cvx_optimize import HOLDINGS, INSTRUMENT_DIMENSION, TRAINING_CVAR, TRAINING_RETURN

RETURN_PERCENTAGE = "return (%)"
CVAR_PERCENTAGE = "cvar (%)"
DOLLAR_HOLDINGS = "holdings ($)"


def add_risk_return_percentage_to_ds(ds: xr.Dataset, initial_budget: float = 1.0) -> xr.Dataset:
    ds = ds.copy()
    ds[RETURN_PERCENTAGE] = ds[TRAINING_RETURN] / initial_budget
    ds[CVAR_PERCENTAGE] = ds[TRAINING_CVAR] / initial_budget
    ds[DOLLAR_HOLDINGS] = ds[HOLDINGS] * ds[START_VALUE]
    return ds


def plot_mean_cvar_frontier(ds: xr.Dataset) -> Figure:
    ds = xr.Dataset({name: ds[name] for name in [CVAR_PERCENTAGE, RETURN_PERCENTAGE]})
    df = ds.to_dataframe().reset_index()
    return px.line(df, x=CVAR_PERCENTAGE, y=RETURN_PERCENTAGE)


def plot_asset_holdings(ds: xr.Dataset) -> Figure:
    ds = xr.Dataset({name: ds[name] for name in [DOLLAR_HOLDINGS, CVAR_PERCENTAGE]})
    df = ds.to_dataframe().reset_index()
    return px.line(df, x=CVAR_PERCENTAGE, y=DOLLAR_HOLDINGS, color=INSTRUMENT_DIMENSION)
