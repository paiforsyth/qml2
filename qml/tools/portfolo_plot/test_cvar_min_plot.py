import xarray as xr
from plotly.graph_objs import Figure

from qml.tools.portfolo_plot.cvar_min_plot import plot_mean_cvar_frontier, add_risk_return_percentage_to_ds, \
    plot_asset_holdings


def test_mean_cvar_frontier_plot(example_efficient_frontier: xr.Dataset):
    plot = plot_mean_cvar_frontier(example_efficient_frontier)
    assert isinstance(plot, Figure)

def test_asset_holdings_plot(example_efficient_frontier: xr.Dataset):
    plot = plot_asset_holdings(example_efficient_frontier)
    assert isinstance(plot, Figure)
