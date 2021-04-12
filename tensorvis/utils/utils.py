import plotly
import pandas as pd
import plotly.graph_objects as go

from ast import literal_eval
from typing import List, Union, Dict


def _color_is_hex(color: str) -> bool:
    return color[0] == "#"


def _str_rgb_color_to_rgba_str(color: str, opacity: float) -> str:
    return "rgba(%d,%d,%d,%f)" % (literal_eval(color[3:]) + (opacity,))


def _hex_to_rgba_string(color: str, opacity: int) -> str:
    rgba = plotly.colors.hex_to_rgb(color) + (opacity,)
    return "rgba(%d,%d,%d,%f)" % rgba


def draw_line(experiment_df: pd.DataFrame, label_name: str, color: str = "#636EFA") -> List[go.Scatter]:
    """
    Return a line plot of the data provided in the dataframe.

    :param experiment_df: Dataframe containing metrics to plot
    :param label_name: Name of experiment to give to legend
    :param color: Color to draw line with
    :return: go.Scatter object with the plotted data
    """

    mean_value_trace = go.Scatter(
        x=experiment_df.index,
        y=experiment_df["mean"],
        mode="lines",
        showlegend=True,
        name=label_name,
        line=dict(color=color),
    )

    mean_plus_std = go.Scatter(
        x=experiment_df.index,
        y=experiment_df["mean"] + experiment_df["std"],
        mode="lines",
        line=dict(width=0),
        name=f"{label_name} upper bound",
        showlegend=False,
    )

    if _color_is_hex(color):
        color = _hex_to_rgba_string(color, 0.3)
    else:
        color = _str_rgb_color_to_rgba_str(color, 0.3)

    mean_minus_std = go.Scatter(
        x=experiment_df.index,
        y=experiment_df["mean"] - experiment_df["std"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        fillcolor=color,
        fill="tonexty",
        name=f"{label_name} lower bound",
    )

    return [mean_value_trace, mean_plus_std, mean_minus_std]


def draw_scatter(
    experiment_df: pd.DataFrame, comps: List[str], color: str, label: str
) -> Union[go.Scatter, go.Scatter3d]:
    """
    Return a 2D or 3D scatter plot of the experiment dataframe.

    :param experiment_df: Dataframe containing embedding to plot
    :param comps: List of embedding components defining columns in the dataframe
    :param color: Color for points in embedding
    :param label: Label for the components plotted
    :return: go.Scatter or go.Scatter3d with the plotted data
    """
    if len(comps) == 2:
        embedding_plot = go.Scatter(
            x=experiment_df[comps[0]],
            y=experiment_df[comps[1]],
            mode="markers",
            marker=dict(color=color),
            name=label,
            showlegend=True,
        )
    else:
        embedding_plot = go.Scatter3d(
            x=experiment_df[comps[0]],
            y=experiment_df[comps[1]],
            z=experiment_df[comps[2]],
            mode="markers",
            marker=dict(color=color),
            name=label,
            showlegend=True,
        )

    return embedding_plot


def separate_exps(
    experiments_df: pd.DataFrame, tags: List[str], start_step: int
) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Takes an experiment dataframe containing at least one experiment and
    returns a list of dataframes each corresponding to a separate experiment.

    :param experiments_df: DataFrame containing at least one experiment
    :param tags: User specified tags corresponding to which metrics to keep
    :param start_step: First step value for which a metric has been logged
    :return: Dict mapping experiment names to experiment dfs
    """

    exp_dfs = {}
    # Case where single experiment in the dataframe so don't consider each run as
    # a separate experiment.
    if experiments_df["run"][0].find("/") == -1:
        return exp_df_to_tags_df(experiments_df, tags, start_step)

    exp_names = set([name_run.split("/")[0] for name_run in experiments_df.run.unique()])
    for exp_name in exp_names:
        experiment_df = exp_df_to_tags_df(
            experiments_df[experiments_df.run.map(lambda x: exp_name + "/" in x)], tags, start_step
        )

        exp_dfs[exp_name] = experiment_df
    return exp_dfs


def exp_df_to_tags_df(experiment_df: pd.DataFrame, tags: List[str], start_step: int) -> pd.DataFrame:
    """
    Given an experiment dataframe and a tags list it returns a dataframe
    with the step column as the index with a column for each run for every
    specified tag with the addition of an average and std column for each tag.

    :param experiment_df: The experiment DataFrame
    :param tags: User specified tags corresponding to which metrics to keep
    :param start_step: First step value for which a metric has been logged
    :return: pd.DataFrame with all run metrics and the corresponding average
        and standard deviation for each step value.
    """

    def split_run_name(row):
        row.run = row.run.split("/")[-1]
        return row

    dfs = []
    experiment_df = experiment_df.apply(split_run_name, axis="columns")
    runs = experiment_df.run.unique()
    for run in runs:
        run_df = experiment_df[experiment_df["run"] == run].loc[:, ["step"] + tags].set_index("step")
        run_df.index.rename("step", inplace=True)
        index = run_df.first_valid_index()
        run_df = run_df.loc[index:]
        # Get and set a value at 0 step
        zeroth_step_vals = []
        # Skip step column
        for tag in tags:
            zeroth_step_vals.append(run_df.at[start_step, tag])
        run_df.at[0, :] = zeroth_step_vals
        run_df = run_df.sort_index()
        run_df.rename(columns=lambda x: f"{run}_{x}", inplace=True)
        dfs.append(run_df)

    df = pd.concat(dfs, axis=1)

    return df
