import matplotlib
import pandas as pd
import seaborn as sns

from typing import List, Dict, Optional

def draw_line(experiment_df: pd.DataFrame, x: str, tag: str, ax: Optional[matplotlib.axes.Axes] = None) -> matplotlib.axes.Axes:
    """
        Return a line plot of the data provided in the dataframe.

        :param experiment_df: Dataframe containing metrics to plot
        :param x: Column to plot on x-axis
        :param tag: Column to plot on y-axis
        :return: matplotlib.axes.Axes object with the plotted data
    """
    return sns.lineplot(data=experiment_df, x=x, y=tag, ax=ax)


def separate_exps(experiments_df: pd.DataFrame, tags: List, start_step: int) -> Dict[str, pd.DataFrame]:
    """
        Takes an experiment dataframe containing at least one experiment and
        returns a list of dataframes each corresponding to a separate experiment.

        :param experiments_df: DataFrame containing at list one experiment
        :param tags: User specified tags corresponding to which metrics to keep
        :param start_step: First step value for which a metric has been logged
        :return: Dict mapping experiment names to experiment dfs
    """

    # if single: return [exp_df_to_tags_df(experiments_df, tags, start_step)]
    exp_dfs = {}

    exp_names = set([name_run.split('/')[0] for name_run in experiments_df.run.unique()])
    experiments = experiments_df.run.apply(lambda run: run.split('/')[0])
    for exp_name in exp_names:
        experiment_df = exp_df_to_tags_df(
            experiments_df[experiments_df.run.map(lambda x: exp_name + '/' in x)],
            tags,
            start_step
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

    dfs = []
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
        dfs.append(run_df)

    df = pd.concat(dfs)
    
    return df