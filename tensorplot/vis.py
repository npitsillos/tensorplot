import click
import pandas as pd
import tensorboard as tb
import plotly.express as px
from .utils.utils import draw


@click.group()
@click.option(
    "-eid",
    "--experiment-id",
    "experiment",
    type=str,
    default=None,
    show_default=True,
    required=False)
@click.option(
    "-f",
    "--file",
    "file",
    type=str,
    default=None,
    show_default=True,
    required=False)
@click.pass_context
def cli(ctx, experiment, file):
    """
        Visualisation Tool for My PhD. Integrates tensorboard with plotly to
        automate result visualisation and customize it.\n
        EXPERIMENT-ID is the ID of the experiment provided when the events
                      file is uploaded to Tensorboard dev.
    """
    ctx.ensure_object(dict)
    ctx.obj["eid"] = experiment
    ctx.obj["file"] = file


@cli.command("download")
def download(ctx):
    """
        Download experiment file as csv.
    """
    # Get experiment data using tensorboard and experiment id
    experiment = tb.data.experimental.ExperimentFromDev(ctx.obj["eid"])

    exp_df = experiment.get_scalars(pivot=True)

    exp_df.to_csv(ctx.obj["eid"])


@cli.command("plot")
@click.option(
    "--vis/--no-vis",
    help="Whether to visualise plot or not.",
    default=True)
@click.option(
    "-r",
    "--run",
    "runs",
    multiple=True,
    help="A series of strings corresponding to which runs to visualise.\
            A run is a subdirectory of the original logdir.")
@click.option(
    "-sc",
    "--scalar",
    "scalars",
    multiple=True,
    help="A list of strings that specifies which metrics to plot.\
            Some runs may have more than one scalar plotted.")
@click.option(
    "-s",
    "--save",
    is_flag=True,
    help="If present will save plots to the path where this was executed.")
@click.option(
    "-pt",
    "--plot-type",
    type=click.Choice(["scatter", "bar", "hist"]),
    help="Type of plot to create.")
def plot(ctx, vis, runs, scalars, save, plot_type):
    """
        Plots the data in the csv file identified by EXPERIMENT_ID\
            another file name.
    """
    # exp_df = pd.read_csv(ctx.obj["eid"] + ".csv")

    # for run in runs:
    #     run_df = exp_df[exp_df["run"].str.contains(run)]
    #     run_df = run_df.pivot_table(
    #         values=("value"),
    #         index=["run", "step"],
    #         columns="tag",
    #         dropna=False,
    #     )

    #     # Taken from https://github.com/tensorflow/tensorboard/blob/master/tensorboard/data/experimental/experiment_from_dev.py
    #     run_df = run_df.reset_index()
    #     run_df.columns.name = None
    #     run_df.columns.names = [None for name in run_df.columns.names]
    # else:
    #     run_df = exp_df[exp_df["run"].str.contains(run)]

    #     for scalar in scalars:
    #         fig = draw(run_df, run, scalar)
    #         if vis:
    #             fig.show()
    #         if save:
    #             fig.write_image("{}.png".format(run))

@cli.command("embedding")
@click.option(
    "-c",
    "--components",
    "comps",
    type=int,
    default=2,
    required=True
)
@click.pass_context
def embedding(ctx, comps):
    """ Plots a scatter plot of the data resulting from an embedding transformation. """

    file = ctx.obj["file"]

    df = pd.read_csv(file)

    if comps == 2:
        fig = px.scatter(df, x="z1", y="z2", color="label")
    else:
        fig = px.scatter_3d(df, x="z1", y="z2", z="z3", color="label")

    fig.update_layout(
        title=file,
        hovermode="x",
        title_x=0.5,
        font=dict(
            family="Courier New, monospace",
            size=18
        )
    )
    fig.show()