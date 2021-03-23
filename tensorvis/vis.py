import os
import click
import pandas as pd
import tensorboard as tb
import seaborn as sns
import matplotlib.pyplot as plt

from tensorvis.utils import DRAW_FN_MAP

@click.group()
@click.pass_context
@click.option(
    "-s",
    "--save",
    is_flag=True,
    help="If present will save plots to the path where this was executed.",
    default=True)
@click.option(
    "--vis/--no-vis",
    help="Whether to visualise plot or not.",
    default=True)
def cli(ctx, save, vis):
    """
        Visualisation Tool for My PhD. Integrates tensorboard with plotly to
        automate result visualisation and customize it.\n
        EXPERIMENT-ID is the ID of the experiment provided when the events
                      file is uploaded to Tensorboard dev.
    """
    sns.set_theme()

    ctx.ensure_object(dict)

    ctx.obj["save"] = save
    ctx.obj["vis"] = vis

@cli.command("upload")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-n",
    "--name",
    "name",
    help="Name to give to experiment",
    default=None)
def upload(path, name):
    """
        Upload experiment to tensorboard dev from path.
        
        PATH is the path to directory holding experiment events.
    """
    import subprocess
    import datetime

    comm = ["tensorboard", "dev", "upload", "--logdir", "--one_shot"]
    log_file_path = os.path.join(os.getcwd(), "experiment_log.csv")

    try:
        comm.insert(4, path)
        if name is not None:
            comm.extend(["--name", name])
        process_ret = subprocess.run(comm, capture_output=True)
        return_str = process_ret.stdout
        data = {
            "date": pd.Series([datetime.date.today().strftime("%d%m%y")], dtype=str),
            "name": pd.Series(["" if name is None else name], dtype=str),
            "id": pd.Series([return_str.decode("utf-8").split("\n")[-2].split()[-1].split('/')[-2]], dtype=str)
        }
        df = pd.DataFrame(data=data)
        if not os.path.exists(log_file_path):
            df.to_csv(log_file_path, index=False)
        else:
            old_df = pd.read_csv(log_file_path)
            new_df = old_df.append(df, ignore_index=True)
            new_df.to_csv(log_file_path, index=False)
    except subprocess.CalledProcessError as err:
        print(err.stderr)


@cli.command("download")
@click.argument("experiment")
def download(experiment):
    """
        Download experiment file as csv.

        EXPERIMENT is the id of the experiment to download from Tensorboard dev.
    """
    # Get experiment data using tensorboard and experiment id
    tb_experiment = tb.data.experimental.ExperimentFromDev(experiment)

    exp_df = tb_experiment.get_scalars()
    exp_df = exp_df.pivot_table(index=["run", "step"], columns="tag", values="value")
    exp_df = exp_df.reset_index()
    exp_df.columns.name = None
    exp_df.columns.names = [None for name in exp_df.columns.names]
    exp_df.to_csv(os.path.join(os.getcwd(), experiment + ".csv"))

@cli.command("plot")
@click.pass_context
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-t",
    "--tags",
    "tags",
    help="A comma separated list of strings that specifies which metrics to plot.\
            Some runs may have more than one scalar plotted.")
@click.option(
    "-es",
    "--eval-step",
    "eval_step",
    help="First step value corresponding to datapoints for success.",
    default=2000)
@click.option(
    "-ms",
    "--max-step",
    "max_step",
    help="Maximum step to plot up to.  Handles cases when algorithms have mismatch\
            in number of steps trained on.",
    type=int
)
@click.option(
    "-pt",
    "--plot-type",
    type=click.Choice(["scatter", "bar", "hist", "line"]),
    help="Type of plot to create.",
    default="line")
def plot(ctx, path, tags, eval_step, max_step, plot_type):
    """
        Plots the data in the csv file identified by EXPERIMENT.
        
        EXPERIMENT is the id of the experiment to download from Tensorboard dev.
    """
    exp_df = pd.read_csv(path, index_col=0)
    # Get average success and episode length for each step
    # over all runs
    tags = tags.split(',')
    exps = exp_df.run.apply(lambda run: run.split('/')[0])
    max_step = max_step if max_step is not None else exp_df["step"].max()
    new_df = exp_df.loc[(exp_df["step"] >= eval_step) & (exp_df["step"] <= max_step)]
    for tag in tags:
        ax = DRAW_FN_MAP[plot_type](new_df, "step", tag, exps)
        if ctx.obj["save"]:
            fig = ax.get_figure()
            fig.savefig(f"{tag}.png")
        if ctx.obj["vis"]:
            plt.show()

@cli.command("embedding")
@click.pass_context
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-c",
    "--components",
    "comps",
    type=int,
)
def embedding(ctx, path, comps):
    """ Plots a scatter plot of the data resulting from an embedding transformation. """

    df = pd.read_csv(path)

    if comps == 2:
        ax = sns.scatter(data=df, x="z1", y="z2", hue="label")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
        ax.scatter(df["z1"], df["z2"], df["z3"])

    if ctx.obj["save"]:
        fig = ax.get_figure()
        fig.savefig(f"{tag}.png")
    if ctx.obj["vis"]:
        plt.show()
 