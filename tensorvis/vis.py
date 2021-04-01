import os
import json
import click
import random
import pandas as pd
import tensorboard as tb
import plotly.express as px
import plotly.graph_objects as go

from tensorvis.utils import separate_exps, DRAW_FN_MAP

DEFAULT_COLORS = {
    "qualitative": px.colors.qualitative.Plotly,
    "sequential": px.colors.sequential.Plotly3,
}


@click.group()
@click.pass_context
@click.option(
    "-s", "--save", is_flag=True, help="If present will save plots to the path where this was executed.", default=True
)
@click.option("--vis/--no-vis", help="Whether to visualise plot or not.", default=True)
def cli(ctx, save, vis):
    """
    Visualisation Tool for My PhD. Integrates tensorboard with plotly to
    automate result visualisation and customize it.\n
    EXPERIMENT-ID is the ID of the experiment provided when the events
                  file is uploaded to Tensorboard dev.
    """

    ctx.ensure_object(dict)

    ctx.obj["save"] = save
    ctx.obj["vis"] = vis
    ctx.obj["root"] = os.path.join(os.environ["HOME"], ".tensorplot")
    config_path = os.path.join(ctx.obj["root"], "config.json")
    if os.path.exists(config_path):
        ctx.obj["colors"] = json.load(open(config_path, "r"))
    else:
        os.mkdir(ctx.obj["root"])
        json.dump(DEFAULT_COLORS, open(config_path, "w"))
        ctx.obj["colors"] = DEFAULT_COLORS
    ctx.obj["config-path"] = config_path


@cli.command("set_palette")
@click.pass_context
@click.option("--qualitative", help="Qualitative color scale string", default="Plotly")
@click.option("--sequential", help="Sualitative color scale string", default="Plotly3")
def set_palette(ctx, qualitative, sequential):
    """
    Sets the palette for all different color scales available in plotly and
    saves it in $HOME/.tensorplot/config.json
    """

    # Hacky way to get all qualitative colorscales
    assert qualitative in [x["y"][0] for x in px.colors.qualitative.swatches()["data"]]
    assert sequential.lower() in px.colors.named_colorscales()

    ctx.obj["colors"]["qualitative"] = getattr(px.colors.qualitative, qualitative)
    ctx.obj["colors"]["sequential"] = getattr(px.colors.sequential, sequential)
    json.dump(ctx.obj["colors"], open(ctx.obj["config-path"], "w"))


@cli.command("upload")
@click.argument("path", type=click.Path(exists=True))
@click.option("-n", "--name", "name", help="Name to give to experiment", default=None)
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
        exp_id = return_str.decode("utf-8").split("\n")[-2].split()[-1].split("/")[-2]
        data = {
            "date": pd.Series([datetime.date.today().strftime("%d%m%y")], dtype=str),
            "name": pd.Series(["" if name is None else name], dtype=str),
            "id": pd.Series([exp_id], dtype=str),
        }
        df = pd.DataFrame(data=data)
        if not os.path.exists(log_file_path):
            df.to_csv(log_file_path, index=False)
        else:
            old_df = pd.read_csv(log_file_path)
            new_df = old_df.append(df, ignore_index=True)
            new_df.to_csv(log_file_path, index=False)
        print(f"Experiment ID returned from tensorboard: {exp_id}")
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
            Some runs may have more than one scalar plotted.",
)
@click.option(
    "-es", "--eval-step", "eval_step", help="First step value corresponding to datapoints for success.", default=2000
)
@click.option(
    "-ms",
    "--max-step",
    "max_step",
    help="Maximum step to plot up to.  Handles cases when algorithms have mismatch\
            in number of steps trained on.",
    type=int,
)
@click.option(
    "--compare", help="Denotes whether experiments are to be compared and appear on the same plot.", is_flag=True
)
@click.option(
    "-pt",
    "--plot-type",
    type=click.Choice(["scatter", "bar", "hist", "line"]),
    help="Type of plot to create.",
    default="line",
)
def plot(ctx, path, tags, eval_step, max_step, compare, plot_type):
    """
    Plots the data in the csv file identified by PATH.

    PATH is the id of the experiment to download from Tensorboard dev.
    """

    exp_df = pd.read_csv(path, index_col=0)
    tags = tags.split(",")
    experiment_dfs = separate_exps(exp_df, tags, eval_step)
    if compare:
        colors = {
            k: col
            for k, col in zip(
                experiment_dfs.keys(), random.sample(ctx.obj["colors"]["qualitative"], len(experiment_dfs.keys()))
            )
        }
    for tag in tags:
        traces = []
        for exp_name in experiment_dfs.keys():
            experiment_df = experiment_dfs[exp_name]
            tag_run_cols = [col for col in experiment_df.columns if tag in col]
            tag_run_df = experiment_df[tag_run_cols].copy(deep=True)
            tag_run_df["mean"] = tag_run_df.mean(axis=1)
            tag_run_df["std"] = tag_run_df.std(axis=1)
            if compare:
                traces.extend(DRAW_FN_MAP[plot_type](tag_run_df, exp_name, colors[exp_name]))
            else:
                traces.extend(DRAW_FN_MAP[plot_type](tag_run_df, exp_name))

            if not compare:
                fig = go.Figure(traces)
                if ctx.obj["vis"]:
                    fig.show()
                if ctx.obj["save"]:
                    fig.write_image(f"{exp_name}_{tag}.png")
                traces = []

        if compare:
            fig = go.Figure(traces)
            if ctx.obj["vis"]:
                fig.show()
            if ctx.obj["save"]:
                fig.write_image(f"{exp_name}_{tag}.png")


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
    pass
    # df = pd.read_csv(path)

    # if comps == 2:
    #     ax = sns.scatter(data=df, x="z1", y="z2", hue="label")
    # else:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     ax.set_xlabel("z1")
    #     ax.set_ylabel("z2")
    #     ax.set_zlabel("z3")
    #     ax.scatter(df["z1"], df["z2"], df["z3"])

    # if ctx.obj["save"]:
    #     fig = ax.get_figure()
    #     fig.savefig(f"{tag}.png")
    # if ctx.obj["vis"]:
    #     plt.show()
