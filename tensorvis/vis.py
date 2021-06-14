import os
import json
import click
import random
import pandas as pd
import tensorboard as tb
import plotly.express as px
import plotly.graph_objects as go

from tensorvis.utils import separate_exps, DRAW_FN_MAP, update_layout

CONFIG = {
    "seed": random.randint(0, 2 ** 10 - 1),
    "default_colors": {
        "qualitative": px.colors.qualitative.Plotly,
        "sequential": px.colors.sequential.Plotly3,
    },
}
TENSORPLOT_ROOT = ".tensorplot"
CONFIG_FILE = "config.json"
EXP_LOG = "experiment_log.csv"


class CustomGroup(click.Group):
    """
    Custom class to grab parameters sent to invoked subcommands.
    """

    def invoke(self, ctx):
        ctx.obj = dict(args=tuple(ctx.args))
        super(CustomGroup, self).invoke(ctx)


@click.group(cls=CustomGroup)
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
    ctx.obj["root"] = os.path.join(os.environ["HOME"], TENSORPLOT_ROOT)
    ctx.obj["exp_log_path"] = os.path.join(ctx.obj["root"], EXP_LOG)
    ctx.obj["config_path"] = os.path.join(ctx.obj["root"], CONFIG_FILE)
    if os.path.exists(ctx.obj["config_path"]):
        config = json.load(open(ctx.obj["config_path"], "r"))
    else:
        os.mkdir(ctx.obj["root"])
        json.dump(CONFIG, open(ctx.obj["config_path"], "w"))
        config = CONFIG
    ctx.obj["colors"] = config["default_colors"]

    # Finally seed random number generator to pick the same colours
    # random.seed(config["seed"])


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
    json.dump(ctx.obj["colors"], open(ctx.obj["config_path"], "w"))


@cli.command("upload")
@click.pass_context
@click.argument("path", type=click.Path(exists=True))
@click.option("-n", "--name", "name", help="Name to give to experiment", default=None)
def upload(ctx, path, name):
    """
    Upload experiment to tensorboard dev from path.

    PATH is the path to directory holding experiment events.
    """
    import subprocess
    import datetime

    comm = ["tensorboard", "dev", "upload", "--logdir", "--one_shot"]
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
        if not os.path.exists(ctx.obj["exp_log_path"]):
            df.to_csv(ctx.obj["exp_log_path"], index=False)
        else:
            old_df = pd.read_csv(ctx.obj["exp_log_path"])
            new_df = old_df.append(df, ignore_index=True)
            new_df.to_csv(ctx.obj["exp_log_path"], index=False)
        print(f"Experiment ID returned from tensorboard: {exp_id}")
    except subprocess.CalledProcessError as err:
        print(err.stderr)


@cli.command("download")
@click.pass_context
@click.argument("name")
def download(ctx, name):
    """
    Download experiment file as csv.
    NAME is the name of the experiment data to download from Tensorboard dev.
    """
    exps_df = pd.read_csv(ctx.obj["exp_log_path"])
    exp_id = exps_df[exps_df["name"] == name]["id"].values[0]

    # Get experiment data using tensorboard and experiment id
    tb_experiment = tb.data.experimental.ExperimentFromDev(exp_id)

    exp_df = tb_experiment.get_scalars()
    exp_df = exp_df.pivot_table(index=["run", "step"], columns="tag", values="value")
    exp_df = exp_df.reset_index()
    exp_df.columns.name = None
    exp_df.columns.names = [None for _ in exp_df.columns.names]
    exp_df.to_csv(os.path.join(ctx.obj["root"], exp_id + ".csv"))


@cli.command("plot")
@click.pass_context
@click.argument("name")
@click.option(
    "-t",
    "--tags",
    "tags",
    help="A comma separated list of strings that specifies which metrics to plot.\
            Some runs may have more than one scalar plotted.",
)
@click.option(
    "--compare", help="Denotes whether experiments are to be compared and appear on the same plot.", is_flag=True
)
@click.option("--variance", help="Whether to plot variance in plots", is_flag=True)
@click.option(
    "-pt",
    "--plot-type",
    type=click.Choice(["scatter", "bar", "hist", "line"]),
    help="Type of plot to create.",
    default="line",
)
@click.option("--title", "title", help="Title for the plot", default="Performance")
def plot(ctx, name, tags, compare, variance, plot_type, title):
    """
    Plots the data in the csv file identified by PATH.

    NAME is the name of the experiment downloaded from Tensorboard dev.
    """

    exps_df = pd.read_csv(ctx.obj["exp_log_path"])
    exp_id = exps_df[exps_df["name"] == name]["id"].values[0]

    exp_df = pd.read_csv(os.path.join(ctx.obj["root"], f"{exp_id}.csv"), index_col=0)
    tags = tags.split(",")
    experiment_dfs = separate_exps(exp_df, tags)
    if isinstance(experiment_dfs, pd.DataFrame):
        experiment_dfs = {name: experiment_dfs}

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
            label_name = exp_name if len(experiment_dfs) > 1 else tag
            experiment_df = experiment_dfs[exp_name]
            tag_run_cols = [col for col in experiment_df.columns if tag in col]
            tag_run_df = experiment_df[tag_run_cols].copy(deep=True)
            tag_run_df.dropna(inplace=True)
            tag_run_df["mean"] = tag_run_df.mean(axis=1)
            tag_run_df["std"] = tag_run_df.std(axis=1)
            if compare:
                traces.extend(DRAW_FN_MAP[plot_type](tag_run_df, label_name, variance, colors[exp_name]))
            else:
                traces.extend(DRAW_FN_MAP[plot_type](tag_run_df, label_name, variance))
                fig = go.Figure(traces)
                fig = update_layout(fig, title, "Episodes", tag.title())
                if ctx.obj["vis"]:
                    fig.show()
                if ctx.obj["save"]:
                    fig.write_image(f"{exp_name}_{tag}.png", width=2000, height=1000, scale=1)
                traces = []

        if compare:
            fig = go.Figure(traces)
            fig = update_layout(fig, title, "Episodes", tag.title())
            if ctx.obj["vis"]:
                fig.show()
            if ctx.obj["save"]:
                fig.write_image(f"{name}_{tag}.png", width=2000, height=1000, scale=1)


@cli.command("embedding")
@click.pass_context
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-c",
    "--components",
    "comps",
    help="A comma separated list of columns to use for plotting",
)
@click.option("-l", "--label", "label", help="Column to be used for the hue parameter")
@click.option("-np", "--num-points", "points", help="Number of data points to plot from each unique label", default=-1)
@click.option("-t", "--title", "title", help="Title for the embedding plot", default="Embedding")
def embedding(ctx, path, comps, label, points, title):
    """ Plots a scatter plot of the data resulting from an embedding transformation. """
    embedding_df = pd.read_csv(path)
    unique = embedding_df[label].unique()
    unique = list(unique)
    colors = {
        k: col
        for k, col in zip(
            unique,
            random.sample(ctx.obj["colors"]["qualitative"], len(embedding_df[label].unique())),
        )
    }
    traces = []
    comps = comps.split(",")
    for comp_label, color in colors.items():
        comp_label_df = embedding_df[embedding_df[label] == comp_label]
        if points != -1:
            comp_label_df = comp_label_df.sample(n=points)
        traces.append(DRAW_FN_MAP["scatter"](comp_label_df, comps, color, comp_label))

    fig = go.Figure(traces)
    # if ctx.obj["save"]:
    #     fig = ax.get_figure()
    #     fig.savefig(f"{tag}.png")
    fig = update_layout(fig, title, comps[0], comps[1])
    if ctx.obj["vis"]:
        fig.show()
