import os
import json
import click
import shutil
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


def complete_exp_names(ctx, param, incomplete):
    names = next(os.walk(os.path.join(os.environ["HOME"], TENSORPLOT_ROOT)))[1]
    return [name for name in names if name.startswith(incomplete)]


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
@click.argument("name")
def upload(ctx, path, name):
    """
    Upload experiment to tensorboard dev from path.

    PATH is the path to directory holding experiment events.\n
    NAME is the name given to the experiment.
    """
    import subprocess
    import datetime

    comm = ["tensorboard", "dev", "upload", "--logdir", "--one_shot"]
    try:
        comm.insert(4, path)
        comm.extend(["--name", name])
        process_ret = subprocess.run(comm, capture_output=True)
        return_str = process_ret.stdout
        exp_id = return_str.decode("utf-8").split("\n")[-2].split()[-1].split("/")[-2]
        data = {
            "date": pd.Series([datetime.date.today().strftime("%d%m%y")], dtype=str),
            "name": pd.Series([name], dtype=str),
            "id": pd.Series([exp_id], dtype=str),
        }
        df = pd.DataFrame(data=data)
        if not os.path.exists(ctx.obj["exp_log_path"]):
            df.to_csv(ctx.obj["exp_log_path"], index=False)
        else:
            old_df = pd.read_csv(ctx.obj["exp_log_path"])
            exps_present = old_df["name"].tolist()
            if name in exps_present:
                idx = old_df.index[old_df["name"] == name]
                old_df.drop(idx, inplace=True)
                old_df.reset_index(drop=True, inplace=True)
            new_df = old_df.append(df, ignore_index=True)
            new_df.to_csv(ctx.obj["exp_log_path"], index=False)

        # Finally create directory in root folder named with `name`
        if not os.path.exists(os.path.join(ctx.obj["root"], name)):
            os.mkdir(os.path.join(ctx.obj["root"], name))

        print(f"Experiment ID returned from tensorboard: {exp_id}")
    except subprocess.CalledProcessError as err:
        print(err.stderr)


@cli.command("download")
@click.pass_context
@click.argument("name", shell_complete=complete_exp_names)
def download(ctx, name):
    """
    Download experiment file as csv.
    NAME is the name of the experiment data to download from Tensorboard dev.
    """
    exps_df = pd.read_csv(ctx.obj["exp_log_path"])
    exp_id = exps_df[exps_df["name"] == name]["id"].values[0]

    # Is this the first time the experiment is downloaded?
    prev_dirs = os.listdir(os.path.join(ctx.obj["root"], name))
    if len(prev_dirs) > 0:
        # Usually only one folder will exist but iterate over whole list
        for f in prev_dirs:
            shutil.rmtree(os.path.join(os.path.join(ctx.obj["root"], name), f))

    exp_path = os.path.join(ctx.obj["root"], name, exp_id)
    if not os.path.exists(exp_path):
        os.mkdir(os.path.join(ctx.obj["root"], name, exp_id))

    # Get experiment data using tensorboard and experiment id
    tb_experiment = tb.data.experimental.ExperimentFromDev(exp_id)

    exp_df = tb_experiment.get_scalars()
    exp_df = exp_df.pivot_table(index=["run", "step"], columns="tag", values="value")
    exp_df = exp_df.reset_index()
    exp_df.columns.name = None
    exp_df.columns.names = [None for _ in exp_df.columns.names]
    exp_df.to_csv(os.path.join(exp_path, exp_id + ".csv"))


@cli.command("plot")
@click.pass_context
@click.argument("name", shell_complete=complete_exp_names)
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
def plot(ctx, name, tags, compare, variance, plot_type):
    """
    Plots the data in the csv file identified by PATH.

    NAME is the name of the experiment downloaded from Tensorboard dev.
    """

    exps_df = pd.read_csv(ctx.obj["exp_log_path"])
    exp_id = exps_df[exps_df["name"] == name]["id"].values[0]
    exp_path = os.path.join(ctx.obj["root"], name, exp_id)
    exp_df = pd.read_csv(os.path.join(exp_path, f"{exp_id}.csv"), index_col=0)
    tags = tags.split(",")
    experiment_dfs = separate_exps(exp_df, tags)

    if compare:
        colors = {
            k: col
            for k, col in zip(
                experiment_dfs.keys(), random.sample(ctx.obj["colors"]["qualitative"], len(experiment_dfs.keys()))
            )
        }
    for tag in tags:
        if compare:
            traces = []

        tag = tag.split("/")  # if distinguishing between training and eval

        # If single experiment and the user is not comparing runs
        # then get run average and save
        if len(experiment_dfs) == 1:
            all_runs = list(experiment_dfs.values())
            all_runs_df = pd.concat(all_runs, axis=1)
            cols = all_runs_df.columns.tolist()
            df_tags = [t for t in cols if "/".join(tag) in t]
            tag_df = all_runs_df[df_tags].copy(deep=True)
            tag_df.dropna(inplace=True)
            tag_df["mean"] = tag_df.mean(axis=1)
            tag_df["std"] = tag_df.std(axis=1)
            fig = go.Figure(DRAW_FN_MAP[plot_type](tag_df, variance=variance))
            fig = update_layout(fig, f"{' '.join(name.split('_')).title()}", " ".join(tag).title())
            if ctx.obj["vis"]:
                fig.show()
            if ctx.obj["save"]:
                fig.write_image(os.path.join(exp_path, f"{'_'.join(tag)}.png"), width=2000, height=1000, scale=1)
            continue

        for exp_name in experiment_dfs.keys():
            experiment_df = experiment_dfs[exp_name]
            tag_run_df = experiment_df.copy(deep=True)
            tag_run_df.dropna(inplace=True)
            tag_run_df["mean"] = tag_run_df.mean(axis=1)
            tag_run_df["std"] = tag_run_df.std(axis=1)

            if compare:
                traces.extend(DRAW_FN_MAP[plot_type](tag_run_df, exp_name, variance, colors[exp_name]))
            else:
                fig = go.Figure(DRAW_FN_MAP[plot_type](tag_run_df, variance=variance))
                fig = update_layout(fig, f"{exp_name.title()}", " ".join(tag).title())
                if ctx.obj["vis"]:
                    fig.show()
                if ctx.obj["save"]:
                    fig.write_image(
                        os.path.join(exp_path, f"{exp_name}_{'_'.join(tag)}.png"), width=2000, height=1000, scale=1
                    )

        if compare:
            fig = go.Figure(traces)
            fig = update_layout(fig, "Model Comparison", tag.title())
            if ctx.obj["vis"]:
                fig.show()
            if ctx.obj["save"]:
                fig.write_image(os.path.join(exp_path, f"{name}_{tag}.png"), width=2000, height=1000, scale=1)


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
