
<h1 align="center">
TensorVis
</h1>

<p align="center">
  <a href="http://makeapullrequest.com">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
  </a>
  <a href="https://github.com/npitsillos/tensorplot/issues"><img src="https://img.shields.io/github/issues/npitsillos/tensorplot.svg"/></a>

  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black">
  </a>  
</p>

<p align="center">
<a href="#overview">Overview</a>
•
<a href="#features">Features</a>
•
<a href="#installation">Installation</a>
•
<a href="#contribute">Contribute</a>
</p>

# Overview
A command line tool to automate the process of grabbing tensorboard events data and visualising them.  This allows for faster result analysis and separation of the experiment logic from the visualisation aspect of the metrics logged in tensorboard.

# Features
* Uploads experiment metrics logged to tensorboard to tensorboard.dev and creates a log of uploaded experiments.
* Downloads experiments from tensorboard.dev to a local csv file.
* Plots experiment metrics.

## Benefits
1. Faster result analysis
2. Less code writting
3. Separate experiments from analysis
4. Allows for more research time

# Installation
```tensorvis``` can be installed using pip with the command:

```
pip install tensorvis
```

This will install ```tensorvis``` in the current python environment and will be available through the terminal.

## Assumptions
There can be many different directory structures when running and logging experiments with tensorboard.  This tool makes several assumptions to make it easier to handle dataframes resulting from downloading experiments.

```tensorvis``` assumes the following directory structure of tensorboard logs within the top level directory ```logs```, where each ```run``` subdirectory contains the events file:

```bash
logs
├── exp_name_1
│   ├── run_1
│   └── run_2
├── exp_name_2
│   ├── run_1
│   ├── run_2
│   └── run_3
└── exp_name_3
    └── run_1
```

> For a description of how the directory structure is represented in a dataframe follow this [link](https://www.tensorflow.org/tensorboard/dataframe_api#loading_tensorboard_scalars_as_a_pandasdataframe).

By default ```tensorvis``` assumes a single experiment directory is provided which corresponds to a single experiment having multiple runs.  All runs from a single experiment will be aggregate and averaged to plot the mean values along with the standard deviation.

# Contribute
Any feedback on ```tensorvis``` is welcomed in order to improve its usage and versatility.  If you have something specific in mind please don't hesitate to create an issue or better yet open a PR!

## Current Contributors
* [npitsillos](https://github.com/npitsillos)