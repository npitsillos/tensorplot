import os
import numpy as np
import pandas as pd

from tensorvis.utils.utils import *
from tensorvis import __version__

TEST_CSV = "ioLYjGW7R4Cyt5i6KAWMkQ.csv"
TEST_TAGS = ["mean_success", "mean_timesteps"]
TEST_START_STEP = 3584
TEST_EXP_NUM = 4
TEST_DUMMY_TAGS = ["success", "timesteps", "reward"]

def test_version():
    assert __version__.__version__ == "0.2.1"


def test_exp_df_to_tags_df():
    """
        Tests whether exp_df_to_tags_df returns a dataframe
        with correct number of columns
    """

    steps = np.arange(100, 1100, 10)
    success_values = np.linspace(0, 1, 100)
    timesteps_values = np.linspace(200, 100, 100)
    rewards = np.random.randint(20, 50, 100)
    # Generate random number of experiment runs
    num_runs = np.random.randint(15)
    dfs = []
    for i in range(num_runs):
        df = pd.DataFrame({
            "step": steps,
            "success": success_values,
            "timesteps": timesteps_values,
            "reward": rewards
        })
        df["run"] = f"run_{i}"
        dfs.append(df)
    
    exp_df = pd.concat(dfs)

    tags_df = exp_df_to_tags_df(exp_df, TEST_DUMMY_TAGS, 100)

    assert len(tags_df.columns) == len(TEST_DUMMY_TAGS)

def test_separate_exps():

    multiple_exps_df = pd.read_csv(TEST_CSV, index_col=0)

    exp_dfs = separate_exps(multiple_exps_df, TEST_TAGS, TEST_START_STEP)

    assert len(exp_dfs) == TEST_EXP_NUM