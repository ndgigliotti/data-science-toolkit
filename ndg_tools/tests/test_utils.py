import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import datasets
from ndg_tools import utils
import pytest


@pytest.mark.filterwarnings("ignore: Function stratified_sample")
def test_balanced_sample():
    rng = np.random.default_rng(15)
    df = np.zeros(shape=(10 ** 4, 10), dtype="int64")
    df[rng.integers(df.shape[0], size=100), -1] = 1
    df[rng.integers(df.shape[0], size=3000), -1] = 2
    df[rng.integers(df.shape[0], size=500), -1] = 3
    columns = [f"feat_{i}" for i in range(df.shape[1])]
    columns[-1] = "tar"
    df = pd.DataFrame(df, columns=columns)
    winners = []
    for _ in range(100):
        samp = df.sample(n=50, random_state=rng.bit_generator)
        bal_samp = utils.balanced_sample(
            df, "tar", n=50, random_state=rng.bit_generator
        )
        old_bal_samp = utils.stratified_sample(
            df, "tar", n=50, random_state=rng.bit_generator
        )
        samp_std = samp["tar"].value_counts().std()
        bal_samp_std = bal_samp["tar"].value_counts().std()
        old_bal_samp_std = old_bal_samp["tar"].value_counts().std()
        if bal_samp_std < old_bal_samp_std:
            winners.append("new")
        elif old_bal_samp_std > bal_samp_std:
            winners.append("old")
        else:
            winners.append("tie")
        assert bal_samp_std < samp_std
        assert bal_samp_std < 10
    assert winners.count("tie") > winners.count("new") > winners.count("old")
