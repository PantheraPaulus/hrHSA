import numpy as np
import pandas as pd

from hsa.rsf.bayesian import prepare_bayesian_rsf_data


def test_prepare_bayesian_rsf_data_preserves_individuals_and_bins():
    df = pd.DataFrame(
        {
            "individual-local-identifier": ["A", "A", "A", "B", "B", "B"],
            "used": [1, 0, 0, 1, 0, 0],
            "elevation": [101, 104, 149, 201, 204, 249],
            "ruggedness": [0.1, 0.12, 0.18, 0.2, 0.22, 0.28],
        }
    )
    result = prepare_bayesian_rsf_data(
        df,
        predictors=["elevation", "ruggedness"],
        binning={"elevation": 50, "ruggedness": 0.1},
    )

    assert result["X"].shape[1] == 2
    assert result["individuals"] == ["A", "B"]
    assert set(result["data"]["individual-local-identifier"]) == {"A", "B"}
    assert result["meta"]["elevation"]["bin_width"] == 50
    assert np.isclose(result["meta"]["elevation"]["mean"], df["elevation"].mean())
    assert result["n_trials"].sum() == len(df)
    assert result["used_count"].sum() == df["used"].sum()
