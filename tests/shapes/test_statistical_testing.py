import pytest
import pandas as pd

from smds.pipeline.statistical_testing.run_statistical_test import (
    run_statistical_validation,
)


@pytest.mark.slow
def test_statistical_testing_main(circular_data_10d, tmp_path):
    """
    Pytest replacement for the old __main__ entrypoint.
    """

    X, y, _ = circular_data_10d

    pivot_df, output_dir = run_statistical_validation(
        X,
        y,
        n_repeats=5,
        n_folds=5,
        experiment_name="test_dummy",
    )

    assert isinstance(pivot_df, pd.DataFrame)
    assert not pivot_df.empty
    assert output_dir.exists()
