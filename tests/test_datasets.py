import pytest
from decision_infovalue.datasets import get_dataset

def test_get_housing_dataset():
    df, meta = get_dataset('housing')
    assert df is not None
    assert meta is not None
    assert isinstance(df.shape, tuple)
    assert len(df) == df.shape[0]
    assert len(df[meta['target_name']]) == df.shape[0]
    assert isinstance(meta, dict) or meta is None

def test_get_recidivism_dataset():
    df, meta = get_dataset('recidivism')
    assert df is not None
    assert meta is not None
    assert isinstance(df.shape, tuple)
    assert len(df) == df.shape[0]
    assert len(df[meta['target_name']]) == df.shape[0]
    assert isinstance(meta, dict) or meta is None

def test_invalid_dataset_name():
    with pytest.raises(ValueError):
        get_dataset('nonexistent') 