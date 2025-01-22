import pytest
import pandas as pd
from json_reader import DataFrameReader  

# Test Data
CALLS_JSON = """
{
    "data": [
        {
            "id": "d84823c1-1d67-45ca-8b75-6b29e4f330a1",
            "attributes": {
                "number": "+441234567890",
                "operator": "EE",
                "date": "2023-01-01",
                "greenList": true,
                "redList": false,
                "riskScore": 0.5
            }
        }
    ]
}
"""

OPERATORS_JSON = """
{
    "data": [
        {
            "prefix": "123",
            "operator": "EE"
        }
    ]
}
"""

@pytest.fixture
def dataframe_reader():
    """
    Fixture to create a DataFrameReader instance with mock JSON data.
    """
    calls_df = pd.read_json(CALLS_JSON)
    operators_df = pd.read_json(OPERATORS_JSON)
    return DataFrameReader(calls_df=calls_df, operators_df=operators_df)

def test_transform_raw_json(dataframe_reader):
    """
    Test the `_transform_raw_json` method to ensure data is flattened correctly.
    """
    transformed_calls = dataframe_reader._transform_raw_json(dataframe_reader.calls_df, 'data')
    assert isinstance(transformed_calls, pd.DataFrame)
    assert 'attributes.number' in transformed_calls.columns
    assert 'attributes.operator' in transformed_calls.columns
