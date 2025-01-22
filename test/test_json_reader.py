import pytest
import pandas as pd
import pandera
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


# sample valid data for pandera
VALID_DATA = pd.DataFrame(
    {
        "id": ["d84823c1-1d67-45ca-8b75-6b29e4f330a1"],
        "date": ["2023-01-01"],
        "operator": ["EE"],
        "number": ["+441234567890"],
        "score": [0.5],
    }
)

# sample invalid data for pandera
INVALID_DATA = pd.DataFrame(
    {
        "id": ["invalid-uuid"],  # Invalid UUID
        "date": ["invalid-date"],  # Invalid date
        "operator": [""],  # Empty operator name
        "number": ["12345"],  # Invalid phone number
        "score": [1.5],  # Score out of range
    }
)


@pytest.fixture
def dataframe_reader():
    """
    Fixture to create a DataFrameReader instance with mock JSON data.
    """
    calls_df = pd.read_json(CALLS_JSON)
    operators_df = pd.read_json(OPERATORS_JSON)
    return DataFrameReader(calls_df=calls_df, operators_df=operators_df)


@pytest.fixture
def dataframe_reader_pandera():
    """
    Create a DataFrameReader instance for testing.
    """
    calls_df = pd.DataFrame({"data": VALID_DATA.to_dict(orient="records")})
    operators_df = pd.DataFrame()  # Empty operators for this test
    return DataFrameReader(calls_df, operators_df)


def test_transform_raw_json(dataframe_reader):
    """
    Test the `_transform_raw_json` method to ensure data is flattened correctly.
    """
    transformed_calls = dataframe_reader._transform_raw_json(
        dataframe_reader.calls_df, "data"
    )
    assert isinstance(transformed_calls, pd.DataFrame)
    assert "attributes.number" in transformed_calls.columns
    assert "attributes.operator" in transformed_calls.columns


def test_pandera_valid_data(dataframe_reader_pandera):
    """
    Test that valid data passes the pandera schema check.
    """
    schema = dataframe_reader_pandera.pandera_check("calls_data")
    valid_data = VALID_DATA.copy()
    valid_data["date"] = pd.to_datetime(valid_data["date"])  # Ensure correct data type

    # Validate the data
    try:
        validated_data = schema.validate(valid_data)
        assert isinstance(validated_data, pd.DataFrame)
        assert len(validated_data) == len(valid_data)
    except pandera.errors.SchemaError as e:
        pytest.fail(f"Valid data failed schema validation: {e}")


def test_pandera_invalid_data(dataframe_reader_pandera):
    """
    Test that invalid data raises a pandera SchemaError.
    """
    schema = dataframe_reader_pandera.pandera_check("calls_data")
    invalid_data = INVALID_DATA.copy()

    # Attempt validation and check for SchemaError
    with pytest.raises(pandera.errors.SchemaError):
        schema.validate(invalid_data)
