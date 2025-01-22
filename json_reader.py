import pandas as pd
import pandera as pa

# paths to the jsons
OPERATORS_JSON_PATH = 'data/operators.json'
CALLS_JSON_PATH = 'data/calls.json'

class DataFrameReader:
    """
    This class has been designed with a feature in mind i.e. we are reading in the raw data with the constructor,
    and then creating private methods to transport the dataframe in ways we would like in a ETL (explore, transporm, L)
    methodology.
    """
    def __init__(self, calls_df: str, operators_df: str) -> None:
        """
        Raw inputs to transform for the json files
        """

        # to help with testing, we can make the constructor from the string or just take in a dataframe 
        if isinstance(calls_df, str):
            self.calls_df = pd.read_json(calls_df)
        else:
            self.calls_df = calls_df

        if isinstance(operators_df, str):
            self.operators_df = pd.read_json(operators_df)
        else:
            self.operators_df = operators_df
        
    def _transform_raw_json(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        The raw data is currently transformed into a dictionary of series, so we need
        to extract out the data dictionary from within (the 'data' key) and then
        flatten the data to get a true pandas dataframe, which we can read into the 
        """
        if column not in df:
            raise ValueError(f"Error - {column} is not within the dataframe")
        flattened_df = pd.json_normalize(df[column])
        df = pd.concat([df.drop(columns=[column]), flattened_df], axis=1)
        return df


    def _divide_phone_number(self, number: str, imputed_str: str = "Withheld") -> None:
        """
        Internal function to transform

        Divide up the numbers in case of having a legitimate number. Otherwise,
        return "Withheld"
        """
        number = str(number)
        if number == 'nan': 
            return imputed_str
        divided_numbers = '-'.join((number[:3], number[3:7] , number[7:]))
        return divided_numbers

    def _extract_numeric(self, x: str) -> float | str:
        """
        Extract the middle number value for comparison 
        """
        try:
            return float(x.split('-')[1])  # split and convert to float
        except (IndexError, ValueError, AttributeError):
            return 'Unknown'  # return None for invalid values

    def _impute_operators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        if the operator cannot be found, "Unknown" should be displayed as the operator field
        if the number is absent, "Withheld" should be displayed as the number field
        the calls are ordered by ascending date
        """
        df['attributes.number'] = df.apply(lambda row: self._divide_phone_number(row['attributes.number']), axis = 1) # divide the numbers
        df["attributes.number.numeric"] = df["attributes.number"].apply(self._extract_numeric) # creating the middle numeric value 
        return df

    def _safe_comparison(self, row) -> bool:
        """
        Do a comparison between the prefix and the operator. If it fits within the boundary, then return value.
        Otherwise, dont do anything
        """
        try:
            num = float(row["attributes.number.numeric"]) # convert to float if possible
            prefix = float(row["attributes.prefix"]) # perform the comparison
            return (num > prefix) and (num < prefix + 999)
        except (ValueError, TypeError):
            return False # return False if comparison is not valid
            
    def _join_table_on_key(self, df_calls: pd.DataFrame, df_operators: pd.DataFrame) -> pd.DataFrame:
        """
        Join dataframe and filter based on the values
        """
        joined_dataframe = pd.merge(df_calls, df_operators, how = "cross")
        dataframe_after_comparison = joined_dataframe[joined_dataframe.apply(self._safe_comparison, axis=1)]
        return dataframe_after_comparison

    def _assign_risk_score(self, green: bool, red: bool, risk_score: float) -> float:
        """
        assign riskscore based on the green, red and risk score
        """
        if green == True and red == True:
            return 0.0
        if red == True and green == False:
            return 1.0
        else:
            return round(risk_score, 1)

    def pandera_check(self, key: str) -> pa.DataFrameSchema:
        """
        Define Pandera schemas for different datasets and validate them.
        """
        # dictionary to store Pandera schemas
        dictionary_pandera_schemas = {
            'calls_data': pa.DataFrameSchema({
                "id": pa.Column(
                    str,
                    checks=[
                        pa.Check.str_length(min_value=36, max_value=36),  # UUID length
                        pa.Check.str_matches(r"^[a-f0-9\-]+$")  # UUID pattern
                    ],
                ),
                "date": pa.Column(
                    pa.DateTime,  # ensure it's a valid datetime object
                    nullable=False,  # ensure no null values
                ),
                "operator": pa.Column(
                    str,
                    checks=pa.Check.str_length(min_value=1),  # non-empty operator names
                ),
                "number": pa.Column(
                    str,
                    checks=[
                        pa.Check.str_startswith("+44", error="Invalid country code"),  # check UK numbers
                        pa.Check(
                            lambda s: s.fillna("withheld"),  # allow and fill missing values
                            error="Invalid phone number format",
                        ),
                    ],
                    nullable=True  # allow missing values
                ),
                "score": pa.Column(
                    float,
                    checks=[
                        pa.Check.ge(0),  # greater than or equal to 0
                        pa.Check.le(1),  # less than or equal to 1
                    ],
                ),
            })
        }

        return dictionary_pandera_schemas[key]

            
    def _total_transformation(self) -> pd.DataFrame:
        """
        Execute the step by step transformation
        """
        flattened_calls_df = self._transform_raw_json(self.calls_df, 'data')
        flattened_operators_df = self._transform_raw_json(self.operators_df, 'data')
        flattened_calls_df['attributes.number'] = flattened_calls_df.apply(lambda row: self._divide_phone_number(row['attributes.number']), axis = 1)
        flattened_calls_df["attributes.number.numeric"] = flattened_calls_df["attributes.number"].apply(self._extract_numeric)
        joined_dataframe = self._join_table_on_key(flattened_calls_df, flattened_operators_df)
        # filter out the columns we want
        joined_dataframe = joined_dataframe[['id_x', 'attributes.number', 'attributes.operator', 'attributes.date',
                                              'attributes.greenList', 'attributes.redList', 'attributes.riskScore']]
        
        joined_dataframe.columns = ["id", "number", "operator", "date", 'greenlist', 'redlist', 'riskscore']
        # convert date to right format
        joined_dataframe['date'] = pd.to_datetime(joined_dataframe['date']).dt.date  # get the YYYY-MM-DD
        joined_dataframe['date'] = pd.to_datetime(joined_dataframe['date'])  # reassign type to date 
        joined_dataframe['score'] = joined_dataframe.apply(lambda row: self._assign_risk_score(row['greenlist'], row['redlist'], row['riskscore']), axis = 1) # compute score
        joined_dataframe = joined_dataframe[['id', 'date', 'operator', 'number', 'score']] # get the right columns
        joined_dataframe = joined_dataframe.sort_values(by='date').reset_index(drop=True) # reindex
        return joined_dataframe

    def publish_result_csv(self) -> None:
        """
        Create, validate, and publish dataset to csv
        """
        schema = self.pandera_check('calls_data') # call pandera schema
        output = self._total_transformation()
        validated_df = schema.validate(output) # validate with pandera schema
        validated_df.to_csv('output.csv', index=False)  
    
if __name__ == "__main__":
    # Ensure that this python file can be directly executred to generate an output csv
    CallsDataGenerator = DataFrameReader(CALLS_JSON_PATH, OPERATORS_JSON_PATH)
    CallsDataGenerator.publish_result_csv()
