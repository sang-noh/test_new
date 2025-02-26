# Instructions

This repository contains the code for reading in the calls/data jsons, and processing the files step by step to create the scored and filtered csvs. 
The convention followed has been to use the up to date conventions of python such as type-hinting,
object orientation and private (python-version) and public methods for executing step-by-step transformation of the data. 

The code is as follows - all the reading and writing code has been written in 
the DataFrameReader class. The 'private' methods have been written with _ in front, to indicate 
that they are only operated within the 'public' methods - which is the publish_result_csv.

The code within the private methods involve transforming the initial json dataset into a pandas dataset, and 
imputing the datasets to create the exceptions such as 'Unknown' and 'Withheld' values. The further transformations 
are udf-type transformations to create new columns for comparative purposes, such as separating out the phonenumber components 
as to be comparable to the area code. This is then cross joined and we filter out the dataset.

The _total_transformation method is used to execute these steps, as well as finally adding in the scores, sorting according to dates, 
and validating the dataset with a pandera schema. The pandera schema checks each column values for numerical ranges, and exceptions 
within the columns as well. The pandera schema has been added as a dictionary, as to ensure further schemas can be added for 
different datasets to create a catalogue for future use. 

To run the code, you can type:

make run 


To run the tests, you can run: 

make test


Further improvements could have been made with time - the static inputs of strings and scores could be changed to flexible inputs 
to ensure that with changing conventions of acceptable score, we can create general code.

