Saving Election Data
=======================================

The `pref_voting.io` module provides functionality for saving and loading election data in different formats. 

A ``Profile`` or ``ProfileWithTies`` can be saved in any of the following formats:

* preflib: The format used by preflib descrbed here: https://www.preflib.org/format#types.
* csv: There are two formats for the csv file: "rank_columns" and "candidate_columns".  The "rank_columns" format is used when the csv file contains a column for each rank and the rows are the candidates at that rank (or "skipped" if the ranked is skipped).  The "candidate_columns" format is used when the csv file contains a column for each candidate and the rows are the rank of the candidates (or the empty string if the candidate is not ranked).
* abif: This format is explained here: https://electowiki.org/wiki/ABIF
* json: Save the election data in a json file, where each ranking is a dictionary with the candidate names as keys and the ranks as values.

The `pref_voting.io` module also provides functionality for reading election data from these formats.  

All of these functions are accessible through `write` and `read` methods in the ``Proile`` and ``ProfileWithTies`` classes. 


A ``SpatialProfile`` can be saved as a json file. 

## Saving Election Data 

```{eval-rst}

.. autofunction:: pref_voting.io.writers.to_preflib_instance

.. autofunction:: pref_voting.io.writers.write_preflib

.. autofunction:: pref_voting.io.writers.write_csv

.. autofunction:: pref_voting.io.writers.write_json

.. autofunction:: pref_voting.io.writers.write_abif

.. autofunction:: pref_voting.io.writers.write

``` 

## Loading Election Data

```{eval-rst}

.. autofunction:: pref_voting.io.readers.preflib_to_profile

.. autofunction:: pref_voting.io.readers.csv_to_profile

.. autofunction:: pref_voting.io.readers.json_to_profile

.. autofunction:: pref_voting.io.readers.abif_to_profile

.. autofunction:: pref_voting.io.readers.abif_to_profile_with_ties

.. autofunction:: pref_voting.io.readers.read

```

## Save Spatial Profile Data

```{eval-rst}

.. autofunction:: pref_voting.io.writers.write_spatial_profile_to_json

```

## Load Spatial Profile Data

```{eval-rst}

.. autofunction:: pref_voting.io.readers.json_to_spatial_profile

```