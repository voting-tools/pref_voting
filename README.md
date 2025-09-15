pref_voting
==========
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07020/status.svg)](https://doi.org/10.21105/joss.07020) [![DOI](https://zenodo.org/badge/578984957.svg)](https://doi.org/10.5281/zenodo.14675583)

[![Tests](https://github.com/voting-tools/pref_voting/actions/workflows/tests.yml/badge.svg)](https://github.com/voting-tools/pref_voting/actions/workflows/tests.yml)


> [!NOTE]
> - [**Documentation**](https://pref-voting.readthedocs.io/)
> - [**Installation**](https://pref-voting.readthedocs.io/en/latest/installation.html)  
> - [**Example Notebooks**](https://github.com/voting-tools/pref_voting/tree/main/examples)  
> - [**Example Elections**](https://github.com/voting-tools/election-analysis)
> - [**► pref_voting web app**](https://pref.tools/pref_voting/)

See the [COMSOC community page](https://comsoc-community.org/tools) for an overview of other software tools related to Computational Social Choice.

## Installation

The package can be installed using the ``pip3`` package manager:

```bash
pip3 install pref_voting
```
**Notes**: 
* The package requires Python 3.10 or higher and has been tested on Python 3.12.

* Since the package uses Numba, refer to the [Numba documentation for the latest supported Python version](https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information).
* If you have both Python 2 and Python 3 installed on your system, make sure to use ``pip3`` instead of pip to install packages for Python 3. Alternatively, you can use ``python3 -m pip`` to ensure you're using the correct version of pip. If you have modified your system's defaults or soft links, adjust accordingly.

See the [installation guide](https://pref-voting.readthedocs.io/en/latest/installation.html) for more detailed instructions.

## Example Usage

A profile (of linear orders over the candidates) is created by initializing a `Profile` class object.  Simply provide a list of rankings (each ranking is a tuple of numbers) and a list giving the number of voters with each ranking:

```python
from pref_voting.profiles import Profile

rankings = [
    (0, 1, 2, 3), # candidate 0 is ranked first, candidate 1 is ranked second, candidate 2 is ranked 3rd, and candidate 3 is ranked last.
    (2, 3, 1, 0), 
    (3, 1, 2, 0), 
    (1, 2, 0, 3), 
    (1, 3, 2, 0)]

rcounts = [5, 3, 2, 4, 3] # 5 voters submitted the first ranking (0, 1, 2, 3), 3 voters submitted the second ranking, and so on.

prof = Profile(rankings, rcounts=rcounts)

prof.display() # display the profile
```

The function `generate_profile` is used to generate a profile for a given number of candidates and voters:  

```python
from pref_voting.generate_profiles import generate_profile

# generate a profile using the Impartial Culture probability model
prof = generate_profile(3, 4) # prof is a Profile object with 3 candidates and 4 voters

# generate a profile using the Impartial Anonymous Culture probability model
prof = generate_profile(3, 4, probmod = "IAC") # prof is a Profile object with 3 candidates and 4 voters 
```

The `Profile` class has a number of methods that can be used to analyze the profile. For example, to determine the margin of victory between two candidates, the plurality scores, the Copeland scores, the Borda scores, the Condorcet winner, the weak Condorcet winner, and the Condorcet loser, and whether the profile is uniquely weighted, use the following code:

```python

prof = Profile([
    [2, 1, 0, 3], 
    [3, 2, 0, 1], 
    [3, 1, 0, 2]], 
    rcounts=[2, 2, 3])

prof.display()

print(f"The margin of 1 over 3 is {prof.margin(1, 3)}")
print(f"The Plurality scores are {prof.plurality_scores()}")
print(f"The Copeland scores are {prof.copeland_scores()}")
print(f"The Borda scores are {prof.borda_scores()}")
print(f"The Condorcet winner is {prof.condorcet_winner()}")
print(f"The weak Condorcet winner is {prof.weak_condorcet_winner()}")
print(f"The Condorcet loser is {prof.condorcet_loser()}")
print(f"The profile is uniquely weighted: {prof.is_uniquely_weighted()}")

```

To use one of the many voting methods, import the function from `pref_voting.voting_methods` and apply it to the profile: 

```python
from pref_voting.generate_profiles import generate_profile
from pref_voting.voting_methods import *

prof = generate_profile(3, 4) # create a profile with 3 candidates and 4 voters
split_cycle(prof) # returns the sorted list of winning candidates
split_cycle.display(prof) # displays the winning candidates

```

Additional notebooks that demonstrate how to use the package can be found in the [examples directory](https://github.com/voting-tools/pref_voting/tree/main/examples)

Some interesting political elections are analyzed using pref_voting in the [election-analysis repository](https://github.com/voting-tools/election-analysis).

Consult the documentation [https://pref-voting.readthedocs.io](https://pref-voting.readthedocs.io) for a complete overview of the package. 


## Testing
 
To ensure that the package is working correctly, you can run the test suite using [pytest](https://docs.pytest.org/en/stable/). The test files are located in the `tests` directory. Follow the instructions below based on your setup.

### Prerequisites

- **Python 3.9 or higher**: Ensure you have a compatible version of Python installed.
- **`pytest`**: Install `pytest` if it's not already installed.

### Running the tests

If you are using **Poetry** to manage your dependencies, run the tests with:

```bash
poetry run pytest

```
 
From the command line, run:

```bash
pytest
```

For more detailed output, add the -v or --verbose flag:

```bash
pytest -v
```

## How to cite
 
If you would like to acknowledge our work in a scientific paper,
please use the following citation:

Wesley H. Holliday and Eric Pacuit (2025). pref_voting: The Preferential Voting Tools package for Python. Journal of Open Source Software, 10(105), 7020. https://doi.org/10.21105/joss.07020

### BibTeX:

```bibtex
@article{HollidayPacuit2025, 
  author = {Wesley H. Holliday and Eric Pacuit}, 
  title = {pref_voting: The Preferential Voting Tools package for Python}, 
  journal = {Journal of Open Source Software},
  year = {2025}, 
  publisher = {The Open Journal}, 
  volume = {10}, 
  number = {105}, 
  pages = {7020}, 
  doi = {10.21105/joss.07020}
}

```

Alternatively, you can cite the archived code repository
at [zenodo](https://doi.org/10.5281/zenodo.14675583).

## Contributing

If you would like to contribute to the project, please see the [contributing guidelines](CONTRIBUTING.md).

## Questions?

Feel free to [send an email](https://pacuit.org/) if you have questions about the project.

## License

[MIT](https://github.com/voting-tools/pref_voting/blob/main/LICENSE.txt)
