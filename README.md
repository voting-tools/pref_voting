pref_voting
==========

## Installation

With pip package manager:

```bash
pip install pref_voting
```
## Documentation

Online documentation is available at [https://pref_voting.readthedocs.io](https://pref_voting.readthedocs.io).

## Profiles and Voting Methods

A profile (of linear orders over the candidates) is created by initializing a Profile class object.  This needs a list of rankings (each ranking is a tuple of numbers), the number of candidates, and a list giving the number of each ranking in the profile:

```python
from pref_voting.profiles import Profile

rankings = [(0, 1, 2, 3), (2, 3, 1, 0), (3, 1, 2, 0), (1, 2, 0, 3), (1, 3, 2, 0)]
rcounts = [5, 3, 2, 4, 3]

prof = Profile(rankings, rcounts=rcounts)
```

The function generate_profile is used to generate a profile for a given number of candidates and voters:  
```python
from pref_voting.generate_profiles import generate_profile

# generate a profile using the Impartial Culture probability model
prof = generate_profile(3, 4) # prof is a Profile object

# generate a profile using the Impartial Anonymous Culture probability model
prof = generate_profile(3, 4, probmod = "IAC") # prof is a Profile object 
```

```python
from pref_voting.profiles import Profile
from pref_voting.voting_methods import *

prof = Profile(rankings, num_cands, rcounts=rcounts)
print(f"{split_cycle.name} winners:  {split_cycle(prof)}")
split_cycle.display(prof)

```

## Versions

- v0.1.10 (2022-08-09): **Initial release** 
- v0.1.13 (2022-11-05): Minor updates and bug fixes 
- v0.1.14 (2022-12-19): Add plurality_scores to ProfileWithTies; add generate ceots function; bug fixes 
- v0.1.21 (2022-12-27): Add instant_runoff_for_truncated_linear_orders and functions to truncate overvotes in a ProfileWithTies


## Questions?

Feel free to [send me an email](https://pacuit.org/) if you have questions about the project.

## License

[MIT](https://github.com/jontingvold/pyrankvote/blob/master/LICENSE.txt)
