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
- v0.1.23 (2022-12-27): Add instant_runoff_for_truncated_linear_orders and functions to truncate overvotes in a ProfileWithTies, add smith_irv_put, document analysis functions
- v0.1.25 (2023-1-11): Add condorcet_irv, condorcet_irv_put; Update documentation; add axioms.py; add display and equality to Ranking class; fix enumerate ceots functions
- v0.1.27 (2023-2-07): Add Borda for ProfileWithTies
- v0.2 (2023-2-15): Add Benham, add anonymize to Profile method, comment out numba to make compatible with Python 3.11, add add_unranked_candidates to ProfileWithTies
- v0.2.1 (2023-2-15): Bug fixes
- v0.2.3 (2023-4-2): Add plurality_with_runoff_with_explanation
- v0.2.4 (2023-4-9): Update generate_truncated_profile so that it implements the IC probability model.
- v0.2.6 (2023-5-10): Add axiom class, dominance axioms, and axiom_violations_data.
- v0.2.8 (2023-5-16): Add description function to Majority Graphs.
- v0.2.11 (2023-5-16): Update implementation of Simple Stable Voting and Stable Voting.
- v0.2.13 (2023-5-24): Improve implementation of split_cycle; Breaking changes: split_cycle_faster renamed split_cycle_Floyd_Warshall and beat_path_faster renamed beat_path_Floyd_Warshall.
- v0.2.17 (2023-5-25): Add to_linear_profile to ProfileWithTies
- v0.3.3 (2023-5-26): Add implementations of UtilityProfile and a number of different utility methods.
- v0.3.4 (2023-5-30): Add write and from_string methods to a UtilityProfile.
- v0.4 (2023-5-31): Add SpatialProfile class and utility functions for generating utility profiles from spatial profiles; add functions to generate a SpatialProfile.
- v0.4.8 (2023-5-31): Add bottom two IRV and Tideman's alternative voting methods.
- v0.4.12 (2023-6-3): Add probabilistic methods.
- v0.5.0 (2023-9-24): Add _Mapping class with Utility and Grade as subclasses of _Mapping, add GradeProfile class, add Score Vote, Approval Vote, and STAR Vote.

- v0.5.4 (2023-10-01): Add median grading voting methods.

## Questions?

Feel free to [send an email](https://pacuit.org/) if you have questions about the project.

## License

[MIT](https://github.com/jontingvold/pyrankvote/blob/master/LICENSE.txt)
