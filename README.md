pref_voting
==========

## Installation

With pip package manager:

```bash
pip install pref_voting
```

## Documentation

Online documentation is available at [https://pref-voting.readthedocs.io](https://pref-voting.readthedocs.io).

## Example Usage

A profile (of linear orders over the candidates) is created by initializing a `Profile` class object.  Simply provide a list of rankings (each ranking is a tuple of numbers) and a list giving the number of voters with each ranking:

```python
from pref_voting.profiles import Profile

rankings = [
    (0, 1, 2, 3), 
    (2, 3, 1, 0), 
    (3, 1, 2, 0), 
    (1, 2, 0, 3), 
    (1, 3, 2, 0)]

rcounts = [5, 3, 2, 4, 3]

prof = Profile(rankings, rcounts=rcounts)
```

The function `generate_profile` is used to generate a profile for a given number of candidates and voters:  

```python
from pref_voting.generate_profiles import generate_profile

# generate a profile using the Impartial Culture probability model
prof = generate_profile(3, 4) # prof is a Profile object

# generate a profile using the Impartial Anonymous Culture probability model
prof = generate_profile(3, 4, probmod = "IAC") # prof is a Profile object 
```

To use one of the many voting methods, import the function from `pref_voting.voting_methods` and apply it to the profile: 

```python
from pref_voting.generate_profiles import generate_profile
from pref_voting.voting_methods import *

prof = generate_profile(3, 4)
split_cycle(prof) # returns the sorted list of winning candidates
split_cycle.display(prof) # display the winning candidates

```

## Questions?

Feel free to [send an email](https://pacuit.org/) if you have questions about the project.

## License

[MIT](https://github.com/jontingvold/pyrankvote/blob/master/LICENSE.txt)
