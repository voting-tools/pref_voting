Interfaces to Other Packages
=======================================

The `pref_voting.interfaces` modules convert between pref_voting election objects and the election objects of two other voting packages:

* [VoteKit](https://votekit.readthedocs.io) (MGGG Redistricting Lab): ranked ballots (with ties and truncation) and score ballots.
* [abcvoting](https://abcvoting.readthedocs.io) (Lackner et al.): approval-based committee (ABC) voting.

This makes it possible, for instance, to generate profiles with VoteKit's ballot generators and analyze them with pref_voting's voting methods, to run VoteKit's elections on pref_voting profiles, or to apply abcvoting's committee rules to pref_voting approval profiles.

The correspondence between the main election objects:

| pref_voting | VoteKit | abcvoting |
|---|---|---|
| ``Profile`` (linear orders) | ``PreferenceProfile`` of ranked ballots | — |
| ``ProfileWithTies`` (ties, truncation) | ``PreferenceProfile`` of ranked ballots | — |
| ``GradeProfile`` with numeric grades | ``PreferenceProfile`` of score ballots | — |
| ``GradeProfile`` with grades 0/1 (approval) | — | ``Profile`` |

Both packages are *optional* dependencies: they are only imported when a conversion function is called, and a plain installation of pref_voting is unchanged.  To use the interfaces, install the extras:

```bash
uv pip install "pref_voting[votekit]"      # VoteKit (requires Python >= 3.11)
uv pip install "pref_voting[abcvoting]"    # abcvoting
uv pip install "pref_voting[interfaces]"   # both
```

**Weights.** VoteKit ballots and abcvoting voters carry (possibly fractional) weights, while pref_voting profiles count voters with integer counts.  Integer-valued weights are converted directly; fractional weights raise a ``ValueError`` unless ``scale_weights=True`` is passed, in which case all weights are multiplied by a common denominator (an exact rescaling that preserves the relative weights of the ballots but changes the total number of voters).

All of the conversions are also available as methods on the profile classes: ``Profile.to_votekit()``, ``ProfileWithTies.to_votekit()`` / ``ProfileWithTies.from_votekit()``, and ``GradeProfile.to_votekit()`` / ``GradeProfile.from_votekit()`` / ``GradeProfile.to_abcvoting()`` / ``GradeProfile.from_abcvoting()``.

For a walkthrough of all the conversions, see the notebook [examples/interfaces_demo.ipynb](https://github.com/voting-tools/pref_voting/blob/main/examples/interfaces_demo.ipynb).

## VoteKit

```{eval-rst}

.. autofunction:: pref_voting.interfaces.votekit_interface.to_votekit_profile

.. autofunction:: pref_voting.interfaces.votekit_interface.votekit_to_profile_with_ties

.. autofunction:: pref_voting.interfaces.votekit_interface.grade_profile_to_votekit

.. autofunction:: pref_voting.interfaces.votekit_interface.votekit_to_grade_profile

```

## abcvoting

```{eval-rst}

.. autofunction:: pref_voting.interfaces.abcvoting_interface.to_abcvoting_profile

.. autofunction:: pref_voting.interfaces.abcvoting_interface.abcvoting_to_grade_profile

```
