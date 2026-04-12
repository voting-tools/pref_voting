Single-Peakedness
==========

Functions to analyze single-peakedness of preference profiles.

A preference profile is *single-peaked* with respect to an axis (linear order over the candidates) if every voter has a unique most-preferred candidate (peak) and the voter's preferences decrease monotonically in both directions from the peak along the axis.

A profile is *k-maverick single-peaked* with respect to an axis if all but *k* voters are single-peaked with respect to that axis. The minimum *k* over all axes measures how far the profile is from being single-peaked.

**Reference:** Faliszewski, Hemaspaandra & Hemaspaandra (2014), "The complexity of manipulative attacks in nearly single-peaked electorates", *Artificial Intelligence* 207, 69-99. [DOI](https://doi.org/10.1016/j.artint.2013.11.004)

## Checking Single-Peakedness of Individual Rankings

```{eval-rst}

.. autofunction:: pref_voting.single_peakedness.is_single_peaked

```

## Profile-Level Analysis

### Counting Mavericks

```{eval-rst}

.. autofunction:: pref_voting.single_peakedness.num_mavericks

```

### Finding the Minimum k

```{eval-rst}

.. autofunction:: pref_voting.single_peakedness.min_k_maverick_single_peaked

```
