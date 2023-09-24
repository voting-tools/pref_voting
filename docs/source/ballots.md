Ballots
=======================================

There are three different types of ballots that are used in preference voting.  

1. A **linear order** of the candidates: In this ballot, voters rank the candidates from most preferred to least preferred. The two main assumptions are that all the candidates must be ranked, and no two candidates can be tied for the same rank.  This ballot is represented as a ``list``.
2. A **(truncated) ranking** of the candidates: In this ballot, voters rank the candidates from most preferred to the least preferred.  Voters are not required to rank all the candidates, and multiple candidates can be tied for the same rank.  This ballot is represented by the ``Ranking`` class described below.
3. An **assignment of grades** to candidates: Given a fixed set of grades (which may be numbers or strings, such as "A", "B", "C", etc.), voters assign a grade from this set to each candidate.  Voters do not need to grade all the candidates, and multiple candidates may be assigned the same grade. This ballot is represented by the ``Grade`` class described below.

In addition to the above ballots, pref_voting includes a class representing **utility functions**.  A utility function is a function that assigns a real number to each candidate (or alternative). 

## Ranking Class

```{eval-rst}
.. autoclass:: pref_voting.rankings.Ranking
    :members: 
```

## Mappings

Both the ``Grade`` class and ``Utility`` class are subclasses of the ``_Mapping`` class representing a partial function from a set of candidates to a set of grades or to any floating point number.  The ``_Mapping`` class is not intended to be used directly, but is used as a base class for the ``Grade`` and ``Utility`` classes.

```{eval-rst}

.. autoclass:: pref_voting.mappings._Mapping
    :members: 

```

### Grade Class

```{eval-rst}

.. autoclass:: pref_voting.mappings.Grade
    :members: 

``` 

### Utility Class

```{eval-rst}

.. autoclass:: pref_voting.mappings.Utility
    :members: 

``` 

