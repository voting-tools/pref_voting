Iterative Methods
=========================

The voting methods defined below all follow a similar procedure: iteratively remove "poorly performing" candidates from the profile until there is a candidate with that is the majority winner. The voting methods differ in how they answer the following two questions: 

1. What does it mean for a candidate to "perform poorly"?  Different definitions of a poorly performing candidate include the fewest number of first-place votes, the most last-place votes, the least Borda score, etc.

2. How to break ties when there are more than one candidate that is identified as "poorly performing"?  One approach is to remove all candidates that are poorly performing at each round (and if all remaining candidates are removed, then the remaining candidates are tied for the win).  A second approach is to use so-called *parallel-universe tiebreaking*. On this approach, a candidate is a winner if that candidate wins using some tiebreaking ordering (a linear order over the set of candidates).

```{eval-rst}

To illustrate the difference with respect to the first question, consider the voting methods Instant Runoff and Coombs.   According to Instant Runoff, a candidate is "poorly performing" if that candidate has the fewest number first-place votes, and according to Coombs, a candidate is "poorly performing" if that candidate has the most last-place votes. 

.. exec_code::

    from pref_voting.profiles import Profile
    from pref_voting.iterative_methods import instant_runoff, coombs
    
    prof = Profile([[2, 1, 0], [0, 2, 1], [1, 2, 0]], [1, 2, 2])

    prof.display()
    instant_runoff.display(prof)
    coombs.display(prof)

```

To illustrate the difference with respect to the first question, consider Instant Runoff and Instant Runoff PUT that uses parallel-universe tiebreaking. 

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile
    from pref_voting.iterative_methods import instant_runoff, instant_runoff_put
    
    prof = Profile([[1, 2, 0], [2, 1, 0], [0, 1, 2]], [1, 1, 1])

    prof.display()
    instant_runoff.display(prof)
    instant_runoff_put.display(prof)

```


## Instant Runoff

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.instant_runoff

```

### Instant Runoff TB

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.instant_runoff_tb

```

### Instant Runoff PUT

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.instant_runoff_put

```

### Instant Runoff with Explanation

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.instant_runoff_with_explanation

```

### Instant Runoff for Truncated Linear Orders

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.instant_runoff_for_truncated_linear_orders

```


## Benham

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.benham

```

### Benham TB

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.benham_tb

```

### Benham PUT

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.benham_put

```

## Plurality With Runoff

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.plurality_with_runoff

```

## Coombs

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.coombs

```
### Coombs TB

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.coombs_tb

```

### Coombs PUT

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.coombs_put

```

### Coombs with Explanation

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.coombs_with_explanation

```


## Baldwin

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.baldwin

```

### Baldwin TB

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.baldwin_tb

```

### Baldwin PUT

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.baldwin_put

```

### Baldwin with Explanation

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.baldwin_with_explanation

```

## Nanson

### Strict Nanson

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.strict_nanson

```

### Strict Nanson with Explanation

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.strict_nanson_with_explanation

```

### Weak Nanson

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.weak_nanson

```

### Weak Nanson with Explanation

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.weak_nanson_with_explanation

```

## Iterative Removal of Condorcet Loser


```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.iterated_removal_cl

```

### Iterative Removal of Condorcet Loser with Explanation

```{eval-rst}

.. autofunction:: pref_voting.iterative_methods.iterated_removal_cl_with_explanation

```


