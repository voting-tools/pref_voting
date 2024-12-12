Margin Methods
=======================================

Suppose that $\mathbf{P}$ is a profile.  We write $\mathcal{M}(\mathbf{P})$ for the margin graph of $\mathbf{P}$. 

A voting method $F$ is **margin-based** if it satisfies the following invariance property: For all $\mathbf{P}, \mathbf{P}'$, if $\mathcal{M}(\mathbf{P})= \mathcal{M}(\mathbf{P}')$, then $F(\mathbf{P}) = F(\mathbf{P}')$. 


## Minimax

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.minimax

```

### Minimax Scores

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.minimax_scores

```

## Beat Path

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.beat_path

```


### Beat Path Defeat

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.beat_path_defeat

```

## Split Cycle

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.split_cycle

```


### Split Cycle Defeat

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.split_cycle_defeat

```

## Ranked Pairs

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.ranked_pairs

```



### Ranked Pairs with Test

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.ranked_pairs_with_test

```

### Ranked Pairs Defeats

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.ranked_pairs_defeats

```

### Stacks

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.is_stack

```


### Ranked Pairs with Tiebreaking

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.ranked_pairs_tb


```

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.ranked_pairs_zt

```


## River

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.river


```

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.river_with_test


```

### River with Tiebreaking

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.river_tb


```

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.river_zt


```


## Stable Voting

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.stable_voting
```

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.stable_voting_with_explanation

```


## Simple Stable Voting

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.simple_stable_voting

```


```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.simple_stable_voting_with_explanation

```


## Loss-Trimmer

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.loss_trimmer

```
 

## Essential Set

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.essential

```
 
## Weighted Covering

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.weighted_covering

```

## Beta-Uncovered Set

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.beta_uncovered_set

```