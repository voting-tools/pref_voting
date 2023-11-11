Margin Methods
=======================================

Suppose that $\mathbf{P}$ is a profile.   We write $\mathcal{M}(\mathbf{P})$ for the margin graph of $\mathbf{P}$. 

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

### Beat Path Floyd-Warshall

 
```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.beat_path_Floyd_Warshall

```

### Beat Path Defeat

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.beat_path_defeat

```

## Split Cycle

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.split_cycle

```

### Split Cycle Floyd-Warshall

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.split_cycle_Floyd_Warshall


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

### Ranked Pairs from Stacks

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.is_stack

```

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.ranked_pairs_from_stacks

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

### Stable Voting Faster

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.stable_voting_faster

```

### Simple Stable Voting

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.simple_stable_voting

```

### Simple Stable Voting Faster

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.simple_stable_voting_faster

```

## Loss-Trimmer

```{eval-rst}

.. autofunction:: pref_voting.margin_based_methods.loss_trimmer

```
 