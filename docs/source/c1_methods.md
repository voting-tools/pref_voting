C1 Methods
=======================================

Suppose that $\mathbf{P}$ is a profile.   We write $M(\mathbf{P})$ for the majority graph of $\mathbf{P}$. 

A voting method $F$ is **C1** if it satisfies the following invariance property: For all $\mathbf{P}, \mathbf{P}'$, if $M(\mathbf{P})= M(\mathbf{P}')$, then $F(\mathbf{P}) = F(\mathbf{P}')$. 


## Condorcet

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.condorcet

```

## Weak Condorcet

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.weak_condorcet

```

## Copeland

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.copeland

```

### Llull

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.llull

```

## Uncovered Set

### Uncovered Set - Gillies

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.uc_gill

```

### Uncovered Set Defeat (Gillies Version)

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.uc_gill_defeat

```

### Uncovered Set - Fishburn

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.uc_fish

```

### Uncovered Set Defeat (Fishburn Version)

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.uc_fish_defeat

```


### Uncovered Set - Bordes

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.uc_bordes

```

### Uncovered Set - McKelvey

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.uc_mckelvey

```

## Top Cycle

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.top_cycle

```

### Top Cycle Defeat 

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.top_cycle_defeat

```

## GOCHA

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.gocha

```


## Banks

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.banks

```

### Banks with Explanation

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.banks_with_explanation

```


## Slater

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.slater

```

### Slater Rankings

```{eval-rst}

.. autofunction:: pref_voting.c1_methods.slater_rankings

```
