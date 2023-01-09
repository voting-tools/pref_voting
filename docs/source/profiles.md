Profiles
=======================================


Suppose that $X$ is a set of $n$ candidates.  A **strict linear order** of $X$, also called a **ranking** of $X$, is a relation $P\subseteq X\times X$ that is 
1. *asymmteric*: for all $x, y\in X$, if $x\mathrel{P} y$ then not $y\mathrel{P}x$, 
2. *transitive*: for all $x, y,z\in X$, if $x\mathrel{P} y$  and $y\mathrel{P} z$ then $x\mathrel{P}z$; and 
3. *connected*: for all $x,y\in X$ with $x\neq y$, either $x\mathrel{P} y$ or $y\mathrel{P}x$. 

When $x\mathrel{P}y$, we say "$x$ is ranked above $y$" or "$x$ is strictly preferred to $y$".   Let $\mathcal{L}(X)$ be the set of all strict linear orders over $X$.  


An **anonymous profile** is function $\mathbf{P}:\mathcal{L}(X)\rightarrow \mathbb{N}$ assigning a non-negative number to each ranking. 

The ``Profile`` class represents an anonymous profile.  There are two important assumptions when defining a ``Profile``. 

1. When there are $n$ candidates,  the  candidates are $0, 1, 2, \ldots, n-1$. 
2. A strict linear order of $X$ is represented by a list of elements from $X$ of length $n$ in which each element of $X$ appears in the list. For instance, ``[1, 0, 2]`` represents the rankings $L$ where:
$1\mathrel{L} 2, 0\mathrel{L}2, \mbox{ and } 1\mathrel{L} 2.$
 
A ``Profile`` is defined by specifying a list of rankings.  It is assumed that any ranking not in this list was not submitted by any voter. 

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile 
    prof = Profile([[0, 1, 2], [2, 1, 0]])
    print(f"The rankings are {prof.rankings}")
    print(f"The number of voters is {prof.num_voters}")
    print(f"The number of candidate is {prof.num_cands}")
    print(f"The candidates are {prof.candidates}")
```

There are two optional keyword parameters for a ``Profile``: 

1. ``rcounts`` is a list of integers specifying the number of voters with a given ranking.  If ``rankings`` is the list of rankings, then it is assumed that the number of voters who submitted ranking ``rankings[i]`` is ``rcounts[i]``.  If ``rcounts`` is not provided, the default value is 1 for each ranking.

2. ``cmap`` is a dictionary mapping the candidates to their names. This is used when displaying a profile. 

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile 
    prof = Profile([[0, 1, 2], [2, 1, 0]], rcounts = [2, 1], cmap = {0:"a", 1:"b", 2:"c"})
    print(f"The rankings are {prof.rankings}")
    print(f"The number of voters is {prof.num_voters}")
    print(f"The number of candidate is {prof.num_cands}")
    print(f"The candidates are {prof.candidates}")
    prof.display()

```

```{warning} 

There are two things to keep in mind when defining a ``Profile``.  

1. The length of ``rcounts`` must be equal to the number of rankings used to define the profile. 

2. You cannot skip a number when defining the rankings. That is, the following will produce an error: 

    ```python 

    prof = Profile([[0, 1, 3], [3, 1, 0]])

    ```

```

## Profile Methods


See the [next section](#profile-class) for a complete list of the methods associated with a ``Profile``.

### Margins

The **margin** of $x$ over $y$ in $\mathbf{P}$ is 

$$

Margin_\mathbf{P}(x,y) = \sum_{L\in\mathcal{L}(X), x\mathrel{L}y}\mathbf{P}(L)\ \ \ - \sum_{L\in\mathcal{L}(X), y\mathrel{L}x}\mathbf{P}(L)

$$

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile 
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [4, 3, 2])
    prof.display()
    print(f"The margin of 0 over 1 is {prof.margin(0,1)}")
    print(f"The margin of 1 over 0 is {prof.margin(1,0)}")
    print(f"The margin of 0 over 2 is {prof.margin(0,2)}")
    print(f"The margin of 2 over 0 is {prof.margin(2,0)}")
    print(f"The margin of 1 over 2 is {prof.margin(1,2)}")
    print(f"The margin of 2 over 1 is {prof.margin(2,1)}")

```

There are a number of other methods that related to the margins of a profile. 

1. $x$ is **majority preferred** to $y$ in $\mathbf{P}$ if $Margin_\mathbf{P}(x,y) > 0$

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile 
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [4, 3, 2])
    prof.display()
    print(f"0 is majority preferred over 1 is {prof.majority_prefers(0, 1)}")
    print(f"0 is majority prefered over 2 is {prof.majority_prefers(0, 2)}")
    print(f"1 is majority prefered over 2 is {prof.majority_prefers(1, 2)}")

```

2. The **margin matrix** for a profile $\mathbf{P}$  is a matrix where the $(i,j)$-entry is $Margin_\mathbf{P}(i,j)$. 

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile 
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [4, 3, 2])
    prof.display()
    print(f"The margin matrix for the profile is: {prof.margin_matrix()}")
    prof.display_margin_matrix()

```

3. The [**margin graph**](margin_graphs.md) is created using ``.margin_graph()`` and can be displayed using ``.display_margin_graph()``.


### Voting-Theoretic Attributes of Profiles

1. Condorcet candidates: 

    - A candidate $x$ is a  **Condorcet winner** if $x$ is majority preferred to every other candidate. 

    - A candidate $x$ is a  **Condorcet loser** if every other candidate is majority preferred to $x$. 

    - A candidate $x$ is a  **weak Condorcet winner** if there is no candidate that is majority preferred to $x$. 

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile 
    
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [3, 1, 1])
    prof.display()
    print(f"The Condorcet winner is {prof.condorcet_winner()}")
    print(f"The Condorcet loser is {prof.condorcet_loser()}")
    print(f"The weak Condorcet winner(s) is(are) {prof.weak_condorcet_winner()}")
    
    prof = Profile([[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]])
    prof.display()
    print(f"The Condorcet winner is {prof.condorcet_winner()}")
    print(f"The Condorcet loser is {prof.condorcet_loser()}")
    print(f"The weak Condorcet winner(s) is(are) {prof.weak_condorcet_winner()}")
    
    prof = Profile([[3, 0, 1, 2], [3, 1, 2, 0], [3, 2, 0, 1]])
    prof.display()
    print(f"The Condorcet winner is {prof.condorcet_winner()}")
    print(f"The Condorcet loser is {prof.condorcet_loser()}")
    print(f"The weak Condorcet winner(s) is(are) {prof.weak_condorcet_winner()}")
    
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [2, 1, 1])
    prof.display()
    print(f"The Condorcet winner is {prof.condorcet_winner()}")
    print(f"The Condorcet loser is {prof.condorcet_loser()}")
    print(f"The weak Condorcet winner(s) is(are) {prof.weak_condorcet_winner()}")

```

2. Scoring candidates: 

    - The **Plurality score** of a candidate $c$ in a profile $\mathbf{P}$ is the number of voters who rank $c$ is first place. 

    - The **Borda score** of candidate $c$ is calculated as follows: the score assigned to $c$ by a ranking is the number of candidates ranked below $c$, and the Borda score of $c$ is the sum of the scores assigned to $c$ by each ranking in the profile.

    - The **Copeland score** of candidate $c$ is calculate as follows:  $c$ receives 1 point for every candidate that  $c$ is majority preferred to, 0 points for every candidate that is tied with $c$, and -1  points for every candidate that is majority preferred to $c$. 

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile 
    
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [3, 2, 1])
    prof.display()
    print(f"The Plurality scores for the candidates are {prof.plurality_scores()}")
    print(f"The Borda scores for the candidates are {prof.borda_scores()}")
    print(f"The Copeland scores for the candidates are {prof.copeland_scores()}")

```

## Profile Class

```{eval-rst}
.. autoclass:: pref_voting.profiles.Profile
    :members: 
```

