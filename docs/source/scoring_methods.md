Positional Scoring Rules
=======================================

Suppose that $\mathbf{P}$ is an anonymous profile of linear orders (i.e., a ``Profile`` object). A **scoring vector** for $m$ candidates is a tuple of numbers $\langle s_1, s_2, \ldots, s_m\rangle$ where for each $l=1,\ldots, m-1$, $s_l \ge s_{l+1}$.  

Suppose that $a\in X(\mathbf{P})$ and $R\in \mathcal{L}(X(\mathbf{P}))$ is linear order on the set of candidates.  The **rank** of $a$ in $R$ is one more than the number of candidates ranked above $a$ (i.e., $|\{b\mid b\in X(\mathbf{P})\mbox{ and } b\mathrel{R}a\}| + 1$).  The **score of $a$** given $R$ is  $score(R,a)=s_r$ where $r$ is the *rank* of $a$ in $R$. 
 
For each anonymous profile $\mathbf{P}$,  $a\in X(\mathbf{P})$ and scoring vector $\vec{s}$ for the number of candidates in $\mathbf{P}$, let 
 
 $$
 Score_\vec{s}(\mathbf{P},x)= \sum_{R\in\mathcal{L}(X(\mathbf{P}))} \mathbf{P}(R) * score(R, x)
 $$

A **positional scoring rule** is defined by specifying a scoring vector for each number of candidates.  The winners in $\mathbf{P}$ according to the positional scoring rule is the set of candidates that maximize their scoring according to the scoring vector for the number of candidates in $\mathbf{P}$.  That is, if $F$ is a positional scoring rule, then for each profile $\mathbf{P}$,  

$$
F(\mathbf{P}) =   \mathrm{argmax}_{a\in X(\mathbf{P})} Score_\vec{s}(\mathbf{P}, a).   
$$

where $\vec{s}$ is the scoring vector for the number of candidates in $\mathbf{P}$. 

The most well-known positional scoring rules are: 

1. Plurality: the positional scoring rule for $\langle 1, 0, \ldots,   0\rangle$.
2. Borda:  the positional scoring rule for ${\langle m-1, m-2, \ldots, 1,  0\rangle}$, where $m$ is the number of candidates.
3. Anti-Plurality:  the positional scoring rule for ${\langle 0, 0, \ldots, -1\rangle}$.

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile
    from pref_voting.scoring_methods import plurality, borda, anti_plurality

    prof = Profile([[0, 2, 1], [0, 1, 2], [2, 1, 0], [1, 2, 0]], [3, 2, 1, 4])

    prof.display()
    plurality.display(prof)
    borda.display(prof)
    anti_plurality.display(prof)

```


```{eval-rst}

An arbitrary scoring rule can be defined using the ``scoring_rule`` function.  For instance, the Two-Approval voting method is a positional scoring rule in which scores are assigned as follows: candidates ranked either in first- or second-place are given a score of 1, otherwise the candidates are given a score of 0.    

.. exec_code::

    from pref_voting.profiles import Profile
    from pref_voting.scoring_methods import plurality, borda, anti_plurality, scoring_rule

    prof = Profile([[0, 2, 1], [0, 1, 2], [2, 1, 0], [1, 2, 0]], [3, 2, 1, 4])
    
    two_approval_score = lambda num_cands, rank: 1 if rank == 1 or rank == 2 else 0

    prof.display()
    scoring_rule.display(prof, score = two_approval_score)  

    # for comparison, display the Plurality winner
    plurality.display(prof)  

One problem with the above code is that the name of the scoring rule is "Scoring Rule" rather than "Two-Approval".  To define a voting method using the ``scoring_rule`` function, use the `@vm` decorator.

.. exec_code::

    from pref_voting.profiles import Profile
    from pref_voting.scoring_methods import plurality, borda, anti_plurality, scoring_rule
    from pref_voting.voting_method import vm
    
    @vm(name="Two-Approval")
    def two_approval(profile, curr_cands = None): 
        """Returns the list of candidates with the largest two-approval score in the profile restricted to curr_cands. 
        """

        two_approval_score = lambda num_cands, rank: 1 if rank == 1 or rank == 2 else 0

        return scoring_rule(profile, curr_cands = curr_cands, score=two_approval_score)

    prof = Profile([[0, 2, 1], [0, 1, 2], [2, 1, 0], [1, 2, 0]], [3, 2, 1, 4])
    
    prof.display()
    two_approval.display(prof)  

```


## Plurality

```{eval-rst}

.. autofunction:: pref_voting.scoring_methods.plurality

```

## Borda

```{eval-rst}

.. autofunction:: pref_voting.scoring_methods.borda

```

### Borda for ProfilesWithTies

```{eval-rst}

.. autofunction:: pref_voting.scoring_methods.borda_for_profile_with_ties

```

## Anti-Plurality

```{eval-rst}

.. autofunction:: pref_voting.scoring_methods.anti_plurality(profile, curr_cands = None)

```

## Scoring Rule

```{eval-rst}

.. autofunction:: pref_voting.scoring_methods.scoring_rule(profile, curr_cands = None, score = lambda num_cands, rank: 1 if rank == 1 else 0)

```


