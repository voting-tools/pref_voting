Overview
===========

## Profiles

A Profile is defined from a list of rankings, where each ranking is a tuple or list of the candidates. 

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile

    prof = Profile([
        [0, 1, 2], 
        [2, 1, 0], 
        [1, 2, 0]
    ])
    print(f"There are {prof.num_voters} voters in the profile.")
    print(f"There are {prof.num_cands} candidates in the profile.")

    # display prof - the header is the number of voters with each ranking
    prof.display()

There are two optional parameters when defining a Profile:

1. rcounts: an array specifying the number of voters who submit each ranking in the profile.
2. cmap: a dictionary mapping candidates to candidate names.

.. exec_code::

    from pref_voting.profiles import Profile

    rankings = [[0, 1, 2], [2, 1, 0], [1, 2, 0]]
    rcounts = [1, 2, 3] 

    #1 voter with the ranking (0,1,2),
    #2 voters with the ranking (2,1,0), and 
    #3 voters with the ranking (1,2,0).

    prof2 = Profile(rankings, rcounts=rcounts)
    print(f"There are {prof2.num_voters} voters in the profile.")
    print(f"There are {prof2.num_cands} candidates in the profile.")

    # display prof2  - the header is the number of voters with each ranking
    prof2.display()

    cmap={0:"a", 1:"b", 2:"c"}

    prof3 = Profile(rankings, rcounts=rcounts, cmap=cmap)

    # the candidate names are used in cmap
    prof3.display()

```

There are a number of useful methods associated with a Profile.  Suppose that $\mathbf{P}$ is a profile, $X(\mathbf{P})$ is the set of candidates in $\mathbf{P}$, and $V(\mathbf{P})$ is the set of voters in $\mathbf{P}$. Let $a,b\in X(\mathbf{P})$.

* The support for $a$ over $b$ is $|\{i\in V(\mathbf{P})\mid a\mathrel{\mathbf{P}_i}b\}|$. 
* The **margin of $a$ over $b$ in $\mathbf{P}$** is $Margin_\mathbf{P}(a,b)=|\{i\in V(\mathbf{P})\mid a\mathrel{\mathbf{P}_i}b\}| -|\{i\in V(\mathbf{P})\mid b\mathrel{\mathbf{P}_i} a\}|$.
* Candidate $a$ is **majority preferred** to $b$ (and $b$ is **majority dispreferred** to $a$) when $Margin_\mathbf{P}(a,b)> 0$.

```{eval-rst}

.. exec_code::

    from pref_voting.profiles import Profile

    rankings = [ (0, 1, 2, 3), (2, 3, 1, 0),  (3, 1, 2, 0), (1, 2, 0, 3), (1, 3, 2, 0)]

    prof = Profile(rankings)

    prof.display()

    c1 = 2
    c2 = 3

    print("")
    print(f"The candidates are {list(prof.candidates)}")
    print(f"support of {c1} over {c2}: ", prof.support(c1,c2))
    print(f"support of {c2} over {c1}: ", prof.support(c2, c1))
    print(f"Margin({c1},{c2}) =  ", prof.margin(c1,c2))
    print(f"Margin({c2},{c1}) =  ", prof.margin(c2,c1))
    print(f"{c1} is majority preferred to {c2} is ", prof.majority_prefers(c1,c2))
    print(f"{c2} is majority preferred to {c1} is ", prof.majority_prefers(c2,c1))
    print(f"The number of voters that rank {c1} in 1st place is ", prof.num_rank(c1, 1))
    print(f"The number of voters that rank {c1} in 2nd place is ", prof.num_rank(c1, 2))
    print(f"The size of a strict majority of voters is ", prof.strict_maj_size())

```

In addition, there are methods for each of the following: 

* Condorcet winner: a candidate who is majority preferred to every other candidate (returns None if the Condorcet winner does not exist);
* weak Condorcet winner: a list of candidates who are not majority dispreferred to any other candidate (returns None if no such candidate exists); 
* Condorcet loser: a candidate who is majority dispreferred to every other candidate (returns None if the Condorcet loser does not exist);
* Plurality scores: a dictionary associating with each candidate its Plurality score;
* Borda scores: a dictionary associating with each candidate its Borda score;
* Copeland scores: a dictionary associating with each candidate its Copeland score.

```{eval-rst}

.. exec_code:: 

    from pref_voting.profiles import Profile

    rankings = [
        (0, 1, 2, 3), 
        (2, 3, 1, 0), 
        (3, 1, 2, 0), 
        (1, 2, 0, 3), 
        (1, 3, 2, 0)
    ]
    prof = Profile(rankings)

    prof.display()
    print("")
    print(f"The Plurality scores are ", prof.plurality_scores())
    print(f"The Borda scores are ", prof.borda_scores())
    print(f"The Copeland scores are ", prof.copeland_scores())
    print(f"The Condorcet winner is ", prof.condorcet_winner())
    print(f"The weak Condorcet winner is ", prof.weak_condorcet_winner())
    print(f"The Condorcet loser is ", prof.condorcet_loser())

```

## Profile with Ties

Use the `ProfileWithTies` class to create a profile in which voters may submit strict weak orderings of the candidates, allowing ties, and/or in which voters do not rank all of the candidates.  To create a profile, specify a list of rankings, the number of candidates, the list of counts for each ranking, and possibly a candidate map (mapping candidates to their names). 

```{eval-rst}

.. exec_code::

    from pref_voting.profiles_with_ties import ProfileWithTies

    a = "a"
    b = "b"
    c = "c"

    rankings = [
        {a:1, b:1, c:1},
        {a:1, b:2, c:2},
        {a:1, b:2, c:3},
        {a:1, b:1, c:2},
        {a:1, b:1, c:1},
        {a:1, b:1, c:2},
        {a:1, c:1, b:2},
        {c:1, b:1, a:2}
    ]
    rcounts = [1, 2, 1, 4, 1, 2, 1, 1]
    prof = ProfileWithTies(rankings, rcounts=rcounts)

    prof.display()

    print("")
    for r,n in zip(prof.rankings, prof.rcounts):
        print(f"{n} voters have the ranking {r}")

    # the support of a over b is the number of voters that rank a strictly above b
    print("\n")
    print(f"support(a, b) = {prof.support(a, b)}")
    print(f"support(a, a) = {prof.support(a, a)}")
    print(f"support(b, a) = {prof.support(b, a)}")
    print(f"support(a, c) = {prof.support(a, c)}")
    print(f"support(c, a) = {prof.support(c, a)}")
    print(f"support(b, c) = {prof.support(b, c)}")
    print(f"support(c, b) = {prof.support(c, b)}")

    # the margin of a over b is the number of voters who rank a strictly above b minus
    # the number of voters who rank b strictly above a
    print("\n")
    print(f"margin(a, b) = {prof.margin(a, b)}")
    print(f"margin(a, a) = {prof.margin(a, a)}")
    print(f"margin(b, a) = {prof.margin(b, a)}")
    print(f"margin(a, c) = {prof.margin(a, c)}")
    print(f"margin(c, a) = {prof.margin(c, a)}")
    print(f"margin(b, c) = {prof.margin(b, c)}")
    print(f"margin(c, b) = {prof.margin(c, b)}")

    # the ratio of a over b is the support of a over b divided by the support of b over a
    print("\n")
    print(f"ratio(a, b) = {prof.ratio(a, b)}")
    print(f"ratio(a, a) = {prof.ratio(a, a)}")
    print(f"ratio(b, a) = {prof.ratio(b, a)}")
    print(f"ratio(a, c) = {prof.ratio(a, c)}")
    print(f"ratio(c, a) = {prof.ratio(c, a)}")
    print(f"ratio(b, c) = {prof.ratio(b, c)}")
    print(f"ratio(c, b) = {prof.ratio(c, b)}")

```

## (Weighted) Majority Graphs

A **majority graph** is a directed asymmetric graph in which the nodes are the candidates and an edge from $a$ to $b$ means that $a$ is majority preferred to $b$. 

```{eval-rst}

A :class:`~pref_voting.weighted_majority_graphs.MajorityGraph` has a number of methods used by voting methods. 

.. exec_code::

    from pref_voting.weighted_majority_graphs import MajorityGraph

    mg = MajorityGraph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])

    print(mg.edges)
    for c1 in mg.candidates:
        for c2 in mg.candidates: 
            print(f"{c1} is majority preferred to {c2}: {mg.majority_prefers(c1, c2)}")
            print(f"{c1} is tied with {c2}: {mg.is_tied(c1, c2)}")

    for c in mg.candidates:
        print(f"The dominators of {c} are {mg.dominators(c)}")
        print(f"The candidates that {c} dominates are {mg.dominates(c)}")
    
    print(f"Copeland scores: {mg.copeland_scores()}")
    print(f"Condorcet winner: {mg.condorcet_winner()}")
    print(f"Weak Condorcet winners: {mg.weak_condorcet_winner()}")
    print(f"Condorcet loser: {mg.condorcet_loser()}")

    print()

    mg2 = MajorityGraph([0, 1, 2], [(0, 1), (1, 2)])

    print(mg2.edges)
    for c1 in mg2.candidates:
        for c2 in mg2.candidates: 
            print(f"{c1} is majority preferred to {c2}: {mg2.majority_prefers(c1, c2)}")
            print(f"{c1} is tied with {c2}: {mg2.is_tied(c1, c2)}")

    for c in mg2.candidates:
        print(f"The dominators of {c} are {mg2.dominators(c)}")
        print(f"The candidates that {c} dominates are {mg2.dominates(c)}")
    
    print(f"Copeland scores: {mg2.copeland_scores()}")
    print(f"Condorcet winner: {mg2.condorcet_winner()}")
    print(f"Weak Condorcet winners: {mg2.weak_condorcet_winner()}")
    print(f"Condorcet loser: {mg2.condorcet_loser()}")

```

A **margin graph** is a weighted directed asymmetric graph in which the nodes are the candidates, an edge from $a$ to $b$ means that $a$ is majority preferred to $b$, and the weight of the edge is the margin of $a$ over $b$. 

```{eval-rst}

A :class:`~pref_voting.weighted_majority_graphs.MarginGraph` has a number of methods used by voting methods. 

.. important:: 

    The weights of a MarginGraph can be any numbers.  However, if the weights are generated by a profile of linear orders, then the weights will have the same parity (which is even if there is any zero margin between distinct candidates). 

.. exec_code::

    from pref_voting.weighted_majority_graphs import MarginGraph

    mg = MarginGraph([0, 1, 2], [(0, 1, 1), (1, 2, 3), (2, 0, 3)])

    print(mg.edges)
    for c1 in mg.candidates:
        for c2 in mg.candidates: 
            print(f"the margin of {c1} over {c2} is {mg.margin(c1, c2)}")
            print(f"{c1} is majority preferred to {c2}: {mg.majority_prefers(c1, c2)}")
            print(f"{c1} is tied with {c2}: {mg.is_tied(c1, c2)}")

    for c in mg.candidates:
        print(f"The dominators of {c} are {mg.dominators(c)}")
        print(f"The candidates that {c} dominates are {mg.dominates(c)}")
    
    print(f"Copeland scores: {mg.copeland_scores()}")
    print(f"Condorcet winner: {mg.condorcet_winner()}")
    print(f"Weak Condorcet winners: {mg.weak_condorcet_winner()}")
    print(f"Condorcet loser: {mg.condorcet_loser()}")

```



```{eval-rst}

Both :class:~pref_voting.weighted_margin_graph.MarginGraph` and :class:~pref_voting.weighted_margin_graph.MajorityGraph` can be generated from a profile.

.. exec_code::

    from pref_voting.profiles import Profile 
    from pref_voting.profiles_with_ties import ProfileWithTies 

    prof = Profile([[0, 1, 2], [2, 0, 1], [1, 0, 2]], rcounts=[2, 1, 2])

    prof.display()

    majg = prof.majority_graph()
    print(f"The majority graph edges are {majg.edges}")

    mg = prof.margin_graph()
    print(f"The margin graph edges are {mg.edges}")

    prof2 = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1:1, 2:1, 0:2}, {2:1, 0:2}], [2, 3, 1])
    prof2.display()

    majg = prof2.majority_graph()
    print(f"The majority graph edges are {majg.edges}")

    mg = prof2.margin_graph()
    print(f"The margin graph edges are {mg.edges}")
    
```
