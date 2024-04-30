Profiles with Ties
=======================================

Use the `ProfileWithTies` class to create a profile in which voters may submit strict weak orderings of the candidates, allowing ties, and/or in which voters do not rank all  the candidates.  To create a profile, specify a list of rankings, the number of candidates, the list of counts for each ranking, and possibly a candidate map (mapping candidates to their names). 

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

## ProfileWithTies Class

```{eval-rst}

.. note::
    The ``Ranking`` class used to the represent the ballots in a ``ProfileWithTies`` is described on the [ballots](ballots) page.

```

```{eval-rst}
.. autoclass:: pref_voting.profiles_with_ties.ProfileWithTies
    :members: 
```

