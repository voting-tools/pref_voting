Overview
===========

The ``pref_voting`` package provides a number of classes that represent different types of elections.

1. A ``Profile`` represents an election where each voter submits a  linear order of the candidates. 
2. A ``ProfileWithTies`` represents an election where each voter submits a (truncated) ranking of the candidates. 
3. A ``GradeProfile`` represents an election where each voter submits a assignment of grades to each candidate. 
4. A ``UtilityProfile`` represents a situation where each voter submits a utility function mapping each candidate (or alternative) to a real number. 
5. A ``SpatialProfile`` represents a situation where each voter and each candidate are assigned an $n$-tuple of numbers from some $n$-dimensional space where each dimension represents different issues. 

In addition, there are a number of classes that represent different types of (weighted) directed graphs that are used when analyzing elections.

1. ``MajorityGraph``: a directed asymmetric graph in which the nodes are the candidates and an edge from $a$ to $b$ means that $a$ is majority preferred to $b$ (i.e., more voters rank $a$ strictly above $b$ than rank $b$ strictly above $a$).
2. ``MarginGraph``: a directed asymmetric graph in which the nodes are the candidates, an edge from $a$ to $b$ means that $a$ is majority preferred to $b$, and the weight of the edge is the margin of $a$ over $b$ (i.e., the number of voters who rank $a$ strictly above $b$ minus the number of voters who rank $b$ strictly above $a$).
3. ``SupportGraph``: a directed asymmetric graph in which the nodes are the candidates, an edge from $a$ to $b$ means that $a$ is majority preferred to $b$, and the weight of the edge is the number of voters who rank $a$ strictly above $b$.

## Helper Function: Creating Elections

```{eval-rst}

.. autofunction:: pref_voting.helper.create_election

```