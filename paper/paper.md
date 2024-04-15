---
title: 'pref_voting: The Preferential Voting Tools package for Python'
tags:
  - Python
  - voting methods
  - voting paradoxes
  - social choice theory
  - utility functions
authors:
  - name: Wesley H. Holliday
    orcid: 0000-0001-6054-9052
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Eric Pacuit
    orcid: 0000-0002-0751-9011
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: Department of Philosophy, University of California, Berkeley, USA
   index: 1
 - name: Department of Philosophy, University of Maryland, USA
   index: 2
date: 11 January 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 
aas-journal: 
---

# Summary

Preferential Voting Tools (`pref_voting`) is a Python package designed for research in and applications of voting theory. The basic problem of voting theory [`@Zwicker2016`] concerns how to combine "inputs" from many individual voters into a single social "output". For example, a common type of input to elicit from each voter is a *ranking* of some set of candidates according to the voter's preferences, while a common type of social output is the selection of a *winning candidate* (or perhaps a set of candidates tied for winning). A *voting method* is then a function that takes in a ranking from each individual and outputs a winning candidate (or set of tied candidates). Other functions may instead output a social ranking of the candidates, or a probability distribution over the candidates, etc., and other input types are also possible, such as grades that voters assign to candidates, or voter utility functions, etc. Faced with any of these types of aggregation functions, voting theorists study a function from several perspectives, including the general principles or "axioms" it satisfies, its statistical behavior according to various probability models for generating voter inputs, its computational complexity, and more. These studies are greatly facilitated by the implementation of algorithms for computing aggregation functions and checking their properties, which are provided in `pref_voting`.

# Statement of need

Research in the burgeoning field of *computational social choice* [`@Brandt2016`] often applies computer-assisted methods to the study of voting methods and other aggregation functions. The aim of `pref_voting` is to contribute to a comprehensive set of tools for such research. Other packages in this area include `abcvoting` [`@Lackner2023`], which focuses on approval-based committee voting,  `prefsampling` [`@Boehmer2024`], which implements  probability models for generating voter rankings, and `prelibtools` [`@Mattei2013`], which provides tools for working with preference data from [PrefLib.org](https://PrefLib.org). The `pref_voting` package provides functionality not available in those previous packages, as desribed below, while also interfacing with other packages. Like `pref_voting`, the `VoteLib` [`@Simbera2021`] and `VoteKit` [`@MGGG2024`] packages provides implementations of a number of voting methods; and like `prefsampling`, `VoteKit` provides tools for generating elections. However, neither package includes all the voting methods and functionality in `pref_voting`, as described below. The `pref_voting` package has already been used in research in computational social choice [`@HKP2024`]. The package can also be used by election administrators to determine election outcomes, as it is used in the backend of the [Stable Voting website](https://stablevoting.org).

# Functionality

## Elections

The `pref_voting` package includes classes for the most important representations of elections, or types of `edata`, used in voting theory: 

 - `Profile`: each voter has a linear order of the candidates; 
 - `ProfileWithTies`: each voter has a ranking of the candidates that may contain ties and may omit some candidates;
 - `GradeProfile`: voters assign to candidates grades from some finite list of grades; 
 - `UtilityProfile`: each voter has a cardinal utility function on the set of candidates; 
 - `SpatialProfile`: each voter and each candidate is placed in a multi-dimensional space;
 - `MajorityGraph`: an edge from one candidate to another represents that a majority of voters prefer the first to the second;
 - `MarginGraph`: a weighted version of a `MajorityGraph`, where the weight on an edge represents the margin of victory (or other measure of strength of majority preference). 

The package also includes methods for transforming one type of representation into another, e.g., turning a `SpatialProfile` into a `UtilityProfile` given a choice of how spatial positions of voters and candidates determine voter utility functions, or turning a `MarginGraph` into a `ProfileWithTies` that induces that `MarginGraph` by solving an associated linear program, and so on. Other methods are included for standard voting-theoretic tests and operations, e.g., testing for the existence of Condorcet winners/losers, removing candidates, and so on. Methods are also included to import from and export to the PrefLib preference data format, the ABIF format, and other data formats.

## Generating elections

For sampling profiles according to standard probability models, `pref_voting` interfaces with the `prefsampling` package. In addition, `pref_voting` contains functions for sampling other types of `edata` listed above, as well as functions for enumerating such objects up to certain equivalence relations.

## Aggregation  methods

Several classes of aggregation methods are built into `pref_voting`:

- `VotingMethod`: given `edata`, outputs a sorted list of candidates, representing tied winners;
- `ProbVotingMethod`: given `edata`, outputs a dictionary whose keys are candidates and whose values are probabilities;
- `SocialWelfareFunctions`: given `edata`, outputs a `Ranking` of the candidates.

Dozens of aggregation methods are implemented in `pref_voting` and organized into standard classes identified in voting theory, e.g., positional scoring rules, iterative methods, margin-based methods (weighted tournament methods), cardinal methods, etc.

## Axioms

The `pref_voting` package also contains an `Axiom` class for functions that check whether an aggregation method satisfies a given axiom with respect to some `edata`. Each axiom comes with a `has_violation` function that checks whether there is at least one violation of the axiom by the aggreation method for the given `edata`, as well as a `find_all_violations` function that enumerates all such violations with the relevant witnessing data. Axioms are divided into several well-known classes from voting theory, e.g., dominance axioms, monotonicity axioms, variable voter axioms, variable candidate axioms, etc.

## Analysis

Finally, `pref_voting` comes with functions that facilitate the analysis of aggregation methods, such as producing data on the frequency of axiom violations in elections generated using one of the available probability models.

# Acknowledgements

We thank Jobst Heitzig and Dominik Peters for helpful contributions and Zoi Terzopoulou for helpful feature requests.

# References