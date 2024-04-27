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
 - name: University of California, Berkeley
   index: 1
 - name: University of Maryland
   index: 2
date: 17 January 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 
aas-journal: 
---

# Summary

Preferential Voting Tools (`pref_voting`) is a Python package designed for research in voting theory [@Dummett1984;@Brams2002;@Tideman2006;@Zwicker2016;@Pacuit2019], a subfield of social choice theory [@Arrow1963;@Fishburn1973;@Kelly1988;@Sen2017], and for practical applications of the theory. The basic problem of voting theory concerns how to combine "inputs" from multiple individual voters into a single social "output". For example, a common type of input from each voter is a *ranking* of some set of candidates, while a common type of social output is the selection of a *winning candidate* (or perhaps a set of candidates tied for winning). A *voting method* is then a function that takes in a ranking from each voter and outputs a winning candidate (or set of tied candidates). Other functions may instead output a social ranking of the candidates [@Arrow1963], or a probability distribution over the candidates [@Brandt2017], etc., and other input types are also possible, such as sets of approved candidates [@Brams2007], or assignments of grades to candidates [@Balinski2010], or real-valued functions on the set of candidates [@Aspremont2002;@Sen2017], etc. Faced with a function of any of these types, voting theorists study the function from several perspectives, including the general principles or "axioms" it satisfies [@Nurmi1987;@Nurmi1999;@Felsenthal2012],  its susceptibility to manipulation by strategic voters [@Taylor2005], its statistical behavior according to probability models for generating voter inputs [@Merrill1988;@Green-Armytage2016], the complexity of the function and related computational problems (e.g., the problem of determining if the function can be manipulated in a given election) [@Faliszewski2009], and more. These studies are greatly facilitated by the implementation of algorithms for computing the relevant functions and checking their properties, which are provided in `pref_voting`.

# Statement of need

Research in the burgeoning field of *computational social choice* (COMSOC) [@Brandt2016;@Geist2017;@Aziz2019] often applies computer-assisted techniques to the study of voting methods and other collective decision procedures. The aim of `pref_voting` is to contribute to a comprehensive set of tools for such research. Other packages in this area include `abcvoting` [@Lackner2023], which focuses on approval-based committee voting,  `preflibtools` [@Mattei2013], which contains tools for working with preference data from [PrefLib.org](https://PrefLib.org), and `prefsampling` [@Boehmer2024], which implements probability models for generating voter rankings. The `pref_voting` package provides functionality not available in these previous packages, while also interfacing with `preflibtools` and `prefsampling`. Like `pref_voting`, the `VoteKit` [@MGGG2024] and `VoteLib` [@Simbera2021] packages provide implementations of a number of voting methods; and like `prefsampling`, `VoteKit` provides tools for generating elections. However, neither package includes all the voting methods and functionality in `pref_voting`, as described below. The `pref_voting` package has already been used in COMSOC research [@HKP2024] and in online COMSOC tools [@Peters2024]. The package can also be used by election administrators to determine election outcomes, as it is used in the [Stable Voting](https://stablevoting.org) website.

# Functionality

## Elections

The `pref_voting` package includes classes for the most important representations of elections, or types of `edata`, used in voting theory: 

 - `Profile`: each voter linearly orders the candidates; 
 - `ProfileWithTies`: each voter ranks the candidates, allowing ties and omissions of candidates;
 - `GradeProfile`: each voter assigns grades from some finite list of grades to selected candidates (with approval ballots as a special case); 
 - `UtilityProfile`: each voter assigns a real number to each candidate; 
 - `SpatialProfile`: each voter and each candidate is placed in a multi-dimensional space;
 - `MajorityGraph`: an edge from candidate A to candidate B represents that more voters rank A above B than vice versa;
 - `MarginGraph`: a weighted version of a `MajorityGraph`, where the weight on an edge represents the margin of victory (or other measure of strength of majority preference). 

The package also includes methods for transforming one type of representation into another, e.g., turning a `SpatialProfile` into a `UtilityProfile` given a choice of how spatial positions of voters and candidates determine voter utility functions [@MerrillGrofman1999], or turning a `MarginGraph` into a minimal `Profile` that induces that `MarginGraph` by solving an associated linear program, and so on. Other methods are included for standard voting-theoretic tests and operations, e.g., testing for the existence of Condorcet winners/losers, removing candidates, and so on. Methods are also included to import from and export to the PrefLib preference data format, the ABIF format [@Lanphier2024], and other data formats.

## Generating elections

For sampling profiles according to standard probability models, `pref_voting` interfaces with the `prefsampling` package. In addition, `pref_voting` contains functions for sampling other types of `edata` listed above, as well as functions for enumerating such objects up to certain equivalence relations.

## Collective decision procedures

Several classes of collective decision procedures are built into `pref_voting`:

- `VotingMethod`: given `edata`, outputs a list of candidates, representing tied winners;
- `ProbVotingMethod`: given `edata`, outputs a dictionary whose keys are candidates and whose values are probabilities;
- `SocialWelfareFunction`: given `edata`, outputs a ranking of the candidates.

Dozens of such functions are implemented in `pref_voting` and organized into standard groups identified in voting theory, e.g., positional scoring rules, iterative methods, margin-based methods (weighted tournament solutions), cardinal methods, etc.

## Axioms

The `pref_voting` package also contains an `Axiom` class for functions that check whether a collective decision procedure satisfies a given axiom with respect to some `edata`. Each axiom comes with a `has_violation` method that checks whether there is at least one violation of the axiom by the procedure for the given `edata`, as well as a `find_all_violations` method that enumerates all such violations together with relevant data. Axioms are divided into several well-known groups from voting theory, e.g., dominance axioms, monotonicity axioms, variable voter axioms, variable candidate axioms, etc.

## Analysis

Finally, `pref_voting` comes with functions that facilitate the analysis of collective decision procedures, such as producing data on the frequency of axiom violations in elections generated using one of the available probability models.


# Acknowledgements

We thank Jobst Heitzig and Dominik Peters for helpful contributions, Zoi Terzopoulou for helpful feature requests, and all three for helpful feedback on this paper.

# References
