Overview
==========

We have two types of aggregation functions: 

  1. Voting Methods: Aggregation methods that output a sorted list of candidates; and
  2. Social Welfare Functions: Aggregation methods that output a [Ranking](ballots.md#ranking-class) of the candidates.

We further categorize aggregation methods based on the input from the voters: 

  1. Ordinal aggregation functions produce sets of candidates or rankings based on a [Profile](profiles.md#profile-class), [ProfilesWithTies](profiles_with_ties.md), [MajorityGraph](weighted_majority_graphs.md#majoritygraph-class), [MarginGraph](weighted_majority_graphs.md#margingraph-class), and/or [SupportGraph](weighted_majority_graphs.md#supportgraph-class) are discussed in the following: 

      * [Positional Scoring Rules](scoring_methods.md)
      * [Iterative Methods](iterative_methods.md)
      * [C1 Methods](c1_methods.md)
      * [Margin Methods](margin_based_methods.md)
      * [Combined Methods](combined_methods.md)
      * [Other Methods](other_methods.md)
      * [Probabilistic Methods](probabilistic_methods.md)


  2. Cardinal aggregation functions that produce sets of candidates or rankings based on a [UtilityProfile](utility_profiles.md) and/or a [GradeProfile](grade_profiles.md) are discussed in the following: 

      * [Utility Methods](utility_methods.md)
      * [Grade Methods](grade_methods.md)


## VotingMethod Class and Decorator

```{eval-rst} 

.. autoclass:: pref_voting.voting_method.VotingMethod
    :members: 

.. autofunction:: pref_voting.voting_method.vm

```

## SocialWelfareFunction Class and Decorator

```{eval-rst} 

.. autoclass:: pref_voting.social_welfare_function.SocialWelfareFunction
    :members: 

.. autofunction:: pref_voting.social_welfare_function.swf

```


## Helper Functions: Converting between Voting Methods and Social Welfare Functions

```{eval-rst} 

.. autofunction:: pref_voting.helper.swf_from_vm


.. autofunction:: pref_voting.helper.vm_from_swf


```
