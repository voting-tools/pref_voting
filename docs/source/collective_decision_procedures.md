Overview
==========

We have three types of collective decision procedures: 

  1. Voting Method: given edata, outputs a sorted list of candidates, representing tied winners;
  2. Probabilistic Voting Method: given edata, outputs a dictionary whose keys are candidates and whose values are probabilities;
  2. Social Welfare Function: given edata, outputs a [Ranking](ballots.md#ranking-class) of the candidates.

We further categorize collective decision procedures based on the input from the voters: 

  1. Ordinal procedures take as input one or more of the following types of edata: [Profile](profiles.md#profile-class), [ProfilesWithTies](profiles_with_ties.md), [MajorityGraph](weighted_majority_graphs.md#majoritygraph-class), [MarginGraph](weighted_majority_graphs.md#margingraph-class), and [SupportGraph](weighted_majority_graphs.md#supportgraph-class). These procedures are discussed in the following: 

      * [Positional Scoring Rules](scoring_methods.md)
      * [Iterative Methods](iterative_methods.md)
      * [C1 Methods](c1_methods.md)
      * [Margin Methods](margin_based_methods.md)
      * [Combined Methods](combined_methods.md)
      * [Other Methods](other_methods.md)
      * [Probabilistic Methods](probabilistic_methods.md)


  2. Cardinal procedures take as input a [UtilityProfile](utility_profiles.md) and/or a [GradeProfile](grade_profiles.md). These procedures are discussed in the following: 

      * [Utility Methods](utility_methods.md)
      * [Grade Methods](grade_methods.md)

## VotingMethod Class and Decorator

```{eval-rst} 

.. autoclass:: pref_voting.voting_method.VotingMethod
    :members: 

.. autofunction:: pref_voting.voting_method.vm

```

## ProbVotingMethod Function Class and Decorator

```{eval-rst} 

.. autoclass:: pref_voting.prob_voting_method.ProbVotingMethod
    :members: 


.. autofunction:: pref_voting.prob_voting_method.pvm

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