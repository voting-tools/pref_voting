Overview
==========

```{eval-rst}

Voting methods are defined as follows::

    @vm(name="Voting Method Name")
    def name_of_voting_method(edata, curr_cands=None):
        # edata can be a Profile, ProfileWithTies, MajorityGraph, and/or MarginGraph
        # if curr_cands is not None, restrict the election to curr_cands
        # voting methods return a sorted list of candidates

```

The decorator ``vm`` adds the following attribute and methods to a voting method: 

1. ``.name``: The human-readable name of the voting method
2. ``.display``: Display the winners 
3. ``.choose``: Choose a random winner from the set of winners
4. ``.set_name``: set the name of the voting method. 

```{eval-rst}


For instance, the voting method :meth:`~pref_voting.scoring_methods.plurality` can be used as follows: 

.. exec_code::

    from pref_voting.profiles import Profile
    from pref_voting.scoring_methods import plurality

    cmap = {0:"a", 1:"b", 2:"c", 3:"d"}

    prof = Profile([[0, 1, 2, 3], [2, 0, 1, 3], [1, 3, 0, 2], [3, 1, 0, 2]], [3, 2, 3, 1])

    prof.display()

    print(plurality.name)
    print(plurality(prof))
    print(plurality(prof, curr_cands = [2, 3]))
    print(plurality.choose(prof))
    plurality.display(prof)
    plurality.display(prof, cmap = cmap)


```
## VotingMethod Class

```{eval-rst} 

.. autoclass:: pref_voting.voting_method.VotingMethod
    :members: 

.. autofunction:: pref_voting.voting_method.vm

```

## Voting Methods

The main voting methods that are implemented with a checkmark in a column if the voting method can compute the winners for the type of input.
 
```{eval-rst} 

.. list-table:: 
    :header-rows: 1
    :stub-columns: 1
    :align: center

    * - Voting Method
      - Profile
      - ProfileWithTies
      - MajorityGraph
      - MarginGraph
    * - Anti Plurality 
    
        :meth:`~pref_voting.scoring_methods.anti_plurality`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Baldwin 

        :meth:`~pref_voting.iterative_methods.baldwin`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Baldwin PUT

        :meth:`~pref_voting.iterative_methods.baldwin_put`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Banks

        :meth:`~pref_voting.other_methods.banks`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Beat Path

        :meth:`~pref_voting.margin_based_methods.beat_path`
      - :math:`\checkmark`
      - :math:`\checkmark`
      - 
      - :math:`\checkmark`
    * - Blacks

        :meth:`~pref_voting.combined_methods.blacks`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Borda

        :meth:`~pref_voting.scoring_methods.borda`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Bucklin

        :meth:`~pref_voting.other_methods.bucklin`
      - :math:`\checkmark`
      -   
      -  
      - 
    * - Condorcet

        :meth:`~pref_voting.c1_methods.condorcet`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Condorcet IRV

        :meth:`~pref_voting.combined_methods.condorcet_irv`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      -  
      -  
    * - Condorcet IRV PUT

        :meth:`~pref_voting.combined_methods.condorcet_irv_put`
      - :math:`\checkmark`
      -   
      -  
      -  
    * - Coombs

        :meth:`~pref_voting.iterative_methods.coombs`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Coombs PUT

        :meth:`~pref_voting.iterative_methods.coombs_put`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Copeland

        :meth:`~pref_voting.c1_methods.copeland`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Daunou

        :meth:`~pref_voting.combined_methods.daunou`
      - :math:`\checkmark`
      -  
      -  
      -
    * - GOCHA

        :meth:`~pref_voting.c1_methods.gocha`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Instant Runoff

        :meth:`~pref_voting.iterative_methods.instant_runoff`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Instant Runoff For Truncated Linear Orders

        :meth:`~pref_voting.iterative_methods.instant_runoff_for_truncated_linear_orders`
      - 
      - :math:`\checkmark`
      -  
      -
    * - Instant Runoff PUT

        :meth:`~pref_voting.iterative_methods.instant_runoff_put`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Iterated Removal of Condorcet Losers

        :meth:`~pref_voting.iterative_methods.iterated_removal_cl`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Iterated Removal of Split Cycle Losers

        :meth:`~pref_voting.iterative_methods.iterated_split_cycle`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Kemmeny-Young

        :meth:`~pref_voting.other_methods.kemmeny_young`
      - :math:`\checkmark`
      -   
      -  
      -      
    * - Llull

        :meth:`~pref_voting.c1_methods.llull`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Majority

        :meth:`~pref_voting.other_methods.majority`
      - :math:`\checkmark`
      -   
      -  
      -  
    * - Minimax

        :meth:`~pref_voting.margin_based_methods.minimax`
      - :math:`\checkmark`
      - :math:`\checkmark`
      - 
      - :math:`\checkmark`
    * - Plurality

        :meth:`~pref_voting.scoring_methods.plurality`
      - :math:`\checkmark`
      -  
      -  
      -
    * - PluralityWRunoff

        :meth:`~pref_voting.iterative_methods.plurality_with_runoff`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Ranked Pairs

        :meth:`~pref_voting.margin_based_methods.ranked_pairs`
      - :math:`\checkmark`
      - :math:`\checkmark`
      - 
      - :math:`\checkmark`
    * - River

        :meth:`~pref_voting.margin_based_methods.river`
      - :math:`\checkmark`
      - :math:`\checkmark`
      - 
      - :math:`\checkmark`
    * - Simple Stable Voting

        :meth:`~pref_voting.margin_based_methods.simple_stable_voting`
      - :math:`\checkmark`
      - :math:`\checkmark`
      - 
      - :math:`\checkmark`
    * - Simplified Bucklin

        :meth:`~pref_voting.other_methods.simplified_bucklin`
      - :math:`\checkmark`
      -   
      -  
      -     
    * - Slater

        :meth:`~pref_voting.other_methods.slater`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      -  
      - :math:`\checkmark`
    * - Smith IRV

        :meth:`~pref_voting.combined_methods.smith_irv`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Split Cycle

        :meth:`~pref_voting.margin_based_methods.split_cycle`
      - :math:`\checkmark`
      - :math:`\checkmark`
      - 
      - :math:`\checkmark`
    * - Stable Voting

        :meth:`~pref_voting.margin_based_methods.stable_voting`
      - :math:`\checkmark`
      - :math:`\checkmark`
      - 
      - :math:`\checkmark`
    * - Strict Nanson

        :meth:`~pref_voting.iterative_methods.strict_nanson`
      - :math:`\checkmark`
      -  
      -  
      -
    * - Top Cycle

        :meth:`~pref_voting.c1_methods.top_cycle`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Uncovered Set (Gillies Version)

        :meth:`~pref_voting.c1_methods.uc_gill`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Uncovered Set (Fishburn Version)

        :meth:`~pref_voting.c1_methods.uc_fish`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Uncovered Set (Bordes Version)

        :meth:`~pref_voting.c1_methods.uc_bordes`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Uncovered Set (McKelvey Version)

        :meth:`~pref_voting.c1_methods.uc_mckelvey`
      - :math:`\checkmark`
      - :math:`\checkmark` 
      - :math:`\checkmark` 
      - :math:`\checkmark`
    * - Weak Nanson

        :meth:`~pref_voting.iterative_methods.weak_nanson`
      - :math:`\checkmark`
      -  
      -  
      -


``` 

## Explanations

```{eval-rst} 

.. list-table:: 
    :header-rows: 1
    :stub-columns: 1
    :align: left

    * - Voting Method
      - Explanation Function
      - Information Provided
    * - Baldwin 

        :meth:`~pref_voting.iterative_methods.baldwin`
      - :meth:`~pref_voting.iterative_methods.baldwin_with_explanation`
      - The candidates that are eliminated and the Borda scores of the remaining candidates (in the profile restricted to candidates that have not been eliminated) 
    * - Banks

        :meth:`~pref_voting.other_methods.banks`
      - :meth:`~pref_voting.other_methods.banks_with_explanation`
      - The maximal chains in the majority graph
    * - Borda

        :meth:`~pref_voting.scoring_methods.borda`
      - Profile method: :meth:`~pref_voting.profiles.Profile.borda_scores`
      - The Borda score for each candidate
    * - Bucklin

        :meth:`~pref_voting.other_methods.bucklin`
      - :meth:`~pref_voting.other_methods.bucklin_with_explanation`
      - The score for each candidate
    * - Coombs

        :meth:`~pref_voting.iterative_methods.coombs`
      - :meth:`~pref_voting.iterative_methods.coombs_with_explanation`
      - The order of elimination of the candidates 
    * - Copeland

        :meth:`~pref_voting.c1_methods.copeland`
      - Profile method: :meth:`~pref_voting.profiles.Profile.copeland_scores`

        Majority/Margin Graph method: :meth:`~pref_voting.weighted_majority_graphs.MajorityGraph.copeland_scores`
      - The Copeland score for each candidate.
    * - Instant Runoff

        :meth:`~pref_voting.iterative_methods.instant_runoff`
      - :meth:`~pref_voting.iterative_methods.instant_runoff_with_explanation`
      -  The order of elimination of the candidates
    * - Iterated Removal of Condorcet Losers

        :meth:`~pref_voting.iterative_methods.iterated_removal_cl`
      - :meth:`~pref_voting.iterative_methods.iterated_removal_cl_with_explanation`
      -  The order of elimination of the candidates
    * - Llull

        :meth:`~pref_voting.c1_methods.llull`
      - Profile method: :meth:`~pref_voting.profiles.Profile.copeland_scores`

        Majority/Margin Graph method: :meth:`~pref_voting.weighted_majority_graphs.MajorityGraph.copeland_scores`

        use score=(1, 0.5, 0)
      - The Llull score for each candidate.
    * - Minimax

        :meth:`~pref_voting.margin_based_methods.minimax`
      - :meth:`~pref_voting.margin_based_methods.minimax_scores`
      - The minimax score for each candidate
    * - Plurality

        :meth:`~pref_voting.scoring_methods.plurality`
      - Profile method: :meth:`~pref_voting.profiles.Profile.plurality_scores`
      - The Plurality score for each candidate
    * - Simplified Bucklin

        :meth:`~pref_voting.other_methods.simplified_bucklin`
      - :meth:`~pref_voting.other_methods.simplified_bucklin_with_explanation`
      -  The score for each candidate 
    * - Strict Nanson

        :meth:`~pref_voting.iterative_methods.strict_nanson`
      - :meth:`~pref_voting.iterative_methods.strict_nanson_with_explanation`
      - The order of elimination of the candidates and the Borda scores of the candidates  (in the profile restricted to candidates that have not been eliminated) 
    * - Weak Nanson

        :meth:`~pref_voting.iterative_methods.weak_nanson`
      - :meth:`~pref_voting.iterative_methods.weak_nanson_with_explanation`
      - The order of elimination of the candidates and the Borda scores of the candidates  (in the profile restricted to candidates that have not been eliminated) 

```

## Defeat Relations

```{eval-rst} 

.. list-table:: 
    :header-rows: 1
    :stub-columns: 1
    :align: left

    * - Function
      - Definition
    * - :meth:`~pref_voting.margin_based_methods.beat_path_defeat`
      - Returns a single networkx DiGraph. For candidates :math:`a` and :math:`b`, a **path** from :math:`a` to :math:`b` is a sequence :math:`x_1, \ldots, x_n` of distinct candidates  with  :math:`x_1=a` and :math:`x_n=b` such that for :math:`1\leq k\leq n-1`, :math:`x_k` is majority preferred to :math:`x_{k+1}`.  The **strength of a path** is the minimal margin along that path.  Candidate :math:`a` defeats :math:`b` according to Beat Path if the the strength of the strongest path from :math:`a` to :math:`b` is greater than the strength of the strongest path from :math:`b` to :math:`a`.
    * - :meth:`~pref_voting.other_methods.kemmeny_young_rankings`
      - Returns a list of lists of candidates.  A Kemmeny-Young ranking is a ranking that minimizes the sum of the Kendall tau distances to the voters’ rankings.
    * - :meth:`~pref_voting.margin_based_methods.ranked_pairs_defeats`
      - Returns a list of networkx DiGraphs. Order the edges in the margin graph from largest to smallest and lock them in in that order, skipping edges that create a cycle.  If there are ties in the margins, break the ties using a tiebreaking rule (i.e., a linear ordering over the edges).  Each tie-breaking rule generates a Ranked Pairs defeat relation where :math:`a` defeats :math:`b` if the edge from :math:`a` to :math:`b` in the margin graph is locked-in using the tiebreaking rule.  
    * - :meth:`~pref_voting.other_methods.slater_rankings`
      - Returns a list of lists of candidates.  A Slater ranking :math:`R` is a linear order  of the candidates that minimises the number of edges in the majority graph have to be turned around before to obtain :math:`R`.
    * - :meth:`~pref_voting.margin_based_methods.split_cycle_defeat`
      - Returns a single networkx DiGraph. A **majority cycle** is a sequence :math:`x_1, \ldots ,x_n` of distinct candidates with :math:`x_1=x_n` such that for :math:`1 \leq k \leq n-1`,  :math:`x_k` is majority preferred to :math:`x_{k+1}`. 1. In each cycle, identify the head-to-head win(s) with the smallest margin of victory in that cycle.  2. After completing step 1 for all cycles, discard the identified wins. All remaining wins count as defeats of the losing candidates.
    * - :meth:`~pref_voting.c1_methods.top_cycle_defeat`
      - Returns a single networkx DiGraph. Candidate :math:`a` defeats candidate :math:`b` when :math:`a` is in the Smith Set, but :math:`b` is not.
    * - :meth:`~pref_voting.c1_methods.uc_gill_defeat`
      - Returns a single networkx DiGraph. Given candidates :math:`a` and :math:`b`, say that :math:`a` defeats :math:`b` in the election if :math:`a` is majority preferred to :math:`b` and :math:`a` left covers :math:`b`: i.e., for all :math:`c`, if :math:`c` is majority preferred to :math:`a`,  then :math:`c` majority preferred to :math:`b`.      
    * - :meth:`~pref_voting.c1_methods.uc_fish_defeat`
      - Returns a single networkx DiGraph. Given candidates :math:`a` and :math:`b`, say that :math:`a` defeats :math:`b` in the election :math:`a` left covers :math:`b`: i.e., for all :math:`c`, if :math:`c` is majority preferred to :math:`a`,  then :math:`c` majority preferred to :math:`b`.     

```