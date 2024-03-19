.. pref_voting documentation master file, created by
   sphinx-quickstart on Fri Jul  8 15:41:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
=======================================

Preferential Voting Tools (``pref_voting``) is a Python library that can be used to study and run elections with different preferential voting methods (graded voting methods and cardinal voting methods are also included for comparison).  In a preferential voting election, each voter submits a *ranking* of the candidates, and the winners are determined based on the submitted rankings.  The rankings may include ties between candidates, and some candidates may be left off the ranking. 
 
The main objective is to create a set of tools that can be used by researchers to study voting methods, teachers to present topics in voting theory, and election administrators to run elections. Use the following website to run an election using the preferential voting method Stable Voting: https://stablevoting.org/ 

The library is developed by Wes Holliday (http://wesholliday.net) and Eric Pacuit (https://pacuit.org). 

**Survey articles about voting methods** 

- E. Pacuit (2019). `Voting methods <https://plato.stanford.edu/entries/voting-methods/>`_, Stanford Encyclopedia of Philosophy. 

- W. Zwicker (2016). `Introduction to the theory of voting <https://www.cambridge.org/core/books/abs/handbook-of-computational-social-choice/introduction-to-the-theory-of-voting/7C7A70249A972A4AC56E8938AD27464E>`_, Handbook of Computational Social Choice.

Related resources
------------------------

- abcvoting (https://abcvoting.readthedocs.io/) - Python library of approval based committee voting rules.

- prefsampling (https://comsoc-community.github.io/prefsampling/) - A Python library for sampling from preference profiles with respect to different probability models.

- PrefLib (https://www.preflib.org/) - A database of election data. 

- Mapel (https://github.com/szufix/mapel) - Mapel (Map of Elections) is a Python package that can be used to simulate elections.  

- https://voting.ml/ - An online tool to study *maximal lotteries* (a Condorcet consistent probabilistic voting method).

- Votelib (https://github.com/simberaj/votelib) - Another Python package that implements a number of voting methods (includes multiwinner methods and some grading systems such as Approval Voting and Majority Judgement).

Contents
-----------------

.. toctree::
   :maxdepth: 2

   self
   installation

.. toctree::
   :maxdepth: 2
   :caption: Elections

   edata_overview
   ballots
   profiles
   profiles_with_ties
   grade_profiles
   utility_profiles
   spatial_profiles
   weighted_majority_graphs
   io
   
.. toctree::
   :maxdepth: 2
   :caption: Generating Elections

   generate_profiles
   generate_weighted_majority_graphs
   generate_utility_profiles
   generate_spatial_profiles

.. toctree::
   :maxdepth: 2
   :caption: Aggregation Methods

   aggregation_methods_overview
   scoring_methods
   iterative_methods
   c1_methods
   margin_based_methods
   combined_methods
   other_methods
   probabilistic_methods
   utility_methods
   grade_methods
   

.. toctree::
   :maxdepth: 2
   :caption: Axioms

   axioms_overview
   dominance_axioms
   monotonicity_axioms
   variable_voter_axioms
   variable_candidate_axioms

.. toctree::
   :maxdepth: 2
   :caption: Analysis

   analysis_overview
   

Index
----------------------

* :ref:`genindex`
