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

How to cite
------------------------

If you would like to acknowledge our work in a scientific paper,
please use the following citation:

Wesley H. Holliday and Eric Pacuit (2025). pref_voting: The Preferential Voting Tools package for Python. Journal of Open Source Software, 10(105), 7020. https://doi.org/10.21105/joss.07020

**Bibtex**:

.. code-block:: bibtex

   @article{HollidayPacuit2025, 
     author = {Wesley H. Holliday and Eric Pacuit}, 
     title = {pref_voting: The Preferential Voting Tools package for Python}, 
     journal = {Journal of Open Source Software},
     year = {2025}, 
     publisher = {The Open Journal}, 
     volume = {10}, 
     number = {105}, 
     pages = {7020}, 
     doi = {10.21105/joss.07020}
   }

Axiom Satisfaction/Violation Database
------------------------

We maintain a database showing whether various voting methods satisfy or violate various axioms: 

restructuredtext* `Dominance Axioms - Profile <axiom_violations_dominance_axioms_profile.html>`_
* `Dominance Axioms - Profile with Ties <axiom_violations_dominance_axioms_profilewithties.html>`_
* `Invariance Axioms - Profile <axiom_violations_invariance_axioms_profile.html>`_
* `Invariance Axioms - Profile with Ties <axiom_violations_invariance_axioms_profilewithties.html>`_
* `Monotonicity Axioms - Profile <axiom_violations_monotonicity_axioms_profile.html>`_
* `Monotonicity Axioms - Profile with Ties <axiom_violations_monotonicity_axioms_profilewithties.html>`_
* `Strategic Axioms - Profile <axiom_violations_strategic_axioms_profile.html>`_
* `Strategic Axioms - Profile with Ties <axiom_violations_strategic_axioms_profilewithties.html>`_
* `Variable Candidate Axioms - Profile <axiom_violations_variable_candidate_axioms_profile.html>`_
* `Variable Candidate Axioms - Profile with Ties <axiom_violations_variable_candidate_axioms_profilewithties.html>`_
* `Variable Voter Axioms - Profile <axiom_violations_variable_voter_axioms_profile.html>`_
* `Variable Voter Axioms - Profile with Ties <axiom_violations_variable_voter_axioms_profilewithties.html>`_


Related resources
------------------------

- VoteKit (https://votekit.readthedocs.io/) - A Python package developed by the MGGG Redistricting Lab (https://mggg.org/) designed to facilitate the study of different election methods.

- prefsampling (https://comsoc-community.github.io/prefsampling/) - A Python library for sampling from preference profiles with respect to different probability models.

- PrefLib (https://www.preflib.org/) - A database of election data. 

- Mapel (https://github.com/szufix/mapel) - Mapel (Map of Elections) is a Python package that can be used to simulate elections.  

- https://voting.ml/ - An online tool to study *maximal lotteries* (a Condorcet consistent probabilistic voting method).

- abcvoting (https://abcvoting.readthedocs.io/) - Python library of approval based committee voting rules.

- Votelib (https://github.com/simberaj/votelib) - Another Python package that implements a number of voting methods (includes multiwinner methods and some grading systems such as Approval Voting and Majority Judgement).


See [https://comsoc-community.org/tools](https://comsoc-community.org/tools) for an overivew of tools for computational social choice.

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
   pairwise_profiles
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
   :caption: Collective Decision Procedures

   collective_decision_procedures
   scoring_methods
   iterative_methods
   c1_methods
   margin_based_methods
   combined_methods
   other_methods
   proportional_methods
   probabilistic_methods
   stochastic_methods
   utility_methods
   grade_methods

.. toctree::
   :maxdepth: 2
   :caption: Axioms

   axioms_overview
   dominance_axioms
   invariance_axioms
   monotonicity_axioms
   strategic_axioms
   variable_voter_axioms
   variable_candidate_axioms
   swf_axioms

.. toctree::
   :maxdepth: 2
   :caption: Analysis

   analysis_overview
   

Index
----------------------

* :ref:`genindex`
