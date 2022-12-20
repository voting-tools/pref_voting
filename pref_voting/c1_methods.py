'''
    File: c1_methods.py
    Author: Eric Pacuit (epacuit@umd.edu)
    Date: January 10, 2022
    Update: July 31, 2022
    
    Implementations of voting methods that work on both profiles and majority graphs.
'''

from pref_voting.voting_method import  *
from pref_voting.helper import get_mg, get_weak_mg
import copy
import math
from itertools import product, permutations
import networkx as nx
import matplotlib.pyplot as plt


@vm(name = "Condorcet")
def condorcet(edata, curr_cands = None):
    """
    Return the Condorcet winner if one exists, otherwise return all the candidates.  A Condorcet winner is a candidate :math:`c` that is majority preferred to every other candidate. 

    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `condorcet_winner` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :meth:`pref_voting.profiles.Profile.condorcet_winner`,  :meth:`pref_voting.profiles_with_ties.ProfileWithTies.condorcet_winner`, :meth:`pref_voting.weighted_majority_graphs.MajorityGraph.condorcet_winner`

    :Example: 

    .. exec_code:: 

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import condorcet
        
        prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [1, 1, 1])

        prof.display()
        print(prof.condorcet_winner())
        condorcet.display(prof)
        condorcet.display(prof.majority_graph())
        condorcet.display(prof.margin_graph())

        prof2 = Profile([[0, 1, 2], [2, 1, 0], [1, 0, 2]], [3, 1, 1])

        prof2.display()
        print(prof2.condorcet_winner())
        condorcet.display(prof2)
        condorcet.display(prof2.majority_graph())
        condorcet.display(prof2.margin_graph())

    """
   
    candidates = edata.candidates if curr_cands is None else curr_cands
    cond_winner = edata.condorcet_winner(curr_cands = curr_cands)
    
    return [cond_winner] if cond_winner is not None else sorted(candidates)


@vm(name = "Copeland")
def copeland(edata, curr_cands = None):
    """The Copeland score for c is the number of candidates that c is majority preferred to minus the number of candidates majority preferred to c.  The Copeland winners are the candidates with the maximum Copeland score in the profile restricted to ``curr_cands``. 

    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `copeland_scores` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :meth:`pref_voting.profiles.Profile.copeland_scores`,  :meth:`pref_voting.profiles_with_ties.ProfileWithTies.copeland_scores`, :meth:`pref_voting.weighted_majority_graphs.MajorityGraph.copeland_scores`


    :Example: 
        
    .. plot:: margin_graphs_examples/mg_ex_copeland_llull.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.c1_methods import copeland
        copeland.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import copeland
        
        prof = Profile([[1, 3, 0, 4, 2], [0, 1, 4, 2, 3], [2, 4, 0, 1, 3], [3, 0, 2, 4, 1],  [4, 3, 1, 0, 2], [2, 3, 0, 1, 4]], [1, 1, 1, 1, 1, 1])
        
        copeland.display(prof)
        print(prof.copeland_scores())


    """    
    c_scores = edata.copeland_scores(curr_cands = curr_cands)
    max_score = max(c_scores.values())
    return sorted([c for c in c_scores.keys() if c_scores[c] == max_score])


@vm(name = "Llull")
def llull(edata, curr_cands = None):
    """The Llull score for a candidate :math:`c` is the number of candidates that :math:`c` is weakly majority preferred to.  This is equivalent to calucating the Copeland scores for a candidate :math:`c` with 1 point for each candidate that :math:`c` is majority preferred to, 1/2 point for each candidate that :math:`c` is tied with, and 0 points for each candidate that is majority preferred to :math:`c`.  The Llull winners are the candidates with the maximum Llull score in the profile restricted to ``curr_cands``. 

    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `copeland_scores` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :meth:`pref_voting.profiles.Profile.copeland_scores`,  :meth:`pref_voting.profiles_with_ties.ProfileWithTies.copeland_scores`, :meth:`pref_voting.weighted_majority_graphs.MajorityGraph.copeland_scores`

    :Example: 
        
    .. plot::  margin_graphs_examples/mg_ex_copeland_llull.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.c1_methods import llull
        llull.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import llull
        
        prof = Profile([[1, 3, 0, 4, 2], [0, 1, 4, 2, 3], [2, 4, 0, 1, 3], [3, 0, 2, 4, 1],  [4, 3, 1, 0, 2], [2, 3, 0, 1, 4]], [1, 1, 1, 1, 1, 1])
        
        llull.display(prof)
        print(prof.copeland_scores(scores=(1, 0.5, 0)))

    """ 

    l_scores = edata.copeland_scores(curr_cands = curr_cands, scores = (1,1,0))
    max_score = max(l_scores.values())
    return sorted([c for c in l_scores.keys() if l_scores[c] == max_score])

def left_covers(dom, c1, c2):
    # left covers: c1 left covers c2 when all the candidates that are majority preferred to c1 are also majority preferred to c2.
    
    # weakly left covers: c1 weakly left covers c2 when all the candidates that are majority preferred to or tied with c1
    # are also majority preferred to or tied with c2.
    
    return dom[c1].issubset(dom[c2])

def right_covers(dom, c1, c2):
    # right covers: c1 right covers c2 when all the candidates that c2  majority preferrs are majority
    # preferred by c1
      
    return dom[c2].issubset(dom[c1])


@vm(name = "Uncovered Set")
def uc_gill(edata, curr_cands = None): 
    """Uncovered Set (Gillies version):  Given candidates :math:`a` and :math:`b`, say that :math:`a` defeats :math:`b` in the election if :math:`a` is majority preferred to :math:`b` and :math:`a` left covers :math:`b`: i.e., for all :math:`c`, if :math:`c` is majority preferred to :math:`a`,  then :math:`c` majority preferred to :math:`b`. The winners are the set of candidates who are undefeated in the election restricted to ``curr_cands``. 
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `dominators` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :func:`pref_voting.c1_methods.uc_fish`,  :func:`pref_voting.c1_methods.uc_bordes`, :func:`pref_voting.c1_methods.uc_mckelvey`

    :Example: 
        
    .. plot::  margin_graphs_examples/mg_ex_uncovered_sets.py
        :context: reset  
        :include-source: True

    .. code-block:: 

        from pref_voting.c1_methods import uc_gill
        uc_gill.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import uc_gill
        
        prof = Profile([[2, 3, 0, 1], [0, 2, 1, 3], [3, 0, 1, 2], [1, 2, 0, 3], [1, 2, 3, 0]], [1, 1, 1, 2, 1])
        
        uc_gill.display(prof)
    
    """

    candidates = edata.candidates if curr_cands is None else curr_cands
    dom = {c: set(edata.dominators(c, curr_cands = curr_cands)) for c in candidates}
    uc_set = list()
    for c1 in candidates:
        is_in_ucs = True
        for c2 in edata.dominators(c1, curr_cands = curr_cands): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))

def uc_gill_defeat(edata, curr_cands = None): 
    """Returns the defeat relation used to find the  Uncovered Set (Gillies version) winners.
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `dominators` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A networkx object in which there is an edge from :math:`a` to :math:`b` when :math:`a` to :math:`b` according to Top Cycle. 

    .. seealso::

        :func:`pref_voting.c1_methods.uc_gill`

    :Example: 
        
        
    .. plot::  margin_graphs_examples/uc_gill_defeat_example.py
        :include-source: True

    
    """
    
    defeat = nx.DiGraph()

    candidates = edata.candidates if curr_cands is None else curr_cands
    
    defeat.add_nodes_from(candidates)
    
    dom = {c: set(edata.dominators(c, curr_cands = curr_cands)) for c in candidates}
    for c1 in candidates:
        for c2 in edata.dominators(c1, curr_cands = curr_cands): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(dom, c2, c1):
                    defeat.add_edge(c2, c1)
    return defeat


@vm(name = "Uncovered Set - Fishburn")
def uc_fish(edata, curr_cands = None): 
    """Uncovered Set (Fishburn version):  Given candidates :math:`a` and :math:`b`, say that :math:`a` defeats :math:`b` in the election :math:`a` left covers :math:`b`: i.e., for all :math:`c`, if :math:`c` is majority preferred to :math:`a`,  then :math:`c` majority preferred to :math:`b`. The winners are the set of candidates who are undefeated in the election restricted to ``curr_cands``. 
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `dominators` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :func:`pref_voting.c1_methods.uc_gill`,  :func:`pref_voting.c1_methods.uc_bordes`, :func:`pref_voting.c1_methods.uc_mckelvey`

    :Example: 
        
        
    .. plot::  margin_graphs_examples/mg_ex_uncovered_sets.py
        :include-source: True


    .. code-block:: 

        from pref_voting.c1_methods import uc_fish
        uc_fish.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import uc_fish
        
        prof = Profile([[2, 3, 0, 1], [0, 2, 1, 3], [3, 0, 1, 2], [1, 2, 0, 3], [1, 2, 3, 0]], [1, 1, 1, 2, 1])
        
        uc_fish.display(prof)
    
    """
    candidates = edata.candidates if curr_cands is None else curr_cands
    dom = {c: set(edata.dominators(c, curr_cands = curr_cands)) for c in candidates}
    uc_set = list()
    for c1 in candidates:
        is_in_ucs = True
        for c2 in candidates:
            if c1 != c2:
                # check if c2 left covers  c1 but c1 does not left cover c2
                if left_covers(dom, c2, c1)  and not left_covers(dom, c1, c2):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))

def uc_fish_defeat(edata, curr_cands = None): 
    """Returns the defeat relation used to find the  Uncovered Set (Fishburn version) winners.
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `dominators` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A networkx object in which there is an edge from :math:`a` to :math:`b` when :math:`a` to :math:`b` according to Top Cycle. 

    .. seealso::

        :func:`pref_voting.c1_methods.uc_fish`


    :Example: 
        
        
    .. plot::  margin_graphs_examples/uc_fish_defeat_example.py
        :include-source: True

    """
    defeat = nx.DiGraph()

    candidates = edata.candidates if curr_cands is None else curr_cands
    
    defeat.add_nodes_from(candidates)
    dom = {c: set(edata.dominators(c, curr_cands = curr_cands)) for c in candidates}
    for c1 in candidates:
        is_in_ucs = True
        for c2 in candidates:
            if c1 != c2:
                # check if c2 left covers  c1 but c1 does not left cover c2
                if left_covers(dom, c2, c1)  and not left_covers(dom, c1, c2):
                    defeat.add_edge(c2, c1)
    return defeat


@vm(name = "Uncovered Set - Bordes")
def uc_bordes(edata, curr_cands = None): 
    """Uncovered Set (Bordes version):  Given candidates :math:`a` and :math:`b`, say that :math:`a` Bordes covers :math:`b` if :math:`a` is majority preferred to :math:`b` and for all :math:`c`, if :math:`c` is majority preferred or tied with :math:`a`, then :math:`c` is majority preferred to or tied with :math:`b`. The winners are the set of candidates who are not Bordes covered in the election restricted to ``curr_cands``. 
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has  `dominators` and `majority_prefers` methods. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :func:`pref_voting.c1_methods.uc_gill`,  :func:`pref_voting.c1_methods.uc_fish`, :func:`pref_voting.c1_methods.uc_mckelvey`

    :Example: 
        
        
    .. plot::  margin_graphs_examples/mg_ex_uncovered_sets.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.c1_methods import uc_bordes
        uc_bordes.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import uc_bordes
        
        prof = Profile([[2, 3, 0, 1], [0, 2, 1, 3], [3, 0, 1, 2], [1, 2, 0, 3], [1, 2, 3, 0]], [1, 1, 1, 2, 1])
        
        uc_bordes.display(prof)
    
    """

    candidates = edata.candidates if curr_cands is None else curr_cands

    dom = {c: set(edata.dominators(c, curr_cands = curr_cands)).union([_c for _c in candidates if (not edata.majority_prefers(c, _c) and not edata.majority_prefers(_c, c))]) for c in candidates}
    
    uc_set = list()
    for c1 in candidates:
        is_in_ucs = True
        for c2 in edata.dominators(c1, curr_cands = curr_cands): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))  

@vm(name = "Uncovered Set - McKelvey")
def uc_mckelvey(edata, curr_cands = None): 
    """Uncovered Set (McKelvey version):  Given candidates :math:`a` and :math:`b`, say that  :math:`a` McKelvey covers :math:`b` if a Gillies covers :math:`b` and :math:`a` Bordes covers :math:`b`. The winners are the set of candidates who are not McKelvey covered in the election restricted to ``curr_cands``. 
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has  `dominators` and `majority_prefers` methods. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        :func:`pref_voting.c1_methods.uc_gill`, :func:`pref_voting.c1_methods.uc_fish`, :func:`pref_voting.c1_methods.uc_bordes`

    :Example: 
        
        
    .. plot::  margin_graphs_examples/mg_ex_uncovered_sets.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.c1_methods import uc_mckelvey
        uc_bordes.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import uc_mckelvey
        
        prof = Profile([[2, 3, 0, 1], [0, 2, 1, 3], [3, 0, 1, 2], [1, 2, 0, 3], [1, 2, 3, 0]], [1, 1, 1, 2, 1])
        
        uc_mckelvey.display(prof)
    
    """
    candidates = edata.candidates if curr_cands is None else curr_cands

    strict_dom = {c: set(edata.dominators(c, curr_cands = curr_cands)) for c in candidates}    
    weak_dom = {c: strict_dom[c].union([_c for _c in candidates if (not edata.majority_prefers(c, _c) and not edata.majority_prefers(_c, c))]) for c in candidates}
    uc_set = list()
    for c1 in candidates:
        is_in_ucs = True
        for c2 in edata.dominators(c1, curr_cands = curr_cands): # consider only c2 predecessors
            if c1 != c2:
                # check if c2 left covers  c1 
                if left_covers(strict_dom, c2, c1) and left_covers(weak_dom, c2, c1):
                    is_in_ucs = False
        if is_in_ucs:
            uc_set.append(c1)
    return list(sorted(uc_set))      

@vm(name = "Top Cycle")
def top_cycle(edata, curr_cands = None):
    """The smallest set of candidates such that every candidate inside the set is majority preferred to every candidate outside the set.  
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `majority_prefers` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        Also known as ``getcha`` and ``smith_set``. 
        
        Related function includes :func:`pref_voting.c1_methods.gocha`

    :Example: 
        
        
    .. plot::  margin_graphs_examples/mg_ex_top_cycle_gocha.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.c1_methods import top_cycle, getcha, smith_set
        top_cycle.display(prof)
        getcha.display(prof)
        smith_set.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import top_cycle, getcha, smith_set
        
        prof = Profile([[1, 2, 0, 3], [1, 3, 0, 2], [3, 1, 0, 2], [0, 3, 1, 2]], [1, 1, 1, 1])
        
        top_cycle.display(prof)
        getcha.display(prof)
        smith_set.display(prof)

        
    """    
    wmg = get_weak_mg(edata, curr_cands = curr_cands)
    scc = list(nx.strongly_connected_components(wmg))
    min_indegree = min([max([wmg.in_degree(n) for n in comp]) for comp in scc])
    smith = [comp for comp in scc if max([wmg.in_degree(n) for n in comp]) == min_indegree][0]
    return sorted(list(smith))

# Create some aliases for Top Cycle
top_cycle.set_name("GETCHA")
getcha = copy.deepcopy(top_cycle)
top_cycle.set_name("Smith Set")
smith_set = copy.deepcopy(top_cycle)

# reset the name Top Cycle
top_cycle.set_name("Top Cycle")

def top_cycle_defeat(edata, curr_cands = None):
    """Return the defeat relation associated with the Top Cycle voting method. 
    
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `majority_prefers` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A networkx object in which there is an edge from :math:`a` to :math:`b` when :math:`a` to :math:`b` according to Top Cycle. 

    .. seealso::

        :func:`pref_voting.c1_methods.top_cycle`

    :Example: 
        
    .. plot::  margin_graphs_examples/top_cycle_defeat.py
        :context: reset  
        :include-source: True
 
    """    

    defeat = nx.DiGraph()
    candidates = edata.candidates if curr_cands is None else curr_cands
    smith_set = top_cycle(edata, curr_cands = candidates)
    
    defeat.add_nodes_from(candidates)
    defeat.add_edges_from([(a, b) for a in candidates for b in candidates if a != b and a in smith_set and b not in smith_set])
    return defeat


@vm(name = "GOCHA")
def gocha(edata, curr_cands = None):
    """The GOCHA set (also known as the Schwartz set) is the smallest set of candidates with the property
    that every candidate inside the set is not majority preferred by every candidate outside the set.
      
    Args:
        edata (Profile, ProfileWithTies, MajorityGraph, MarginGraph): Any election data that has a `majority_prefers` method. 
        curr_cands (List[int], optional): If set, then find the winners for the profile restricted to the candidates in ``curr_cands``

    Returns: 
        A sorted list of candidates

    .. seealso::

        Also known as ``schwartz_set``. 
        
        Related function includes :func:`pref_voting.c1_methods.top_cycle`

    :Example: 
          
    .. plot::  margin_graphs_examples/mg_ex_top_cycle_gocha.py
        :context: reset  
        :include-source: True


    .. code-block:: 

        from pref_voting.c1_methods import top_cycle, gocha, schwartz_set

        gocha.display(prof)
        schwartz_set.display(prof)


    .. exec_code:: 
        :hide_code:

        from pref_voting.profiles import Profile
        from pref_voting.c1_methods import gocha, schwartz_set
        
        prof = Profile([[1, 2, 0, 3], [1, 3, 0, 2], [3, 1, 0, 2], [0, 3, 1, 2]], [1, 1, 1, 1])
        
        gocha.display(prof)
        schwartz_set.display(prof)
    
    """    
    
    mg = get_mg(edata, curr_cands = curr_cands)
    transitive_closure =  nx.algorithms.dag.transitive_closure(mg)
    schwartz = set()
    for ssc in nx.strongly_connected_components(transitive_closure):
        if not any([transitive_closure.has_edge(c2,c1) 
                    for c1 in ssc for c2 in transitive_closure.nodes if c2 not in ssc]):
            schwartz =  schwartz.union(ssc)
    return sorted(list(schwartz))

# Create some aliases for GOCHA
gocha.set_name("Schwartz Set")
schwartz_set = copy.deepcopy(gocha)

# reset the name GETCHA
gocha.set_name("GOCHA")

c1_vms = [
    condorcet,
    copeland,
    llull,
    uc_gill,
    uc_fish,
    uc_bordes,
    uc_mckelvey,
    top_cycle,
    gocha,
]

defeat_methods = [
    top_cycle_defeat,
    uc_gill_defeat,
    uc_fish_defeat
]
