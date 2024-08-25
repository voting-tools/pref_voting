'''
    File: generate_margin_graphs.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: July 14, 2022
    Updated: December 19, 2022
    
    Functions to generate a margin graph
    
'''


import networkx as nx
from itertools import combinations
from helper import sublists, compositions, enumerate_compositions, convex_lexicographic_sublists
from pref_voting.weighted_majority_graphs import MarginGraph
import numpy as np
from scipy.stats import multivariate_normal

def generate_edge_ordered_tournament(num_cands, parity="even"): 
    """Generate a random uniquely weighted MarginGraph for ``num_cands`` candidates.  

    :param num_cands: the number of candidates
    :type num_cands: int
    :returns: a uniquely weighted margin graph
    :rtype: MarginGraph

    .. note:: This function randomly generates a tournament with a linear order over the edges.  A **tournament** is an asymmetric directed graph with an edge between every two nodes.  The linear order of the edges is represented by assigning to each edge a number  :math:`2, \ldots, 2*n`, where :math:`n` is the number of the edges. 
    """

    assert parity in ["even", "odd"], "The parity should be either 'even' or 'odd'."

    mg = nx.DiGraph()
    mg.add_nodes_from(range(num_cands))
    _edges = list()
    for c1 in range(num_cands): 
        for c2 in range(c1+1, num_cands): 
            if np.random.choice([True, False]): 
                _edges.append((c1, c2))
            else: 
                _edges.append((c2, c1))
                   
    edges = list()
    edge_indices = list(range(len(_edges)))
    np.random.shuffle(edge_indices)
    
    for i, e_idx in enumerate(edge_indices):
        edges.append((_edges[e_idx][0], _edges[e_idx][1], 2 * (i+1) if parity == 'even' else 2 * i+1)) 
    
    return MarginGraph(range(num_cands), edges)

def generate_margin_graph(num_cands, weight_domain = None, parity = 'even'): 
    """Generate a random MarginGraph (allowing for ties in the margins) for ``num_cands`` candidates.  

    Args:
        num_cands (int): the number of candidates

    Returns:
        MarginGraph

    """

    assert parity in ['even', 'odd'], "Parity must be 'even' or 'odd'."
    assert weight_domain is None or isinstance(weight_domain, list) and len(weight_domain) > 0, "The weight_domain must be a list with at least one element."
    
    candidates = list(range(num_cands))
    edges = list()
    pairs_of_cands = list(combinations(candidates, 2))

    if weight_domain is None and parity == 'even': 
        weight_domain = [2 * pidx for pidx in range(len(pairs_of_cands) + 1)]
    elif weight_domain is None and parity == 'odd': 
        weight_domain = [2 * pidx + 1 for pidx in range(len(pairs_of_cands) + 1)]
    # 

    for c1, c2 in pairs_of_cands:

        margin = np.random.choice(weight_domain)
        
        if margin != 0: 
            if np.random.choice([True, False]): 
                edges.append((c1, c2, margin))
            else:
                edges.append((c2, c1, margin))

    return MarginGraph(candidates, edges)

def generate_margin_graph_bradley_terry(num_cands, num_voters, score_prob_mod = lambda c: np.random.uniform(0,1)):
    """Generates a margin graph for num_cands candidates by first sampling candidate scores from score_prob_mod and then sampling votes from the Bradley-Terry model using the sampled scores.

    Args:
        num_cands (int): Number of candidates
        num_voters (int): Number of voters
        score_prob_mod (function, optional): A function that takes a candidate and returns a score. Defaults to lambda c: np.random.uniform(0,1).

    Returns:
        MarginGraph: A margin graph
    """

    candidates = list(range(num_cands))
    pairs_of_cands = list(combinations(candidates, 2))

    cand_score = dict()
    for c in candidates:
        cand_score[c] = score_prob_mod(c)

    edges = list()
    
    for c1, c2 in pairs_of_cands:

        support_c1_c2 = 0
        support_c2_c1 = 0

        for n in range(num_voters):
            vote = np.random.choice([1,0], p = [cand_score[c1] / (cand_score[c1] + cand_score[c2]), cand_score[c2] / (cand_score[c1] + cand_score[c2])])

            if vote == 1:
                support_c1_c2 += 1
            else:
                support_c2_c1 += 1
        
        if support_c1_c2 > support_c2_c1:
            edges.append((c1,c2, support_c1_c2 - support_c2_c1))

        if support_c2_c1 > support_c1_c2:
            edges.append((c2,c1, support_c2_c1 - support_c1_c2))

    return MarginGraph(candidates, edges)

### 

# Turn a code into a pair
def depair(pair_vector, k):
    return pair_vector[k]

# This function defines the i,jth entry of the covariance matrix
def entries(pair_vector, i,j):
    x = depair(pair_vector, i)
    y = depair(pair_vector, j)
    if x[0] == y[0] and x[1] == y[1]:
        return 1
    if x[1] == y[0]:
        return -1/3
    if x[1] == y[1]:
        return 1/3
    if x[0] == y[0]:
        return 1/3
    if x[0] == y[1]:
        return -1/3
    return 0

def generate_covariance_matrix(num_candidates):
    
    num_pairs = num_candidates *(num_candidates -1)//2

    # Store the vector mapping codes to pairs
    pair_vector = [0]*num_pairs

    # Populate the vector of pairs
    k=0
    for i in range(num_candidates):
        for j in range(i+1,num_candidates):
            pair_vector[k] = [i,j]
            k = k+1

    # Populate the covariance matrix
    cov = np.empty((num_pairs,num_pairs))
    for i in range(num_pairs):
        for j in range(num_pairs):
            cov[i,j] = entries(pair_vector, i,j)
            
    return cov


def generate_edge_ordered_tournament_infinite_limit(num_candidates, cov_matrix = None): 
    """
    Using the ideas from Section 9 of the paper 
    *An Analysis of Random Elections with Large Numbers of Voters* by Matthew Harrison-Trainor 
    (https://arxiv.org/abs/2009.02979) and the code provided at  
    https://github.com/MatthewHT/RandomMarginGraphs/, generate a qualitative margin graph for 
    ``num_candidates`` candidates.
    
    .. important:: 
        
        The weights of the generated margin graphs are real numbers, representing a linear ordering of the edges. 
        Only qualitative margin graph invariant voting methods, such as Split Cycle, Beat Path, Minimax, 
        Ranked Pairs, etc., should be used on the generated graphs. 
        
    Args:
        
        num_candidates (int): the number of candidates
        
    Returns: 
    
        MarginGraph
    
    """
    
    candidates = range(num_candidates)
    cov_matrix = cov_matrix if cov_matrix is not None else generate_covariance_matrix(num_candidates)
    # random_var is a random variable with the multivariate normal distribution of margin graphs
    random_var = multivariate_normal(None, cov_matrix)
    rv = random_var.rvs()
    
    def pair(p):
        return p[1]-2*p[0]-1 + (num_candidates)*(num_candidates+1)//2 - (num_candidates-p[0])*(num_candidates-p[0]+1)//2

    mg = [[-np.inf for _ in candidates] for _ in candidates]
    
    for c1 in candidates:
        for c2 in candidates:
            if (c1 < c2 and rv[pair([c1,c2])] > 0):
                mg[c1][c2] = rv[pair([c1,c2])]
            if (c1 > c2 and rv[pair([c2,c1])] < 0):
                mg[c1][c2] = -rv[pair([c2,c1])]
            if (c1 == c2):
                mg[c1][c2] = 0

    w_edges = [(c1, c2, mg[c1][c2]) 
               for c1 in candidates 
               for c2 in candidates if c1 != c2 if mg[c1][c2] > 0]
    
    return MarginGraph(candidates, w_edges)

## Generating Canonical MarginGraphs without Tied Margins

def _enumerate_ceots(num_cands, num_edges, partial_ceot, used_nodes, next_node):

    # Given a partial ceot P, we can extend it with any new edge (A,B) satisfying one of the following conditions:

    # 1. A and B have both already been used in edges in P, but neither (A,B) nor (B,A) is in P;
    # 2. A has already been used in an edge in P, and B is the next integer after the largest integer in an edge in P.
    # 3. A is the next integer after the largest integer in an edge in P, and B has already been used in an edge in P;
    # 4. A is the next integer after the largest integer in an edge in P, and B is the next integer after A.

    if len(partial_ceot) == num_edges:
        yield partial_ceot

    if len(partial_ceot) < num_edges:

        if next_node == num_cands:
            available_nodes = used_nodes

        if next_node < num_cands:
            available_nodes = used_nodes + [next_node]

        for n in available_nodes:

            if n == next_node and next_node < num_cands - 1: # If n == next_node, we are in Case 3 or Case 4 above
                
                available_nodes = used_nodes + [next_node + 1]

            for m in available_nodes:
                
                if not n==m and not (n,m) in partial_ceot and not (m,n) in partial_ceot:
                
                    new_ceot = [edge for edge in partial_ceot] + [(n,m)]

                    if not (n == next_node or m == next_node): # Then we are in Case 1 above

                        yield from _enumerate_ceots(num_cands,num_edges,new_ceot,used_nodes,next_node)

                    if (n == next_node or m == next_node) and not m == next_node + 1: # Then we are in Case 2 or 3 above

                        new_used_nodes = list(set(used_nodes + [n,m]))
                        new_next_node = next_node + 1
                        
                        yield from _enumerate_ceots(num_cands,num_edges,new_ceot,new_used_nodes,new_next_node)

                    if m == next_node + 1: # Then we are in Case 4 above

                        new_used_nodes = list(set(used_nodes + [n,m]))
                        new_next_node = next_node + 2
                        
                        yield from _enumerate_ceots(num_cands,num_edges,new_ceot,new_used_nodes,new_next_node)

def _enumerate_ceots_as_edgelist(num_cands):

    num_edges = (num_cands * (num_cands -1))//2

    partial_ceot = [(0,1)]

    used_nodes = [0,1]

    next_node = 2

    yield from _enumerate_ceots(num_cands,num_edges,partial_ceot,used_nodes,next_node)

    
def enumerate_canonical_edge_ordered_tournaments(num_cands, parity = "even"):  
    """
    A *canonical* edge-ordered tournament (ceot) is a representative from an isomorphism class of  
    linearly edge-ordered tournaments.  Enumerate all ceots for ``num_cands`` candidates, representing 
    a ceot as a ``MarginGraph`` where the margins represent the linear order of the edges.  
    
    Args:
        num_cands (int): the number of candidates
        parity (str, optional): The parity of the margins, either 'even' or 'odd'.

    Returns:
        A generator of ``MarginGraph`` for ``num_candidates``
        
    .. warning:: It is only feasible to finish the enumeration for up to 5 candidates.    

    """
    
    assert parity in ["odd", "even"], "parity must be either 'odd' or 'even'"
    
    for ceot in _enumerate_ceots_as_edgelist(num_cands): 
        yield MarginGraph(list(range(num_cands)), 
                          [(e[0], e[1], 2 * (eidx + 1) if parity == "even" else 2 * eidx + 1) 
                           for eidx, e in enumerate(reversed(ceot))])

            
def enumerate_uniquely_weighted_margin_graphs(num_cands, weight_domain):  
    """
    Enumerate all representatives from isomorphism classes of uniquely-weighted margin graphs with weights drawn from ``weight_domain``. 
    
    Args:
        num_cands (int): the number of candidates
        weight_domain (List[int]): The list of weights in the margin graph.
        
    Returns:
        A generator of ``MarginGraph`` for ``num_candidates``
        

    .. warning:: It is only feasible to finish the enumeration for up to 5 candidates.    

    """
    
    weight_domain = sorted(weight_domain)
    
    num_edges = (num_cands * (num_cands - 1)) // 2
    
    for ceot in _enumerate_ceots_as_edgelist(num_cands): 
        
        for weight_list in sublists(weight_domain, num_edges): 
            yield MarginGraph(list(range(num_cands)), 
                              [(e[0], e[1], weight_list[eidx]) for eidx, e in enumerate(reversed(ceot))])


## Generating Canonical MarginGraphs with Tied Margins

def _enumerate_cweots_as_edgelist(num_cands, include_weak_tournaments=True):
    
    #Enumerate each canonical weakly edge ordered tournament as a list of lists of tied edges.
    #If include_weak_tournaments = True, then allow weak tournaments in which two nodes may have no edge between them.   
    
    def edge_match(e1, e2):
        return e1['weight'] == e2['weight']

    cweots = dict() # For isomorphism checking, keep track of the cweots generated so far.
    
    if include_weak_tournaments:
        cweots_with_absent_edges = dict() # For isomorphism checking, keep track of the cweots for which some edges are absent.

    for ceot in tqdm(list(_enumerate_ceots_as_edgelist(num_cands))):

        # The sorted list of number of wins by each candidate will be a useful invariant for isomorphism checking below.
        win_vector = tuple(sorted([len([edge for edge in ceot if edge[0] == i]) for i in range(num_cands)]))
        
        # Given ceot, we will generate many cweots as follows:

        # 1. Collect all the convex lexicographic sublists of ceot in order as [L1,...,Ln]. 
        # It suffices to only consider convex lexicographic sublists because for any cweot, 
        # we can obtain a ceot by breaking all ties between edges in a tied group lexicographically.

        l_sublists = convex_lexicographic_sublists(ceot)

        # 2. Within each L_i, we want to consider all ways of making consecutive edges tied.
        # Such a way is given by a composition of the integer len(L_i). 
        # Thus, we first iterate over all compositions of len(L_i),...,len(L_n).

        int_list = [len(s) for s in l_sublists]

        # 3. Since the above approach overgenerates cweots, we will check for isomorphism before adding a cweot to our list.

        for compositions in enumerate_compositions(int_list):

            cases = [False, True] if include_weak_tournaments else [False]

            for consider_weak_tourns in cases:
                
                cweot = []

                for idx, s in enumerate(l_sublists):
                    
                    composition = compositions[idx]
    
                    for n in composition:
                        cweot.append(s[:n])
                        s=s[n:]

                # If we are considering weak tournaments in this case, we remove the last tied group of edges and compute the sorted win-loss vector.
                if consider_weak_tourns:
                    win_loss_vector = tuple(sorted([(len([edge for edge in ceot if edge[0] == i and edge not in cweot[-1]]),len([edge for edge in ceot if edge[1] == i and edge not in cweot[-1]])) for i in range(num_cands)]))
                    cweot = cweot[:-1]
                    
                G = nx.DiGraph()
                weight = len(ceot)
                for group in cweot:
                    for edge in group:
                        G.add_edge(edge[0], edge[1], weight=weight)
                    weight = weight-1

                add_graph = True
                
                # Next we check whether G is isomorphic to a cweot G2 already generated.
                # We only need to check those cweots G2 that have (i) the same sorted Copeland scores and 
                # (ii) the same list of numbers of edges in each tied group as G, 
                # since these are necessary conditions for isomorphism.

                tie_sizes = tuple([len(s) for s in cweot])

                if not consider_weak_tourns:

                    invariant = (win_vector, tie_sizes)
                    
                    if invariant not in cweots.keys():
                        cweots[invariant] = []
              
                    for idx, G2 in enumerate(cweots[invariant]):
                        if nx.is_isomorphic(G, G2, edge_match=edge_match):  
                            add_graph = False
                            break 

                    if add_graph:   
                        cweots[invariant].append(G)    
                        yield cweot

                if consider_weak_tourns:

                    invariant = (win_loss_vector, tie_sizes)

                    if invariant not in cweots_with_absent_edges.keys():
                        cweots_with_absent_edges[invariant] = []

                    for idx, G2 in enumerate(cweots_with_absent_edges[invariant]):
                        if nx.is_isomorphic(G, G2, edge_match=edge_match):  
                            add_graph = False
                            break

                    if add_graph:
                        cweots_with_absent_edges[invariant].append(G)
                        yield cweot

def enumerate_canonical_weakly_edge_ordered_tournaments(num_cands, parity = "even", include_weak_tournaments = True):  
    """
    A *canonical* weakly edge-ordered tournament (cweot) is a representative from an isomorphism class of  
    weakly edge-ordered tournaments.  Enumerate all cweots for ``num_cands`` candidates, representing 
    a cweot as a ``MarginGraph`` where the margins represent the order of the edges.

    If include_weak_tournaments = True, then allow weak tournaments in which two nodes may have no edge between them.
    
    Args:
        num_cands (int): the number of candidates
        parity (str, optional): The parity of the margins, either 'even' or 'odd'.

    Returns:
        A generator of ``MarginGraph`` for ``num_candidates``
        
    .. warning:: It is only feasible to finish the enumeration for up to 4 candidates.    

    """
    
    assert parity in ["odd", "even"], "parity must be either 'odd' or 'even'"
    
    for cweot in _enumerate_cweots_as_edgelist(num_cands, include_weak_tournaments=include_weak_tournaments): 

        weighted_edges = list()

        for idx, group in enumerate(reversed(cweot)):
            for e in group:
                weighted_edge = (e[0], e[1], 2 * (idx + 1) if parity == "even" else 2 * idx + 1) 
                weighted_edges.append(weighted_edge)

        yield MarginGraph(list(range(num_cands)), weighted_edges)

def enumerate_margin_graphs(num_cands, weight_domain, include_weak_tournaments = True):  
    """
    Enumerate all representatives from isomorphism classes of margin graphs with weights drawn from ``weight_domain``. 
    
    Args:
        num_cands (int): the number of candidates
        weight_domain (List[int]): The list of weights in the margin graph.
        
    Returns:
        A generator of ``MarginGraph`` for ``num_candidates``
        
    .. warning:: It is only feasible to finish the enumeration for up to 4 candidates.    

    """
    
    weight_domain = sorted(weight_domain)
    
    for cweot in _enumerate_cweots_as_edgelist(num_cands, include_weak_tournaments = include_weak_tournaments): 

        for weight_list in sublists(weight_domain, len(cweot)):

            weighted_edges = list()

            for idx, group in enumerate(reversed(cweot)):
                for e in group:
                    weighted_edge = (e[0], e[1], weight_list[idx]) 
                    weighted_edges.append(weighted_edge)
            
            yield MarginGraph(list(range(num_cands)), weighted_edges)