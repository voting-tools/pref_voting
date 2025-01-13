
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.rankings import Ranking
from itertools import combinations, chain, permutations
import copy

def display_mg(edata): 
    if type(edata) == Profile or type(edata) == ProfileWithTies: 
        edata.display_margin_graph()
    else: 
        edata.display()

def list_to_string(cands, cmap): 
    return "{" + ', '.join([cmap[c] for c in cands]) + "}"


def swap_candidates(ranking, c1, c2):
    """
    Swap two candidates in a ranking.
    :param ranking: either a tuple or a list of candidates or a Ranking object
    :param c1: candidate 1
    :param c2: candidate 2
    :return: a new ranking (a tuple) with c1 and c2 swapped
    """

    if isinstance(ranking, Ranking):

        rmap = ranking.rmap

        if c1 not in rmap or c2 not in rmap:
            raise ValueError("One of the candidates is not in the ranking")
        
        # swap the values associated with c1 and c2
        new_rmap = rmap.copy()
        new_rmap[c1], new_rmap[c2] = new_rmap[c2], new_rmap[c1]
        new_ranking = Ranking(new_rmap)
        
    elif isinstance(ranking, (list, tuple)):
        new_ranking = []
        for c in ranking:
            if c == c1:
                new_ranking.append(c2)
            elif c == c2:
                new_ranking.append(c1)
            else:
                new_ranking.append(c)
        new_ranking = tuple(new_ranking)
    return new_ranking



def equal_size_partitions_with_duplicates(lst):
    """
    Generate all partitions of a list into two distinct subsets of equal size, 
    including cases where the input list contains duplicates and elements 
    that do not support ordering (<).
    
    Parameters:
        lst (list): The input list to partition. Must have an even number of elements.
        
    Returns:
        list of tuples: A list of tuples, where each tuple contains two lists of equal size.
    """
    if len(lst) % 2 != 0:
        raise ValueError("The input list must have an even number of elements.")
    
    n = len(lst) // 2
    partitions = []
    
    seen = set()
    
    for subset in combinations(lst, n):
        complement = lst[:]
        for item in subset:
            complement.remove(item)
        
        partition_key = frozenset([frozenset(subset), frozenset(complement)])
        if partition_key not in seen:
            seen.add(partition_key)
            partitions.append((list(subset), complement))
    
    return partitions


def get_rank(ranking, c):
    """
    Get the (normalized) rank of a candidate in a ranking.
    :param ranking: either a tuple or a list of candidates or a Ranking object
    :param c: candidate
    :return: the rank of c in the ranking
    """
    if isinstance(ranking, Ranking):
        norm_ranking = copy.deepcopy(ranking)
        norm_ranking.normalize_ranks()
        return norm_ranking.rmap[c]
    elif isinstance(ranking, (list, tuple)):
        return ranking.index(c)
    else:
        raise ValueError("Invalid input type")
    

# generate all subsets of a set, use combinations
def powerset(lst):
    s = list(lst)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def linear_orders_with_reverse(cands): 

    lin_orders = list(permutations(cands))
    lin_orders_with_reverse = []
    for lin_order in lin_orders:
        lin_orders_with_reverse.append((lin_order, lin_order[::-1]))
    return lin_orders_with_reverse

def remove_first_occurrences(rankings, r1, r2):
    removed_r1 = False
    removed_r2 = False
    result = []

    for r in rankings:
        if r == r1 and not removed_r1:
            removed_r1 = True  # Skip the first r1
        elif r == r2 and not removed_r2:
            removed_r2 = True  # Skip the first r2
        else:
            result.append(r)  # Keep all other elements

    return result
