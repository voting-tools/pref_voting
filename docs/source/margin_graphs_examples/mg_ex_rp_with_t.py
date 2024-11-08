
from pref_voting.weighted_majority_graphs import MarginGraph

mg = MarginGraph([0, 1, 2, 3], [(1, 2, 2), (1, 3, 2), (2, 0, 2)])

mg.display()