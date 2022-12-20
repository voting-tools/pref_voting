
from pref_voting.weighted_majority_graphs import MarginGraph
 
mg = MarginGraph([0, 1, 2, 3], [(0, 1, 10), (0, 2, 2), (1, 3, 4), (2, 1, 6), (2, 3, 8), (3, 0, 4)])

mg.display()

