
from pref_voting.profiles import Profile
            
prof = Profile([[2, 3, 0, 1], [0, 2, 1, 3], [3, 0, 1, 2], [1, 2, 0, 3], [1, 2, 3, 0]], [1, 1, 1, 2, 1])

prof.display_margin_graph()
