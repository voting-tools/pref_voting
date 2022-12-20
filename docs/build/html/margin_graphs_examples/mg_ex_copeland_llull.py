from pref_voting.profiles import Profile
            
prof = Profile([[1, 3, 0, 4, 2], [0, 1, 4, 2, 3], [2, 4, 0, 1, 3], [3, 0, 2, 4, 1],  [4, 3, 1, 0, 2], [2, 3, 0, 1, 4]], [1, 1, 1, 1, 1, 1])

prof.display_margin_graph()
