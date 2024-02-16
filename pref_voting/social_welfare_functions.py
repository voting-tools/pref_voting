from pref_voting.utility_methods import sum_utilitarian_ranking, relative_utilitarian_ranking, maximin_ranking, lexicographic_maximin_ranking, nash_ranking, utilitarian_swfs

from pref_voting.scoring_methods import plurality_ranking, borda_ranking, scoring_swfs

# List of all social welfare functions
social_welfare_functions = utilitarian_swfs + scoring_swfs

