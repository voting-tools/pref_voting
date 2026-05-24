from pref_voting.stochastic_methods import RGCR
from pref_voting.grade_profiles import GradeProfile
import networkx as nx
import pytest
import numpy as np
import logging

logging.getLogger("RGCR").setLevel(logging.NOTSET)
logging.getLogger("test").setLevel(logging.INFO)

def random_ordinal_ranking(gprofile:GradeProfile, curr_cands=None):
	x = gprofile.to_ranking_profile().majority_graph().to_networkx()
	return list(nx.topological_sort(x))

def mean_estimator(gprofile:GradeProfile, curr_cands=None):
	gprofile = GradeProfile([g.mapping for g in gprofile._grades], gprofile.grades.tolist(), candidates=gprofile.candidates)
	if curr_cands is None:
		curr_cands = gprofile.candidates
	return sorted(curr_cands, key=lambda c: gprofile.avg(c) if gprofile.has_grade(c) else 0, reverse=True)

def median_estimator(gprofile:GradeProfile, curr_cands=None):
	if curr_cands is None:
		curr_cands = gprofile.candidates
	return sorted(curr_cands, key=lambda c: gprofile.median(c) if gprofile.has_grade(c) else 0, reverse=True)


logger = logging.getLogger("test")

def is_topological_order(profile, ranking):
	G = profile.to_ranking_profile().majority_graph().to_networkx()
	if len(ranking) != len(G.nodes) or set(ranking) != set(G.nodes):
		logger.error("is_top_ord: Ranking does not contain the same candidates as the graph. len(ranking)=%g, len(G.nodes)=%g", len(ranking), len(G.nodes))
		logger.error("is_top_ord: Ranking candidates: %s, all candidates: %s", ranking, list(G.nodes))
		return False
		
	# שמירת המיקום של כל צומת ברשימה
	index_map = {node: i for i, node in enumerate(ranking)}
	
	# בדיקה שעבור כל קשת, צומת המקור מופיע לפני צומת היעד
	for u, v in G.edges():
		if index_map[u] > index_map[v]:
			logger.error("is_top_ord: Edge (%s, %s) violates the topological order.", u, v)
			return False
	logger.info("is_top_ord: Ranking is a valid topological order.")
	return True

def create_random_legal_gprofile(size=5, num_voters=10, rev_prob=0.3): # creates a random gprofile which is legal (i.e. does not contain cycles). the 'true' order is 0 < 1 < ... < size-1.
	candidates = list(range(size))
	voters = []
	for _ in range(num_voters):
		voter = {}
		val = 0
		for c in candidates:
			if np.random.rand() < rev_prob:
				voter[c] = val + np.random.randint(0, 10)+1
				val = voter[c] # ensure that the scores are non-decreasing
		voters.append(voter)
	logger.debug("Created random legal graph with %s", voters)
	return GradeProfile(voters, np.arange(0, 10*size+1, 1), candidates=candidates)


def test_topological_order():
	for i in range(1,100,10):
		n = np.random.rand() * 5*i
		gprofile = create_random_legal_gprofile(size=i, num_voters=int(n))
		logger.info("Test topological order with %g candidates and %g voters", i, int(n))
		ranking = RGCR(gprofile)
		assert is_topological_order(gprofile, ranking)

@pytest.mark.parametrize("profile, expected_sol, expected_prob", [
	(GradeProfile([{1: 7}, {2: 3}], range(0, 10), candidates=[1, 2]), [1, 2], 0.9),
	(GradeProfile([{1: 4, 2: 8}, {2: 6, 3: 2}], range(0, 10), candidates=[1, 2, 3]), [2, 1, 3], 5/6),
	(GradeProfile([{1: 3, 2: 4, 5: 5}, {1: 5, 3: 6}, {4: 2, 5: 10}, {2: 7, 6: 10}], range(0, 11), candidates=[1,2,3,4,5,6]), [6,5,4,3,2,1], 7/198)
])
def test_probability(profile, expected_sol, expected_prob): #test approximation to the probability, only for small inputs.
	prob = 0
	trials = 10000
	for _ in range(trials):
		solution = RGCR(profile)
		if solution == expected_sol:
			prob += 1
	assert abs(prob/trials - expected_prob) < 0.05

@pytest.mark.parametrize("w, expected_prob", [
	(lambda x: x/(1+x), 5/6), #the default w
	(lambda x: 3*x/(1+3*x), 13/14),
	(lambda x: 0.1*x/(1+0.1*x), 7/12)
])
def test_probability_with_diff_w(w, expected_prob):
	profile = GradeProfile([{1: 4, 2: 8}, {2: 6, 3: 2}], np.arange(0, 10, 1), candidates=[1, 2, 3])
	expected_sol = [2,1,3]
	prob = 0
	trials = 10000
	for _ in range(trials):
		if RGCR(profile, w = w) == expected_sol:
			prob += 1
	assert abs(prob/trials - expected_prob) < 0.05

@pytest.mark.parametrize("gprofile", [
	(GradeProfile([{1: 4, 2: 8}, {2: 6, 1: 7}], range(0, 10), candidates=[1, 2, 3])), # cycle
	(GradeProfile([{1: 3, 2: 4, 5: 5}, {2: 6, 3: 7}, {3: 2, 1: 5}], range(0, 11), candidates=[1,2,3,4,5])) # cycle
])
def test_illegal_input(gprofile):
	with pytest.raises(ValueError):
		RGCR(gprofile)

@pytest.mark.parametrize("w", [lambda x: x, lambda x: x**2, lambda x: np.sqrt(x), lambda x: 1-x/(1+x)]) #w must be increasing and return value in [0,1].
def test_illegal_w(w):
	with pytest.raises(ValueError):
		RGCR(GradeProfile([{1: 7}, {2: 3}, {3: 5}, {4: 3}], range(0, 10), candidates=[1, 2, 3, 4]), w=w)

@pytest.mark.parametrize("estimator", [random_ordinal_ranking, mean_estimator, median_estimator])
def test_strict_uniform_dominance(estimator):
	rgcr_success = 0
	another_estimator_success = 0
	trials = 1000
	for i in range(1, trials):
		voters = 10
		items = np.random.randint(voters, voters*3)
		gprofile = create_random_legal_gprofile(size=items, num_voters=voters)
		rgcr_ranking = RGCR(gprofile)
		another_ranking = estimator(gprofile)
		if rgcr_ranking == list(range(items)).reverse(): # the true order is always 0 < 1 < ...
			logger.info("RGCR found the true order in trial %g", i)
			rgcr_success += 1
		if another_ranking == list(range(items)).reverse():
			logger.info("Another estimator found the true order in trial %g", i)
			another_estimator_success += 1
	assert rgcr_success > another_estimator_success