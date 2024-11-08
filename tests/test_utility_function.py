import pytest
import numpy as np
from pref_voting.mappings import Utility

@pytest.fixture
def utility_instance():
    utils = {1: 100, 2: 200, 3: 300}
    return Utility(utils)

# def test_init_conflicting_args():
#     utils = {1: 100, 2: 200}
#     with pytest.raises(ValueError):
#         Utility(utils, domain={1, 2}, candidates={1, 2})

# def test_init_correct_domain_derivation():
#     utils = {1: 100, 2: 200}
#     ut = Utility(utils)
#     assert ut.domain == [1, 2]

# def test_init_with_domain():
#     utils = {1: 100, 2: 200}
#     ut = Utility(utils, domain={1, 2, 3})
#     assert ut.domain == {1, 2, 3}

# def test_candidates(utility_instance):
#     assert utility_instance.candidates == [1, 2, 3]

# def test_items_with_util(utility_instance):
#     assert utility_instance.items_with_util(100) == [1]
#     assert utility_instance.items_with_util(500) == []

# def test_has_utility(utility_instance):
#     assert utility_instance.has_utility(1)
#     assert not utility_instance.has_utility(4)

# def test_remove_cand(utility_instance):
#     removed = utility_instance.remove_cand(2)
#     assert 2 not in removed.domain
#     assert removed.domain == [1, 3]

# def test_to_approval_ballot(utility_instance):
#     ballot = utility_instance.to_approval_ballot()
#     assert isinstance(ballot, Grade)
#     assert set(ballot.candidates) == {1, 2, 3}

# def test_to_k_approval_ballot(utility_instance):
#     with patch('numpy.random.rand', return_value=0.5):
#         ballot = utility_instance.to_k_approval_ballot(2)
#         assert len([x for x in ballot.candidates if ballot.cmap[x] == 1]) <= 2

# def test_ranking(utility_instance):
#     rank = utility_instance.ranking()
#     assert isinstance(rank, Ranking)
#     assert rank.is_linear(num_cands=3)

# def test_extended_ranking(utility_instance):
#     rank = utility_instance.extended_ranking()
#     assert isinstance(rank, Ranking)
#     assert rank.is_linear(num_cands=3)  # Adjust based on how extended_ranking is supposed to work

# def test_has_tie(utility_instance):
#     assert not utility_instance.has_tie()
#     assert not utility_instance.has_tie(use_extended=True)

# def test_is_linear(utility_instance):
#     assert utility_instance.is_linear(num_cands=3)

# def test_represents_ranking(utility_instance):
#     # This requires a proper ranking instance which matches the utility setup
#     ranking = utility_instance.ranking()  # Using the same utility instance's ranking for simplicity
#     assert utility_instance.represents_ranking(ranking)

# def test_transformation(utility_instance):
#     # Test a simple transformation, e.g., squaring the utility
#     transformed = utility_instance.transformation(lambda x: x ** 2)
#     assert transformed.val(1) == 10000
#     assert transformed.val(2) == 40000

# def test_linear_transformation(utility_instance):
#     transformed = utility_instance.linear_transformation(a=2, b=-100)
#     assert transformed.val(1) == 100
#     assert transformed.val(2) == 300

# def test_normalize_by_range(utility_instance):
#     normalized = utility_instance.normalize_by_range()
#     assert normalized.val(1) == 0
#     assert normalized.val(3) == 1

# def test_normalize_by_standard_score(utility_instance):
#     normalized = utility_instance.normalize_by_standard_score()
#     expected_scores = [
#         (utility_instance.val(x) - np.mean([100, 200, 300])) / np.std([100, 200, 300])
#         for x in utility_instance.candidates
#     ]
#     assert all(np.isclose(normalized.val(x), score) for x, score in zip(utility_instance.candidates, expected_scores))

# def test_expectation(utility_instance):
#     prob = {1: 0.5, 2: 0.3, 3: 0.2}
#     expected_util = utility_instance.expectation(prob)
#     assert expected_util == 0.5 * 100 + 0.3 * 200 + 0.2 *
