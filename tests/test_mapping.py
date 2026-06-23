import pytest
import numpy as np
from pref_voting.mappings import _Mapping, Utility, Grade
from pref_voting.rankings import Ranking
# Assume needed imports and setups for _Mapping are done here.

@pytest.fixture
def simple_mapping():
    return _Mapping({1: 100, 2: 200, 3: 300})

def test_initialization():
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2}, codomain={10, 20})
    assert mapping.domain == {1, 2}
    assert mapping.codomain == {10, 20}
    assert mapping.val(1) == 10

def test_val(simple_mapping):
    assert simple_mapping.val(1) == 100

    with pytest.raises(AssertionError) as excinfo:
        simple_mapping.val(4)
    assert "4 not in the domain [1, 2, 3]" in str(excinfo.value) 

def test_has_value(simple_mapping):
    assert simple_mapping.has_value(1)
    assert not simple_mapping.has_value(4)

def test_defined_domain(simple_mapping):
    assert sorted(simple_mapping.defined_domain) == [1, 2, 3]
    mapping2 = _Mapping({1: 10, 2: 20}, domain=[0, 1, 2, 3])
    assert sorted(mapping2.defined_domain) == [1, 2]

def test_inverse_image(simple_mapping):
    assert simple_mapping.inverse_image(100) == [1]
    assert simple_mapping.inverse_image(400) == []

def test_image(simple_mapping):
    assert simple_mapping.image() == [100, 200, 300]
    assert simple_mapping.image([1, 3]) == [100, 300]

def test_range(simple_mapping):
    assert simple_mapping.range == [100, 200, 300]

def test_average(simple_mapping):
    assert simple_mapping.average() == np.mean([100, 200, 300])

def test_median(simple_mapping):
    assert simple_mapping.median() == np.median([100, 200, 300])

def test_compare(simple_mapping):
    assert simple_mapping.compare(1, 2) == -1
    assert simple_mapping.compare(2, 2) == 0
    assert simple_mapping.compare(2, 1) == 1
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert mapping.compare(1, 2) == -1
    assert mapping.compare(2, 2) == 0
    assert mapping.compare(2, 1) == 1
    assert mapping.compare(3, 1) is None
    assert mapping.compare(1, 3) is None
    assert mapping.compare(3, 4) is None

def test_extended_compare():
    
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert mapping.extended_compare(1, 2) == -1
    assert mapping.extended_compare(2, 2) == 0
    assert mapping.extended_compare(2, 1) == 1
    assert mapping.extended_compare(3, 1) == -1
    assert mapping.extended_compare(1, 3) == 1
    assert mapping.extended_compare(3, 4) == 0

def test_strict_pref(simple_mapping):
    assert simple_mapping.strict_pref(3, 1)
    assert not simple_mapping.strict_pref(1, 3)
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.strict_pref(1, 2) 
    assert not mapping.strict_pref(2, 2) 
    assert mapping.strict_pref(2, 1)
    assert not mapping.strict_pref(3, 1) 
    assert not mapping.strict_pref(1, 3) 
    assert not mapping.strict_pref(3, 4) 

def test_extended_strict_pref():
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.extended_strict_pref(1, 2) 
    assert not mapping.extended_strict_pref(2, 2) 
    assert mapping.extended_strict_pref(2, 1)
    assert not mapping.extended_strict_pref(3, 1) 
    assert mapping.extended_strict_pref(1, 3) 
    assert not mapping.extended_strict_pref(3, 4) 

def test_indiff(simple_mapping):
    assert not simple_mapping.indiff(1, 2)
    assert simple_mapping.indiff(2, 2)

    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.indiff(1, 2) 
    assert mapping.indiff(2, 2) 
    assert not mapping.indiff(3, 1) 
    assert not mapping.indiff(1, 3) 
    assert not mapping.indiff(3, 4) 

def test_extended_indiff():
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.extended_indiff(1, 2) 
    assert mapping.extended_indiff(2, 2) 
    assert not mapping.extended_indiff(3, 1) 
    assert not mapping.extended_indiff(1, 3) 
    assert mapping.extended_indiff(3, 4) 

def test_weak_pref(simple_mapping):
    assert simple_mapping.weak_pref(3, 1)
    assert not simple_mapping.weak_pref(1, 3)
    assert simple_mapping.weak_pref(3, 3)
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.weak_pref(1, 2) 
    assert  mapping.weak_pref(2, 2) 
    assert mapping.weak_pref(2, 1)
    assert not mapping.weak_pref(3, 1) 
    assert not mapping.weak_pref(1, 3) 
    assert not mapping.weak_pref(3, 4) 

def test_extended_weak_pref():
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.extended_weak_pref(1, 2) 
    assert  mapping.extended_weak_pref(2, 2) 
    assert mapping.extended_weak_pref(2, 1)
    assert not mapping.extended_weak_pref(3, 1) 
    assert  mapping.extended_weak_pref(1, 3) 
    assert  mapping.extended_weak_pref(3, 4) 


def test_indifference_class(simple_mapping):
    assert simple_mapping._indifference_classes(simple_mapping.domain) == [[1], [2], [3]]
    assert simple_mapping._indifference_classes([2, 3]) == [[2], [3]]
    mapping = _Mapping({1: 10, 2: 10}, domain={1, 2, 3, 4})
    assert mapping._indifference_classes([1, 2]) == [[1, 2]]
    assert mapping._indifference_classes([1, 3]) == [[1]]
    assert mapping._indifference_classes([3, 4]) == []

    assert mapping._indifference_classes([1, 2, 3, 4], use_extended=True) == [[1, 2], [3, 4]]

    assert mapping._indifference_classes([1, 2], use_extended=True) == [[1, 2]]
    assert mapping._indifference_classes([1, 3], use_extended=True) == [[1], [3]]
    assert mapping._indifference_classes([3, 4], use_extended=True) == [[3, 4]]

def test_sorted_domain(simple_mapping):
    assert simple_mapping.sorted_domain(simple_mapping.domain) == [[3], [2], [1]]
    mapping = _Mapping({1: 10, 2: 10}, domain={1, 2, 3, 4})
    assert mapping.sorted_domain() == [[1, 2]]
    assert mapping.sorted_domain(extended=True) == [[1, 2], [3, 4]]

def test_as_dict(simple_mapping):
    assert simple_mapping.as_dict() == {1: 100, 2: 200, 3: 300}

def test_display_str(simple_mapping):
    
    assert simple_mapping.display_str("F") == "F(1) = 100, F(2) = 200, F(3) = 300"

def test_call(simple_mapping):
    assert simple_mapping(1) == 100

def test_repr(simple_mapping):
    assert repr(simple_mapping) == "{1: 100, 2: 200, 3: 300}"

def test_str(simple_mapping):

    assert str(simple_mapping) == "1:100, 2:200, 3:300"


# ===========================================================================
#  Utility (subclass of _Mapping)
# ===========================================================================

@pytest.fixture
def util():
    return Utility({0: 5, 1: 3, 2: 1})

def test_utility_basics(util):
    assert util.candidates == [0, 1, 2]
    assert util.has_utility(0) is True
    assert util.has_utility(9) is False
    assert util.items_with_util(5) == [0]

def test_utility_domain_and_candidates_are_mutually_exclusive():
    with pytest.raises(ValueError):
        Utility({0: 1}, domain=[0], candidates=[0])

def test_utility_candidates_kwarg_sets_domain():
    u = Utility({0: 1}, candidates=[0, 1])
    assert u.candidates == [0, 1]

def test_utility_ranking(util):
    assert util.ranking().rmap == {0: 1, 1: 2, 2: 3}
    assert util.has_tie() is False
    assert util.is_linear(3) is True

def test_utility_ranking_with_tie():
    ut = Utility({0: 5, 1: 5, 2: 1})
    assert ut.ranking().rmap == {0: 1, 1: 1, 2: 2}
    assert ut.has_tie() is True
    assert ut.is_linear(3) is False

def test_utility_extended_ranking_places_unrated_last():
    u = Utility({0: 5, 1: 3}, domain=[0, 1, 2])
    assert u.extended_ranking().rmap == {0: 1, 1: 2, 2: 3}

def test_utility_remove_cand(util):
    assert util.remove_cand(1).as_dict() == {0: 5, 2: 1}

def test_utility_transformation(util):
    assert util.transformation(lambda u: u * 2).as_dict() == {0: 10, 1: 6, 2: 2}

def test_utility_linear_transformation(util):
    # regression: linear_transformation re-applied self.val to the value and crashed
    assert util.linear_transformation(a=2, b=1).as_dict() == {0: 11, 1: 7, 2: 3}

def test_utility_normalize_by_range(util):
    assert util.normalize_by_range().as_dict() == {0: 1.0, 1: 0.5, 2: 0.0}

def test_utility_normalize_by_range_equal_values():
    # max == min -> all zeros
    assert Utility({0: 5, 1: 5}).normalize_by_range().as_dict() == {0: 0, 1: 0}

def test_utility_normalize_by_standard_score(util):
    out = util.normalize_by_standard_score().as_dict()
    assert out[0] == pytest.approx(1.224744, abs=1e-5)
    assert out[1] == pytest.approx(0.0, abs=1e-9)
    assert out[2] == pytest.approx(-1.224744, abs=1e-5)

def test_utility_expectation(util):
    assert util.expectation({0: 0.5, 1: 0.5}) == pytest.approx(4.0)

def test_utility_expectation_rejects_out_of_domain_prob(util):
    with pytest.raises(AssertionError):
        util.expectation({9: 1.0})

def test_utility_represents_ranking(util):
    assert util.represents_ranking(Ranking({0: 1, 1: 2, 2: 3})) is True
    # util has 0 strictly over 1, but this ranking says they are tied
    assert util.represents_ranking(Ranking({0: 1, 1: 1, 2: 2})) is False

def test_utility_represents_ranking_extended(util):
    assert util.represents_ranking(Ranking({0: 1, 1: 2, 2: 3}), use_extended=True) is True

def test_utility_to_approval_ballot():
    u = Utility({0: 5, 1: 4, 2: 0})  # average 3, so 0 and 1 are above average
    # prob 1.0 -> keep approving every above-average candidate
    assert u.to_approval_ballot(prob_to_cont_approving=1.0).as_dict() == {0: 1, 1: 1, 2: 0}
    # prob 0.0 -> stop after the top candidate
    assert u.to_approval_ballot(prob_to_cont_approving=0.0).as_dict() == {0: 1, 1: 0, 2: 0}

def test_utility_to_k_approval_ballot():
    u = Utility({0: 5, 1: 4, 2: 0})
    # k = 1 caps the approval set at the single top candidate
    assert u.to_k_approval_ballot(1, prob_to_cont_approving=1.0).as_dict() == {0: 1, 1: 0, 2: 0}
    # k = 2 with prob 1.0 approves both above-average candidates
    assert u.to_k_approval_ballot(2, prob_to_cont_approving=1.0).as_dict() == {0: 1, 1: 1, 2: 0}

def test_utility_from_linear_ranking():
    u = Utility.from_linear_ranking([2, 0, 1], seed=42)
    # earlier in the list -> higher utility
    assert u(2) > u(0) > u(1)
    assert u.ranking().rmap == {2: 1, 0: 2, 1: 3}

@pytest.mark.parametrize("bad", [
    "not a list",        # not a list/tuple
    [0, 1, 1],           # not unique
])
def test_utility_from_linear_ranking_validates(bad):
    with pytest.raises(ValueError):
        Utility.from_linear_ranking(bad)

def test_utility_str(util):
    assert str(util) == "U(0) = 5, U(1) = 3, U(2) = 1"


# ===========================================================================
#  Grade (subclass of _Mapping)
# ===========================================================================

@pytest.fixture
def grade():
    return Grade({0: 1, 1: 0, 2: 1}, [0, 1])

def test_grade_basics(grade):
    assert grade.graded_candidates == [0, 1, 2]
    assert grade.candidates_with_grade(1) == [0, 2]
    assert grade.has_grade(0) is True
    assert grade.has_grade(9) is False

def test_grade_init_rejects_grade_not_in_grades():
    with pytest.raises(AssertionError):
        Grade({0: 5}, [0, 1])  # grade 5 is not among the allowed grades

def test_grade_init_rejects_candidate_outside_candidates():
    with pytest.raises(AssertionError):
        Grade({0: 1, 9: 1}, [0, 1], candidates=[0, 1])

def test_grade_ranking(grade):
    # grade 1 (candidates 0, 2) ranks above grade 0 (candidate 1)
    assert grade.ranking().rmap == {0: 1, 2: 1, 1: 2}
    assert grade.has_tie() is True
    assert grade.is_linear(3) is False

def test_grade_linear_ranking():
    gl = Grade({0: 2, 1: 1, 2: 0}, [0, 1, 2])
    assert gl.ranking().rmap == {0: 1, 1: 2, 2: 3}
    assert gl.is_linear(3) is True

def test_grade_extended_ranking_places_ungraded_last():
    g = Grade({0: 2, 1: 1}, [0, 1, 2], candidates=[0, 1, 2])
    assert g.extended_ranking().rmap == {0: 1, 1: 2, 2: 3}

def test_grade_remove_cand(grade):
    assert grade.remove_cand(1).as_dict() == {0: 1, 2: 1}

def test_grade_str(grade):
    assert str(grade) == "grade(0) = 1, grade(1) = 0, grade(2) = 1"

def test_utility_to_k_approval_ballot_stops_on_probability():
    # prob 0.0 with k >= 2 enters the loop but breaks immediately on the draw
    u = Utility({0: 5, 1: 4, 2: 0})
    assert u.to_k_approval_ballot(2, prob_to_cont_approving=0.0).as_dict() == {0: 1, 1: 0, 2: 0}

def test_utility_represents_ranking_false_unrated_candidate():
    # ranking mentions candidate 2, which the utility does not rate
    assert Utility({0: 5, 1: 3}).represents_ranking(Ranking({0: 1, 1: 2, 2: 3})) is False

def test_utility_represents_ranking_false_strict_mismatch():
    # ranking says 0 > 1 strictly, but the utility ties them
    assert Utility({0: 5, 1: 5, 2: 1}).represents_ranking(Ranking({0: 1, 1: 2, 2: 3})) is False

def test_utility_represents_ranking_extended_false_strict_mismatch():
    # extended: ranking strictly prefers 0 to 1, utility ties them
    assert Utility({0: 5, 1: 5}).represents_ranking(Ranking({0: 1, 1: 2}), use_extended=True) is False

def test_utility_represents_ranking_extended_false_indiff_mismatch():
    # extended: ranking is indifferent between 0 and 1, utility strictly prefers 0
    assert Utility({0: 5, 1: 3}).represents_ranking(Ranking({0: 1, 1: 1}), use_extended=True) is False
