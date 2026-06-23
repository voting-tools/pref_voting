import pytest

from pref_voting.axiom_helpers import (
    display_mg,
    list_to_string,
    swap_candidates,
    equal_size_partitions_with_duplicates,
    get_rank,
    powerset,
    linear_orders_with_reverse,
    remove_first_occurrences,
)
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MarginGraph
from pref_voting.rankings import Ranking


# --- list_to_string ------------------------------------------------------

def test_list_to_string():
    assert list_to_string([0, 2], {0: "a", 1: "b", 2: "c"}) == "{a, c}"
    assert list_to_string([], {0: "a"}) == "{}"


# --- swap_candidates -----------------------------------------------------

def test_swap_candidates_list():
    assert swap_candidates([0, 1, 2], 0, 2) == (2, 1, 0)

def test_swap_candidates_tuple():
    assert swap_candidates((0, 1, 2), 1, 2) == (0, 2, 1)

def test_swap_candidates_ranking_does_not_mutate_original():
    r = Ranking({0: 1, 1: 2, 2: 3})
    swapped = swap_candidates(r, 0, 2)
    assert swapped.rmap == {0: 3, 1: 2, 2: 1}
    assert r.rmap == {0: 1, 1: 2, 2: 3}  # original untouched

def test_swap_candidates_ranking_missing_candidate_raises():
    r = Ranking({0: 1, 1: 2})
    with pytest.raises(ValueError):
        swap_candidates(r, 0, 9)


# --- equal_size_partitions_with_duplicates -------------------------------

def test_equal_size_partitions():
    parts = equal_size_partitions_with_duplicates([1, 2, 3, 4])
    assert parts == [([1, 2], [3, 4]), ([1, 3], [2, 4]), ([1, 4], [2, 3])]

def test_equal_size_partitions_with_duplicates():
    # duplicates must not create spurious distinct partitions
    parts = equal_size_partitions_with_duplicates([1, 1, 2, 2])
    assert parts == [([1, 1], [2, 2]), ([1, 2], [1, 2])]

def test_equal_size_partitions_odd_length_raises():
    with pytest.raises(ValueError):
        equal_size_partitions_with_duplicates([1, 2, 3])


# --- get_rank ------------------------------------------------------------

def test_get_rank_ranking_is_normalized():
    # {0:1, 1:3, 2:5} normalizes to ranks 1,2,3 -> candidate 1 has rank 2
    assert get_rank(Ranking({0: 1, 1: 3, 2: 5}), 1) == 2

def test_get_rank_ranking_does_not_mutate():
    r = Ranking({0: 1, 1: 3, 2: 5})
    get_rank(r, 1)
    assert r.rmap == {0: 1, 1: 3, 2: 5}  # deep-copied internally

def test_get_rank_list_is_zero_based_index():
    assert get_rank([2, 0, 1], 0) == 1
    assert get_rank([2, 0, 1], 2) == 0

def test_get_rank_invalid_type_raises():
    with pytest.raises(ValueError):
        get_rank(42, 0)


# --- powerset ------------------------------------------------------------

def test_powerset():
    assert list(powerset([1, 2])) == [(), (1,), (2,), (1, 2)]

def test_powerset_empty():
    assert list(powerset([])) == [()]


# --- linear_orders_with_reverse -----------------------------------------

def test_linear_orders_with_reverse():
    out = linear_orders_with_reverse([0, 1])
    assert out == [((0, 1), (1, 0)), ((1, 0), (0, 1))]

def test_linear_orders_with_reverse_pairs_each_order_with_its_reverse():
    for order, rev in linear_orders_with_reverse([0, 1, 2]):
        assert rev == order[::-1]


# --- remove_first_occurrences -------------------------------------------

def test_remove_first_occurrences():
    # removes only the FIRST occurrence of each of r1 and r2
    assert remove_first_occurrences([1, 2, 1, 2, 3], 1, 2) == [1, 2, 3]

def test_remove_first_occurrences_absent():
    assert remove_first_occurrences([3, 4, 5], 1, 2) == [3, 4, 5]


# --- display_mg (smoke; these DRAW a graph, headless under the Agg backend) ---

def test_display_mg_profile():
    # Profile/ProfileWithTies route to edata.display_margin_graph()
    display_mg(Profile([[0, 1, 2], [2, 1, 0]]))

def test_display_mg_profile_with_ties():
    display_mg(ProfileWithTies([{0: 1, 1: 2, 2: 3}]))

def test_display_mg_margin_graph():
    # non-profile edata falls to the else branch (edata.display())
    display_mg(MarginGraph([0, 1, 2], [(0, 1, 2), (1, 2, 2), (0, 2, 2)]))
