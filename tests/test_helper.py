import random

import numpy as np
import pytest

from pref_voting.helper import (
    get_mg,
    get_weak_mg,
    swf_from_vm,
    vm_from_swf,
    create_election,
    SPO,
    weak_orders,
    weak_compositions,
    compositions,
    enumerate_compositions,
    sublists,
    convex_lexicographic_sublists,
    powerset,
)
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.weighted_majority_graphs import MarginGraph
from pref_voting.rankings import Ranking
from pref_voting.scoring_methods import plurality
from pref_voting.c1_methods import top_cycle


# ---------------------------------------------------------------------------
#  get_mg / get_weak_mg
# ---------------------------------------------------------------------------

def test_get_mg_from_profile():
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0]])  # 0 > 1 > 2 majority
    mg = get_mg(prof)
    assert mg.has_edge(0, 1) and mg.has_edge(1, 2) and mg.has_edge(0, 2)

def test_get_mg_from_profile_curr_cands():
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0]])
    mg = get_mg(prof, curr_cands=[1, 2])
    assert set(mg.nodes()) == {1, 2}
    assert mg.has_edge(1, 2)

def test_get_mg_from_margin_graph():
    marg = MarginGraph([0, 1, 2], [(0, 1, 2), (1, 2, 2), (0, 2, 2)])
    full = get_mg(marg)
    assert full.has_edge(0, 1)
    restricted = get_mg(marg, curr_cands=[0, 1])
    assert set(restricted.nodes()) == {0, 1}

def test_get_weak_mg_from_profile_adds_tie_edges():
    prof = Profile([[0, 1], [1, 0]])  # 0 and 1 are tied
    wmg = get_weak_mg(prof)
    assert wmg.has_edge(0, 1) and wmg.has_edge(1, 0)

def test_get_weak_mg_from_profile_curr_cands():
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0]])
    wmg = get_weak_mg(prof, curr_cands=[1, 2])
    assert set(wmg.nodes()) == {1, 2}
    assert wmg.has_edge(1, 2)

def test_get_weak_mg_from_margin_graph_full():
    marg = MarginGraph([0, 1, 2], [(0, 1, 2), (1, 2, 2), (0, 2, 2)])
    wmg = get_weak_mg(marg)
    assert set(wmg.nodes()) == {0, 1, 2}
    assert wmg.has_edge(0, 1)

def test_get_weak_mg_does_not_mutate_input_graph():
    # Bug 4.1: get_weak_mg aliased edata.mg and added tie-edges in place, corrupting
    # the input graph. With 0 ~ 1 tied, the buggy version permanently adds (0,1),(1,0)
    # to marg.mg. This must NOT happen -- the input graph must be untouched.
    marg = MarginGraph([0, 1, 2], [(0, 2, 2), (1, 2, 2)])  # 0 ~ 1 are tied
    before = sorted(marg.mg.edges())
    get_weak_mg(marg)
    assert sorted(marg.mg.edges()) == before, "get_weak_mg mutated its input graph"

def test_top_cycle_does_not_mutate_input_graph():
    # Bug 4.1, via a caller: top_cycle calls get_weak_mg, so the aliasing bug also
    # corrupts the input graph through it.
    marg = MarginGraph([0, 1, 2], [(1, 2, 4), (2, 0, 2)])  # 0 ~ 1 tied
    before = sorted(marg.mg.edges())
    top_cycle(marg)
    assert sorted(marg.mg.edges()) == before, "top_cycle mutated the input graph"

def test_get_weak_mg_from_margin_graph_curr_cands():
    marg = MarginGraph([0, 1, 2], [(0, 1, 2), (1, 2, 2), (0, 2, 2)])
    wmg = get_weak_mg(marg, curr_cands=[0, 1])
    assert set(wmg.nodes()) == {0, 1}


# ---------------------------------------------------------------------------
#  swf_from_vm / vm_from_swf
# ---------------------------------------------------------------------------

def test_swf_from_vm_default_tie_breaker():
    # plurality ties 0 and 1, so with no tie-breaker they share the top rank
    prof = Profile([[0, 1], [1, 0]])
    swf = swf_from_vm(plurality)
    r = swf(prof)
    assert r.rmap == {0: 0, 1: 0}

def test_swf_from_vm_alphabetic_tie_breaker():
    prof = Profile([[0, 1], [1, 0]])
    swf = swf_from_vm(plurality, tie_breaker="alphabetic")
    assert swf(prof).rmap == {0: 0, 1: 1}

def test_swf_from_vm_random_tie_breaker():
    prof = Profile([[0, 1], [1, 0]])
    swf = swf_from_vm(plurality, tie_breaker="random")
    random.seed(0)
    r = swf(prof)
    # a random tie-break still produces a strict order over both candidates
    assert sorted(r.rmap.keys()) == [0, 1]
    assert sorted(r.rmap.values()) == [0, 1]

def test_swf_from_vm_orders_distinct_winners():
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0]])  # 0 plurality winner
    r = swf_from_vm(plurality, tie_breaker="alphabetic")(prof)
    assert r.rmap == {0: 0, 1: 1, 2: 2}

def test_vm_from_swf_selects_top_ranked():
    prof = Profile([[0, 1, 2], [0, 1, 2], [1, 2, 0]])
    swf = swf_from_vm(plurality, tie_breaker="alphabetic")
    vm = vm_from_swf(swf)
    assert vm(prof) == [0]


# ---------------------------------------------------------------------------
#  create_election
# ---------------------------------------------------------------------------

def test_create_election_from_tuples_returns_profile():
    prof = create_election([(0, 1, 2), (2, 1, 0)])
    assert isinstance(prof, Profile)

def test_create_election_from_dicts_returns_profile_with_ties():
    prof = create_election([{0: 1, 1: 2, 2: 3}])
    assert isinstance(prof, ProfileWithTies)

def test_create_election_from_dicts_with_candidates_and_extended():
    prof = create_election([{0: 1, 1: 2}], candidates=[0, 1, 2],
                           using_extended_strict_preference=True)
    assert isinstance(prof, ProfileWithTies)
    assert prof.using_extended_strict_preference

def test_create_election_empty_warns_and_returns_profile(capsys):
    prof = create_election([])
    assert isinstance(prof, Profile)
    assert "empty" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  SPO (strict partial order)
# ---------------------------------------------------------------------------

def test_spo_transitive_closure_and_linear_order():
    s = SPO(3)
    s.add(0, 1)
    s.add(1, 2)
    # transitive closure infers 0 P 2
    assert bool(s.to_numpy()[0][2]) is True
    assert s.initial_elements() == [0]
    assert s.to_list() == [0, 1, 2]

def test_spo_to_list_with_cmap():
    s = SPO(3)
    s.add(0, 1)
    s.add(1, 2)
    assert s.to_list(cmap={0: "a", 1: "b", 2: "c"}) == ["a", "b", "c"]

def test_spo_to_list_none_when_not_linear():
    s = SPO(3)
    s.add(0, 1)  # 2 is incomparable to 0 and 1
    assert s.to_list() is None

def test_spo_to_networkx():
    s = SPO(3)
    s.add(0, 1)
    s.add(1, 2)
    g = s.to_networkx()
    assert sorted(g.edges()) == [(0, 1), (0, 2), (1, 2)]
    g2 = s.to_networkx(cmap={0: "a", 1: "b", 2: "c"})
    assert sorted(g2.edges()) == [("a", "b"), ("a", "c"), ("b", "c")]

def test_spo_add_is_idempotent():
    s = SPO(2)
    s.add(0, 1)
    s.add(0, 1)  # adding again is a no-op
    assert s.preds[1] == [0]

def test_spo_multi_step_transitive_closure():
    # build 0<1 and 2<3, then link them with 1<2; closure must infer all pairs
    s = SPO(4)
    s.add(0, 1)
    s.add(2, 3)
    s.add(1, 2)
    P = s.to_numpy()
    assert bool(P[0][2]) and bool(P[0][3]) and bool(P[1][3])
    assert s.to_list() == [0, 1, 2, 3]

def test_spo_diamond_reregisters_existing_pair():
    # 0<3 is implied via both 1 and 2; the second path re-registers an existing pair
    s = SPO(4)
    s.add(0, 1)
    s.add(0, 2)
    s.add(1, 3)
    s.add(2, 3)
    assert bool(s.to_numpy()[0][3]) is True


# ---------------------------------------------------------------------------
#  Combinatorial generators
# ---------------------------------------------------------------------------

def test_weak_orders_empty():
    assert list(weak_orders([])) == [{}]

def test_weak_orders_two_elements():
    orders = [dict(sorted(o.items())) for o in weak_orders([0, 1])]
    # 0 above 1, 1 above 0, and the two tied
    assert {0: 0, 1: 1} in orders
    assert {0: 1, 1: 0} in orders
    assert {0: 0, 1: 0} in orders
    assert len(orders) == 3

def test_weak_compositions():
    assert list(weak_compositions(2, 2)) == [[0, 2], [1, 1], [2, 0]]

def test_compositions():
    assert list(compositions(3)) == [[1, 1, 1], [1, 2], [2, 1], [3]]

def test_enumerate_compositions_single():
    assert list(enumerate_compositions([2])) == [[[1, 1]], [[2]]]

def test_enumerate_compositions_multiple():
    assert list(enumerate_compositions([1, 2])) == [[[1], [1, 1]], [[1], [2]]]

def test_sublists():
    assert list(sublists([0, 1, 2], 2)) == [[0, 1], [0, 2], [1, 2]]

def test_convex_lexicographic_sublists_all_sorted():
    assert convex_lexicographic_sublists([1, 2, 3]) == [[1, 2, 3]]

def test_convex_lexicographic_sublists_splits():
    assert convex_lexicographic_sublists([1, 2, 1, 3]) == [[1, 2], [1, 3]]

def test_convex_lexicographic_sublists_break_at_last():
    assert convex_lexicographic_sublists([3, 1]) == [[3], [1]]

def test_powerset():
    assert list(powerset([1, 2])) == [(), (1,), (2,), (1, 2)]
