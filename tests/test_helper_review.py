"""Regression tests for bugs found in pref_voting/helper.py.

Run:  .venv/bin/python -m pytest review_tests/test_helper_review.py -v
"""
from pref_voting.weighted_majority_graphs import MarginGraph
from pref_voting.helper import get_weak_mg
from pref_voting.c1_methods import top_cycle


def test_get_weak_mg_does_not_mutate_graph_input():
    """Bug 4.1: get_weak_mg aliased edata.mg and added tie-edges to it, mutating
    the original MajorityGraph/MarginGraph. Methods like top_cycle/getcha/smith_set
    that call get_weak_mg would corrupt the input graph."""
    mg = MarginGraph([0, 1, 2], [(1, 2, 4), (2, 0, 2)])  # 0 and 1 are tied (no edge)
    before = sorted(mg.mg.edges())
    get_weak_mg(mg)
    assert sorted(mg.mg.edges()) == before, "get_weak_mg mutated the input graph"


def test_top_cycle_does_not_mutate_graph_input():
    mg = MarginGraph([0, 1, 2], [(1, 2, 4), (2, 0, 2)])
    before = sorted(mg.mg.edges())
    _ = top_cycle(mg)
    assert sorted(mg.mg.edges()) == before, "top_cycle mutated the input graph"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
