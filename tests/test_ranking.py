import pytest

from pref_voting.rankings import Ranking, break_ties_alphabetically


@pytest.fixture
def linear_ranking():
    return Ranking({0: 1, 1: 3, 2: 2})


@pytest.fixture
def ranking_with_tie():
    return Ranking({0: 1, 1: 1, 2: 5})


@pytest.fixture
def truncated_ranking():
    return Ranking({0: 1, 1: 3})


def test_ranking_initialization():
    rmap = {0: 1, 1: 3, 2: 2}
    cmap = {0: "Alice", 1: "Bob", 2: "Charlie"}
    rank = Ranking(rmap, cmap)
    assert rank.rmap == rmap
    assert rank.cmap == cmap


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 3, 2: 2}, [0, 1, 2]),
        ({0: 1, 1: 1, 2: 2}, [0, 1, 2]),
        ({0: 1, 1: 3}, [0, 1]),
    ],
)
def test_ranking_cands(rmap, expected):
    assert Ranking(rmap).cands == expected


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 3, 2: 2}, [1, 2, 3]),
        ({0: 1, 1: 1, 2: 5}, [1, 5]),
        ({0: 1, 1: 3}, [1, 3]),
    ],
)
def test_ranking_ranks(rmap, expected):
    assert Ranking(rmap).ranks == expected


@pytest.mark.parametrize(
    "rmap, r, expected",
    [
        ({0: 1, 1: 3, 2: 2}, 1, [0]),
        ({0: 1, 1: 1, 2: 5}, 1, [0, 1]),
        ({0: 1, 1: 1, 2: 5}, 2, []),
        ({0: 1, 1: 1, 2: 5}, 5, [2]),
        ({0: 1, 1: 3}, 2, []),
        ({0: 1, 1: 3}, 3, [1]),
    ],
)
def test_cands_at_rank(rmap, r, expected):
    assert Ranking(rmap).cands_at_rank(r) == expected


@pytest.mark.parametrize(
    "rmap, c1, c2, expected",
    [
        ({0: 1, 1: 3, 2: 2}, 0, 1, True),
        ({0: 1, 1: 3, 2: 2}, 1, 0, False),
        ({0: 1, 1: 3}, 0, 1, True),
        ({0: 1, 1: 3}, 2, 3, False),
        ({0: 1, 1: 3}, 0, 2, False),
    ],
)
def test_strict_pref(rmap, c1, c2, expected):
    assert Ranking(rmap).strict_pref(c1, c2) == expected


@pytest.mark.parametrize(
    "rmap, c1, c2, expected",
    [
        ({0: 1, 1: 3, 2: 2}, 0, 1, True),
        ({0: 1, 1: 3, 2: 2}, 1, 0, False),
        ({0: 1, 1: 3}, 0, 1, True),
        ({0: 1, 1: 3}, 2, 3, False),
        ({0: 1, 1: 3}, 0, 2, True),
    ],
)
def test_extended_strict_pref(rmap, c1, c2, expected):
    assert Ranking(rmap).extended_strict_pref(c1, c2) == expected


@pytest.mark.parametrize(
    "rmap, c1, c2, expected",
    [
        ({0: 1, 1: 1, 2: 2}, 0, 1, True),
        ({0: 1, 1: 1, 2: 2}, 1, 0, True),
        ({0: 1, 1: 1, 2: 2}, 1, 2, False),
        ({0: 1, 1: 3}, 0, 2, False),
        ({0: 1, 1: 3}, 2, 3, False),
    ],
)
def test_indiff(rmap, c1, c2, expected):
    assert Ranking(rmap).indiff(c1, c2) == expected


@pytest.mark.parametrize(
    "rmap, c1, c2, expected",
    [
        ({0: 1, 1: 1, 2: 2}, 0, 1, True),
        ({0: 1, 1: 1, 2: 2}, 1, 0, True),
        ({0: 1, 1: 1, 2: 2}, 1, 2, False),
        ({0: 1, 1: 3}, 0, 2, False),
        ({0: 1, 1: 3}, 2, 3, True),
    ],
)
def test_extended_indiff(rmap, c1, c2, expected):
    assert Ranking(rmap).extended_indiff(c1, c2) == expected


@pytest.mark.parametrize(
    "rmap, c1, c2, expected",
    [
        ({0: 1, 1: 1, 2: 2}, 0, 1, True),
        ({0: 1, 1: 1, 2: 2}, 1, 0, True),
        ({0: 1, 1: 1, 2: 2}, 1, 2, True),
        ({0: 1, 1: 1, 2: 2}, 2, 1, False),
        ({0: 1, 1: 3}, 0, 2, False),
        ({0: 1, 1: 3}, 2, 3, False),
    ],
)
def test_weak_pref(rmap, c1, c2, expected):
    assert Ranking(rmap).weak_pref(c1, c2) == expected


@pytest.mark.parametrize(
    "rmap, c1, c2, expected",
    [
        ({0: 1, 1: 1, 2: 2}, 0, 1, True),
        ({0: 1, 1: 1, 2: 2}, 1, 0, True),
        ({0: 1, 1: 1, 2: 2}, 1, 2, True),
        ({0: 1, 1: 1, 2: 2}, 2, 1, False),
        ({0: 1, 1: 3}, 0, 2, True),
        ({0: 1, 1: 3}, 2, 3, True),
    ],
)
def test_extended_weak_pref(rmap, c1, c2, expected):
    assert Ranking(rmap).extended_weak_pref(c1, c2) == expected


@pytest.mark.parametrize(
    "cand, expected_cands, expected_ranks",
    [
        (0, [1, 2], [2, 3]),
        (1, [0, 2], [1, 2]),
        (2, [0, 1], [1, 3]),
        (3, [0, 1, 2], [1, 2, 3]),
    ],
)
def test_remove_cand_linear_ranking(
    cand, expected_cands, expected_ranks, linear_ranking
):
    r2 = linear_ranking.remove_cand(cand)
    assert r2.cands == expected_cands
    assert r2.ranks == expected_ranks


@pytest.mark.parametrize(
    "cand, expected_cands, expected_ranks",
    [
        (0, [1, 2], [1, 5]),
        (1, [0, 2], [1, 5]),
        (2, [0, 1], [1]),
        (3, [0, 1, 2], [1, 5]),
    ],
)
def test_remove_cand_ranking_with_tie(
    cand, expected_cands, expected_ranks, ranking_with_tie
):
    r2 = ranking_with_tie.remove_cand(cand)
    assert r2.cands == expected_cands
    assert r2.ranks == expected_ranks


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 2, 2: 2}, [0]),
        ({0: 1, 1: 1, 2: 2}, [0, 1]),
        ({0: 3, 1: 3, 2: 5}, [0, 1]),
        ({0: 3}, [0]),
        ({}, []),
    ],
)
def test_first(rmap, expected):
    assert Ranking(rmap).first() == expected


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 2, 2: 2}, [1, 2]),
        ({0: 1, 1: 1, 2: 2}, [2]),
        ({0: 3, 1: 3, 2: 5}, [2]),
        ({0: 3, 1: 5, 2: 5}, [1, 2]),
        ({0: 3}, [0]),
        ({}, []),
    ],
)
def test_last(rmap, expected):
    assert Ranking(rmap).last() == expected


def test_is_empty():
    assert Ranking({}).is_empty()
    assert not Ranking({0: 1, 1: 1}).is_empty()


def test_has_tie(linear_ranking, ranking_with_tie, truncated_ranking):
    assert not linear_ranking.has_tie()
    assert ranking_with_tie.has_tie()
    assert not truncated_ranking.has_tie()
    assert not Ranking({}).has_tie()


def test_has_overvote(linear_ranking, ranking_with_tie, truncated_ranking):
    assert not linear_ranking.has_overvote()
    assert ranking_with_tie.has_overvote()
    assert not truncated_ranking.has_overvote()
    assert not Ranking({}).has_overvote()


def test_is_linear(linear_ranking, ranking_with_tie, truncated_ranking):
    assert linear_ranking.is_linear(3)
    assert not linear_ranking.is_linear(2)
    assert not ranking_with_tie.is_linear(3)
    assert not ranking_with_tie.is_linear(2)
    assert not truncated_ranking.is_linear(3)
    assert truncated_ranking.is_linear(2)
    assert not Ranking({}).is_linear(1)


def test_to_linear(linear_ranking, ranking_with_tie, truncated_ranking):
    assert linear_ranking.to_linear() == (0, 2, 1)
    assert ranking_with_tie.to_linear() is None
    assert truncated_ranking.to_linear() == (0, 1)
    assert Ranking({}).to_linear() == ()


def test_has_skipped_rank(linear_ranking, ranking_with_tie):
    assert not linear_ranking.has_skipped_rank()
    assert Ranking({0: 1, 1: 4, 2: 4}).has_skipped_rank()
    assert not Ranking({0: 1, 1: 1, 2: 2}).has_skipped_rank()
    assert Ranking({0: 1, 1: 1, 2: 3}).has_skipped_rank()


def test_to_indiff_list():
    r = Ranking({0: 1, 1: 1, 2: 2})
    assert r.to_indiff_list() == ((0, 1), (2,))


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 2, 2: 3, 3: 3}, {0: 1, 1: 2}),
        ({0: 1, 1: 1, 2: 2}, {}),
        ({0: 1, 1: 2, 2: 5, 3: 3}, {0: 1, 1: 2, 2: 5, 3: 3}),
        ({0: 1, 1: 2, 2: 2, 3: 3}, {0: 1}),
    ],
)
def test_truncate_overvote(rmap, expected):
    r = Ranking(rmap)
    r.truncate_overvote()
    assert r.rmap == expected


def test_AAdom():
    r = Ranking({0: 1, 1: 3, 2: 1, 3: 4})
    assert r.AAdom([0, 2], [1, 3])
    assert r.AAdom([0, 1], [1, 3])
    assert not r.AAdom([0, 1], [2, 3])
    assert r.AAdom([0], [1, 2, 3])
    assert r.AAdom([0], [0, 1, 2, 3])
    assert not r.AAdom([0, 3], [0, 1])


def test_strong_dom():
    r = Ranking({0: 1, 1: 3, 2: 1, 3: 4})
    assert r.strong_dom([0, 2], [1, 3])
    assert r.strong_dom([0, 1], [1, 3])
    assert not r.strong_dom([0, 1], [2, 3])
    assert not r.strong_dom([0], [1, 2, 3])
    assert not r.strong_dom([0], [0, 1, 2, 3])
    assert not r.strong_dom([0, 3], [0, 1])


def test_weak_dom():
    r = Ranking({0: 1, 1: 3, 2: 1, 3: 4})
    assert r.weak_dom([0, 2], [1, 3])
    assert r.weak_dom([0, 1], [1, 3])
    assert not r.weak_dom([0, 1], [2, 3])
    assert r.weak_dom([0], [1, 2, 3])
    assert r.weak_dom([0], [0, 1, 2, 3])
    assert not r.weak_dom([0, 3], [0, 1])


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 2, 2: 3}, {0: 1, 1: 2, 2: 3}),
        ({0: 1000, 1: -10, 2: 0}, {0: 3, 1: 1, 2: 2}),
        ({0: 1, 1: 5, 2: 5, 3: 3}, {0: 1, 1: 3, 2: 3, 3: 2}),
        ({0: 1, 1: 1, 2: 4, 3: 4}, {0: 1, 1: 1, 2: 2, 3: 2}),
    ],
)
def test_normalize_ranks(rmap, expected):
    r = Ranking(rmap)
    r.normalize_ranks()
    assert r.rmap == expected


def test_display(capsys):

    ranking = Ranking({0: 1, 1: 3, 3: 3})
    ranking.display()

    # Get the captured output
    captured = capsys.readouterr()
    print(captured.out)
    expected_output = """\
+-----+
|  0  |
| 1 3 |
+-----+
"""
    assert captured.out.strip() == expected_output.strip()


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 2, 2: 3, 3: 3}, "0 1 ( 2  3 )"),
        ({0: 1, 1: 1, 2: 2}, "( 0  1 ) 2"),
        ({0: 1, 1: 2, 2: 3}, "0 1 2"),
    ],
)
def test_str(rmap, expected):
    assert str(Ranking(rmap)).strip() == expected.strip()


def test_get_item():
    r = Ranking({0: 1, 1: 2, 2: 3})
    assert r[0] == 0
    assert r[1] == 1
    assert r[2] == 2
    r = Ranking({0: 1, 1: 1, 2: 3})
    assert r[0] == [0, 1]
    assert r[1] == 2


@pytest.mark.parametrize(
    "rmap1, rmap2, expected",
    [
        ({0: 1, 1: 2, 2: 3, 3: 3}, {0: 1, 1: 2, 2: 3, 3: 3}, True),
        ({0: 1, 1: 2, 2: 3, 3: 3}, {0: 2, 1: 1, 2: 3, 3: 3}, False),
        ({0: 1, 1: 2, 2: 3}, {0: -10, 1: 20, 2: 300}, True),
    ],
)
def test_eq(rmap1, rmap2, expected):
    assert (Ranking(rmap1) == Ranking(rmap2)) == expected


def test_eq2():
    rmap1 = {0: 1, 1: 1}
    rmap2 = {1: 1, 0: 1}
    assert Ranking(rmap1) == Ranking(rmap2)


# ---------------------------------------------------------------------------
#  Coverage additions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rmap, expected", [({0: 1, 1: 3, 2: 2}, 3), ({0: 1, 1: 3}, 2), ({}, 0)]
)
def test_num_ranked_candidates(rmap, expected):
    assert Ranking(rmap).num_ranked_candidates() == expected


@pytest.mark.parametrize(
    "rmap, num_cands, expected",
    [
        ({0: 1}, 4, True),  # favorite ranked, rest unranked -> bullet
        (
            {0: 1, 1: 2, 2: 2, 3: 2},
            4,
            True,
        ),  # favorite alone on top, rest tied bottom -> bullet
        (
            {0: 2, 1: 3, 2: 3, 3: 3},
            4,
            True,
        ),  # same ballot, non-normalized ranks -> bullet
        ({0: 1, 1: 1, 2: 1, 3: 1}, 4, False),  # everyone tied at top -> NOT a bullet
        ({0: 1, 1: 1, 2: 1}, 4, False),  # several tied at top -> not a bullet
        (
            {0: 1, 1: 2},
            4,
            False,
        ),  # favorite + one ranked, rest unranked -> not a bullet
        (
            {0: 1, 1: 2, 2: 2},
            4,
            False,
        ),  # favorite + 2 tied but cand 3 unranked -> not a bullet
        ({0: 1, 1: 2, 2: 2}, 3, True),  # favorite + all others tied -> bullet
        ({0: 1, 1: 2}, 2, False),  # complete 2-candidate order -> not a bullet
        ({0: 1}, 1, False),  # only one candidate -> voting for everyone, not a bullet
        ({}, 4, False),
    ],
)  # empty ranking -> not a bullet
def test_is_bullet_vote(rmap, num_cands, expected):
    assert Ranking(rmap).is_bullet_vote(num_cands) == expected


def test_is_ranked(truncated_ranking):
    assert truncated_ranking.is_ranked(0) is True
    assert truncated_ranking.is_ranked(1) is True
    assert truncated_ranking.is_ranked(2) is False


def test_is_tied():
    r = Ranking({0: 1, 1: 1, 2: 2})
    assert r.is_tied([0, 1]) is True
    assert r.is_tied([0, 2]) is False
    assert r.is_tied([2]) is True


@pytest.mark.parametrize(
    "rmap, num_cands, expected",
    [
        ({0: 1, 1: 3}, 3, True),  # linear but ranks fewer than num_cands
        ({0: 1, 1: 2, 2: 3}, 3, False),  # ranks all -> not truncated
        ({0: 1, 1: 1}, 3, False),  # has a tie -> not linear
        ({}, 3, True),
    ],
)
def test_is_truncated_linear(rmap, num_cands, expected):
    assert Ranking(rmap).is_truncated_linear(num_cands) == expected


@pytest.mark.parametrize(
    "indiff_list, expected",
    [
        ([(0, 1), (2,)], {0: 1, 1: 1, 2: 2}),
        ([(0,), (1,), (2,)], {0: 1, 1: 2, 2: 3}),
        ([], {}),
    ],
)
def test_from_indiff_list(indiff_list, expected):
    assert Ranking.from_indiff_list(indiff_list).rmap == expected


@pytest.mark.parametrize(
    "linear_order, expected", [([2, 0, 1], {2: 1, 0: 2, 1: 3}), ([0], {0: 1}), ([], {})]
)
def test_from_linear_order(linear_order, expected):
    assert Ranking.from_linear_order(linear_order).rmap == expected


def test_from_indiff_list_roundtrips():
    r = Ranking({0: 1, 1: 1, 2: 2})
    assert Ranking.from_indiff_list(r.to_indiff_list()) == r


def test_to_weak_order_fills_unranked_at_bottom():
    r = Ranking({0: 1, 1: 2}, cmap={0: "a", 1: "b"})
    w = r.to_weak_order([0, 1, 2, 3])
    # unranked candidates 2, 3 share the bottom rank (max_rank + 1)
    assert w.rmap == {0: 1, 1: 2, 2: 3, 3: 3}
    # missing candidates get a string cmap entry
    assert w.cmap == {0: "a", 1: "b", 2: "2", 3: "3"}


def test_to_weak_order_does_not_mutate_original():
    # regression for Bug 3.1: to_weak_order aliased self.rmap and rewrote it
    r = Ranking({0: 1, 1: 2})
    before = dict(r.rmap)
    r.to_weak_order([0, 1, 2, 3])
    assert r.rmap == before


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 2, 2: 3}, {0: 3, 1: 2, 2: 1}),
        ({0: 1, 1: 1, 2: 2}, {0: 2, 1: 2, 2: 1}),
        ({0: 1, 1: 2}, {0: 2, 1: 1}),
    ],
)
def test_reverse(rmap, expected):
    assert Ranking(rmap).reverse().rmap == expected


def test_break_tie():
    r = Ranking({0: 1, 1: 1, 2: 2})
    # breaking the {0,1} tie with order (1, 0) -> 1 first, then 0, then 2
    assert r.break_tie((1, 0)).rmap == {1: 1, 0: 2, 2: 3}
    # a lin_order that matches no indifference class leaves the ranking unchanged
    assert r.break_tie((5, 6)) == r


@pytest.mark.parametrize(
    "rmap, expected",
    [
        ({0: 1, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2}),  # numeric, already sorted
        ({"c": 1, "a": 1, "b": 1}, {"a": 0, "b": 1, "c": 2}),  # Bug 3.2: must sort
        ({0: 1, 1: 2, 2: 3}, {0: 0, 1: 1, 2: 2}),
    ],
)  # already strict
def test_break_ties_alphabetically(rmap, expected):
    assert break_ties_alphabetically(Ranking(rmap)).rmap == expected


def test_eq_different_lengths():
    # __eq__ short-circuits when the number of rank levels differs
    assert (Ranking({0: 1, 1: 2}) == Ranking({0: 1})) is False
    assert (Ranking({0: 1}) == Ranking({0: 1, 1: 2})) is False


def test_hashable_and_usable_in_set():
    r1 = Ranking({0: 1, 1: 2, 2: 3})
    r2 = Ranking({0: 1, 1: 2, 2: 3})
    assert hash(r1) == hash(r2)
    assert len({r1, r2}) == 1
    # usable as a dict key
    d = {r1: "x"}
    assert d[r2] == "x"


def test_hash_consistent_with_eq_for_tied_rankings():
    # __eq__ compares each rank level as a set (order-independent), so __hash__ must
    # also be order-independent within a tie class. Otherwise two EQUAL rankings built
    # with different key-insertion order hash differently and break set/dict membership.
    # Python requires:  a == b  implies  hash(a) == hash(b).
    a = Ranking({0: 1, 1: 1})
    b = Ranking({1: 1, 0: 1})  # same ranking, different key-insertion order
    assert a == b
    assert hash(a) == hash(b)
    assert b in {a}
    assert len({a, b}) == 1
    assert {a: "x"}[b] == "x"
