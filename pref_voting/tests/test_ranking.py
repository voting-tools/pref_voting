from pref_voting.rankings import Ranking
import pytest

@pytest.fixture
def linear_ranking():
    return Ranking({0:1, 1:3, 2:2})

@pytest.fixture
def ranking_with_tie():
    return Ranking({0:1, 1:1, 2:5})

@pytest.fixture
def truncated_ranking():
    return Ranking({0:1, 1:3})

def test_ranking_initialization():
    rmap = {0:1, 1:3, 2:2}
    cmap = {0: "Alice", 1: "Bob", 2: "Charlie"}
    rank = Ranking(rmap, cmap)
    assert rank.rmap == rmap
    assert rank.cmap == cmap


@pytest.mark.parametrize("rmap, expected", [
    ({0:1, 1:3, 2:2}, [0, 1, 2]),
    ({0:1, 1:1, 2:2}, [0, 1, 2]),
    ({0:1, 1:3}, [0, 1])])
def test_ranking_cands(rmap, expected):
    assert Ranking(rmap).cands == expected

@pytest.mark.parametrize("rmap, expected", [
    ({0:1, 1:3, 2:2}, [1, 2, 3]),
    ({0:1, 1:1, 2:5}, [1, 5]),
    ({0:1, 1:3}, [1, 3])])
def test_ranking_ranks(rmap, expected):
    assert Ranking(rmap).ranks == expected

@pytest.mark.parametrize("rmap, r, expected", [
    ({0:1, 1:3, 2:2}, 1, [0]),
    ({0:1, 1:1, 2:5}, 1,  [0, 1]),
    ({0:1, 1:1, 2:5}, 2,  []),
    ({0:1, 1:1, 2:5}, 5,  [2]),
    ({0:1, 1:3}, 2, []),
    ({0:1, 1:3}, 3, [1])])
def test_cands_at_rank(rmap, r, expected):
    assert Ranking(rmap).cands_at_rank(r) == expected

@pytest.mark.parametrize("rmap, c1, c2, expected", [
    ({0:1, 1:3, 2:2}, 0, 1, True),
    ({0:1, 1:3, 2:2}, 1, 0, False),
    ({0:1, 1:3}, 0, 1, True),
    ({0:1, 1:3}, 2, 3, False),
    ({0:1, 1:3}, 0, 2, False)])
def test_strict_pref(rmap, c1, c2, expected):
    assert Ranking(rmap).strict_pref(c1, c2) == expected

@pytest.mark.parametrize("rmap, c1, c2, expected", [
    ({0:1, 1:3, 2:2}, 0, 1, True),
    ({0:1, 1:3, 2:2}, 1, 0, False),
    ({0:1, 1:3}, 0, 1, True),
    ({0:1, 1:3}, 2, 3, False),
    ({0:1, 1:3}, 0, 2, True)])
def test_extended_strict_pref(rmap, c1, c2, expected):
    assert Ranking(rmap).extended_strict_pref(c1, c2) == expected


@pytest.mark.parametrize("rmap, c1, c2, expected", [
    ({0:1, 1:1, 2:2}, 0, 1, True),
    ({0:1, 1:1, 2:2}, 1, 0, True),
    ({0:1, 1:1, 2:2}, 1, 2, False),
    ({0:1, 1:3}, 0, 2, False),
    ({0:1, 1:3}, 2, 3, False)])
def test_indiff(rmap, c1, c2, expected):
    assert Ranking(rmap).indiff(c1, c2) == expected


@pytest.mark.parametrize("rmap, c1, c2, expected", [
    ({0:1, 1:1, 2:2}, 0, 1, True),
    ({0:1, 1:1, 2:2}, 1, 0, True),
    ({0:1, 1:1, 2:2}, 1, 2, False),
    ({0:1, 1:3}, 0, 2, False),
    ({0:1, 1:3}, 2, 3, True)])
def test_extended_indiff(rmap, c1, c2, expected):
    assert Ranking(rmap).extended_indiff(c1, c2) == expected


@pytest.mark.parametrize("rmap, c1, c2, expected", [
    ({0:1, 1:1, 2:2}, 0, 1, True),
    ({0:1, 1:1, 2:2}, 1, 0, True),
    ({0:1, 1:1, 2:2}, 1, 2, True),
    ({0:1, 1:1, 2:2}, 2, 1, False),
    ({0:1, 1:3}, 0, 2, False),
    ({0:1, 1:3}, 2, 3, False)])
def test_weak_pref(rmap, c1, c2, expected):
    assert Ranking(rmap).weak_pref(c1, c2) == expected

@pytest.mark.parametrize("rmap, c1, c2, expected", [
    ({0:1, 1:1, 2:2}, 0, 1, True),
    ({0:1, 1:1, 2:2}, 1, 0, True),
    ({0:1, 1:1, 2:2}, 1, 2, True),
    ({0:1, 1:1, 2:2}, 2, 1, False),
    ({0:1, 1:3}, 0, 2, True),
    ({0:1, 1:3}, 2, 3, True)])
def test_extended_weak_pref(rmap, c1, c2, expected):
    assert Ranking(rmap).extended_weak_pref(c1, c2) == expected


@pytest.mark.parametrize("cand, expected_cands, expected_ranks", [
    (0, [1, 2], [2, 3]),
    (1, [0, 2], [1, 2]),
    (2, [0, 1], [1, 3]),
    (3, [0, 1, 2], [1, 2, 3])])
def test_remove_cand_linear_ranking(cand, expected_cands, expected_ranks, linear_ranking):
    r2 = linear_ranking.remove_cand(cand)
    assert r2.cands == expected_cands
    assert r2.ranks == expected_ranks

@pytest.mark.parametrize("cand, expected_cands, expected_ranks", [
    (0, [1, 2], [1, 5]),
    (1, [0, 2], [1, 5]),
    (2, [0, 1], [1]),
    (3, [0, 1, 2], [1, 5])])
def test_remove_cand_ranking_with_tie(cand, expected_cands, expected_ranks, ranking_with_tie):
    r2 = ranking_with_tie.remove_cand(cand)
    assert r2.cands == expected_cands
    assert r2.ranks == expected_ranks

@pytest.mark.parametrize("rmap, expected", [
    ({0:1, 1:2, 2:2}, [0]),
    ({0:1, 1:1, 2:2}, [0, 1]),
    ({0:3, 1:3, 2:5}, [0, 1]),
    ({0:3}, [0]),
    ({}, []),])
def test_first(rmap, expected):
    assert Ranking(rmap).first() == expected

@pytest.mark.parametrize("rmap, expected", [
    ({0:1, 1:2, 2:2}, [1, 2]),
    ({0:1, 1:1, 2:2}, [2]),
    ({0:3, 1:3, 2:5}, [2]),
    ({0:3, 1:5, 2:5}, [1, 2]),
    ({0:3}, [0]),
    ({}, []),])
def test_last(rmap, expected):
    assert Ranking(rmap).last() == expected

def test_is_empty():
    assert Ranking({}).is_empty()
    assert not Ranking({0:1, 1:1}).is_empty()

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
    assert Ranking({}).to_linear()  == ()

def test_has_skipped_rank(linear_ranking, ranking_with_tie):
    assert not linear_ranking.has_skipped_rank()
    assert Ranking({0:1, 1:4, 2:4}).has_skipped_rank()
    assert not Ranking({0:1, 1:1, 2:2}).has_skipped_rank()
    assert Ranking({0:1, 1:1, 2:3}).has_skipped_rank()

def test_to_indiff_list():
    r = Ranking({0:1, 1:1, 2:2})
    assert r.to_indiff_list() == ((0, 1), (2,))

@pytest.mark.parametrize("rmap, expected", [
    ({0:1, 1:2, 2:3, 3:3}, {0:1, 1:2}),
    ({0:1, 1:1, 2:2}, {}),
    ({0:1, 1:2, 2:5, 3:3}, {0:1, 1:2, 2:5, 3:3}),
    ({0:1, 1:2, 2:2, 3:3}, {0:1}),])
def test_truncate_overvote(rmap, expected):
    r = Ranking(rmap)
    r.truncate_overvote()
    assert r.rmap == expected

def test_AAdom():
    r = Ranking({0:1, 1:3, 2:1, 3:4})
    assert r.AAdom([0, 2], [1, 3]) 
    assert r.AAdom([0, 1], [1, 3]) 
    assert not r.AAdom([0, 1], [2, 3]) 
    assert r.AAdom([0], [1, 2, 3])
    assert r.AAdom([0], [0,1, 2, 3])
    assert not r.AAdom([0, 3], [0, 1])


def test_strong_dom():
    r = Ranking({0:1, 1:3, 2:1, 3:4})
    assert r.strong_dom([0, 2], [1, 3]) 
    assert r.strong_dom([0, 1], [1, 3]) 
    assert not r.strong_dom([0, 1], [2, 3]) 
    assert not r.strong_dom([0], [1, 2, 3])
    assert not r.strong_dom([0], [0, 1, 2, 3])
    assert not r.strong_dom([0, 3], [0, 1])


def test_weak_dom():
    r = Ranking({0:1, 1:3, 2:1, 3:4})
    assert r.weak_dom([0, 2], [1, 3]) 
    assert r.weak_dom([0, 1], [1, 3]) 
    assert not r.weak_dom([0, 1], [2, 3]) 
    assert  r.weak_dom([0], [1, 2, 3])
    assert  r.weak_dom([0], [0, 1, 2, 3])
    assert not r.weak_dom([0, 3], [0, 1])


@pytest.mark.parametrize("rmap, expected", [
    ({0:1, 1:2, 2:3}, {0:1, 1:2, 2:3}),
    ({0:1000, 1:-10, 2:0}, {0:3, 1:1, 2:2}),
    ({0:1, 1:5, 2:5, 3:3}, {0:1, 1:3, 2:3, 3:2}),
    ({0:1, 1:1, 2:4, 3:4}, {0:1, 1:1, 2:2, 3:2}),])
def test_normalize_ranks(rmap, expected):
    r = Ranking(rmap)
    r.normalize_ranks()
    assert r.rmap == expected

def test_display(capsys):
    
    ranking = Ranking({0:1, 1:3, 3:3})
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

@pytest.mark.parametrize("rmap, expected", [
    ({0:1, 1:2, 2:3, 3:3}, "0 1 ( 2  3 )"),
    ({0:1, 1:1, 2:2}, "( 0  1 ) 2"), 
    ({0:1, 1:2, 2:3}, "0 1 2"),])
def test_str(rmap, expected):
    assert str(Ranking(rmap)).strip() == expected.strip()

def test_get_item():
    r = Ranking({0:1, 1:2, 2:3})
    assert r[0] == 0
    assert r[1] == 1
    assert r[2] == 2
    r = Ranking({0:1, 1:1, 2:3})
    assert r[0] == [0,1]
    assert r[1] == 2


@pytest.mark.parametrize("rmap1, rmap2, expected", [
    ({0:1, 1:2, 2:3, 3:3}, {0:1, 1:2, 2:3, 3:3}, True),
    ({0:1, 1:2, 2:3, 3:3}, {0:2, 1:1, 2:3, 3:3}, False),
    ({0:1, 1:2, 2:3}, {0:-10, 1:20, 2:300}, True),])
def test_eq(rmap1, rmap2, expected): 
    assert (Ranking(rmap1) == Ranking(rmap2)) == expected   

def test_eq2(): 
    rmap1 = {0:1, 1:1}
    rmap2 = {1:1, 0:1}
    assert Ranking(rmap1) == Ranking(rmap2) 
