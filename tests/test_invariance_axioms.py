import pytest

from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.voting_method import VotingMethod
from pref_voting.scoring_methods import borda
from pref_voting.invariance_axioms import (
    homogeneity,
    upward_homogeneity,
    downward_homogeneity,
    block_invariance,
    upward_block_preservation,
    downward_block_preservation,
    preferential_equality,
    tiebreaking_compensation,
    invariance_axioms,
)


# Stub voting methods whose winners depend on the size of the electorate, which
# lets us deterministically trigger the homogeneity / block axioms.
def _vm_new_winner_when_large(threshold):
    # a loser (1) becomes a co-winner once the electorate reaches `threshold`
    return VotingMethod(
        lambda e, curr_cands=None: [0, 1] if e.num_voters >= threshold else [0],
        name="NewWinnerWhenLarge")

def _vm_winner_lost_when_large(threshold):
    # a winner (1) drops out once the electorate reaches `threshold`
    return VotingMethod(
        lambda e, curr_cands=None: [0, 1] if e.num_voters < threshold else [0],
        name="WinnerLostWhenLarge")

# winner = top candidate of the first ranking -> sensitive to ranking content/order
first_top = VotingMethod(
    lambda e, curr_cands=None: [e.rankings[0][0]]
    if isinstance(e.rankings[0], (list, tuple))
    else [sorted(e.rankings[0].cands_at_rank(min(e.rankings[0].ranks)))[0]],
    name="FirstTop")


@pytest.fixture
def prof():
    return Profile([[0, 1, 2], [1, 0, 2]], [1, 1])  # 2 voters


# ---------------------------------------------------------------------------
#  Homogeneity family (multiplying each ballot by num_copies)
# ---------------------------------------------------------------------------

def test_homogeneity_violation(prof):
    vm = _vm_new_winner_when_large(4)  # 2 voters -> [0]; doubled -> 4 voters -> [0,1]
    assert homogeneity.has_violation(prof, vm) is True
    assert homogeneity.find_all_violations(prof, vm) == [1]

def test_homogeneity_no_violation(prof):
    assert homogeneity.has_violation(prof, borda) is False
    assert homogeneity.find_all_violations(prof, borda) == []

def test_downward_homogeneity_detects_violation(prof):
    # Bug 5.7 regression: the Downward branch body was the bare no-op `violation`
    # instead of `violation = True`, so this ALWAYS returned False before the fix.
    vm = _vm_new_winner_when_large(4)  # loser 1 wins after doubling
    assert downward_homogeneity.has_violation(prof, vm) is True
    assert downward_homogeneity.find_all_violations(prof, vm) == [1]
    # the same scenario is NOT an upward-homogeneity violation (0 stays a winner)
    assert upward_homogeneity.has_violation(prof, vm) is False

def test_downward_homogeneity_no_false_positive(prof):
    assert downward_homogeneity.has_violation(prof, borda) is False

def test_upward_homogeneity_violation(prof):
    vm = _vm_winner_lost_when_large(4)  # winner 1 lost after doubling
    assert upward_homogeneity.has_violation(prof, vm) is True
    assert upward_homogeneity.find_all_violations(prof, vm) == [1]

def test_homogeneity_on_profile_with_ties():
    pwt = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1: 1, 0: 2, 2: 3}], [1, 1],
                          candidates=[0, 1, 2])
    vm = _vm_new_winner_when_large(4)
    assert homogeneity.has_violation(pwt, vm) is True

def test_homogeneity_verbose_runs(prof, capsys):
    downward_homogeneity.has_violation(prof, _vm_new_winner_when_large(4), verbose=True)
    assert "Violation" in capsys.readouterr().out

def test_homogeneity_profile_with_ties_extended_strict_preference():
    # covers the using_extended_strict_preference branch of the ties path
    pwt = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1: 1, 0: 2, 2: 3}], [1, 1],
                          candidates=[0, 1, 2])
    pwt.use_extended_strict_preference()
    assert homogeneity.has_violation(pwt, _vm_new_winner_when_large(4)) is True


# ---------------------------------------------------------------------------
#  Block family (adding a block of all linear orders)
# ---------------------------------------------------------------------------

def test_block_invariance_violation(prof):
    # 3 candidates -> a block adds 6 rankings -> 2 voters becomes 8
    vm = _vm_new_winner_when_large(8)
    assert block_invariance.has_violation(prof, vm) is True
    assert block_invariance.find_all_violations(prof, vm) == [1]

def test_block_invariance_no_violation(prof):
    assert block_invariance.has_violation(prof, borda) is False

def test_downward_block_preservation_violation(prof):
    vm = _vm_new_winner_when_large(8)
    assert downward_block_preservation.has_violation(prof, vm) is True
    assert downward_block_preservation.find_all_violations(prof, vm) == [1]

def test_upward_block_preservation_violation(prof):
    vm = _vm_winner_lost_when_large(8)
    assert upward_block_preservation.has_violation(prof, vm) is True
    assert upward_block_preservation.find_all_violations(prof, vm) == [1]

def test_block_on_profile_with_ties():
    pwt = ProfileWithTies([{0: 1, 1: 2, 2: 3}, {1: 1, 0: 2, 2: 3}], [1, 1],
                          candidates=[0, 1, 2])
    vm = _vm_new_winner_when_large(8)
    assert block_invariance.has_violation(pwt, vm) is True

def test_block_verbose_runs(prof, capsys):
    block_invariance.has_violation(prof, _vm_new_winner_when_large(8), verbose=True)
    assert "Violation" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  Preferential equality
# ---------------------------------------------------------------------------

@pytest.fixture
def pe_prof():
    # several rankings with 0 directly above 1; first_top is sensitive to swaps
    return Profile([[0, 1, 2], [0, 1, 2], [2, 0, 1], [2, 0, 1]])

def test_preferential_equality_violation(pe_prof):
    assert preferential_equality.has_violation(pe_prof, first_top) is True

def test_preferential_equality_find_all(pe_prof):
    violations = preferential_equality.find_all_violations(pe_prof, first_top)
    assert len(violations) > 0
    # each violation is a (prof, prof_I, prof_J) triple
    assert all(len(v) == 3 for v in violations)

def test_preferential_equality_no_violation(pe_prof):
    assert preferential_equality.has_violation(pe_prof, borda) is False

def test_preferential_equality_profile_with_ties():
    pwt = ProfileWithTies(
        [{0: 1, 1: 2, 2: 3}, {0: 1, 1: 2, 2: 3}, {2: 1, 0: 2, 1: 3}, {2: 1, 0: 2, 1: 3}],
        [1, 1, 1, 1], candidates=[0, 1, 2])
    assert preferential_equality.has_violation(pwt, first_top) is True

def test_preferential_equality_yx_path():
    # a profile where candidate 1 sits directly ABOVE candidate 0 exercises the
    # symmetric "yx" half of the function (the pair is iterated as (0, 1))
    yx_prof = Profile([[1, 0, 2], [1, 0, 2], [2, 1, 0], [2, 1, 0]])
    assert preferential_equality.has_violation(yx_prof, first_top) is True
    assert len(preferential_equality.find_all_violations(yx_prof, first_top)) > 0

def test_preferential_equality_verbose_runs(pe_prof, capsys):
    preferential_equality.has_violation(pe_prof, first_top, verbose=True)
    assert "original profile" in capsys.readouterr().out

def test_preferential_equality_find_all_verbose_runs(pe_prof, capsys):
    preferential_equality.find_all_violations(pe_prof, first_top, verbose=True)
    assert "original profile" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  Tiebreaking compensation
# ---------------------------------------------------------------------------

@pytest.fixture
def tc_prof():
    # two ballots that tie {0, 1} at the top
    return ProfileWithTies([{0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 2}], [1, 1],
                           candidates=[0, 1, 2])

def test_tiebreaking_compensation_violation(tc_prof):
    assert tiebreaking_compensation.has_violation(tc_prof, first_top) is True

def test_tiebreaking_compensation_find_all(tc_prof):
    assert len(tiebreaking_compensation.find_all_violations(tc_prof, first_top)) > 0

def test_tiebreaking_compensation_no_violation(tc_prof):
    assert tiebreaking_compensation.has_violation(tc_prof, borda) is False

def test_tiebreaking_compensation_profile_returns_false():
    # for a (linear) Profile there are no ties to break -> always False
    assert tiebreaking_compensation.has_violation(Profile([[0, 1, 2]]), first_top) is False
    assert tiebreaking_compensation.find_all_violations(Profile([[0, 1, 2]]), first_top) == []

def test_tiebreaking_compensation_verbose_runs(tc_prof, capsys):
    tiebreaking_compensation.has_violation(tc_prof, first_top, verbose=True)
    assert "After breaking the tie" in capsys.readouterr().out

def test_tiebreaking_compensation_find_all_verbose_runs(tc_prof, capsys):
    tiebreaking_compensation.find_all_violations(tc_prof, first_top, verbose=True)
    assert "After breaking the tie" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  Module registry
# ---------------------------------------------------------------------------

def test_invariance_axioms_list():
    names = {ax.name for ax in invariance_axioms}
    assert {"Homogeneity", "Downward Homogeneity", "Block Invariance",
            "Preferential Equality", "Tiebreaking Compensation"}.issubset(names)
