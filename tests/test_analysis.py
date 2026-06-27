import numpy as np
import pandas as pd
import pytest

import pref_voting.analysis as analysis_mod
from pref_voting.analysis import (
    axiom_violations_data,
    binomial_confidence_interval,
    condorcet_efficiency_data,
    estimated_std_error,
    estimated_variance_of_sampling_dist,
    find_profiles_with_different_winners,
    means_with_estimated_standard_error,
    resoluteness_data,
    social_utility_performance,
)
from pref_voting.monotonicity_axioms import monotonicity
from pref_voting.profiles import Profile
from pref_voting.scoring_methods import borda, plurality
from pref_voting.utility_profiles import UtilityProfile
from pref_voting.voting_method import VotingMethod

# ---------------------------------------------------------------------------
#  find_profiles_with_different_winners  (Bug 8.1)
#
#  These monkeypatch generate_profile to return a fixed profile and use stub VMs
#  with fixed winners, so the search is deterministic and fast (no randomness).
# ---------------------------------------------------------------------------


def test_find_profiles_detects_disagreement_when_set_order_differs(monkeypatch):
    """Bug 8.1: 'all winning sets distinct' was tested as list(set(wss)) == list(wss),
    which depends on set iteration order. With winners [1] and [0],
    wss = [(1,), (0,)] -> list(set(wss)) may be [(0,), (1,)] != [(1,), (0,)], so the
    buggy check REJECTS a genuinely-disagreeing profile. Fixed check len(set)==len keeps it."""
    m1 = VotingMethod(lambda edata, curr_cands=None: [1], name="M1")
    m2 = VotingMethod(lambda edata, curr_cands=None: [0], name="M2")

    wss = [(1,), (0,)]
    assert list(set(wss)) != list(wss)  # the exact order-dependent trap
    assert len(set(wss)) == len(wss)  # genuinely distinct

    fixed_prof = Profile([[0, 1], [1, 0]], [1, 1])
    monkeypatch.setattr(analysis_mod, "generate_profile", lambda *a, **k: fixed_prof)

    profs = find_profiles_with_different_winners(
        [m1, m2],
        numbers_of_candidates=[2],
        numbers_of_voters=[2],
        show_profiles=False,
        show_margin_graphs=False,
        show_winning_sets=False,
        num_trials=1,
    )
    assert len(profs) == 1


def test_find_profiles_rejects_identical_winners(monkeypatch):
    m1 = VotingMethod(lambda edata, curr_cands=None: [0], name="M1")
    m2 = VotingMethod(lambda edata, curr_cands=None: [0], name="M2")
    fixed_prof = Profile([[0, 1], [1, 0]], [1, 1])
    monkeypatch.setattr(analysis_mod, "generate_profile", lambda *a, **k: fixed_prof)
    profs = find_profiles_with_different_winners(
        [m1, m2],
        numbers_of_candidates=[2],
        numbers_of_voters=[2],
        show_profiles=False,
        show_margin_graphs=False,
        show_winning_sets=False,
        num_trials=1,
    )
    assert len(profs) == 0


def test_find_profiles_return_single(monkeypatch):
    # return_multiple_profiles=False returns the first matching profile, not a list
    m1 = VotingMethod(lambda edata, curr_cands=None: [1], name="M1")
    m2 = VotingMethod(lambda edata, curr_cands=None: [0], name="M2")
    fixed_prof = Profile([[0, 1], [1, 0]], [1, 1])
    monkeypatch.setattr(analysis_mod, "generate_profile", lambda *a, **k: fixed_prof)
    result = find_profiles_with_different_winners(
        [m1, m2],
        numbers_of_candidates=[2],
        numbers_of_voters=[2],
        show_profiles=False,
        show_margin_graphs=False,
        show_winning_sets=False,
        return_multiple_profiles=False,
        num_trials=1,
    )
    assert isinstance(result, Profile)


def test_find_profiles_with_display_options(monkeypatch, capsys):
    # exercise the show_* display branches (Agg backend keeps plotting headless)
    m1 = VotingMethod(lambda edata, curr_cands=None: [1], name="M1")
    m2 = VotingMethod(lambda edata, curr_cands=None: [0], name="M2")
    fixed_prof = Profile([[0, 1], [1, 0]], [1, 1])
    monkeypatch.setattr(analysis_mod, "generate_profile", lambda *a, **k: fixed_prof)
    profs = find_profiles_with_different_winners(
        [m1, m2],
        numbers_of_candidates=[2],
        numbers_of_voters=[2],
        show_profiles=True,
        show_margin_graphs=True,
        show_winning_sets=True,
        show_rankings_counts=True,
        num_trials=1,
    )
    assert len(profs) == 1


def test_find_profiles_all_unique_winners_filter(monkeypatch):
    # all_unique_winners=True requires every method to have a singleton winner;
    # a tie ([0,1]) disqualifies the profile
    m1 = VotingMethod(lambda edata, curr_cands=None: [0, 1], name="M1")
    m2 = VotingMethod(lambda edata, curr_cands=None: [0], name="M2")
    fixed_prof = Profile([[0, 1], [1, 0]], [1, 1])
    monkeypatch.setattr(analysis_mod, "generate_profile", lambda *a, **k: fixed_prof)
    profs = find_profiles_with_different_winners(
        [m1, m2],
        numbers_of_candidates=[2],
        numbers_of_voters=[2],
        all_unique_winners=True,
        show_profiles=False,
        show_margin_graphs=False,
        show_winning_sets=False,
        num_trials=1,
    )
    assert profs == []


# ---------------------------------------------------------------------------
#  Statistical helpers (pure, fast)
# ---------------------------------------------------------------------------


def test_estimated_variance_of_sampling_dist():
    # row [1,2,3,4,5]: sample var (ddof=1) = 2.5; /n=5 -> 0.5
    # row [2,2,2,2, nan]: nan excluded, zero spread -> 0.0
    arr = np.array([[1.0, 2, 3, 4, 5], [2.0, 2, 2, 2, np.nan]])
    out = estimated_variance_of_sampling_dist(arr)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.0)


def test_estimated_variance_single_value_is_nan():
    # n=1 after dropping nan -> n*(n-1)=0 -> nan (no crash)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = estimated_variance_of_sampling_dist(np.array([[5.0, np.nan]]))
    assert np.isnan(out[0])


def test_estimated_variance_with_explicit_mean():
    arr = np.array([[1.0, 2, 3, 4, 5]])
    out = estimated_variance_of_sampling_dist(
        arr, mean_for_each_experiment=np.array([3.0])
    )
    assert out[0] == pytest.approx(0.5)


def test_estimated_std_error():
    out = estimated_std_error(np.array([[1.0, 2, 3, 4, 5]]))
    assert out[0] == pytest.approx(np.sqrt(0.5))


def test_binomial_confidence_interval():
    low, high = binomial_confidence_interval([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])  # 6/10
    assert 0.0 <= low < 0.6 < high <= 1.0
    assert low == pytest.approx(0.2624, abs=1e-3)
    assert high == pytest.approx(0.8784, abs=1e-3)


def test_binomial_confidence_interval_confidence_level():
    # a wider confidence level gives a wider interval
    lo95, hi95 = binomial_confidence_interval([1, 0, 1, 1, 0], confidence_level=0.95)
    lo80, hi80 = binomial_confidence_interval([1, 0, 1, 1, 0], confidence_level=0.80)
    assert (hi95 - lo95) > (hi80 - lo80)


def test_means_with_estimated_standard_error_immediate_convergence():
    # constant samples -> std error 0 -> converges at the first check
    gen = lambda num_samples, step: np.ones((2, num_samples))
    means, errs, variances, num_trials = means_with_estimated_standard_error(
        gen, max_std_error=0.1, initial_trials=10, step_trials=10, min_num_trials=10
    )
    assert list(means) == [1.0, 1.0]
    assert list(errs) == [0.0, 0.0]
    assert list(variances) == [0.0, 0.0]
    assert num_trials == 10


def test_means_with_estimated_standard_error_loops_until_max_num_trials(capsys):
    # an unreachable max_std_error forces the while loop; max_num_trials bounds it.
    # verbose=True also exercises the in-loop progress prints.
    gen = lambda num_samples, step: np.tile([0.0, 1.0], (2, num_samples // 2 + 1))[
        :, :num_samples
    ]
    means, errs, variances, num_trials = means_with_estimated_standard_error(
        gen,
        max_std_error=0.0,
        initial_trials=10,
        step_trials=10,
        min_num_trials=10,
        max_num_trials=30,
        verbose=True,
    )
    assert num_trials == 30
    assert "Number of trials" in capsys.readouterr().out


def test_means_with_estimated_standard_error_verbose(capsys):
    gen = lambda num_samples, step: np.ones((2, num_samples))
    means_with_estimated_standard_error(
        gen,
        max_std_error=0.1,
        initial_trials=10,
        step_trials=10,
        min_num_trials=10,
        verbose=True,
    )
    assert "Initial number of trials" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  Data-collection drivers (run serially with tiny parameters so they are fast)
# ---------------------------------------------------------------------------


def test_condorcet_efficiency_data():
    df = condorcet_efficiency_data(
        [plurality],
        numbers_of_candidates=[3],
        numbers_of_voters=[5],
        min_num_samples=20,
        max_num_samples=20,
        use_parallel=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert set(["voting_method", "condorcet_efficiency", "num_samples"]).issubset(
        df.columns
    )
    assert 0.0 <= df["condorcet_efficiency"].iloc[0] <= 1.0


def test_resoluteness_data():
    df = resoluteness_data(
        [plurality, borda],
        numbers_of_candidates=[3],
        numbers_of_voters=[5],
        num_trials=20,
        use_parallel=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # one row per voting method
    assert df["avg_num_winners"].min() >= 1.0


def test_axiom_violations_data():
    df = axiom_violations_data(
        [monotonicity],
        [plurality],
        numbers_of_candidates=[3],
        numbers_of_voters=[5],
        num_trials=10,
        use_parallel=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["num_violations"].iloc[0] >= 0


# social utilities {0: 1, 1: 0, 2: -1}: mean = 0, max = 1, so the normalizer is 1
social_utilities = {0: 1.0, 1: 0.0, 2: -1.0}


def test_utilitarian_winner_scores_one():
    assert social_utility_performance(social_utilities, [0]) == pytest.approx(1.0)


def test_random_baseline_scores_zero():
    # candidate 1 has exactly the mean social utility
    assert social_utility_performance(social_utilities, [1]) == pytest.approx(0.0)


def test_worse_than_random_is_negative():
    assert social_utility_performance(social_utilities, [2]) == pytest.approx(-1.0)


def test_tied_winners_are_averaged():
    assert social_utility_performance(social_utilities, [0, 1]) == pytest.approx(
        0.5
    )  # mean(1, 0)
    assert social_utility_performance(social_utilities, [0, 2]) == pytest.approx(
        0.0
    )  # mean(1, -1)


def test_probabilistic_winner_is_expected_utility():
    assert social_utility_performance(
        social_utilities, {0: 0.5, 2: 0.5}
    ) == pytest.approx(0.0)
    assert social_utility_performance(
        social_utilities, {0: 0.75, 1: 0.25}
    ) == pytest.approx(0.75)


def test_set_and_uniform_probability_dict_agree():
    assert social_utility_performance(social_utilities, [0, 1, 2]) == pytest.approx(
        social_utility_performance(social_utilities, {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    )


def test_accepts_utility_profile():
    # one voter whose utilities are exactly the social utilities above
    uprof = UtilityProfile([{0: 1.0, 1: 0.0, 2: -1.0}])
    assert social_utility_performance(uprof, [0]) == pytest.approx(1.0)
    assert social_utility_performance(uprof, [2]) == pytest.approx(-1.0)


def test_utility_profile_matches_precomputed_mapping():
    # two voters; social utility = average utility over voters (column means)
    uprof = UtilityProfile([{0: 2.0, 1: 0.0, 2: -1.0}, {0: 0.0, 1: 0.0, 2: 1.0}])
    precomputed = {0: 1.0, 1: 0.0, 2: 0.0}
    for winners in ([0], [1], [2], [0, 1], {0: 0.6, 2: 0.4}):
        assert social_utility_performance(uprof, winners) == pytest.approx(
            social_utility_performance(precomputed, winners)
        )


def test_social_utility_is_voter_weighted():
    # candidate 0 averages (3*0 + 1*4)/4 = 1.0; candidate 1 is constant 0
    uprof = UtilityProfile([{0: 0.0, 1: 0.0}, {0: 4.0, 1: 0.0}], ucounts=[3, 1])
    # social utilities are {0: 1.0, 1: 0.0}: mean = 0.5, max = 1.0, normalizer = 0.5
    assert social_utility_performance(uprof, [0]) == pytest.approx(1.0)
    assert social_utility_performance(uprof, [1]) == pytest.approx(-1.0)


def test_all_equal_social_utility_returns_one():
    # the normalizer is 0 (every candidate optimal); any winner -> 1.0
    assert social_utility_performance({0: 5.0, 1: 5.0, 2: 5.0}, [1]) == 1.0
    assert social_utility_performance({0: 5.0, 1: 5.0, 2: 5.0}, {0: 0.5, 2: 0.5}) == 1.0


def test_invariant_to_constant_shift_of_utilities():
    # adding a constant to all social utilities leaves the score unchanged (shifts cancel)
    shifted = {c: u + 10 for c, u in social_utilities.items()}
    for winners in ([0], [2], [0, 1], {0: 0.5, 1: 0.5}):
        assert social_utility_performance(shifted, winners) == pytest.approx(
            social_utility_performance(social_utilities, winners)
        )
