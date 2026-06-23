import numpy as np
import pytest
from pref_voting.spatial_profiles import SpatialProfile
from pref_voting.utility_profiles import UtilityProfile
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Setup sample data
cand_pos = {1: np.array([0.1, 0.2]), 2: np.array([0.3, 0.4])}
voter_pos = {1: np.array([0.5, 0.6]), 2: np.array([0.7, 0.8])}

@pytest.fixture
def sample_profile():
    """Fixture to create a sample SpatialProfile instance for tests."""
    return SpatialProfile(cand_pos, voter_pos)

def test_init_with_valid_data():
    cand_pos = {1: np.array([0.1, 0.2]), 2: np.array([0.3, 0.4])}
    voter_pos = {1: np.array([0.5, 0.6]), 2: np.array([0.7, 0.8])}
    sp = SpatialProfile(cand_pos, voter_pos)
    assert sp.candidates == sorted(cand_pos.keys()), "Candidates not initialized correctly."
    assert sp.voters == sorted(voter_pos.keys()), "Voters not initialized correctly."
    assert sp.cand_pos == cand_pos, "Candidate positions not stored correctly."
    assert sp.voter_pos == voter_pos, "Voter positions not stored correctly."
    assert sp.num_dims == 2, "Number of dimensions not calculated correctly."

def test_init_with_unequal_dimensions():
    cand_pos = {1: np.array([0.1, 0.2]), 2: np.array([0.3, 0.4, 0.5])}  # Different dimensions
    voter_pos = {1: np.array([0.5, 0.6]), 2: np.array([0.7, 0.8])}
    with pytest.raises(AssertionError, match="All candidate positions must have the same number of dimensions."):
        SpatialProfile(cand_pos, voter_pos)

def test_init_with_no_candidates():
    cand_pos = {}
    voter_pos = {1: np.array([0.5, 0.6]), 2: np.array([0.7, 0.8])}
    with pytest.raises(AssertionError, match="There must be at least one candidate."):
        SpatialProfile(cand_pos, voter_pos)

def test_init_with_no_voters():
    cand_pos = {1: np.array([0.1, 0.2]), 2: np.array([0.3, 0.4])}
    voter_pos = {}
    with pytest.raises(AssertionError, match="There must be at least one voter."):
        SpatialProfile(cand_pos, voter_pos)

def test_init_with_different_dimensions_between_candidates_and_voters():
    cand_pos = {1: np.array([0.1, 0.2, 0.3])}  # 3 dimensions
    voter_pos = {1: np.array([0.5, 0.6])}  # 2 dimensions
    with pytest.raises(AssertionError, match="Candidate and voter positions must have the same number of dimensions."):
        SpatialProfile(cand_pos, voter_pos)

# Testing method for voter_position
def test_voter_position_with_valid_voter(sample_profile):
    for voter, pos in voter_pos.items():
        np.testing.assert_array_equal(sample_profile.voter_position(voter), pos, err_msg=f"Incorrect position for voter {voter}")

# Testing method for candidate_position
def test_candidate_position_with_valid_candidate(sample_profile):
    for candidate, pos in cand_pos.items():
        np.testing.assert_array_equal(sample_profile.candidate_position(candidate), pos, err_msg=f"Incorrect position for candidate {candidate}")

def test_to_utility_profile_with_default_function(sample_profile):
    up = sample_profile.to_utility_profile()
    assert type(up) == UtilityProfile

def test_to_utility_profile_with_defined_function(sample_profile):
    def util_fnc(v1, v2):
        return 10 * np.linalg.norm(v1 - v2)
    up = sample_profile.to_utility_profile(utility_function = util_fnc)
    assert type(up) == UtilityProfile

# Testing write method
def test_to_string(sample_profile):
    expected_string = "C-1:0.1,0.2_C-2:0.3,0.4_V-1:0.5,0.6_V-2:0.7,0.8"  
    assert sample_profile.to_string() == expected_string, "The output string does not match the expected format."

# Testing from_string class method
def test_from_string():
    input_string = "C-1:0.1,0.2_C-2:0.3,0.4_V-1:0.5,0.6_V-2:0.7,0.8" 
    sp = SpatialProfile.from_string(input_string)
    assert sp.candidates == sorted(cand_pos.keys()), "Candidates not initialized correctly."
    assert sp.voters == sorted(voter_pos.keys()), "Voters not initialized correctly."
    for voter, pos in voter_pos.items():
        np.testing.assert_array_equal(sp.voter_position(voter), pos, err_msg=f"Incorrect position for voter {voter}")
    for candidate, pos in cand_pos.items():
        np.testing.assert_array_equal(sp.candidate_position(candidate), pos, err_msg=f"Incorrect position for candidate {candidate}")

def test_to_from_string(sample_profile):
    str = sample_profile.to_string()
    sp = SpatialProfile.from_string(str)
    for voter, pos in voter_pos.items():
        np.testing.assert_array_equal(sp.voter_position(voter), pos, err_msg=f"Incorrect position for voter {voter}")
    for candidate, pos in cand_pos.items():
        np.testing.assert_array_equal(sp.candidate_position(candidate), pos, err_msg=f"Incorrect position for candidate {candidate}")

def test_view_does_not_raise_errors_with_valid_data(sample_profile):
    sample_profile.view() 
    sample_profile.view(show_cand_labels=True, show_voter_labels=True) 

    sp2 = SpatialProfile({0:[0.1], 1:[0.2]}, {0:[0.3], 1:[0.75], 2:[0.55]})
    sp2.view(show_cand_labels=True, show_voter_labels=True) 

    sp2 = SpatialProfile({0:[0.1, 0.2, 0.3], 1:[0.2, 0.25, 0.35]}, {0:[0.1, 0.2, 0.3], 1:[0.25, 0.75, 0.25], 2:[0.1, 0.85, 0.55]})
    sp2.view()
    # 3D view with labels exercises the 3D voter/candidate label branches
    sp2.view(show_cand_labels=True, show_voter_labels=True)

def test_display_runs_without_errors(sample_profile, capsys):
    try:
        sample_profile.display()
    except Exception as e:
        pytest.fail(f"Display method raised an unexpected exception: {e}")
    captured = capsys.readouterr()
    assert captured.out, "Expected some output from display method, but got nothing."


# ---------------------------------------------------------------------------
#  Counts, candidate types, candidate mutation
# ---------------------------------------------------------------------------

def test_num_cands_and_num_voters(sample_profile):
    assert sample_profile.num_cands == 2
    assert sample_profile.num_voters == 2

def test_candidate_type_defaults_to_unknown(sample_profile):
    for c in sample_profile.candidates:
        assert sample_profile.candidate_type(c) == "unknown"

def test_set_candidate_types(sample_profile):
    sample_profile.set_candidate_types({1: "left", 2: "right"})
    assert sample_profile.candidate_type(1) == "left"
    assert sample_profile.candidate_type(2) == "right"

def test_set_candidate_types_requires_all_candidates(sample_profile):
    with pytest.raises(AssertionError, match="must be specified for all candidates"):
        sample_profile.set_candidate_types({1: "left"})

def test_add_candidate_single():
    # use 0-indexed candidates so the new name (= num_cands) does not collide
    sp = SpatialProfile({0: np.array([0.1, 0.2]), 1: np.array([0.3, 0.4])},
                        {0: np.array([0.5, 0.6])})
    sp.add_candidate([0.9, 0.9])
    assert sp.candidates == [0, 1, 2]
    np.testing.assert_array_equal(sp.candidate_position(2), [0.9, 0.9])
    assert sp.num_cands == 3

def test_add_candidate_multiple():
    sp = SpatialProfile({0: np.array([0.1, 0.2]), 1: np.array([0.3, 0.4])},
                        {0: np.array([0.5, 0.6])})
    sp.add_candidate([[0.5, 0.5], [0.6, 0.6]], add_multiple_candidates=True)
    assert sp.candidates == [0, 1, 2, 3]
    np.testing.assert_array_equal(sp.candidate_position(2), [0.5, 0.5])
    np.testing.assert_array_equal(sp.candidate_position(3), [0.6, 0.6])

def test_add_candidate_wrong_dimension():
    sp = SpatialProfile({0: np.array([0.1, 0.2])}, {0: np.array([0.5, 0.6])})
    with pytest.raises(AssertionError):
        sp.add_candidate([0.9, 0.9, 0.9])  # 3 dims into a 2-dim profile

def test_move_candidate(sample_profile):
    sample_profile.move_candidate(1, [0.0, 0.0])
    np.testing.assert_array_equal(sample_profile.candidate_position(1), [0.0, 0.0])

def test_move_candidate_wrong_dimension(sample_profile):
    with pytest.raises(AssertionError):
        sample_profile.move_candidate(1, [0.0, 0.0, 0.0])

def test_move_candidate_unknown_candidate(sample_profile):
    with pytest.raises(AssertionError, match="is not in the profile"):
        sample_profile.move_candidate(99, [0.0, 0.0])


# ---------------------------------------------------------------------------
#  to_utility_profile with an uncertainty function (the stochastic branch)
# ---------------------------------------------------------------------------

def test_to_utility_profile_with_uncertainty(sample_profile):
    # uncertainty_function returns (std, rho) consumed by generate_covariance
    def uncertainty(prof, c, v):
        return (0.1, 0.0)

    np.random.seed(0)
    up = sample_profile.to_utility_profile(uncertainty_function=uncertainty)
    assert type(up) == UtilityProfile

    up2, virtual_positions = sample_profile.to_utility_profile(
        uncertainty_function=uncertainty, return_virtual_cand_positions=True)
    assert type(up2) == UtilityProfile
    assert set(virtual_positions.keys()) == set(sample_profile.candidates)

def test_to_utility_profile_with_uncertainty_batch(sample_profile):
    def uncertainty(prof, c, v):
        return (0.1, 0.0)

    np.random.seed(0)
    up = sample_profile.to_utility_profile(uncertainty_function=uncertainty, batch=True)
    assert type(up) == UtilityProfile
