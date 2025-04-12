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

def test_display_runs_without_errors(sample_profile, capsys):
    try:
        sample_profile.display()
    except Exception as e:
        pytest.fail(f"Display method raised an unexpected exception: {e}")
    captured = capsys.readouterr()
    assert captured.out, "Expected some output from display method, but got nothing."
