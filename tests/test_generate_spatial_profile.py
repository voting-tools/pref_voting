import pytest
import numpy as np
from pref_voting.spatial_profiles import SpatialProfile
from numpy.testing import assert_array_almost_equal
from pref_voting.generate_spatial_profiles import generate_covariance, generate_spatial_profile, generate_spatial_profile_polarized, generate_spatial_profile_polarized_cands_randomly_polarized_voters


def test_generate_covariance_basic():
    cov = generate_covariance(n_dimensions=2, std=1, rho=0.5)
    expected_cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert_array_almost_equal(cov, expected_cov, decimal=5)

def test_generate_covariance_invalid_std():
    with pytest.raises(AssertionError):
        generate_covariance(n_dimensions=2, std=-1, rho=0.5)

def test_generate_covariance_invalid_rho():
    with pytest.raises(AssertionError):
        generate_covariance(n_dimensions=2, std=1, rho=-0.1)

def test_generate_covariance_invalid_n_dimensions():
    with pytest.raises(AssertionError):
        generate_covariance(n_dimensions=0, std=1, rho=0.5)

def test_generate_spatial_profile_dimensions():
    profile = generate_spatial_profile(num_cands=10, num_voters=20, num_dims=3)
    assert type(profile) == SpatialProfile
    assert profile.num_dims == 3
    assert profile.candidates == list(range(10))
    assert profile.voters == list(range(20))

def test_generate_spatial_profile_custom_covariance():
    cov = np.eye(3) * 2
    profile = generate_spatial_profile(num_cands=5, num_voters=5, num_dims=3, cand_cov=cov, voter_cov=cov)
    assert type(profile) == SpatialProfile
    assert profile.num_dims == 3
    assert profile.candidates == list(range(5))
    assert profile.voters == list(range(5))


def test_generate_spatial_profile_polarized_structure():
    cand_clusters = [([0, 0], np.eye(2), 5), ([10, 10], np.eye(2), 5)]
    voter_clusters = [([5, 5], np.eye(2), 10)]
    profile = generate_spatial_profile_polarized(cand_clusters, voter_clusters)
    assert len(profile.cand_pos) == 10  
    assert len(profile.voter_pos) == 10

def test_generate_spatial_profile_polarized_cands_randomly_polarized_voters_structure():
    cand_clusters = [([0, 0], np.eye(2), 5), ([10, 10], np.eye(2), 5)]
    voter_distributions = [([5, 5], np.eye(2), 0.75), ([15, 15], np.eye(2), 0.25)]
    profile = generate_spatial_profile_polarized_cands_randomly_polarized_voters(cand_clusters, num_voters=20, voter_distributions=voter_distributions)
    assert len(profile.cand_pos) == 10  # 5 candidates from each cluster
    assert len(profile.voter_pos) == 20
