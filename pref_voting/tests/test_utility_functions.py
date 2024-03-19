import pytest
import numpy as np
from scipy.spatial import distance
from pref_voting.utility_functions import mixed_rm_utility, rm_utility, linear_utility, quadratic_utility, city_block_utility, shepsle_utility, matthews_utility


def test_mixed_rm_utility():
    v_pos = np.array([1.0, 0.0], dtype=np.float32)
    c_pos = np.array([0.0, 1.0], dtype=np.float32)
    result = mixed_rm_utility(v_pos, c_pos, beta=0.75)
    assert isinstance(result, np.floating) or isinstance(result, float)

def test_rm_utility():
    v_pos = np.array([1, 0], dtype=np.float32)
    c_pos = np.array([0, 1], dtype=np.float32)
    result = rm_utility(v_pos, c_pos)
    assert isinstance(result, np.floating) or isinstance(result, float)

def test_linear_utility():
    v_pos = np.array([1, 0], dtype=np.float32)
    c_pos = np.array([0, 1], dtype=np.float32)
    result = linear_utility(v_pos, c_pos)
    assert isinstance(result, np.floating) or isinstance(result, float)

def test_quadratic_utility():
    v_pos = np.array([1, 0], dtype=np.float32)
    c_pos = np.array([0, 1], dtype=np.float32)
    result = quadratic_utility(v_pos, c_pos)
    assert isinstance(result, np.floating) or isinstance(result, float)

def test_city_block_utility():
    v_pos = np.array([1.0, 2.0], dtype=np.float32)
    c_pos = np.array([4.0, 1.0], dtype=np.float32)
    result = city_block_utility(v_pos, c_pos)
    print(result)
    assert isinstance(result, np.floating) or isinstance(result, float)

def test_shepsle_utility():
    v_pos = np.array([0, 0], dtype=np.float32)
    c_pos = np.array([1, 1], dtype=np.float32)
    result = shepsle_utility(v_pos, c_pos)
    assert isinstance(result, np.floating) or isinstance(result, float)

def test_matthews_utility():
    v_pos = np.array([2, 2], dtype=np.float32)
    c_pos = np.array([1, 1], dtype=np.float32)
    result = matthews_utility(v_pos, c_pos)
    assert isinstance(result, np.floating) or isinstance(result, float)
