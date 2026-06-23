import numpy as np
import pytest

from pref_voting.generate_utility_profiles import (
    generate_utility_profile_uniform,
    generate_utility_profile_normal,
    generate_spatial_utility_profile,
)
from pref_voting.utility_profiles import UtilityProfile


# ---------------------------------------------------------------------------
#  generate_utility_profile_uniform
# ---------------------------------------------------------------------------

def test_uniform_single_profile():
    np.random.seed(0)
    up = generate_utility_profile_uniform(3, 4)
    assert isinstance(up, UtilityProfile)
    assert up.num_cands == 3
    assert up.num_voters == 4
    # uniform utilities lie in [0, 1)
    for u in up.utilities:
        for c in range(3):
            assert 0.0 <= u(c) < 1.0

def test_uniform_multiple_profiles():
    np.random.seed(0)
    ups = generate_utility_profile_uniform(3, 4, num_profiles=3)
    assert isinstance(ups, list)
    assert len(ups) == 3
    assert all(isinstance(up, UtilityProfile) for up in ups)
    assert all(up.num_cands == 3 and up.num_voters == 4 for up in ups)


# ---------------------------------------------------------------------------
#  generate_utility_profile_normal
# ---------------------------------------------------------------------------

def test_normal_default():
    np.random.seed(1)
    up = generate_utility_profile_normal(3, 5)
    assert isinstance(up, UtilityProfile)
    assert up.num_cands == 3
    assert up.num_voters == 5

def test_normal_normalize_range():
    np.random.seed(1)
    up = generate_utility_profile_normal(3, 5, normalize="range")
    assert isinstance(up, UtilityProfile)
    # range-normalized: every voter's utilities lie in [0, 1] with both endpoints hit
    for u in up.utilities:
        vals = [u(c) for c in range(3)]
        assert min(vals) == pytest.approx(0.0)
        assert max(vals) == pytest.approx(1.0)

def test_normal_normalize_score():
    np.random.seed(1)
    up = generate_utility_profile_normal(3, 5, normalize="score")
    assert isinstance(up, UtilityProfile)
    # standard-score normalized: each voter's utilities have mean ~0
    for u in up.utilities:
        assert np.mean([u(c) for c in range(3)]) == pytest.approx(0.0, abs=1e-9)

def test_normal_multiple_profiles():
    np.random.seed(2)
    ups = generate_utility_profile_normal(3, 4, num_profiles=2)
    assert isinstance(ups, list)
    assert len(ups) == 2
    assert all(isinstance(up, UtilityProfile) for up in ups)


# ---------------------------------------------------------------------------
#  generate_spatial_utility_profile
# ---------------------------------------------------------------------------

ALL_UTILITY_FUNCTIONS = ["RM", "Linear", "Quadratic", "Shepsle", "City Block", "Matthews"]

@pytest.mark.parametrize("uf", ALL_UTILITY_FUNCTIONS)
def test_spatial_all_utility_functions_run(uf):
    # Bug 6.1: the parametrized branch did partial(func, util_parm), which PREPENDS
    # the parameter as the first positional arg (v_pos), so RM (param=1) and any
    # explicit-param call raised a numba TypingError. The parameter must be passed
    # AFTER the positions: _utility_fnc(v_pos, c_pos, util_parm). This generates
    # under every model, which fails on the buggy code (at least for RM).
    np.random.seed(0)
    up = generate_spatial_utility_profile(3, 4, num_dims=2, utility_function=uf)
    assert isinstance(up, UtilityProfile)
    assert up.num_cands == 3
    assert up.num_voters == 4

def test_spatial_param_actually_affects_output():
    # the explicit parameter must be applied (passed as beta), not ignored
    np.random.seed(0)
    up1 = generate_spatial_utility_profile(3, 5, utility_function="RM", utility_function_param=0.0)
    np.random.seed(0)
    up2 = generate_spatial_utility_profile(3, 5, utility_function="RM", utility_function_param=1.0)
    vals1 = [float(up1._utilities[0](c)) for c in range(3)]
    vals2 = [float(up2._utilities[0](c)) for c in range(3)]
    assert vals1 != vals2, "RM parameter had no effect -> not being applied"

def test_spatial_paramless_function_uses_no_param_branch():
    # Quadratic has param None and no explicit param -> the else branch (utility_fnc
    # = _utility_fnc directly, no lambda wrapper)
    np.random.seed(3)
    up = generate_spatial_utility_profile(3, 4, utility_function="Quadratic")
    assert isinstance(up, UtilityProfile)
    assert up.num_cands == 3

def test_spatial_one_dimension():
    np.random.seed(4)
    up = generate_spatial_utility_profile(3, 4, num_dims=1, utility_function="Quadratic")
    assert isinstance(up, UtilityProfile)
    assert up.num_cands == 3
    assert up.num_voters == 4
