"""
Tests for the spatial-profile-from-binned-distribution prototype.

Runnable two ways:
    pytest test_generate_spatial_profile_from_binned_distribution.py
    python test_generate_spatial_profile_from_binned_distribution.py   # standalone, no pytest
"""

from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: no display needed for the plot smoke tests

from pref_voting.generate_spatial_profile_from_binned_distribution import (
    BinnedDistribution,
)
from pref_voting.generate_spatial_profile_from_binned_distribution import (
    generate_spatial_profile_from_binned_distribution as gen,
)
from pref_voting.spatial_profiles import SpatialProfile
from pref_voting.utility_functions import linear_utility

# ── helpers ─────────────────────────────────────────────────────────────────


def uniform5():
    """A 1D 5-bin distribution on [-0.5, 0.5] with atkinson-style regions."""
    return BinnedDistribution.from_binned(
        [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5],
        [1, 1, 1, 1, 1],
        regions={"left": [0, 1], "centrist": [2], "right": [3, 4]},
    )


def quad2d():
    """A 2D distribution: a 2x2 grid of unit boxes over [-1, 1]^2, with the four
    quadrants as regions."""
    lows = [[-1, -1], [0, -1], [-1, 0], [0, 0]]
    highs = [[0, 0], [1, 0], [0, 1], [1, 1]]
    return BinnedDistribution.from_boxes(
        lows, highs, [1, 1, 1, 1], regions={"ll": [0], "lr": [1], "ul": [2], "ur": [3]}
    )


def points_in_region(vd, region, pts):
    """Boolean mask: which of ``pts`` (shape (n, num_dims)) lie in the region's bins."""
    pts = np.atleast_2d(pts)
    inside = np.zeros(len(pts), dtype=bool)
    for b in vd.regions[region]:
        inside |= np.all((pts >= vd.bin_lows[b]) & (pts <= vd.bin_highs[b]), axis=1)
    return inside


def cand_pos(sp, c):
    return np.asarray(sp.candidate_position(c), dtype=float)


# ── BinnedDistribution ──────────────────────────────────────────────────────


def test_from_binned_basics():
    vd = uniform5()
    assert vd.num_dims == 1
    assert vd.num_bins == 5
    assert np.allclose(vd.support, [[-0.5, 0.5]])
    pts = vd.sample(10000, np.random.default_rng(0))
    assert pts.shape == (10000, 1)
    assert pts.min() >= -0.5 and pts.max() <= 0.5


def test_from_binned_normalizes_probs():
    vd = BinnedDistribution.from_binned([0, 1, 2], [2, 2])  # -> [0.5, 0.5]
    assert np.allclose(vd.bin_probs, [0.5, 0.5])


def test_from_ces_loads_with_default_regions():
    vd = BinnedDistribution.from_ces("AK", source="atkinson")
    assert vd.num_bins == 5
    assert vd.regions == {"left": [0, 1], "centrist": [2], "right": [3, 4]}
    vd7 = BinnedDistribution.from_ces("AK", source="mccune")
    assert vd7.num_bins == 7
    assert vd7.regions == {"left": [0, 1, 2], "centrist": [3], "right": [4, 5, 6]}


def test_sample_in_region_stays_in_region():
    vd = uniform5()
    rng = np.random.default_rng(0)
    for region in ("left", "centrist", "right"):
        pts = vd.sample_in_region(region, 5000, rng)
        assert points_in_region(vd, region, pts).all()


def test_sample_in_region_zero_mass_still_populates():
    # 'left' bins carry no probability mass, but the region can still be sampled
    vd = BinnedDistribution.from_binned(
        [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5],
        [0, 0, 1, 1, 1],
        regions={"left": [0, 1], "centrist": [2], "right": [3, 4]},
    )
    pts = vd.sample_in_region("left", 1000, np.random.default_rng(0))
    assert points_in_region(vd, "left", pts).all()


def test_sample_in_region_unknown_raises():
    vd = uniform5()
    try:
        vd.sample_in_region("up", 1, np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown region should raise ValueError")


# ── 2D distributions work the same way ───────────────────────────────────────


def test_2d_distribution_random_and_structured():
    vd = quad2d()
    assert vd.num_dims == 2 and vd.num_bins == 4
    assert np.allclose(vd.support, [[-1, 1], [-1, 1]])

    # random model: 2D positions
    sp = gen(4, 60, vd, seed=0)
    assert sp.num_dims == 2
    assert np.asarray(sp.voter_position(sp.voters[0])).shape == (2,)

    # structured by quadrant
    sp = gen(4, 60, vd, candidate_counts={"ll": 1, "lr": 1, "ul": 1, "ur": 1}, seed=0)
    assert sorted(sp.candidate_type(c) for c in sp.candidates) == [
        "ll",
        "lr",
        "ul",
        "ur",
    ]
    for c in sp.candidates:
        assert points_in_region(vd, sp.candidate_type(c), cand_pos(sp, c)).all()


# ── random candidate model ───────────────────────────────────────────────────


def test_random_model_shape_and_types():
    sp = gen(4, 101, uniform5(), seed=0)
    assert isinstance(sp, SpatialProfile)
    assert sp.num_cands == 4 and sp.num_voters == 101
    assert sp.num_dims == 1
    # no structure -> candidate_types default to 'unknown'
    assert all(sp.candidate_type(c) == "unknown" for c in sp.candidates)


def test_num_profiles_returns_list():
    sps = gen(3, 50, uniform5(), num_profiles=5, seed=0)
    assert isinstance(sps, list) and len(sps) == 5
    assert all(isinstance(sp, SpatialProfile) for sp in sps)


# ── structured candidate model: exact counts ─────────────────────────────────


def test_structured_counts_exact_composition_and_positions():
    vd = uniform5()
    sp = gen(
        4, 101, vd, candidate_counts={"centrist": 1, "left": 2, "right": 1}, seed=1
    )
    types = [sp.candidate_type(c) for c in sp.candidates]
    assert sorted(types) == ["centrist", "left", "left", "right"]
    for c in sp.candidates:
        assert points_in_region(vd, sp.candidate_type(c), cand_pos(sp, c)).all()


def test_structured_counts_must_sum_to_num_cands():
    try:
        gen(4, 10, uniform5(), candidate_counts={"left": 1, "right": 1}, seed=0)
    except AssertionError:
        pass
    else:
        raise AssertionError("counts not summing to num_cands should raise")


def test_structured_region_prob_independent_of_mass():
    # 'left' holds almost no mass, but a count of 2 still places 2 left candidates there
    vd = BinnedDistribution.from_binned(
        [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5],
        [0.001, 0.001, 0.5, 0.3, 0.198],
        regions={"left": [0, 1], "centrist": [2], "right": [3, 4]},
    )
    sp = gen(3, 10, vd, candidate_counts={"left": 2, "centrist": 1}, seed=0)
    lefties = [c for c in sp.candidates if sp.candidate_type(c) == "left"]
    assert len(lefties) == 2
    for c in lefties:
        assert points_in_region(vd, "left", cand_pos(sp, c)).all()


# ── structured candidate model: probabilities ────────────────────────────────


def test_structured_probs_positions_in_region_and_freqs():
    vd = uniform5()
    probs = {"centrist": 0.5, "left": 0.25, "right": 0.25}
    # over many candidates, type frequencies track the given probs (not the mass)
    sp = gen(4000, 10, vd, candidate_type_probs=probs, seed=3)
    freq = Counter(sp.candidate_type(c) for c in sp.candidates)
    assert abs(freq["centrist"] / 4000 - 0.5) < 0.03
    assert abs(freq["left"] / 4000 - 0.25) < 0.03
    for c in sp.candidates:
        assert points_in_region(vd, sp.candidate_type(c), cand_pos(sp, c)).all()


def test_structured_probs_must_sum_to_one():
    try:
        gen(4, 10, uniform5(), candidate_type_probs={"left": 0.5, "right": 0.2}, seed=0)
    except AssertionError:
        pass
    else:
        raise AssertionError("probs not summing to 1 should raise")


# ── error handling ───────────────────────────────────────────────────────────


def test_both_counts_and_probs_raises():
    try:
        gen(
            4,
            10,
            uniform5(),
            candidate_counts={"left": 4},
            candidate_type_probs={"left": 1.0},
            seed=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("giving both counts and probs should raise")


def test_structured_without_regions_raises():
    plain = BinnedDistribution.from_binned([0, 1, 2], [1, 1])  # no regions
    try:
        gen(2, 10, plain, candidate_counts={"left": 2}, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("structured model without regions should raise")


def test_unknown_region_name_raises():
    try:
        gen(2, 10, uniform5(), candidate_counts={"up": 2}, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown region name should raise")


# ── randomness conventions ───────────────────────────────────────────────────


def test_seed_is_reproducible():
    a = gen(4, 200, uniform5(), candidate_counts={"left": 2, "right": 2}, seed=42)
    b = gen(4, 200, uniform5(), candidate_counts={"left": 2, "right": 2}, seed=42)
    assert np.allclose(
        [cand_pos(a, c) for c in a.candidates], [cand_pos(b, c) for c in b.candidates]
    )
    va = np.array([a.voter_position(v) for v in a.voters])
    vb = np.array([b.voter_position(v) for v in b.voters])
    assert np.allclose(va, vb)


def test_rng_threads_across_calls():
    # threading one rng gives independent (different) profiles; seeding it makes the
    # whole sequence reproducible
    def two(seed):
        rng = np.random.default_rng(seed)
        return [gen(4, 100, uniform5(), rng=rng) for _ in range(2)]

    p1, p2 = two(7)
    assert not np.allclose(
        [cand_pos(p1, c) for c in p1.candidates],
        [cand_pos(p2, c) for c in p2.candidates],
    )
    q1, _ = two(7)  # same seed -> identical sequence
    assert np.allclose(
        [cand_pos(p1, c) for c in p1.candidates],
        [cand_pos(q1, c) for c in q1.candidates],
    )


# ── integration ──────────────────────────────────────────────────────────────


def test_plugs_into_utility_pipeline():
    sp = gen(
        4,
        201,
        BinnedDistribution.from_ces("AK", source="mccune"),
        candidate_type_probs={"centrist": 0.5, "left": 0.25, "right": 0.25},
        seed=0,
    )
    uprof = sp.to_utility_profile(utility_function=linear_utility)
    prof = uprof.to_ranking_profile()
    assert prof.num_cands == 4
    assert prof.num_voters == 201


# ── plotting (smoke tests) ───────────────────────────────────────────────────


def test_plot_1d_returns_axes():
    import matplotlib.pyplot as plt

    ax = uniform5().plot()
    assert ax is not None
    assert ax.get_xlabel() == "position" and ax.get_ylabel() == "density"
    plt.close("all")


def test_plot_1d_color_by_region_has_legend():
    import matplotlib.pyplot as plt

    ax = uniform5().plot(color_by_region=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(labels) == {"left", "centrist", "right"}
    plt.close("all")


def test_plot_2d_density_and_regions():
    import matplotlib.pyplot as plt

    quad2d().plot()  # heatmap + colorbar
    ax = quad2d().plot(color_by_region=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(labels) == {"ll", "lr", "ul", "ur"}
    plt.close("all")


def test_plot_3d_raises():
    vd = BinnedDistribution.from_boxes([[0, 0, 0]], [[1, 1, 1]], [1.0])  # 3D, one bin
    try:
        vd.plot()
    except ValueError:
        pass
    else:
        raise AssertionError("plot of a >2D distribution should raise ValueError")


# ── standalone runner ───────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
