"""
Spatial profiles from empirical (binned) voter distributions — prototype of a proposed
``pref_voting`` addition, analogous to the generators in
``pref_voting.generate_spatial_profiles``.

Scope: **binned empirical distributions** in any dimension. A distribution is a set of
**bins** (axis-aligned boxes), piecewise-uniform within each bin, whose bins are grouped
into named **regions** (e.g. ``left``/``centrist``/``right``, or quadrants in 2D). Regions
live on the *distribution* because the bin layout already carries the meaning (atkinson
bin 2 *is* "Independent"); a parametric distribution has no natural regions (least of all
in higher dimensions), so it is out of scope here. 1D is just the special case where each
box is an interval; the CES 2020 distributions used by Atkinson, Foley & Gantz and by
McCune et al. are 1D.

Two pieces:

* :class:`BinnedDistribution` — a binned distribution with named regions.
  ``vd.sample(num, rng)`` draws positions; ``vd.sample_in_region(region, num, rng)`` draws
  positions from the distribution *restricted to a region's bins*.

* :func:`generate_spatial_profile_from_binned_distribution` — builds
  :class:`pref_voting.spatial_profiles.SpatialProfile` objects, with two candidate models:

  - **random**: candidate positions are i.i.d. draws from ``cand_dist`` (defaults to the
    voter distribution).
  - **structured**: each candidate's region is chosen by the given counts/probabilities
    (independent of how much probability mass the region holds), then its position is
    drawn from the distribution restricted to that region's bins. No rejection sampling.

Randomness follows the dual ``seed=``/``rng=`` convention: pass ``seed`` for a quick
reproducible run, or thread an ``rng`` (a ``numpy.random.Generator``) across calls /
``rng.spawn()`` for parallel work.
"""

import json
import os

import numpy as np

from pref_voting.spatial_profiles import SpatialProfile

# Resolve relative to this module so from_ces works regardless of the current directory.
# In pref_voting this file ships in the package's data/ dir and is loaded with
# importlib.resources (see from_ces below and pref_voting/voting_method.py for the pattern).
CES_DISTRIBUTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "voter_distributions", "ces2020_voter_distributions.json"
)


class BinnedDistribution:
    """A binned empirical distribution: a set of axis-aligned bins (boxes),
    piecewise-uniform within each bin, with bins grouped into named regions.

    Any object with a ``num_dims`` attribute and a ``sample(num, rng)`` method returning a
    ``(num, num_dims)`` array can serve as a voter distribution for the *random* candidate
    model; the *structured* model additionally needs ``regions`` and ``sample_in_region``.

    Args:
        bin_lows: ``(num_bins, num_dims)`` lower corner of each bin (a ``(num_bins,)``
            array is accepted as 1D).
        bin_highs: ``(num_bins, num_dims)`` upper corner of each bin.
        bin_probs: probabilities for the bins (normalized if necessary).
        regions (dict, optional): maps a region name to the list of bin indices it spans,
            e.g. ``{"left": [0, 1], "centrist": [2], "right": [3, 4]}``.
        name (str, optional): a human-readable name.
        metadata (dict, optional): provenance information.
    """

    def __init__(
        self, bin_lows, bin_highs, bin_probs, regions=None, name=None, metadata=None
    ):
        bin_lows = np.asarray(bin_lows, dtype=float)
        bin_highs = np.asarray(bin_highs, dtype=float)
        if bin_lows.ndim == 1:  # 1D convenience
            bin_lows = bin_lows[:, None]
            bin_highs = bin_highs[:, None]
        bin_probs = np.asarray(bin_probs, dtype=float)
        assert bin_lows.shape == bin_highs.shape, (
            "bin_lows and bin_highs must have the same shape"
        )
        assert len(bin_probs) == len(bin_lows), "need one probability per bin"
        assert np.all(bin_highs > bin_lows), (
            "each bin must have positive width in every dimension"
        )
        assert np.all(bin_probs >= 0) and bin_probs.sum() > 0, (
            "bin probs must be >= 0 and not all zero"
        )

        self.bin_lows = bin_lows
        self.bin_highs = bin_highs
        self.bin_probs = bin_probs / bin_probs.sum()
        self.num_bins, self.num_dims = bin_lows.shape
        self.regions = {r: list(bins) for r, bins in regions.items()} if regions else {}
        self.name = name or "binned distribution"
        self.metadata = metadata or {}

        for region, bins in self.regions.items():
            assert all(0 <= b < self.num_bins for b in bins), (
                f"region {region!r} refers to bin indices outside 0..{self.num_bins - 1}"
            )

    @property
    def support(self):
        """``(num_dims, 2)`` array of the (lo, hi) extent in each dimension."""
        return np.stack([self.bin_lows.min(axis=0), self.bin_highs.max(axis=0)], axis=1)

    def _sample_bins(self, bins, probs, num, rng):
        """Draw ``num`` positions: choose among ``bins`` with probabilities ``probs``,
        then uniform within the chosen bin (box)."""
        chosen = rng.choice(bins, size=num, p=probs)
        return rng.uniform(self.bin_lows[chosen], self.bin_highs[chosen])

    def sample(self, num, rng=None):
        """Returns a ``(num, num_dims)`` array of positions from the full distribution."""
        rng = np.random.default_rng() if rng is None else rng
        return self._sample_bins(np.arange(self.num_bins), self.bin_probs, num, rng)

    def sample_in_region(self, region, num, rng=None):
        """Returns a ``(num, num_dims)`` array of positions from the distribution
        restricted to ``region``'s bins (bin chosen in proportion to its probability,
        then uniform within the bin). If the region's bins carry no probability mass, the
        bin is chosen in proportion to its *volume* instead, so a zero-mass region can
        still be populated."""
        rng = np.random.default_rng() if rng is None else rng
        if region not in self.regions:
            raise ValueError(
                f"unknown region {region!r}; this distribution defines: {sorted(self.regions)}"
            )
        bins = np.asarray(self.regions[region], dtype=int)
        probs = self.bin_probs[bins]
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:  # zero-mass region: fall back to volume-weighted choice over its bins
            vol = np.prod(self.bin_highs[bins] - self.bin_lows[bins], axis=1)
            probs = vol / vol.sum()
        return self._sample_bins(bins, probs, num, rng)

    def plot(self, ax=None, color_by_region=False, cmap="viridis"):
        """Plot the distribution's density (1D or 2D only).

        1D: a histogram of bin densities (bar height = ``bin_prob / bin_width``). 2D: a
        heatmap of bin densities (each bin shaded by ``bin_prob / bin_area``). With
        ``color_by_region=True``, bins are instead colored by the region they belong to
        (1D) or each region is shaded a distinct color (2D), and a legend is drawn; bins
        in no region are gray.

        Returns the matplotlib ``Axes``.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection

        if self.num_dims not in (1, 2):
            raise ValueError("plot supports only 1D and 2D distributions")
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3.2) if self.num_dims == 1 else (6, 5))

        region_of = {b: r for r, bins in self.regions.items() for b in bins}
        region_names = list(self.regions)
        palette = plt.get_cmap("tab10").colors
        region_color = {
            r: palette[i % len(palette)] for i, r in enumerate(region_names)
        }

        def region_legend():
            if color_by_region and region_names:
                ax.legend(
                    handles=[
                        mpatches.Patch(color=region_color[r], label=r)
                        for r in region_names
                    ],
                    frameon=False,
                    fontsize=9,
                )

        sizes = self.bin_highs - self.bin_lows

        if self.num_dims == 1:
            lows, widths = self.bin_lows[:, 0], sizes[:, 0]
            density = self.bin_probs / widths
            if color_by_region:
                colors = [
                    region_color.get(region_of.get(b), "0.7")
                    for b in range(self.num_bins)
                ]
            else:
                colors = "0.6"
            ax.bar(
                lows,
                density,
                width=widths,
                align="edge",
                color=colors,
                edgecolor="white",
            )
            ax.set_xlabel("position")
            ax.set_ylabel("density")
        else:
            rects = [
                mpatches.Rectangle(self.bin_lows[b], sizes[b, 0], sizes[b, 1])
                for b in range(self.num_bins)
            ]
            if color_by_region:
                facecolors = [
                    region_color.get(region_of.get(b), "0.7")
                    for b in range(self.num_bins)
                ]
                ax.add_collection(
                    PatchCollection(rects, facecolors=facecolors, edgecolor="white")
                )
            else:
                pc = PatchCollection(rects, cmap=cmap, edgecolor="white")
                pc.set_array(self.bin_probs / np.prod(sizes, axis=1))  # density per bin
                ax.add_collection(pc)
                ax.figure.colorbar(pc, ax=ax, label="density")
            (xlo, xhi), (ylo, yhi) = self.support
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.set_xlabel("dimension 0")
            ax.set_ylabel("dimension 1")
            ax.set_aspect("equal")

        region_legend()
        ax.set_title(self.name)
        return ax

    def __repr__(self):
        return (
            f"BinnedDistribution({self.name!r}, num_dims={self.num_dims}, "
            f"num_bins={self.num_bins}, regions={sorted(self.regions)})"
        )

    @classmethod
    def from_boxes(
        cls, bin_lows, bin_highs, bin_probs, regions=None, name=None, metadata=None
    ):
        """A binned distribution from explicit bin boxes (any dimension)."""
        return cls(
            bin_lows,
            bin_highs,
            bin_probs,
            regions=regions,
            name=name,
            metadata=metadata,
        )

    @classmethod
    def from_binned(cls, bin_edges, bin_probs, regions=None, name=None, metadata=None):
        """A 1D distribution that is piecewise-uniform on consecutive ``bin_edges``
        (length ``k + 1`` for ``k`` bins), with optional named ``regions``."""
        edges = np.asarray(bin_edges, dtype=float)
        assert len(edges) == len(bin_probs) + 1, "need one more edge than bins"
        assert np.all(np.diff(edges) > 0), "bin edges must be increasing"
        return cls(
            edges[:-1],
            edges[1:],
            bin_probs,
            regions=regions,
            name=name,
            metadata=metadata,
        )

    @classmethod
    def from_ces(
        cls,
        region,
        source="atkinson",
        bin_edges=None,
        regions=None,
        data_path=CES_DISTRIBUTIONS_PATH,
    ):
        """Loads a stored CES 2020 voter distribution (1D).

        Args:
            region (str): a two-letter state abbreviation (e.g. "AK"), or "US".
            source (str): "atkinson" (5-bin pid7 party ID) or "mccune" (7-bin CC20_340a ideology).
            bin_edges (array, optional): overrides the stored bin edges (one more entry than the number of bins).
            regions (dict, optional): overrides the default left/centrist/right grouping.
        """
        with open(data_path) as f:
            data = json.load(f)
        assert source in data["distributions"], (
            f"unknown source {source!r}; available: {list(data['distributions'])}"
        )
        dist = data["distributions"][source]
        assert region in dist["regions"], (
            f"unknown region {region!r}; available: {sorted(dist['regions'])}"
        )
        probs = dist["regions"][region]
        edges = np.asarray(
            dist["bin_edges"] if bin_edges is None else bin_edges, dtype=float
        )
        assert len(edges) == len(probs) + 1, (
            f"bin_edges must have {len(probs) + 1} entries for the {len(probs)}-bin {source!r} source"
        )
        if regions is None:
            regions = _default_ces_regions(source, len(probs))
        return cls.from_binned(
            edges,
            probs,
            regions=regions,
            name=f"{source} CES2020 {region}",
            metadata={
                "source": source,
                "region": region,
                "num_respondents": dist["num_respondents"][region],
                "description": dist["description"],
            },
        )


def _default_ces_regions(source, num_bins):
    """Default left/centrist/right bin grouping for the CES sources: the bin straddling  0 is 'centrist', bins below it 'left', bins above it 'right'."""
    if num_bins == 5:  # atkinson: Dem, leanDem, Independent, leanRep, Rep
        return {"left": [0, 1], "centrist": [2], "right": [3, 4]}
    if num_bins == 7:  # mccune: 7-point ideology, middle bin straddles 0
        return {"left": [0, 1, 2], "centrist": [3], "right": [4, 5, 6]}
    raise ValueError(
        f"no default region grouping for a {num_bins}-bin {source!r} distribution; "
        f"pass regions=..."
    )


def _candidate_regions_per_profile(
    num_cands, candidate_counts, candidate_type_probs, rng
):
    """The region name for each of the ``num_cands`` structured candidates."""
    if candidate_counts is not None:
        assert sum(candidate_counts.values()) == num_cands, (
            f"candidate_counts must sum to num_cands={num_cands}, got {sum(candidate_counts.values())}"
        )
        return [r for r, count in candidate_counts.items() for _ in range(count)]
    names = list(candidate_type_probs)
    probs = np.asarray([candidate_type_probs[n] for n in names], dtype=float)
    assert np.all(probs >= 0) and np.isclose(probs.sum(), 1.0), (
        "candidate_type_probs must be non-negative and sum to 1"
    )
    return [names[i] for i in rng.choice(len(names), size=num_cands, p=probs)]


def generate_spatial_profile_from_binned_distribution(
    num_cands,
    num_voters,
    voter_dist,
    cand_dist=None,
    candidate_counts=None,
    candidate_type_probs=None,
    num_profiles=1,
    seed=None,
    rng=None,
):
    """Generate spatial profiles with voter and candidate positions from binned
    distributions.

    Args:
        num_cands (int): the number of candidates.
        num_voters (int): the number of voters.
        voter_dist (BinnedDistribution): the voters' distribution.
        cand_dist (BinnedDistribution, optional): the candidates' distribution; defaults
            to ``voter_dist`` (candidates come from the voters' population).
        candidate_counts (dict, optional): exact composition, mapping region name to a
            count (the counts must sum to ``num_cands``). Selects the *structured* model.
        candidate_type_probs (dict, optional): probabilistic composition, mapping region
            name to a probability (summing to 1); each candidate's region is drawn
            independently. Selects the *structured* model.
        num_profiles (int): the number of profiles to generate.
        seed (int, optional): seed for a fresh generator (used when ``rng`` is None).
        rng (numpy.random.Generator, optional): generator to draw from; takes precedence
            over ``seed``.

    With neither ``candidate_counts`` nor ``candidate_type_probs`` (the *random* model),
    candidate positions are i.i.d. draws from ``cand_dist``. In the *structured* model,
    each candidate's region is chosen by the given counts/probabilities and its position
    is drawn from ``cand_dist`` restricted to that region's bins; the regions are recorded
    in ``SpatialProfile.candidate_types``.

    Returns:
        A SpatialProfile (or a list of SpatialProfiles if ``num_profiles > 1``).
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    cand_dist = voter_dist if cand_dist is None else cand_dist
    assert cand_dist.num_dims == voter_dist.num_dims, (
        "voter_dist and cand_dist must have the same number of dimensions"
    )

    if candidate_counts is not None and candidate_type_probs is not None:
        raise ValueError(
            "give at most one of candidate_counts (exact) or candidate_type_probs "
            "(probabilistic); for the random candidate model, give neither."
        )
    structured = candidate_counts is not None or candidate_type_probs is not None
    if structured:
        if not getattr(cand_dist, "regions", None):
            raise ValueError(
                "the structured candidate model needs a binned cand_dist with named "
                "regions; this distribution has none."
            )
        requested = set(candidate_counts or candidate_type_probs)
        unknown = requested - set(cand_dist.regions)
        if unknown:
            raise ValueError(
                f"unknown candidate region(s) {sorted(unknown)}; cand_dist defines "
                f"{sorted(cand_dist.regions)}."
            )

    profs = []
    for _ in range(num_profiles):
        voter_pos = voter_dist.sample(num_voters, rng)

        if not structured:
            cand_pos = cand_dist.sample(num_cands, rng)
            cand_types = None
        else:
            regions = _candidate_regions_per_profile(
                num_cands, candidate_counts, candidate_type_probs, rng
            )
            cand_pos = np.concatenate(
                [cand_dist.sample_in_region(r, 1, rng) for r in regions]
            )
            cand_types = {cidx: r for cidx, r in enumerate(regions)}

        profs.append(
            SpatialProfile(
                {cidx: cand_pos[cidx] for cidx in range(num_cands)},
                {vidx: voter_pos[vidx] for vidx in range(num_voters)},
                candidate_types=cand_types,
            )
        )

    return profs[0] if num_profiles == 1 else profs
