# Changelog

All notable changes to **pref_voting** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions are numbered sequentially; this project does **not** follow strict Semantic Versioning. Because pref_voting is a research package that updates frequently, breaking changes may appear in ordinary releases.  They are marked **⚠ breaking** below so users can check before upgrading.


## [Unreleased]

## [1.18.1] - 2026-06-27

### Added

- `mappings.Utility.to_truncated_ranking(method, ...)` and `utility_profiles.UtilityProfile.to_truncated_ranking_profile(method, ...)`: convert utilities to truncated ranked ballots (a `Ranking` / `ProfileWithTies`). Two scale-free criteria:  `method="radius"` ranks every candidate whose utility is within `radius` of the voter's favorite (`u(best) - u(c) < radius`); `method="gap"` ranks best-to-worst, stopping once two adjacent utility levels are closer than `min_gap`. `require_at_least_one` (default `True`) keeps a would-be-empty ballot's favorite, otherwise that voter abstains and is dropped (so `num_voters` reflects turnout).
- `generate_spatial_profile_from_binned_distribution.generate_spatial_profile_from_binned_distribution(...)`: builds `SpatialProfile`s from a `BinnedDistribution`, with a random candidate model (positions i.i.d. from the distribution) and a structured model (`candidate_counts` for exact composition or `candidate_type_probs` for a probabilistic mix; each candidate's region chosen by the given counts/probabilities, then its position drawn from that region's bins). Generates `num_profiles` profiles (returns a single `SpatialProfile`, or a list when `num_profiles > 1`). Randomness via the `seed=`/`rng=` convention.
- `analysis.social_utility_performance(utilities, winners)`: the social utility performance of a winning set, normalized so the utilitarian winner(s) score 1 and a uniformly random candidate scores 0. Accepts a `UtilityProfile` or a precomputed `{candidate: social_utility}` mapping, and a winning set (mean over tied winners) or a `{candidate: probability}` mapping (expected utility).

## [1.18.0] - 2026-06-24

> Contains one or more **⚠ breaking** changes (see below) — existing code may need small updates.

### Changed
- **⚠ breaking** `Ranking.is_bullet_vote` now takes a required `num_cands` argument and reflects the correct definition: a vote for a single candidate alone — either the favorite ranked with everyone else unranked, or the favorite alone on top with all other candidates tied at the bottom. (Callers `ProfileWithTies.num_bullet_votes` and `PrefGradeProfile.num_bullet_votes` updated accordingly.)
- **⚠ breaking** `approval` and `dis_and_approval` now accept any grade set that is a subset of `{0, 1}` / `{-1, 0, 1}` (so "approvals-only" ballots are allowed), and treat a candidate left ungraded by a voter as `0`. A candidate graded by nobody now scores `0` and is included, so it can beat an all-negative candidate (dis&approval) or tie in an all-abstain election.
- `score_voting` gained an `ungraded_score` parameter controlling whether candidates graded by no voter are excluded (default) or given a fixed score.
- Build and publish now use **uv** (`uv build` / `uv publish`); CI (`tests.yml`) runs on uv instead of Poetry. The build backend remains `poetry-core`.

### Fixed
- `io/readers.abif_to_profile`: look candidates up by the ABIF id **token**, not the display name, so profiles with custom candidate names round-trip correctly.
- `Ranking.to_weak_order`: copy the rank map instead of aliasing it, so the original ranking is no longer mutated.
- `Ranking.__hash__`: made order-independent within indifference classes to satisfy the  hash/`__eq__` contract; previously equal tied rankings could hash differently and break `set`/`dict` membership.
- `rankings.break_ties_alphabetically`: sort the candidates at each rank so ties are  actually broken alphabetically (affects Copeland / Plurality / Borda rankings with non-integer or unsorted candidates).
- `mappings.Utility.expectation`: use the valid accessor (`self.val`) — the method raised `AttributeError` on every call.
- `mappings.Utility.to_k_approval_ballot`: enforce the `k` cap before appending, so `k=1`   no longer approves more than one candidate. (3.5)
- `mappings.Utility.linear_transformation`: pass the utility value to the transform instead of re-looking-it-up, fixing an `AssertionError` that made the method unusable.
- `weighted_majority_graphs`: `cycles`/`has_cycle` with `curr_cands` now subgraph the actual graph instead of a fresh empty one (they previously always reported "no cycle").
- `weighted_majority_graphs.SupportGraph.display`: draw on its own fresh 2D axes, fixing a crash when a 3D axes was left current (e.g. after a spatial-profile `view()`).
- `helper.get_weak_mg`: copy the input graph before adding tie edges, so a `MajorityGraph`/`MarginGraph` passed in is no longer mutated (corrupting `top_cycle`/`getcha`/`smith_set`).
- `other_axioms.reversal_symmetry`: import `MarginGraph` (was a `NameError` on every call).
- `axiom.Axiom`: add the missing `self` parameter to `satisfies`, `violates`,
  `add_satisfying_vms`, `add_violating_vms`. (5.2)
- `create_methods`: add the missing `VotingMethod` import used by `compose`/`faceoff`.
- `generate_weighted_majority_graphs`: add missing `permutations`, `MajorityGraph`, and `tqdm` imports used by the enumeration functions.
- `invariance_axioms`: `downward_homogeneity` now detects violations (the branch body was a
  no-op `violation` statement instead of `violation = True`).
- `generate_utility_profiles.generate_spatial_utility_profile`: pass the utility-function parameter in the correct (last) position, fixing the `RM` model and any parametrized model.
- `analysis.find_profiles_with_different_winners`: use an order-independent distinctness test (`len(set(wss)) == len(wss)`), so profiles where methods disagree are no longer silently skipped.
- `utility_profiles.UtilityProfile.util_avg` and `util_grade_profile.UtilGradeProfile.util_avg`: return the voter-weighted average instead of an unweighted mean over groups.
- `utility_profiles.UtilityProfile.display(show_totals=True)`: index the candidate map as a dict instead of calling it.
- `profiles_with_ties.ProfileWithTies.display`: no longer crashes on profiles containing an empty (abstention) ranking.
- `profiles.Profile.randomly_truncate`: preserve the ranked order of kept candidates.
- `grade_methods.cumulative_voting`: validate that **each ballot** sums to
  `max_total_grades` (the previous check compared the grade-set sum and could never pass for the default).
- `grade_methods.score_voting`: guard against an all-abstain profile (previously raised `ValueError` from `max([])`).
- Removed invalid escape sequences in several docstrings that triggered `SyntaxWarning` (e.g. `\ldots`, `\sum`) by marking them raw strings.

### Added
- `tqdm` added as a runtime dependency (it was imported but undeclared).
- `pyflakes` added as a dev dependency, plus a `tests/test_no_undefined_names.py` guard that fails if any module references an undefined name (catches the missing-import class of bug statically).
- Substantially expanded the test suite, including new/extended tests for `io`, `mappings`, `helper`, `rankings`, `grade_methods`, `analysis`, `generate_utility_profiles`, `pairwise_profile`, `spatial_profile`, the weighted-majority graphs, and the `axiom`, `axiom_helpers`, and `invariance_axioms` modules.

---

_Changes prior to this changelog are not documented here; see the [commit history](https://github.com/voting-tools/pref_voting/commits/main) and the [PyPI release history](https://pypi.org/project/pref_voting/#history)._
