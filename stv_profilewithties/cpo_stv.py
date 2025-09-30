from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Set, Tuple
from itertools import combinations
from collections import defaultdict
import math
import random

from pref_voting.profiles_with_ties import ProfileWithTies

Candidate = int
Committee = Tuple[Candidate, ...]


def _topset_in_profilewithties(ranking, accept: Set[Candidate]) -> List[Candidate]:
    rmap = getattr(ranking, "rmap", None)
    if not rmap:
        return []
    ranks = [r for c, r in rmap.items() if c in accept]
    if not ranks:
        return []
    rmin = min(ranks)
    return [c for c, r in rmap.items() if c in accept and r == rmin]


def _nextset_in_profilewithties(ranking, accept: Set[Candidate], exclude: Candidate) -> List[Candidate]:
    rmap = getattr(ranking, "rmap", None)
    if not rmap:
        return []
    pool = [(c, r) for c, r in rmap.items() if c != exclude and c in accept]
    if not pool:
        return []
    rmin = min(r for _, r in pool)
    return [c for c, r in pool if r == rmin]


def _quota(N: float, k: int, rule: str) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    rule = rule.lower()
    if rule == "nb":
        return N / float(k + 1)
    if rule == "droop_int":
        return math.floor(N / float(k + 1)) + 1.0
    if rule == "exact_droop":
        return N / float(k + 1)
    if rule == "hare":
        return N / float(k)
    raise ValueError(f"Unknown quota rule: {rule}")


def _compare_committees_cpo_stv(
    profile: ProfileWithTies,
    A: Committee,
    B: Committee,
    quota: float,
    tol: float = 1e-12,
    max_iters: int = 200
) -> Tuple[float, float]:
    S: Set[Candidate] = set(A) | set(B)
    I: Set[Candidate] = set(A) & set(B)

    rankings, rcounts = profile.rankings_counts
    bal_alloc: List[Dict[Candidate, float]] = [defaultdict(float) for _ in rankings]

    for i, (ranking, w) in enumerate(zip(rankings, rcounts)):
        tops = _topset_in_profilewithties(ranking, S)
        if tops:
            share = float(w) / len(tops)
            for t in tops:
                bal_alloc[i][t] += share

    def curr_totals() -> Dict[Candidate, float]:
        totals: Dict[Candidate, float] = {c: 0.0 for c in S}
        for alloc in bal_alloc:
            for c, w in alloc.items():
                totals[c] = totals.get(c, 0.0) + w
        return totals

    totals = curr_totals()

    iters = 0
    while True:
        iters += 1
        if iters > max_iters:
            break
        changed = False
        for c in sorted(I, key=str):
            tc = totals.get(c, 0.0)
            excess = tc - quota
            if excess > tol and tc > tol:
                ratio = excess / tc
                for i, ranking in enumerate(rankings):
                    w_c = bal_alloc[i].get(c, 0.0)
                    if w_c <= 0.0:
                        continue
                    delta = w_c * ratio
                    if delta <= tol:
                        continue
                    next_set = _nextset_in_profilewithties(ranking, S, exclude=c)
                    bal_alloc[i][c] -= delta
                    if next_set:
                        share = delta / len(next_set)
                        for nxt in next_set:
                            bal_alloc[i][nxt] += share
                    changed = True
                if changed:
                    totals = curr_totals()
        if not changed:
            break

    sumA = sum(totals.get(x, 0.0) for x in A)
    sumB = sum(totals.get(y, 0.0) for y in B)
    return sumA, sumB


def _tuple_committee(cands: Iterable[Candidate]) -> Committee:
    return tuple(sorted(cands))


def _condorcet_committee(margins: Dict[Committee, Dict[Committee, float]]) -> Optional[Committee]:
    committees = list(margins.keys())
    for S in committees:
        wins_all = True
        for T in committees:
            if S == T:
                continue
            if margins[S].get(T, 0.0) <= 0.0:
                wins_all = False
                break
        if wins_all:
            return S
    return None


def _condorcet_committee_on_demand(
    profile: ProfileWithTies,
    sets: List[Committee],
    quota: float,
    tol: float = 1e-12,
) -> Optional[Committee]:
    cache: Dict[Tuple[Committee, Committee], float] = {}

    def margin(A: Committee, B: Committee) -> float:
        key = (A, B)
        if key in cache:
            return cache[key]
        sumA, sumB = _compare_committees_cpo_stv(profile, A, B, quota, tol=tol)
        m = sumA - sumB
        cache[key] = m
        cache[(B, A)] = -m
        return m

    if not sets:
        return None

    cand_idx = 0
    for i in range(1, len(sets)):
        if margin(sets[i], sets[cand_idx]) > 0.0:
            cand_idx = i

    c = sets[cand_idx]
    for j in range(cand_idx):
        if margin(c, sets[j]) <= 0.0:
            return None
    return c


def cpo_stv(
    profile: ProfileWithTies,
    k: int,
    quota_rule: str = "nb",
    return_pairwise: bool = False,
    tol: float = 1e-12,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Candidate], Optional[Dict]]:
    if not isinstance(profile, ProfileWithTies):
        raise TypeError("cpo_stv expects a pref_voting.ProfileWithTies")

    all_cands: List[Candidate] = list(profile.candidates)
    n = float(profile.num_voters)
    if k <= 0 or k > len(all_cands):
        raise ValueError("Invalid committee size k.")

    if hasattr(profile, "use_extended_strict_preference"):
        profile.use_extended_strict_preference()

    sets: List[Committee] = [_tuple_committee(S) for S in combinations(all_cands, k)]
    m = len(sets)
    if m == 0:
        return [], None

    quota = _quota(n, k, quota_rule)

    if not return_pairwise:
        winner = _condorcet_committee_on_demand(profile, sets, quota, tol=tol)
        if winner is not None:
            return list(winner), None

    margins: Dict[Committee, Dict[Committee, float]] = {S: {} for S in sets}
    for i in range(m):
        A = sets[i]
        for j in range(i + 1, m):
            B = sets[j]
            sumA, sumB = _compare_committees_cpo_stv(profile, A, B, quota, tol=tol)
            margins[A][B] = sumA - sumB
            margins[B][A] = sumB - sumA

    winner = _condorcet_committee(margins)
    if winner is None:
        win_counts = {S: sum(1 for v in margins[S].values() if v > 0.0) for S in margins}
        max_wins = max(win_counts.values()) if win_counts else 0
        tied = [S for S, w in win_counts.items() if w == max_wins]
        if len(tied) == 1:
            winner = tied[0]
        else:
            def best_smallest_loss_value(S: Committee) -> float:
                losses = [v for v in margins[S].values() if v < 0.0]
                return max(losses) if losses else 0.0
            tied.sort(key=lambda S: (best_smallest_loss_value(S), S))
            winner = tied[-1] if tied else None

    if winner is None:
        winner = sets[0]

    return list(winner), ({"quota": quota, "margins": margins, "sets": sets} if return_pairwise else None)
