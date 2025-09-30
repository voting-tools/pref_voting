'''
    File: proportional_methods.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: September 29, 2025
    
    Implementations of voting methods for proportional representation.
'''
import os
import math
import itertools
import collections
import random

from pref_voting.weighted_majority_graphs import MarginGraph
from pref_voting.margin_based_methods import minimax
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.voting_method import vm
from pref_voting.voting_method_properties import ElectionTypes
from pref_voting.profiles import Profile

EPS = 1e-12
TRACE = bool(int(os.environ.get("STV_TRACE", "0") or "0"))

def _t(msg):
    if TRACE:
        print(msg)

# ---------- Piece model ----------

class RankingPiece:
    """
    A piece represents a fractional portion of a voter's ballot weight allocated to a specific candidate.
    
    In STV (Single Transferable Vote), when candidates receive surplus votes above the quota or when 
    candidates are eliminated, ballot weights must be transferred to other candidates according to 
    voter preferences. Rather than transferring whole ballots, the system creates "pieces", which are fractional 
    portions of ballot weight that can be allocated independently.
    
    For example, if a candidate receives 120 votes but only needs 100 to meet quota, the surplus 20 
    votes are transferred as pieces with reduced weight (20/120 = 1/6 of original weight) to the 
    next preferences on those ballots.
    
    Each piece tracks:
    - ranking: The original voter's preference ranking (Ranking object)
    - weight: The fractional weight of this piece (0.0 to 1.0)
    - current_rank: The preference level this piece is currently at
    - cand: The candidate this piece is currently allocated to
    """
    __slots__ = ("ranking", "weight", "current_rank", "cand", "arrived_value")
    def __init__(self, ranking, weight, current_rank, cand, arrived_value=None):
        self.ranking = ranking
        self.weight = weight
        self.current_rank = current_rank
        self.cand = cand
        # 'arrived_value' is the value credited to this candidate when this piece ARRIVED.
        self.arrived_value = weight if arrived_value is None else arrived_value
    def clone_to(self, cand, new_rank, weight):
        # When a piece moves to a new candidate, its arrival value at that candidate is the amount moved.
        return RankingPiece(self.ranking, weight, new_rank, cand, arrived_value=weight)

class ParcelIndex:
    """
    Track which pieces belong to which parcel for last-parcel transfer rules.
    
    In some STV variants (like Australian Senate rules), surplus transfers use only the 
    "last parcel" of votes received by a candidate, rather than all votes. This class 
    tracks which pieces arrived in which order so the last parcel can be identified.
    
    A "parcel" is a group of pieces that arrived together during a single transfer operation.
    """
    def __init__(self):
        self._last = collections.defaultdict(list)
    def start_new_parcel(self, cand):
        self._last[cand] = []
    def note_arrival(self, cand, piece_idx):
        self._last[cand].append(piece_idx)
    def last_parcel(self, cand):
        return self._last.get(cand, [])
    def clear_parcel(self, cand):
        self._last[cand] = []
    def remap_indices(self, mapping):
        if not mapping:
            return
        for cand, idxs in list(self._last.items()):
            remapped = []
            for idx in idxs:
                if idx in mapping:
                    remapped.append(mapping[idx])
            self._last[cand] = remapped

def _initial_pieces_from_profile(profile, recipients, parcels):
    """
    Create initial ranking pieces from ProfileWithTies.
    
    Converts a ProfileWithTies into the piece-based representation used by STV algorithms.
    Each voter's ranking becomes one or more pieces allocated to their most preferred 
    available candidates. If multiple candidates are tied at the top rank, the ballot 
    weight is split equally among them (see approval_stv for a different approach).
    
    Args:
        profile: ProfileWithTies object containing voter rankings
        recipients: Set of candidates eligible to receive pieces
        parcels: ParcelIndex to track piece arrival order
        
    Returns:
        List of RankingPiece objects representing the initial allocation
    """
    pieces = []
    rankings, rcounts = profile.rankings_counts
    for ranking, count in zip(rankings, rcounts):
        if count <= 0: 
            continue
        rmap = ranking.rmap
        first_rank = None
        first_cands = []
        for c, r in rmap.items():
            if r is not None and c in recipients:
                if first_rank is None or r < first_rank:
                    first_rank = r; first_cands = [c]
                elif r == first_rank:
                    first_cands.append(c)
        if first_cands:
            share = float(count) / len(first_cands)
            for c in first_cands:
                p = RankingPiece(ranking, share, first_rank, c, arrived_value=share)
                parcels.note_arrival(c, len(pieces))
                pieces.append(p)
    return pieces

def _tally_from_pieces(pieces, restrict_to=None):
    """
    Tally the total weight allocated to each candidate from a collection of pieces.
    
    Args:
        pieces: List of RankingPiece objects
        restrict_to: Optional set of candidates to include in tally
        
    Returns:
        Dictionary mapping candidates to their total allocated weight
    """
    t = collections.defaultdict(float)
    for p in pieces:
        if p.weight <= EPS:
            continue
        if restrict_to is None or p.cand in restrict_to:
            t[p.cand] += p.weight
    return t

def _next_prefs_from_ranking(ranking, recipients, current_rank):
    """Find next preferences in a ranking after current_rank that are in recipients."""
    rmap = ranking.rmap
    next_ranks = [r for c, r in rmap.items() if r is not None and r > current_rank and c in recipients]
    if not next_ranks:
        return [], -1
    next_rank = min(next_ranks)
    next_cands = [c for c, r in rmap.items() if r == next_rank and c in recipients]
    return next_cands, next_rank

def _move_piece_forward(piece, recipients):
    """Move a ranking piece forward to next preferences."""
    nxt, new_rank = _next_prefs_from_ranking(piece.ranking, recipients, piece.current_rank)
    if not nxt:
        return []
    share = 1.0 / float(len(nxt))
    return [(c, share, new_rank) for c in nxt]

# ---------- Surplus & elimination ----------

def _transfer_surplus_inclusive(pieces, elect, quota, recipients, parcels,
                                drain_all=True, last_parcel_only=False, ers_rounding=False):
    """
    Inclusive Gregory: drain a common fraction from the winner's pile and move the drained
    mass to next available continuing preferences.
      • drain_all=True: drain the fraction from *all* ballots in the pile. Portions with
        no next preference exhaust.  (Used for WIG and last‑parcel.)
      • drain_all=False: “compensation” (ERS/NB) – only donors that have a next
        preference are drained; the fraction is increased so the surplus is fully removed.
      • last_parcel_only=True: only drain pieces in the most recent parcel (Senatorial).
    Returns True if any weight moved.
    """
    tall = _tally_from_pieces(pieces, restrict_to=(set(recipients) | {elect}))
    surplus = tall.get(elect, 0.0) - quota
    if surplus <= EPS:
        return False

    if last_parcel_only:
        donor_idxs = list(parcels.last_parcel(elect))
    else:
        donor_idxs = [i for i, p in enumerate(pieces) if p.cand == elect and p.weight > EPS]
    if not donor_idxs:
        return False

    if drain_all:
        total_weight = sum(pieces[i].weight for i in donor_idxs)
        if total_weight <= EPS:
            return False
        frac = min(1.0, surplus / total_weight)
        any_moved = False
        opened = set()  # ensure a *new* parcel is opened for each recipient
        for i in donor_idxs:
            p = pieces[i]
            drain = p.weight * frac
            if ers_rounding:
                # ERS practice: round transfer values down to hundredth
                drain = math.floor(drain * 100.0) / 100.0
            if drain <= EPS:
                continue
            forwards = _move_piece_forward(p, recipients)
            if forwards:
                share = drain / float(len(forwards))
                for nxt_c, _, new_idx in forwards:
                    if nxt_c not in opened:
                        parcels.start_new_parcel(nxt_c)
                        opened.add(nxt_c)
                    pieces.append(p.clone_to(nxt_c, new_idx, share))
                    parcels.note_arrival(nxt_c, len(pieces)-1)
            # drain even if no forward (exhaust)
            p.weight -= drain
            any_moved = True
        if last_parcel_only:
            parcels.clear_parcel(elect)
        return any_moved

    # compensation (ERS/NB)
    donors = []
    nxt_cache = {}
    for i in donor_idxs:
        forwards = _move_piece_forward(pieces[i], recipients)
        if forwards:
            donors.append(i)
            nxt_cache[i] = forwards
    total_transferable = sum(pieces[i].weight for i in donors)
    if total_transferable <= EPS:
        return False
    frac = min(1.0, surplus / total_transferable)
    any_moved = False
    opened = set()  # also open new parcels under compensation variant
    for i in donors:
        p = pieces[i]
        drain = p.weight * frac
        if ers_rounding:
            # ERS practice: round transfer values down to hundredth
            drain = math.floor(drain * 100.0) / 100.0
        if drain <= EPS:
            continue
        forwards = nxt_cache[i]
        share = drain / float(len(forwards))
        for nxt_c, _, new_idx in forwards:
            if nxt_c not in opened:
                parcels.start_new_parcel(nxt_c)
                opened.add(nxt_c)
            pieces.append(p.clone_to(nxt_c, new_idx, share))
            parcels.note_arrival(nxt_c, len(pieces)-1)
        p.weight -= drain
        any_moved = True
    if last_parcel_only:
        parcels.clear_parcel(elect)
    return any_moved

def _transfer_surplus_scottish(pieces, elect, quota, recipients, parcels, *, decimals=5):
    """
    Scottish STV surplus transfer (SSI 2007/42):
      For each ballot piece currently credited to the elected candidate, compute
        transfer = truncate_to_decimals((surplus * piece_value_when_received) / total, decimals)
      and push that amount to the next available preference(s) among *continuing* candidates.
      Portions with no next available preference do not transfer (Rule 48(1)(b)).
      (I.e., one truncation of A/B to `decimals`; A = surplus * piece_value_when_received,
      B = total currently credited to the winner.)  Largest‑surplus‑first is handled by caller (Rule 49).

    Returns: True if any weight moved; False otherwise.
    """
    recipients = set(recipients)

    # Total currently credited to the elected candidate and their surplus over quota.
    tall = _tally_from_pieces(pieces, restrict_to=(recipients | {elect}))
    total = tall.get(elect, 0.0)
    surplus = total - quota
    if surplus <= EPS or total <= EPS:
        return False

    scale = 10 ** decimals
    any_moved = False
    opened = set()

    # Consider only pieces currently credited to the elected candidate.
    donor_idxs = [i for i, p in enumerate(pieces) if p.cand == elect and p.weight > EPS]
    if not donor_idxs:
        return False

    for i in donor_idxs:
        p = pieces[i]

        # Only papers that have a next continuing preference are "transferable" (Rule 48(1)(b)).
        forwards = _move_piece_forward(p, recipients)
        if not forwards:
            continue

        # Per‑piece transfer = floor( (surplus * value_when_received / total) * 10^d ) / 10^d
        # Use arrived_value as the statute’s “value ... when received”.
        base = p.arrived_value
        drain = math.floor(((surplus * base / total) * scale) + EPS) / float(scale)
        if drain <= EPS:
            continue

        # Never move more than is still credited to this piece.
        drain = min(drain, p.weight)
        if drain <= EPS:
            continue

        # Deduct from the elected candidate and forward equally among next preferences.
        p.weight -= drain
        share = drain / float(len(forwards))
        for nxt_c, _, new_rank in forwards:
            if nxt_c not in opened:
                parcels.start_new_parcel(nxt_c)  # start a fresh parcel for each recipient in this transfer
                opened.add(nxt_c)
            pieces.append(p.clone_to(nxt_c, new_rank, share))
            parcels.note_arrival(nxt_c, len(pieces) - 1)

        any_moved = True

    return any_moved

def _eliminate_lowest(pieces, continuing, parcels, tie_break_key=None):
    if not continuing:
        return None, pieces
    tallies = _tally_from_pieces(pieces, restrict_to=continuing)
    min_t = float('inf'); lowest = []
    for c in continuing:
        t = tallies.get(c, 0.0)
        if t < min_t - EPS:
            min_t = t; lowest = [c]
        elif abs(t - min_t) <= EPS:
            lowest.append(c)
    if not lowest:
        return None, pieces
    if len(lowest) > 1:
        key = tie_break_key or (lambda x: str(x))
        lowest.sort(key=key)
    elim = lowest[0]
    continuing.remove(elim)

    new_pieces = []
    old_to_new = {}
    pending_notes = []
    for old_idx, p in enumerate(pieces):
        if p.cand != elim:
            new_idx = len(new_pieces)
            new_pieces.append(p)
            old_to_new[old_idx] = new_idx
            continue
        forwards = _move_piece_forward(p, continuing)
        if not forwards:
            continue
        opened = set()
        share = p.weight / float(len(forwards))
        for nxt_c, _, new_idx in forwards:
            if nxt_c not in opened:
                parcels.start_new_parcel(nxt_c)
                opened.add(nxt_c)
            created_idx = len(new_pieces)
            new_pieces.append(p.clone_to(nxt_c, new_idx, share))
            pending_notes.append((nxt_c, created_idx))
        p.weight = 0.0
    parcels.remap_indices(old_to_new)
    for cand, idx in pending_notes:
        parcels.note_arrival(cand, idx)
    return elim, new_pieces

def _nb_quota(total_weight, num_seats):
    return total_weight / float(num_seats + 1)

def _droop_int_quota(total_weight, num_seats):
    """Integer Droop quota used in Scottish STV (SSI 2007/42, Rule 46)."""
    if num_seats <= 0:
        return math.inf
    return math.floor(total_weight / float(num_seats + 1)) + 1

# ---------- Public STV variants ----------

@vm(name="STV-Scottish", input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def stv_scottish(profile, num_seats=2, curr_cands=None, decimals=5, rng=None):
    """
    Scottish STV per SSI 2007/42 (https://www.legislation.gov.uk/ssi/2007/42):
      • Rule 46: Integer Droop quota  q = floor(N/(k+1)) + 1
      • Rule 48(3): per‑ballot transfer = truncate[(surplus × ballot value when received) / total].
      • Rule 49: If multiple surpluses, transfer largest first; if equal, use *history tie‑break*,
                 else decide by lot.
      • Rule 50: Exclusions transfer at current transfer value.
      • Rule 51: Exclusion ties resolved by *history tie‑break*, else decide by lot.
      • Rule 52: If continuing == vacancies remaining, elect them all; no further transfers.

    Ballot ties (not allowed in actual Scottish STV) are supported by equal splitting of weight.

    Args:
        profile : Profile or ProfileWithTies
        num_seats : int
        curr_cands : iterable or None
        decimals : int
            Truncation precision for Rule 48(3). Default 5.
        rng : random.Random-like or None
            Source of randomness for “by lot” decisions. If None, uses Python's `random` module.

    Returns:
        list: Elected candidates (sorted by name for determinism).

    .. warning::
        STV implementations have not yet been thoroughly vetted.
    """
    if isinstance(profile, Profile):
        profile = profile.to_profile_with_ties()

    rand = rng if rng is not None else random

    # helpers

    def droop_int_quota(total_weight, k):
        if k <= 0:
            return math.inf
        return math.floor(total_weight / float(k + 1)) + 1

    def snapshot():
        """Record end-of-stage totals for *all* candidates currently carrying weight."""
        history.append(_tally_from_pieces(pieces))

    def history_prefer(cands, prefer="highest"):
        """
        Apply the statute's 'most recent preceding stage where unequal' rule.
        Returns a single candidate or None if still tied across all previous stages.
        """
        tied = list(cands)
        # Walk backward over completed stages (the current moment is *after* the last snapshot)
        for snap in reversed(history):
            vals = [(c, snap.get(c, 0.0)) for c in tied]
            if not vals:
                break
            if prefer == "highest":
                extreme = max(v for _, v in vals)
                narrowed = [c for c, v in vals if abs(v - extreme) <= EPS]
            else:  # prefer == "lowest"
                extreme = min(v for _, v in vals)
                narrowed = [c for c, v in vals if abs(v - extreme) <= EPS]
            # If we strictly narrowed the field, keep going (maybe down to a singleton)
            if 0 < len(narrowed) < len(tied):
                tied = narrowed
                if len(tied) == 1:
                    return tied[0]
            # else: all equal at this snapshot; look further back
        return None  # equal at all earlier stages

    def eliminate_and_transfer(elim, continuing, parcels):
        """Eliminate `elim` and push their pieces forward at current values."""
        new_pieces = []
        old_to_new = {}
        pending_notes = []
        for old_idx, p in enumerate(pieces):
            if p.cand != elim:
                new_idx = len(new_pieces)
                new_pieces.append(p)
                old_to_new[old_idx] = new_idx
                continue
            forwards = _move_piece_forward(p, continuing)
            if not forwards:
                continue
            opened = set()
            share = p.weight / float(len(forwards))
            for nxt_c, _, new_rank in forwards:
                if nxt_c not in opened:
                    parcels.start_new_parcel(nxt_c)
                    opened.add(nxt_c)
                created_idx = len(new_pieces)
                new_pieces.append(p.clone_to(nxt_c, new_rank, share))
                pending_notes.append((nxt_c, created_idx))
            p.weight = 0.0
        parcels.remap_indices(old_to_new)
        for cand, idx in pending_notes:
            parcels.note_arrival(cand, idx)
        return new_pieces

    # set up
    continuing = set(profile.candidates) if curr_cands is None else set(curr_cands)
    winners = []
    parcels = ParcelIndex()
    pieces = _initial_pieces_from_profile(profile, continuing, parcels)

    # constant quota for the whole count (Rule 46)
    _, rcounts = profile.rankings_counts
    total_votes = sum(float(c) for c in rcounts)
    quota = droop_int_quota(total_votes, num_seats)

    # History of end-of-stage totals (for Rules 49 & 51)
    history = []
    snapshot()  # First stage: after initial allocation of first preferences

    # main count
    safety = 0
    while len(winners) < num_seats:
        safety += 1
        if safety > 50000:
            raise RuntimeError("stv_scottish: loop safety tripped – no progress")

        # Current totals among continuing
        tallies_c = _tally_from_pieces(pieces, restrict_to=continuing)

        # Elect anyone at/above quota
        elected_now = [c for c in list(continuing) if tallies_c.get(c, 0.0) >= quota - EPS]
        if elected_now:
            # Mark as elected (they stop being 'continuing')
            for c in sorted(elected_now, key=lambda x: str(x)):
                continuing.remove(c)
                winners.append(c)

            # Rule 49: transfer surpluses one at a time, always the largest *among all elected*
            # candidates who currently exceed quota (ties by history, else lot).
            stuck = set()
            while True:
                tall_now = _tally_from_pieces(pieces)  # include everyone carrying weight
                # Anyone already elected whose current total still exceeds quota?
                elig = [c for c in winners if tall_now.get(c, 0.0) - float(quota) > EPS and c not in stuck]
                if not elig:
                    break

                # Pick the largest surplus; if tied, apply the statute’s history tie-break; else decide by lot.
                surpluses = {c: tall_now[c] - float(quota) for c in elig}
                max_s = max(surpluses.values())
                tied = [c for c in elig if abs(surpluses[c] - max_s) <= EPS]
                if len(tied) > 1:
                    chosen = history_prefer(tied, prefer="highest")
                    if chosen is None:
                        chosen = rand.choice(tied)
                else:
                    chosen = tied[0]

                moved = _transfer_surplus_scottish(
                    pieces, chosen, float(quota),
                    recipients=continuing, parcels=parcels, decimals=decimals
                )
                if moved:
                    snapshot()  # each successful surplus transfer creates a new "stage" for the Rule 49/51 history
                else:
                    # nothing to move (no next prefs etc.); don't loop forever on this candidate
                    stuck.add(chosen)

            # After finishing the surpluses from this stage, check last‑vacancy rule.
            if len(continuing) <= num_seats - len(winners):
                winners.extend(sorted(continuing, key=lambda x: str(x)))
                break

            continue  # start a fresh stage

        if not elected_now:
            tall_now = _tally_from_pieces(pieces)  # totals for everyone carrying weight
            surplusers = [c for c in winners if tall_now.get(c, 0.0) - float(quota) > EPS]

            if surplusers:
                # Rule 49: pick the largest surplus; tie by history, else lot
                max_s = max(tall_now[c] - float(quota) for c in surplusers)
                tied = [c for c in surplusers if abs((tall_now[c]-float(quota)) - max_s) <= EPS]
                if len(tied) > 1:
                    chosen = history_prefer(tied, prefer="highest") or rand.choice(tied)
                else:
                    chosen = tied[0]

                moved = _transfer_surplus_scottish(
                    pieces, chosen, float(quota),
                    recipients=continuing, parcels=parcels, decimals=decimals
                )
                if moved:
                    snapshot()      # each successful surplus transfer is its own stage
                    continue        # try again before excluding anyone

        # No one newly elected → consider last vacancies (Rule 52)
        if len(continuing) <= num_seats - len(winners):
            winners.extend(sorted(continuing, key=lambda x: str(x)))
            break

        # Exclude the current lowest (Rule 50) with Rule 51 history tie-break
        tallies_c = _tally_from_pieces(pieces, restrict_to=continuing)
        min_t = min(tallies_c.get(c, 0.0) for c in continuing)
        lowest = [c for c in continuing if abs(tallies_c.get(c, 0.0) - min_t) <= EPS]

        if len(lowest) > 1:
            elim = history_prefer(lowest, prefer="lowest")
            if elim is None:
                elim = rand.choice(lowest)  # decide by lot if tied at all previous stages
        else:
            elim = lowest[0]

        continuing.remove(elim)
        pieces = eliminate_and_transfer(elim, continuing, parcels)
        snapshot()  # each exclusion is a new stage

    return sorted(winners, key=lambda x: str(x))

@vm(name="STV-NB", input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def stv_nb(profile, num_seats = 2, curr_cands=None, quota_rule="nb", mann_strict=False, drain_all=False, tie_break_key=None, *, ers_rounding=False):
    """
    Single Transferable Vote — Newland–Britton (ERS) surplus rule (“NB”) with rational Droop quota.

    Summary
    -------
    Uses the NB (rational Droop) quota n/(k+1) and the ERS/Newland–Britton *compensation* rule.
    When a candidate exceeds quota, only ballot pieces that can transfer (i.e., have a next
    available preference among continuing candidates) are drained; pieces that cannot transfer
    are left untouched. The drain fraction α is chosen so the total drained from transferable
    pieces equals the surplus, with α ≤ 1 per piece. This offsets non‑transferables rather than
    letting surplus “disappear.” If many ballots are non‑transferable, an elected candidate may
    remain above quota after the surplus step. (In contrast, WIG drains the same fraction from
    *all* pieces, including those that cannot move; Meek lowers the effective quota via keep
    factors.)

    Counting details
    ----------------
    • Quota: NB (rational Droop) quota = total_weight / (seats + 1).
    If ers_rounding=True, quota is rounded up (to integer if >100, else to hundredth) per ERS practice.
    Optional "Mann strictness" (mann_strict=True) requires strictly more than the NB quota.
    • Surpluses: one at a time, largest surplus first among newly elected.
    • Recipients: transfers go only to continuing (unelected) candidates.
    • If no surplus: eliminate the current lowest and transfer at current weights; tie broken by
    `tie_break_key`.
    • Ballot ties: ties on ballots are supported by equal splitting of weight.
    • Last vacancies: if continuing == seats_remaining, elect them all.

    References: Tideman ("The Single Transferable Vote", 1995) on ERS compensation vs Meek; and 
    Tideman & Richardson ("Better voting methods through technology: The refinement-manageability trade-off in the single transferable vote", 2000).

    Args:
        profile: A Profile or ProfileWithTies object containing voter rankings
        num_seats (int): Number of seats to fill
        curr_cands: List of candidates to consider, defaults to all candidates in profile
        quota_rule (str): Quota calculation rule, defaults to "nb" (rational Droop)
        mann_strict (bool): Whether to use strict Mann-style elimination, defaults to False
        drain_all (bool): Whether to drain all ballots, defaults to False
        tie_break_key: Function for tie-breaking, defaults to None
        ers_rounding: If True, use ERS manual count rounding as described by Tideman and Richardson (2000): quota rounded up (to integer if >100,
        else to hundredth) and transfer values rounded down to hundredth. If False, defaults to rational Droop.

    Returns:
        list: List of elected candidates

    .. warning::
        STV implementations have not yet been thoroughly vetted.
    """
    if isinstance(profile, Profile):
        profile = profile.to_profile_with_ties()
    
    candidates_list = list(profile.candidates) if curr_cands is None else curr_cands
    continuing = set(candidates_list)
    winners = []
    parcels = ParcelIndex()
    pieces = _initial_pieces_from_profile(profile, continuing, parcels)

    # Calculate total weight from profile
    rankings, rcounts = profile.rankings_counts
    total_weight = sum(float(count) for count in rcounts)
    if total_weight <= EPS or not continuing or num_seats <= 0:
        return []
    
    rule = (quota_rule or "nb").lower()
    if rule == "nb":
        raw_quota = _nb_quota(total_weight, num_seats)       # rational Droop
        if ers_rounding:
            # ERS practice: round quota up (to integer if >100, else to hundredth)
            if raw_quota > 100.0:
                quota = math.ceil(raw_quota)
            else:
                quota = math.ceil(raw_quota * 100.0) / 100.0
        else:
            quota = raw_quota
    elif rule == "droop":
        quota = _droop_int_quota(total_weight, num_seats)     # integer Droop
    else:
        raise ValueError(f'Unknown quota_rule "{quota_rule}". Use "nb" or "droop".')

    safety = 0
    while len(winners) < num_seats:
        safety += 1
        if safety > 20000:
            raise RuntimeError("stv_nb: loop safety tripped – no progress")

        tallies_c = _tally_from_pieces(pieces, restrict_to=continuing)
        elected_now = [c for c in list(continuing)
                       if (tallies_c.get(c, 0.0) > quota + EPS if mann_strict
                           else tallies_c.get(c, 0.0) >= quota - EPS)]
        if elected_now:
            for c in sorted(elected_now, key=lambda x: str(x)):
                continuing.remove(c)
                winners.append(c)
            _t(f"Elected now: {elected_now}")

            # Surplus transfers: recipients = continuing only
            stuck = set()
            while True:
                tall_all = _tally_from_pieces(pieces, restrict_to=set(continuing) | set(elected_now))
                surplusers = [c for c in elected_now if tall_all.get(c, 0.0) - quota > EPS and c not in stuck]
                if not surplusers:
                    break
                elect = max(surplusers, key=lambda c: (tall_all.get(c, 0.0) - quota, str(c)))
                moved = _transfer_surplus_inclusive(
                    pieces, elect, quota, recipients=continuing, parcels=parcels,
                    drain_all=drain_all, last_parcel_only=False, ers_rounding=ers_rounding
                )
                _t(f"Transfer surplus from {elect}: moved={moved}")
                if not moved:
                    stuck.add(elect)

            if len(continuing) <= num_seats - len(winners):
                winners.extend(sorted(continuing, key=lambda x: str(x)))
                break

            continue

        if len(continuing) <= num_seats - len(winners):
            winners.extend(sorted(continuing, key=lambda x: str(x)))
            break

        elim, new_pieces = _eliminate_lowest(pieces, continuing, parcels, tie_break_key=tie_break_key)
        if elim is None:
            break
        _t(f"Eliminate: {elim}")
        pieces = new_pieces

    return sorted(winners, key=lambda x: str(x))

@vm(name="STV-WIG", input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def stv_wig(profile, num_seats = 2, curr_cands=None, quota_rule="nb", drain_all=True, tie_break_key=None):
    """
    STV with **Weighted Inclusive Gregory** (WIG) surplus transfers.

    Surpluses: drain the same fraction from every ballot in a winner’s pile; forward to next
    available continuing choices (exhaust otherwise); only surpluses of candidates elected in this stage are processed; previously elected winners are not revisited later. Elimination transfers at current weights. Transfers are exact (no ERS‑style rounding).

    Ballot ties are supported by equal splitting of weight.

    Quota options:
      • quota_rule="nb"     → rational Droop: total_weight / (seats + 1)
      • quota_rule="droop"  → integer Droop: floor(total_weight / (seats + 1)) + 1

    Note: WIG + Droop is common in public counts that use inclusive Gregory.

    References: Tideman ("The Single Transferable Vote", 1995) and Tideman & Richardson ("Better voting methods through technology: The
    refinement-manageability trade-off in the single transferable vote", 2000).

    Args:
        profile: A Profile or ProfileWithTies object containing voter rankings
        num_seats (int): Number of seats to fill
        curr_cands: List of candidates to consider, defaults to all candidates in profile
        quota_rule (str): Quota calculation rule, defaults to "nb" (rational Droop)
        tie_break_key: Function for tie-breaking, defaults to None

    Returns:
        list: List of elected candidates

    .. warning::
        STV implementations have not yet been thoroughly vetted.
    """
    if isinstance(profile, Profile):
        profile = profile.to_profile_with_ties()

    candidates_list = list(profile.candidates) if curr_cands is None else list(curr_cands)
    continuing = set(candidates_list)
    winners = []
    parcels = ParcelIndex()
    pieces = _initial_pieces_from_profile(profile, continuing, parcels)

    if num_seats <= 0 or not continuing:
        return []

    # Compute a constant quota for the count
    rankings, rcounts = profile.rankings_counts
    total_weight = sum(float(count) for count in rcounts)

    rule = (quota_rule or "nb").lower()
    if rule == "nb":
        quota = _nb_quota(total_weight, num_seats)
    elif rule == "droop":
        quota = _droop_int_quota(total_weight, num_seats)
    else:
        raise ValueError(f'Unknown quota_rule "{quota_rule}". Use "nb" or "droop".')

    safety = 0
    while len(winners) < num_seats:
        safety += 1
        if safety > 20000:
            raise RuntimeError("stv_wig: loop safety tripped – no progress")

        tallies_c = _tally_from_pieces(pieces, restrict_to=continuing)

        # Elect everyone at/above quota
        elected_now = [c for c in list(continuing) if tallies_c.get(c, 0.0) >= quota - EPS]
        if elected_now:
            for c in sorted(elected_now, key=lambda x: str(x)):
                continuing.remove(c)
                winners.append(c)

            # Transfer surpluses (largest surplus first), WIG (drain_all=True)
            stuck = set()
            while True:
                tall_all = _tally_from_pieces(pieces, restrict_to=set(continuing) | set(elected_now))
                surplusers = [c for c in elected_now if tall_all.get(c, 0.0) - quota > EPS and c not in stuck]
                if not surplusers:
                    break
                elect = max(surplusers, key=lambda c: (tall_all.get(c, 0.0) - quota, str(c)))
                moved = _transfer_surplus_inclusive(
                    pieces, elect, quota, recipients=continuing, parcels=parcels,
                    drain_all=True, last_parcel_only=False
                )
                if not moved:
                    stuck.add(elect)

            # If remaining candidates equal remaining seats, elect them all
            if len(continuing) <= num_seats - len(winners):
                winners.extend(sorted(continuing, key=lambda x: str(x)))
                break

            continue

        # No election this round — eliminate the lowest and redistribute
        if len(continuing) <= num_seats - len(winners):
            winners.extend(sorted(continuing, key=lambda x: str(x)))
            break

        elim, pieces = _eliminate_lowest(pieces, continuing, parcels, tie_break_key=tie_break_key)
        if elim is None:
            break

    return sorted(winners, key=lambda x: str(x))

@vm(name="STV-Last-Parcel", input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def stv_last_parcel(profile, num_seats = 2, curr_cands=None, quota_rule="nb", tie_break_key=None):
    """
    Single Transferable Vote using the "last parcel" or "senatorial" transfer rule.
    
    This is a variant of STV where surplus transfers work differently from the standard method.
    When a candidate has more votes than the quota (surplus), instead of transferring a proportion
    of all their votes, only the most recent "parcel" (bundle) of votes that put them over the
    quota is transferred. This simulates the practice used in some senatorial elections. Only surpluses 
    of candidates elected in this stage are processed; previously elected winners are not revisited later. 
    Only the NB (rational Droop) quota is implemented for this variant
    
    The last parcel rule can produce different results than standard STV because it treats
    different bundles of votes differently based on when they arrived at the candidate.

    Ballot ties are supported by equal splitting of weight.

    References: Tideman ("The Single Transferable Vote", 1995) and Tideman & Richardson ("Better voting methods through technology: The
    refinement-manageability trade-off in the single transferable vote", 2000).

    Args:
        profile: A Profile or ProfileWithTies object containing voter rankings
        num_seats (int): Number of seats to fill
        curr_cands: List of candidates to consider, defaults to all candidates in profile
        quota_rule (str): Quota calculation rule, defaults to "nb" (rational Droop)
        tie_break_key: Function for tie-breaking, defaults to None

    Returns:
        list: List of elected candidates

    .. warning::
        STV implementations have not yet been thoroughly vetted.
    """
    if isinstance(profile, Profile):
        profile = profile.to_profile_with_ties()
    
    candidates_list = list(profile.candidates) if curr_cands is None else curr_cands
    continuing = set(candidates_list)
    winners = []
    parcels = ParcelIndex()
    pieces = _initial_pieces_from_profile(profile, continuing, parcels)

    # Calculate total weight from profile
    rankings, rcounts = profile.rankings_counts
    total_weight = sum(float(count) for count in rcounts)
    if total_weight <= EPS or not continuing or num_seats <= 0:
        return []
    if quota_rule.lower() != "nb":
        raise ValueError("Only NB quota is implemented.")
    quota = _nb_quota(total_weight, num_seats)

    safety = 0
    while len(winners) < num_seats:
        safety += 1
        if safety > 20000:
            raise RuntimeError("stv_last_parcel: loop safety tripped – no progress")

        tallies_c = _tally_from_pieces(pieces, restrict_to=continuing)
        elected_now = [c for c in list(continuing) if tallies_c.get(c, 0.0) >= quota - EPS]
        if elected_now:
            for c in sorted(elected_now, key=lambda x: str(x)):
                continuing.remove(c)
                winners.append(c)
            _t(f"[LP] Elected now: {elected_now}")

            for c in elected_now:
                moved = _transfer_surplus_inclusive(
                    pieces, c, quota, recipients=continuing, parcels=parcels,
                    drain_all=True, last_parcel_only=True
                )
                _t(f"[LP] Transfer surplus (last parcel) from {c}: moved={moved}")
            continue

        if len(continuing) <= num_seats - len(winners):
            winners.extend(sorted(continuing, key=lambda x: str(x)))
            break

        elim, new_pieces = _eliminate_lowest(pieces, continuing, parcels, tie_break_key=tie_break_key)
        if elim is None:
            break
        _t(f"[LP] Eliminate: {elim}")
        pieces = new_pieces

    return sorted(winners, key=lambda x: str(x))

# ---------- Meek STV ----------

def _meek_flow_one_ballot(tiers, keep, continuing):
    remaining = 1.0
    out = []
    for tier in tiers:
        avail = [c for c in tier if c in continuing]
        if not avail:
            continue
        share = remaining / float(len(avail))
        spilled = 0.0
        for c in avail:
            k = keep.get(c, 1.0)
            kept = k * share
            out.append((c, kept))
            spilled += (1.0 - k) * share
        remaining = spilled
        if remaining <= EPS:
            break
    return out

def _meek_tally_from_profile(profile, keep, continuing):
    """Meek tally working directly with ProfileWithTies."""
    t = collections.defaultdict(float)
    rankings, rcounts = profile.rankings_counts
    
    for ranking, count in zip(rankings, rcounts):
        if count <= 0:
            continue
        rmap = ranking.rmap
        by_rank = collections.defaultdict(list)
        for c, r in rmap.items():
            if r is not None:
                by_rank[int(r)].append(c)
        tiers = []
        for r in sorted(by_rank):
            tiers.append(sorted(by_rank[r], key=lambda x: str(x)))
        
        if tiers:
            for c, a in _meek_flow_one_ballot(tiers, keep, continuing):
                t[c] += a * float(count)
    return t

@vm(name="STV-Meek", input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def stv_meek(profile, num_seats = 2, curr_cands=None, tol=1e-10, max_iter=2000, tie_break_key=None):
    """
    Meek Single Transferable Vote using retention factors for surplus handling.
    
    Meek STV is an STV variant that uses "retention factors" to handle surplus transfers.
    Instead of transferring physical ballot papers, each candidate has a "keep factor" that
    determines what fraction of votes they retain. When a candidate is elected, their keep
    factor is reduced so they only keep exactly the quota, and the remainder flows to next
    preferences.
    
    This method iteratively adjusts keep factors until the system reaches equilibrium.
    Elected candidates remain "continuing" throughout the process with reduced keep factors. 
    At each iteration, quota = (active weight currently credited to continuing candidates) / (k+1); 
    if no further keep‑factor reductions bring all winners to quota, eliminate the current lowest and repeat.

    Ballot ties are supported by equal splitting of weight.

    References: Tideman ("The Single Transferable Vote", 1995) and Tideman & Richardson ("Better voting methods through technology: The
    refinement-manageability trade-off in the single transferable vote", 2000).

    Args:
        profile: A Profile or ProfileWithTies object containing voter rankings
        num_seats (int): Number of seats to fill
        curr_cands: List of candidates to consider, defaults to all candidates in profile
        tol (float): Tolerance for convergence, defaults to 1e-10
        max_iter (int): Maximum number of iterations, defaults to 2000
        tie_break_key: Function for tie-breaking, defaults to None

    Returns:
        list: List of elected candidates

    .. warning::
        STV implementations have not yet been thoroughly vetted.
    """
    if isinstance(profile, Profile):
        profile = profile.to_profile_with_ties()
    
    candidates_list = list(profile.candidates) if curr_cands is None else curr_cands
    continuing = set(candidates_list)
    keep = {c: 1.0 for c in continuing}

    # Calculate total weight from profile
    rankings, rcounts = profile.rankings_counts
    total_weight = sum(float(count) for count in rcounts)
    if total_weight <= EPS or not continuing or num_seats <= 0:
        return []

    while len(continuing) > num_seats:
        # Adjust keep factors (winners remain continuing)
        for _ in range(max_iter):
            tallies = _meek_tally_from_profile(profile, keep, continuing)
            active_total = sum(tallies.get(c, 0.0) for c in continuing)
            quota = active_total / float(num_seats + 1) if active_total > EPS else 0.0

            changed = False
            for c in list(continuing):
                t = tallies.get(c, 0.0)
                if t > quota + tol and keep.get(c, 1.0) > 0.0:
                    # Standard Meek step: monotone, non‑increasing keep
                    new_keep = min(keep[c], quota / t)
                    if keep[c] - new_keep > tol:
                        keep[c] = new_keep
                        changed = True
            if not changed:
                break

        # Eliminate the current lowest
        tallies = _meek_tally_from_profile(profile, keep, continuing)
        min_t = float('inf'); lowest = []
        for c in continuing:
            t = tallies.get(c, 0.0)
            if t < min_t - EPS:
                min_t = t; lowest = [c]
            elif abs(t - min_t) <= EPS:
                lowest.append(c)
        if len(lowest) > 1:
            key = tie_break_key or (lambda x: str(x))
            lowest.sort(key=key)
        elim = lowest[0]
        continuing.remove(elim)
        keep[elim] = 0.0
        _t(f"[Meek] Eliminate: {elim} (t={min_t:.6f})")

    return sorted(list(continuing), key=lambda x: str(x))[:num_seats]

@vm(name="STV-Warren", input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def stv_warren(profile, num_seats = 2, curr_cands=None, tol=1e-10, tie_break_key=None):
    """
    Warren's equal‑price STV (Tideman 1995; Tideman & Richardson 2000):
      • In each stage, allocate each ballot to its current top among continuing (split ties).
      • While any candidate exceeds the (dynamic) quota, compute a uniform price p_c so that
        sum_b min(w_{b,c}, p_c) = quota; cap every ballot's contribution to c at p_c and push
        the released weight to the next available preference(s); iterate until no one > quota.
      • If no one is over quota and too many remain, eliminate the current lowest and repeat.
      • Quota in each stage = (active weight among continuing) / (k+1).

    Args:
        profile: A Profile or ProfileWithTies object containing voter rankings
        num_seats (int): Number of seats to fill
        curr_cands: List of candidates to consider, defaults to all candidates in profile
        tol (float): Tolerance for convergence, defaults to 1e-10
        tie_break_key: Function for tie-breaking, defaults to None

    References: Tideman ("The Single Transferable Vote", 1995) and Tideman & Richardson ("Better voting methods through technology: The
    refinement-manageability trade-off in the single transferable vote", 2000).

    Returns:
        list: List of elected candidates

    .. warning::
        STV implementations have not yet been thoroughly vetted.
    """
    if isinstance(profile, Profile):
        profile = profile.to_profile_with_ties()
    continuing = list(profile.candidates) if curr_cands is None else list(curr_cands)
    rankings, rcounts = profile.rankings_counts

    def topset(ranking, accept):
        rmap = ranking.rmap
        ranks = [r for c, r in rmap.items() if c in accept and r is not None]
        if not ranks: return []
        rmin = min(ranks)
        return [c for c, r in rmap.items() if c in accept and r == rmin]

    def nextset(ranking, accept, exclude):
        rmap = ranking.rmap
        pool = [(c, r) for c, r in rmap.items() if c != exclude and c in accept and r is not None]
        if not pool: return []
        rmin = min(r for _, r in pool)
        return [c for c, r in pool if r == rmin]

    while len(continuing) > num_seats:
        # (1) fresh per‑ballot allocations to current tops
        alloc = [collections.defaultdict(float) for _ in rankings]
        S = set(continuing)
        for i, ranking in enumerate(rankings):
            tops = topset(ranking, S)
            if tops:
                share = 1.0 / float(len(tops))
                for t in tops:
                    alloc[i][t] += share

        # (2) shrink over‑quota piles by equal price until none > quota
        max_equalize_iters = 2000
        iters = 0

        while True:
            # recompute totals and the current quota
            totals = collections.defaultdict(float)
            for i, A in enumerate(alloc):
                m = float(rcounts[i])
                for c, per in A.items():
                    totals[c] += per * m
            active_total = sum(totals.get(c, 0.0) for c in continuing)
            quota = active_total / float(num_seats + 1) if num_seats > 0 else float('inf')

            over = [c for c in continuing if totals.get(c, 0.0) > quota + tol]
            if not over:
                break  # all piles are at/below quota

            iters += 1
            if iters > max_equalize_iters:
                # Safety valve: stop rather than silently under‑equalizing.
                break

            changed = False

            # Warren equal‑price: for each over‑quota candidate c, find p_c so
            #   sum_i m_i * min(w_ic, p_c) = quota, where w_ic = per‑ballot share to c.
            for c in sorted(over, key=lambda x: str(x)):
                per_list = []
                for i, A in enumerate(alloc):
                    w = A.get(c, 0.0)
                    if w > EPS:
                        per_list.append((w, float(rcounts[i])))
                if not per_list:
                    continue

                # bisection for p_c
                lo, hi = 0.0, max(w for w, _ in per_list)
                for _ in range(64):
                    mid = 0.5 * (lo + hi)
                    s = 0.0
                    for w, m in per_list:
                        s += m * (w if w < mid else mid)
                    if s > quota:
                        hi = mid
                    else:
                        lo = mid
                p_c = lo

                # cap at p_c and push released weight forward (exhaust if no next)
                for i, ranking in enumerate(rankings):
                    per = alloc[i].get(c, 0.0)
                    if per <= EPS:
                        continue
                    new_per = min(per, p_c)
                    delta_per = per - new_per
                    if delta_per > tol:
                        alloc[i][c] = new_per
                        nxt = nextset(ranking, set(continuing), exclude=c)
                        if nxt:
                            share = delta_per / float(len(nxt))  # m cancels with later re‑weight
                            for d in nxt:
                                alloc[i][d] += share
                        # else: delta exhausts, lowering active_total and the next quota
                        changed = True

            # If nothing changed numerically, we’re as close as floating‑point permits
            if not changed:
                break

        # (3) eliminate lowest if too many remain
        totals = collections.defaultdict(float)
        for i, A in enumerate(alloc):
            m = float(rcounts[i])
            for c, per in A.items():
                totals[c] += per * m

        if len(continuing) == num_seats:
            break

        min_t = float('inf'); lowest = []
        for c in continuing:
            t = totals.get(c, 0.0)
            if t < min_t - EPS:
                min_t = t; lowest = [c]
            elif abs(t - min_t) <= EPS:
                lowest.append(c)
        if len(lowest) > 1:
            key = tie_break_key or (lambda x: str(x))
            lowest.sort(key=key)
        elim = lowest[0]
        continuing.remove(elim)

    return sorted(list(continuing), key=lambda x: str(x))[:num_seats]

@vm(name="Approval-STV", input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def approval_stv(profile, num_seats=2, curr_cands=None, quota_rule="droop",
                 select_tiebreak=None, elim_tiebreak=None, rng=None):
    """
    Approval‑STV (Delemazure & Peters 2024, https://arxiv.org/abs/2404.11407):

    In each round, a ballot supports all candidates it ranks *top* among the remaining
    candidates. Let B_i be the remaining budget of ballot i (start at 1 per voter).
    Let q be the quota.

    Loop until k winners:
      1) For every continuing candidate c, compute support S(c) = sum_{i: c in top_i} B_i.
      2) If some c has enough support (strictly > q for Droop; >= q for Hare):
         Elect such a c (by default the one with largest S); charge supporters exactly q
         in total by multiplying each supporter's budget by (S(c) - q)/S(c) (Gregory);
         remove c.
      3) Otherwise eliminate a candidate with the smallest S(c); remove it.

    Quotas
    -------
      quota_rule="droop"     → q = n / (k+1), elect if support > q
      quota_rule="droop_int" → q = floor(n / (k+1)) + 1, elect if support > q
      quota_rule="hare"      → q = n / k,     elect if support ≥ q

    Notes
    -----
    • Matches the budget‑flow pseudocode in Fig. 12 (Approval‑STV) using Gregory charging.
    • Equals Approval‑IRV when k=1 with Hare quota (but not with Droop; see Remark 5.1).

    Args:
        profile: A Profile or ProfileWithTies object containing voter rankings
        num_seats (int): Number of seats to fill
        curr_cands: List of candidates to consider, defaults to all candidates in profile
        quota_rule (str): Quota rule to use, defaults to "droop"
        select_tiebreak: Function for tie-breaking, defaults to None
        elim_tiebreak: Function for tie-breaking, defaults to None
        rng: Random number generator, defaults to None

    Returns:
        list: List of elected candidates

    .. warning::
        Approval‑STV implementation has not yet been thoroughly vetted. 
    """

    if isinstance(profile, Profile):
        profile = profile.to_profile_with_ties()
    rankings, rcounts = profile.rankings_counts

    continuing = list(profile.candidates) if curr_cands is None else [c for c in curr_cands if c in profile.candidates]
    winners = []

    # Total number of voters (with multiplicities)
    n = float(sum(rcounts))
    if n <= EPS or num_seats <= 0 or not continuing:
        return []

    # Select quota and election inequality
    if quota_rule == "droop":
        quota = n / float(num_seats + 1); strict = True
    elif quota_rule == "droop_int":
        quota = math.floor(n / float(num_seats + 1)) + 1; strict = True
    elif quota_rule == "hare":
        quota = n / float(num_seats); strict = False
    else:
        raise ValueError("quota_rule must be one of {'droop','droop_int','hare'}")

    # Budgets are tracked per ranking type, scaled by multiplicity
    budgets = [float(c) for c in rcounts]

    def topset(ranking, accept):
        rmap = ranking.rmap
        ranks = [r for c, r in rmap.items() if c in accept and r is not None]
        if not ranks:
            return []
        rmin = min(ranks)
        return [c for c, r in rmap.items() if c in accept and r == rmin]

    def support_budgets(accept):
        S = collections.defaultdict(float)
        A = set(accept)
        for i, ranking in enumerate(rankings):
            b = budgets[i]
            if b <= EPS:
                continue
            tops = topset(ranking, A)
            for c in tops:
                S[c] += b
        for c in accept:
            S.setdefault(c, 0.0)
        return S

    def electable(S):
        if strict:
            return [c for c, v in S.items() if v > quota + EPS]
        else:
            return [c for c, v in S.items() if v + EPS >= quota]

    def charge_supporters(chosen, S, accept):
        """Multiply each supporter's budget by a common factor so the total charge is exactly q."""
        total = S.get(chosen, 0.0)
        if total <= EPS:
            return
        factor = max(0.0, (total - quota) / total)
        A = set(accept)
        for i, ranking in enumerate(rankings):
            if budgets[i] <= EPS:
                continue
            if chosen in topset(ranking, A):
                budgets[i] *= factor

    rand = rng or random.Random(0)

    while len(winners) < num_seats and continuing:
        # Early finish: fill remaining seats if #continuing == seats_left
        if len(continuing) <= num_seats - len(winners):
            winners.extend(sorted(continuing, key=lambda x: str(x)))
            break

        S = support_budgets(continuing)

        # Elect if any candidate's supporters exceed the quota
        elig = electable(S)
        if elig:
            # default: pick largest support, deterministic by name if tied
            if select_tiebreak is None:
                maxv = max(S[c] for c in elig)
                tied = [c for c in elig if abs(S[c] - maxv) <= EPS]
                chosen = sorted(tied, key=lambda x: str(x))[0]
            else:
                best = max(select_tiebreak(c) for c in elig)
                tied = [c for c in elig if abs(select_tiebreak(c) - best) <= EPS]
                chosen = rand.choice(sorted(tied, key=lambda x: str(x)))
            winners.append(chosen)
            # supporters are with respect to the pre‑removal set (which includes `chosen`)
            charge_supporters(chosen, S, continuing + [chosen])
            continuing.remove(chosen)
            _t(f"[Approval‑STV] Elect {chosen}; winners: {winners}")
            continue

        # Otherwise, eliminate a lowest‑supported candidate
        minv = min(S[c] for c in continuing)
        lowest = [c for c in continuing if abs(S[c] - minv) <= EPS]
        if len(lowest) > 1:
            if elim_tiebreak is None:
                elim = sorted(lowest, key=lambda x: str(x))[0]
            else:
                mink = min(elim_tiebreak(c) for c in lowest)
                tied = [c for c in lowest if abs(elim_tiebreak(c) - mink) <= EPS]
                elim = rand.choice(sorted(tied, key=lambda x: str(x)))
        else:
            elim = lowest[0]
        continuing.remove(elim)
        _t(f"[Approval‑STV] Eliminate {elim}")

    return sorted(winners, key=lambda x: str(x))

# ---------- CPO-STV ----------

def _committee_margin_pwt(A, B, profile, inpair_surplus="meek"):
    """
    Compute the pairwise margin (A over B) for CPO‑STV using a *pair‑specific* quota.
    Steps (Tideman 1995/Tideman & Richardson 2000):
      • Restrict to S = A ∪ B. Allocate each ballot to its top(s) in S (split ties equally).
      • Let I = A ∩ B. Transfer *only* surpluses of candidates in I until no I‑member
        exceeds the pair‑quota; do NOT transfer surpluses of candidates outside I.
      • Pair‑quota at each iteration = (usable weight inside S) / (k+1), where k = |A|.
        Weight that has no next preference within S exhausts, lowering the next quota.
      • inpair_surplus = "meek" (ratio shrink) or "warren" (equal‑price).
    Returns: float margin = sum(A) − sum(B).
    """
    rankings, rcounts = profile.rankings_counts

    def _topset_in_ranking(ranking, accept_set):
        rmap = ranking.rmap
        ranks = [r for c, r in rmap.items() if c in accept_set and r is not None]
        if not ranks:
            return []
        rmin = min(ranks)
        return [c for c, r in rmap.items() if c in accept_set and r == rmin]

    def _nextset_in_ranking(ranking, accept_set, exclude):
        rmap = ranking.rmap
        pool = [(c, r) for c, r in rmap.items()
                if c != exclude and c in accept_set and r is not None]
        if not pool:
            return []
        rmin = min(r for _, r in pool)
        return [c for c, r in pool if r == rmin]

    S = set(A) | set(B)
    I = set(A) & set(B)
    k = len(A)

    # Per‑row allocations (already multiplied by row multiplicities)
    bal_alloc = [collections.defaultdict(float) for _ in rankings]
    for i, (ranking, m) in enumerate(zip(rankings, rcounts)):
        tops = _topset_in_ranking(ranking, S)
        if not tops:
            continue
        share = float(m) / float(len(tops))
        for t in tops:
            bal_alloc[i][t] += share

    tol = 1e-12
    max_iters = 10000
    rule = (inpair_surplus or "meek").lower()

    for _ in range(max_iters):
        # Recompute totals and the *pair‑specific* quota from current usable weight in S.
        totals = collections.defaultdict(float)
        for alloc in bal_alloc:
            for c, w in alloc.items():
                totals[c] += w
        usable = sum(totals.get(c, 0.0) for c in S)
        if k == 0 or usable <= tol:
            break
        quota = usable / float(k + 1)

        changed = False
        for c in sorted(I, key=lambda x: str(x)):
            tc = totals.get(c, 0.0)
            excess = tc - quota
            if excess <= tol or tc <= tol:
                continue

            if rule == "warren":
                # Equal‑price per Warren: choose p_c with Σ_i min(w_ic, p_c) = quota.
                w_list = []
                for i, alloc in enumerate(bal_alloc):
                    w_c = alloc.get(c, 0.0)
                    if w_c <= 0.0:
                        continue
                    m = float(rcounts[i])
                    per = w_c / m
                    w_list.append((per, m))
                if not w_list:
                    continue
                lo, hi = 0.0, max(w for w, _ in w_list)
                for _ in range(64):
                    mid = 0.5 * (lo + hi)
                    s = 0.0
                    for w, m in w_list:
                        s += m * (w if w < mid else mid)
                    if s > quota:
                        hi = mid
                    else:
                        lo = mid
                p_c = lo

                # Cap and push inside S; if no next in S, the delta exhausts.
                for i, ranking in enumerate(rankings):
                    w_c = bal_alloc[i].get(c, 0.0)
                    if w_c <= 0.0:
                        continue
                    m = float(rcounts[i])
                    per = w_c / m
                    new_per = min(per, p_c)
                    delta_total = (per - new_per) * m
                    if delta_total <= tol:
                        continue
                    bal_alloc[i][c] = new_per * m
                    nxt = _nextset_in_ranking(ranking, S, exclude=c)
                    if nxt:
                        share = delta_total / float(len(nxt))
                        for nx in nxt:
                            bal_alloc[i][nx] = bal_alloc[i].get(nx, 0.0) + share
                    changed = True

            else:
                # Meek‑like ratio shrink: remove the same *fraction* from each piece for c.
                ratio = excess / tc
                for i, ranking in enumerate(rankings):
                    w_c = bal_alloc[i].get(c, 0.0)
                    if w_c <= 0.0:
                        continue
                    delta = w_c * ratio
                    if delta <= tol:
                        continue
                    bal_alloc[i][c] = w_c - delta
                    nxt = _nextset_in_ranking(ranking, S, exclude=c)
                    if nxt:
                        share = delta / float(len(nxt))
                        for nx in nxt:
                            bal_alloc[i][nx] = bal_alloc[i].get(nx, 0.0) + share
                    changed = True

        if not changed:
            break

    # Final totals & margin
    totals = collections.defaultdict(float)
    for alloc in bal_alloc:
        for c, w in alloc.items():
            totals[c] += w
    score_A = sum(totals.get(c, 0.0) for c in A)
    score_B = sum(totals.get(c, 0.0) for c in B)
    return score_A - score_B

@vm(name="CPO-STV", input_types=[ElectionTypes.PROFILE, ElectionTypes.PROFILE_WITH_TIES])
def cpo_stv(profile, num_seats = 2, curr_cands=None, inpair_surplus="meek", fallback_vm=minimax):
    """
    CPO-STV (Comparison of Pairs of Outcomes) - a Condorcet-consistent proportional method.
    
    Unlike traditional STV which eliminates candidates sequentially, CPO-STV considers all
    possible committees (combinations) of the required size and compares them pairwise.
    
    For any two k‑member sets A and B, restrict each ballot to S = A ∪ B, allocate the
    ballot to its highest ranked available candidate in S, and then transfer **only**
    the surpluses of candidates in the intersection I = A ∩ B (never from candidates
    who appear in only one of the two compared sets). The margin of A vs. B is the sum of
    votes for A’s members minus the sum for B’s.

    Within each A vs B comparison, the quota is q = U/(k+1), where k = |A| and 
    U is the total weight currently credited to candidates in S 
    (i.e., not yet exhausted relative to S). When, at the point of transfer, 
    a ballot has no remaining ranked candidate in S, its remaining weight is 
    treated as exhausted for this comparison, which reduces U on subsequent iterations.
    
    The winning committee is the one that beats all other possible committees
    in these pairwise comparisons. This makes CPO-STV "Condorcet-consistent" - if there's
    a committee that is majority-preferred to every other committee, CPO-STV will find it.
    If there is no such Condorcet committee, then the fallback voting method is used to
    pick the winning committee based on the pairwise margins between committees.
    
    This method is computationally intensive as it must examine C(candidates, seats) committees.

    References: Tideman ("The Single Transferable Vote", 1995) and Tideman & Richardson ("Better voting methods through technology: The
    refinement-manageability trade-off in the single transferable vote", 2000).

    Args:
        profile: A Profile or ProfileWithTies object containing voter rankings
        num_seats (int): Number of seats to fill
        curr_cands: List of candidates to consider, defaults to all candidates in profile
        inpair_surplus (str): Surplus handling method for pairwise comparisons, defaults to "meek"
        fallback_vm: Fallback voting method for tie-breaking, defaults to minimax

    Returns:
        list: List of elected candidates forming the winning committee

    .. warning::
        This implementation of CPO-STV has not yet been thoroughly vetted.
    """
    if isinstance(profile, Profile):
        profile = profile.to_profile_with_ties()
    
    curr_cands = list(profile.candidates) if curr_cands is None else curr_cands
    committees = list(itertools.combinations(curr_cands, num_seats))
    
    if len(committees) <= 1:
        return list(committees[0]) if committees else []

    # For efficiency, we first check for a Condorcet committee using an algorithm that does not require constructing the full margin graph.
    condorcet_committee_exists = True
    C = committees[0]
    for A in committees:
        if _committee_margin_pwt(A, C, profile, inpair_surplus=inpair_surplus) > 0:
            C = A

    for B in committees:
        if C != B and not _committee_margin_pwt(C, B, profile, inpair_surplus=inpair_surplus) > 0:
            condorcet_committee_exists = False
            break

    if condorcet_committee_exists:
        return list(C)

    # If no Condorcet committee exists, we construct the full margin graph and use the fallback voting method to find the winning committee.
    weighted_edges = []
    for i, A in enumerate(committees):
        for B in committees[i+1:]:
            m = _committee_margin_pwt(A, B, profile, inpair_surplus=inpair_surplus)
            if m > 0:
                weighted_edges.append((A, B, m))
            elif m < 0:
                weighted_edges.append((B, A, abs(m)))

    mg = MarginGraph(committees, weighted_edges)
    winners = fallback_vm(mg)
    
    return sorted(random.choice(winners))