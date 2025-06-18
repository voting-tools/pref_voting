"""
Run manipulation experiments and record them with experiments_csv
-----------------------------------------------------------------
    python -m experiments.run_rc_manipulations
Resumes automatically if it crashes: rerun the same command.
"""
# --- imports -----------------------------------------------------------
from __future__ import annotations
import sys, random
from experiments_csv import *

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import experiments_csv
from strategic_voting.strategic_voting_algorithms import (
    borda, make_x_approval, algorithm1_single_voter, algorithm2_coalitional,
)

# --- configuration -----------------------------------------------------
RESULT_DIR   = pathlib.Path("results")
CSV_SINGLE   = "rc_single.csv"
CSV_COAL     = "rc_coal.csv"

RESULT_DIR.mkdir(exist_ok=True)

N_SEEDS      = 50 # number of random instances per grid point
M_VALUES      = list(range(3, 9))      # number of candidates (m)
N_TEAM_LIST   = list(range(2, 8))      # number of teams (voters)
K_VALUES      = [2, 3, 4]            # number of coalitions (k)

# Scoring-rule generators (m is passed at runtime)
RULES = {
    "plurality"   : lambda m: make_x_approval(1),
    "2-approval"  : lambda m: make_x_approval(2),
    "veto"        : lambda m: make_x_approval(m-1),
    "borda"       : lambda m: borda,
}
# N_SEEDS * len(M_VALUES) * len(N_TEAM_LIST) * len(K_VALUES) * len(RULES) = 100 * 10 * 12 * 4 * 4 = 192,000 instances in total
# N_SEEDS * len(M_VALUES) * len(N_TEAM_LIST) * len(K_VALUES) * len(RULES) = 50 * 7 * 7 * 3 * 4 = 29,400 instances in total

# ───────────────────────────────────────────────────────────────────────

# function that runs ONE experiment instance
def rc_single_trial(m: int, n_team: int, rule: str, seed: int):
    random.seed(seed)
    cands = [chr(65+i) for i in range(m)]
    team  = [random.sample(cands, m) for _ in range(n_team)]
    opp   = random.sample(cands, m)
    pref = random.choice(opp)

    ok, _ = algorithm1_single_voter(
        RULES[rule](m),           # scoring function
        team, opp, pref
    )
    return {
        "ok": ok, "m": m, "n_team": n_team, "rule": rule
    }


def rc_coalition_trial(m: int, n_team: int, k: int, rule: str, seed: int):
    if rule == "borda":           # skip borda for coalitions
        return None               # experiments_csv ignores None rows
    random.seed(seed)
    cands = [chr(65+i) for i in range(m)]
    team  = [random.sample(cands, m) for _ in range(n_team)]
    opp   = random.sample(cands, m)
    pref  = opp[1]

    ok, _ = algorithm2_coalitional(
        RULES[rule](m), team, opp, pref, k
    )
    return {
        "ok": ok, "m": m, "n_team": n_team, "k": k, "rule": rule
    }

# --- grids -------------------------------------------------------------
single_grid = {
    "m": M_VALUES,
    "n_team": N_TEAM_LIST,
    "rule": list(RULES.keys()),
    "seed": range(N_SEEDS),
}

coalition_grid = {
    "m": M_VALUES,
    "n_team": N_TEAM_LIST,
    "k": K_VALUES,
    "rule": [r for r in RULES if r != "borda"],  # veto / plurality / 2-approval
    "seed": range(N_SEEDS),
}


# --- launch ------------------------------------------------------------
if __name__ == "__main__":
    # SINGLE-VOTER
    ex1 = experiments_csv.Experiment(RESULT_DIR, CSV_SINGLE, backup_folder=None)
    ex1.clear_previous_results()
    ex1.run_with_time_limit(rc_single_trial, single_grid, time_limit=60)

    # COALITION
    ex2 = experiments_csv.Experiment(RESULT_DIR, CSV_COAL, backup_folder=None)
    ex2.clear_previous_results()
    ex2.run_with_time_limit(rc_coalition_trial, coalition_grid, time_limit=60)

    # ----------  PLOTS  -------------------------------------------------


    # (A) single-voter success %
    df = pd.read_csv(RESULT_DIR / CSV_SINGLE)
    summary = (df.groupby(["rule", "m"]).agg(success_pct=("ok", "mean")).reset_index())
    summary["success_pct"] *= 100

    fig, ax = plt.subplots(figsize=(6,4))
    for rule, g in summary.groupby("rule"):
        ax.plot(g["m"], g["success_pct"], marker="o", label=rule)
    ax.set(xlabel="number of candidates (m)",
           ylabel="manipulation success (%)",
           ylim=(0, 100))
    ax.legend(title="voting rule")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "success_vs_m_single.png", dpi=300)

    # (B) coalition – one figure per k
    dfc = pd.read_csv(RESULT_DIR / CSV_COAL)
    for k, gk in dfc.groupby("k"):
        summ_k = (gk.groupby(["rule", "m"]).agg(success_pct=("ok", "mean")).reset_index())
        summ_k["success_pct"] *= 100
        fig, ax = plt.subplots(figsize=(6,4))
        for rule, g in summ_k.groupby("rule"):
            ax.plot(g["m"], g["success_pct"], marker="o", label=rule)
        ax.set(xlabel="number of candidates (m)",
               ylabel="success (%)",
               ylim=(0, 100),
               title="k = {}".format(k))
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULT_DIR / ("success_vs_m_coal_k{}.png".format(k)),dpi=300)
    print("Done!  Plots saved under", RESULT_DIR)
