"""
Run manipulation experiments and record them with experiments_csv
-----------------------------------------------------------------
    python -m experiments.run_rc_manipulations
Resumes automatically if it crashes: rerun the same command.
"""
# --- imports -----------------------------------------------------------
from __future__ import annotations
import sys, random, pathlib
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

# --- helper: make sure we always have runtime_ms -----------------------
def add_runtime_ms(df: pd.DataFrame) -> pd.DataFrame:
    if "runtime_ms" not in df.columns:
        if "runtime" not in df.columns:
            raise KeyError("Neither 'runtime_ms' nor 'runtime' found in the CSV")
        df = df.copy()
        df["runtime_ms"] = df["runtime"] * 1000      # seconds → milliseconds
    return df

# --- launch ------------------------------------------------------------
if __name__ == "__main__":
    plots_only = True  # set to True to skip running the trials
    # SINGLE-VOTER
    if not plots_only:
        # SINGLE-VOTER
        ex1 = experiments_csv.Experiment(RESULT_DIR, CSV_SINGLE, backup_folder=None)
        ex1.clear_previous_results()
        ex1.run_with_time_limit(rc_single_trial, single_grid, time_limit=60)

        # COALITION
        ex2 = experiments_csv.Experiment(RESULT_DIR, CSV_COAL, backup_folder=None)
        ex2.clear_previous_results()
        ex2.run_with_time_limit(rc_coalition_trial, coalition_grid, time_limit=60)

    # ----------  PLOTS  ----------------------------------------------------
    plt.rcParams["font.size"] = 9


    def stacked_plot(tbl, title, fname):
        tbl_succ = tbl.pivot(index="m", columns="rule", values="success_pct")
        tbl_time = tbl.pivot(index="m", columns="rule", values="runtime_ms")

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                       figsize=(5.5, 5),
                                       gridspec_kw={"height_ratios": [2, 1]})

        tbl_succ.plot(ax=ax1, marker="o", lw=1.2)
        ax1.set(ylabel="success (%)", ylim=(0, 100), title=title)
        ax1.legend(ncol=2, fontsize=8, loc="upper center")

        tbl_time.plot(ax=ax2, marker="x", ls="--")
        ax2.set(xlabel="candidates m", ylabel="runtime (ms)", yscale="log")
        ax2.yaxis.set_major_formatter(lambda v, p: f"{v * 1e3:.0f}")
        ax2.grid(axis="y", ls=":", alpha=.3)
        ax2.legend().remove()

        fig.tight_layout()
        fig.savefig(fname, dpi=300)
        plt.close(fig)


    # B1 ─ single-voter -----------------------------------------------------
    df = add_runtime_ms(pd.read_csv(RESULT_DIR / CSV_SINGLE))
    summ = (df.groupby(["rule", "m"])
            .agg(success_pct=("ok", "mean"), runtime_ms=("runtime_ms", "mean"))
            .reset_index())
    summ["success_pct"] *= 100
    stacked_plot(summ, "single voter",
                 RESULT_DIR / "success_vs_m_single_w_time.png")

    # B2 ─ coalition --------------------------------------------------------
    dfc = add_runtime_ms(pd.read_csv(RESULT_DIR / CSV_COAL))
    for k, gk in dfc.groupby("k"):
        tbl = (gk.groupby(["rule", "m"])
               .agg(success_pct=("ok", "mean"), runtime_ms=("runtime_ms", "mean"))
               .reset_index())
        tbl["success_pct"] *= 100
        stacked_plot(tbl, f"coalition k={k}",
                     RESULT_DIR / f"success_vs_m_coal_k{k}_w_time.png")
    print("Done!  (stacked version)")

