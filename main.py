"""
CivicSim — Bayesian demographic agent generator.

API: given location (e.g. "alameda,california") and n_agents (e.g. 25),
samples from probabilistic model age → race → occupation → income using data in /data,
and outputs a table: agent, age, race, income, occupation.
"""

import os
import argparse
import pandas as pd
import numpy as np


def _data_dir(location: str) -> str:
    """Return path to data directory for this location. Normalize for future multi-county."""
    root = os.path.dirname(os.path.abspath(__file__))
    # For now we only have one geography (alameda,california); data lives in data/
    return os.path.join(root, "data")


def load_distributions(location: str):
    """
    Load the four distribution tables for the given location.
    Returns (df_age, df_race, df_occupation, df_income).
    """
    data_dir = _data_dir(location)
    df_age = pd.read_csv(os.path.join(data_dir, "age.csv"))
    df_race = pd.read_csv(os.path.join(data_dir, "race.csv"))
    df_occupation = pd.read_csv(os.path.join(data_dir, "occupation.csv"))
    df_income = pd.read_csv(os.path.join(data_dir, "income.csv"))
    return df_age, df_race, df_occupation, df_income


def _normalize_to_probs(weights: np.ndarray) -> np.ndarray:
    """Normalize counts/weights to probabilities (sum = 1)."""
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def _proportional_alloc(n: int, weights: np.ndarray) -> np.ndarray:
    """
    Allocate n units across len(weights) categories proportionally (largest-remainder).
    Returns integer counts that sum to n.
    """
    w = np.asarray(weights, dtype=float)
    p = w / w.sum()
    exact = p * n
    counts = np.floor(exact).astype(int)
    remainder = n - int(counts.sum())
    frac = exact - counts
    if remainder > 0:
        idx = np.argsort(frac)[::-1][:remainder]
        counts[idx] += 1
    return counts


def sample_agents(
    location: str,
    n_agents: int,
    *,
    seed: int | None = None,
    diverse: bool = True,
) -> pd.DataFrame:
    """
    Generate n_agents from the model age → race → occupation → income.
    diverse=True: stratify so the sample matches population distribution (diverse).
    diverse=False: independent random draws.
    """
    if seed is not None:
        np.random.seed(seed)

    df_age, df_race, df_occupation, df_income = load_distributions(location)
    categories_age = df_age["category"].values
    categories_race = df_race["category"].values
    categories_occupation = df_occupation["category"].values
    categories_income = df_income["category"].values

    if diverse:
        counts_age = _proportional_alloc(n_agents, df_age["count"].values.astype(float))
        counts_race = _proportional_alloc(n_agents, df_race["value"].values.astype(float))
        counts_occ = _proportional_alloc(n_agents, df_occupation["totalestimate"].values.astype(float))
        counts_inc = _proportional_alloc(n_agents, df_income["estimated"].values.astype(float))
        ages = np.repeat(categories_age, counts_age)
        races = np.repeat(categories_race, counts_race)
        occupations = np.repeat(categories_occupation, counts_occ)
        incomes = np.repeat(categories_income, counts_inc)
        # Shuffle in place; use list to avoid pandas StringArray shuffle warning
        ages = np.array(ages.tolist())
        races = np.array(races.tolist())
        occupations = np.array(occupations.tolist())
        incomes = np.array(incomes.tolist())
        np.random.shuffle(ages)
        np.random.shuffle(races)
        np.random.shuffle(occupations)
        np.random.shuffle(incomes)
        rows = [
            {"agent": i + 1, "age": ages[i], "race": races[i], "income": incomes[i], "occupation": occupations[i]}
            for i in range(n_agents)
        ]
    else:
        p_age = _normalize_to_probs(df_age["count"].values)
        p_race = _normalize_to_probs(df_race["value"].values)
        p_occupation = _normalize_to_probs(df_occupation["totalestimate"].values)
        p_income = _normalize_to_probs(df_income["estimated"].values)
        rows = []
        for agent_id in range(1, n_agents + 1):
            rows.append({
                "agent": agent_id,
                "age": np.random.choice(categories_age, p=p_age),
                "race": np.random.choice(categories_race, p=p_race),
                "occupation": np.random.choice(categories_occupation, p=p_occupation),
                "income": np.random.choice(categories_income, p=p_income),
            })

    return pd.DataFrame(rows)[["agent", "age", "race", "income", "occupation"]]


def check_representativeness(
    location: str,
    sample_df: pd.DataFrame,
    population_dfs: tuple | None = None,
) -> dict:
    """
    Compare the sample of agents to the population distribution.
    Returns a dict of per-variable DataFrames: category, pop_pct, sample_count, sample_pct.
    """
    if population_dfs is None:
        population_dfs = load_distributions(location)
    df_age, df_race, df_occupation, df_income = population_dfs
    n = len(sample_df)

    def report(sample_col: str, df_pop: pd.DataFrame, value_col: str) -> pd.DataFrame:
        pop_pct = _normalize_to_probs(df_pop[value_col].values) * 100
        counts = sample_df[sample_col].value_counts()
        sample_pct = np.zeros(len(df_pop))
        for i, cat in enumerate(df_pop["category"].values):
            sample_pct[i] = counts.get(cat, 0) / n * 100
        return pd.DataFrame({
            "category": df_pop["category"].values,
            "pop_pct": np.round(pop_pct, 1),
            "sample_count": [int(counts.get(c, 0)) for c in df_pop["category"].values],
            "sample_pct": np.round(sample_pct, 1),
        })

    return {
        "age": report("age", df_age, "count"),
        "race": report("race", df_race, "value"),
        "occupation": report("occupation", df_occupation, "totalestimate"),
        "income": report("income", df_income, "estimated"),
    }


def print_representativeness_report(report: dict, n_agents: int) -> None:
    """Print population vs sample so you can check if the 25 agents are representative."""
    print(format_representativeness_report(report, n_agents))


def format_representativeness_report(report: dict, n_agents: int) -> str:
    """Return the representativeness report as a string (for writing to file)."""
    lines = [
        "",
        "=" * 60,
        f"Representativeness check (population vs. sample of n={n_agents})",
        "Pop % = population share from /data; Sample % = share among generated agents.",
        "=" * 60,
    ]
    for variable in ["age", "race", "occupation", "income"]:
        df = report[variable]
        lines.append(f"\n--- {variable.upper()} ---")
        lines.append(df.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate agents from Bayesian demographic model (age→race→occupation→income)."
    )
    parser.add_argument(
        "--location",
        type=str,
        default="alameda,california",
        help="Location, e.g. alameda,california (used to select data in /data)",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=25,
        help="Number of agents to generate (IDs 1 to n_agents)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Also print population vs. sample report",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        default=True,
        help="Stratify so agents match population (default: True)",
    )
    parser.add_argument(
        "--no-diverse",
        action="store_false",
        dest="diverse",
        help="Use random sampling",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="FILE",
        help="Write agent table to FILE",
    )
    args = parser.parse_args()

    df = sample_agents(args.location, args.n_agents, seed=args.seed, diverse=args.diverse)
    print(df.to_string(index=False))

    # Always write agent table to agents.txt (agent, age, race, income, occupation)
    out_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents.txt")
    with open(out_txt, "w") as f:
        f.write(df.to_string(index=False) + "\n")

    report = None
    if args.validate:
        report = check_representativeness(args.location, df)
        print_representativeness_report(report, args.n_agents)

    if args.output:
        path = args.output
        agent_table_str = df.to_string(index=False)
        if path.lower().endswith(".csv"):
            df.to_csv(path, index=False)
            if report is not None:
                report_path = path[:-4] + "_report.txt"
                with open(report_path, "w") as f:
                    f.write(format_representativeness_report(report, args.n_agents))
                print(f"\nWrote {os.path.abspath(path)} and {os.path.abspath(report_path)}")
            else:
                print(f"\nWrote {os.path.abspath(path)}")
        else:
            content = agent_table_str + "\n"
            if report is not None:
                content += format_representativeness_report(report, args.n_agents)
            with open(path, "w") as f:
                f.write(content)
            print(f"\nWrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
