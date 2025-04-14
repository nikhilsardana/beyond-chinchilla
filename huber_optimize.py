"""
Calculate the Chinchilla coefficients for your dataset based on your empirical training runs
by minimizing the huber loss, following the algorithm in the original paper.
"""

import numpy as np
import polars as pl
import scipy
from scipy.special import logsumexp, huber
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools
import argparse


def huber_loss(params, delta, df):
    # optimizing over params, df is constant
    (
        alpha,
        beta,
        e,
        a,
        b,
    ) = params
    huber_sum = 0

    for row in df.rows(named=True):
        N_i = row["Parameters"]
        D_i = row["Tokens"]
        L_i = row["Smoothed Loss"]
        assert L_i != 0 and D_i != 0 and N_i != 0
        residual = logsumexp(
            [a - alpha * np.log(N_i), b - beta * np.log(D_i), e]
        ) - np.log(L_i)
        huber_sum += huber(delta, residual)
    return huber_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the Chinchilla coefficients for your dataset based on your empirical training runs by minimizing the huber loss, following the algorithm in the original paper."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="CSV file with your result data from your training runs. This result data must have a `Smoothed Loss` column, a `Parameters` column (for model size), a `Tokens` column (training data tokens), and a 'Tokens/Param' column.",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        required=False,
        help="Optional. Only fit coefficients on runs with tokens/parameter <= `cutoff`. If not provided, uses all runs.",
    )

    args = parser.parse_args()

    df = pl.scan_csv(args.data, has_header=True).collect()

    assert (
        "Parameters" in df.columns
    ), "DataFrame must contain a `Parameters` column with the model Parameter count."
    assert (
        "Smoothed Loss" in df.columns
    ), "DataFrame must contain a `Smoothed Loss` column with the smoothed final training loss."
    assert (
        "Tokens" in df.columns
    ), "DataFrame must contain a `Tokens` column with the amount of training data tokens."
    assert (
        "Tokens/Params" in df.columns
    ), "DataFrame must contain a `Tokens/Param` column."

    if args.cutoff:
        df = df.filter(pl.col("Tokens/Params") <= args.cutoff)

    print(len(df))

    print(sorted(df["Tokens/Params"].unique()))
    print(df["Parameters"].unique())

    alpha_sweep = [0, 0.5, 1, 1.5, 2]
    beta_sweep = [0, 0.5, 1, 1.5, 2]
    e_sweep = [-1, -0.5, 0, 0.5, 1]
    a_sweep = [0, 5, 10, 15, 20, 25]
    b_sweep = [0, 5, 10, 15, 20, 25]
    delta = 1e-3
    # Sweep to find the parameters that minimize the Huber loss.
    min_hl = float("inf")
    search_space = list(
        itertools.product(alpha_sweep, beta_sweep, e_sweep, a_sweep, b_sweep)
    )

    def optimize_task(initial_guess):
        result = scipy.optimize.minimize(
            huber_loss, initial_guess, (delta, df), method="L-BFGS-B"
        )
        hl = huber_loss(result.x, delta, df)
        return result, hl

    results = Parallel(n_jobs=-1)(
        delayed(optimize_task)(guess) for guess in tqdm(search_space)
    )

    min_hl = float("inf")

    # Find the overall minimum
    for result, hl in results:
        if hl < min_hl:
            min_hl = hl
            opt_alpha, opt_beta, opt_e, opt_a, opt_b = result.x

    print([opt_alpha, opt_beta, np.exp(opt_a), np.exp(opt_b), np.exp(opt_e)])
