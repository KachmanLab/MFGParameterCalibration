"""
Based on the data collected by both the U.S. Influenza Collaborating Laboratories (ICL) and National Respiratory and
Enteric Virus Surveillance System (NREVSS).
"""

from typing import Tuple
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


TRAIN_MAX = 2022
TEST_MIN = TRAIN_MAX


def find_epidemic_onset(I_obs, threshold=0.005, min_increase=3):
    for t in range(len(I_obs) - min_increase):
        if I_obs[t] > threshold:
            if all(I_obs[t + k + 1] > I_obs[t + k] for k in range(min_increase)):
                return t
    return np.argmax(I_obs)  # Return the highest peak if no onset is found


def realign_curve(I_obs, t_start, T_out=49):
    """
    Re-index I_obs so that t=0 corresponds to t_start.
    Pads with zeros at the end if the remaining curve is shorter than T_out,
    and truncates if longer.
    """
    I_aligned = I_obs[t_start:]

    # Pad with zeros if necessary (epidemic ended before T_out)
    if len(I_aligned) < T_out:
        I_aligned = np.concatenate([I_aligned, np.zeros(T_out - len(I_aligned))])

    return I_aligned[:T_out]


def get_custom_period(df, year_i):
    # Calculate the start date of week 35 in year_i
    start_date = datetime.fromisocalendar(year_i, 35, 1)  # Monday of week 35

    # Calculate the end date of week 50 in year i+1
    end_date = datetime.fromisocalendar(year_i + 1, 40, 7)  # Sunday of week 30
    return df[(df["date"] >= start_date) & (df["date"] <= end_date)]


def date_generator(start_date: str, n: int):
    """
    Generator that yields numpy arrays of datetimes,
    each 7 days apart from the last.

    Args:
        start_date: Starting date string in 'YYYY-MM-DD' format
        n: Number of datetime entries to generate
    """
    start = np.datetime64(start_date)
    for i in range(n):
        yield start + np.timedelta64(i * 7, "D")


def get_ICL_NREVSS_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    df = pd.read_csv(config["load_from_dir"])
    ts, mu_infected, mu_init, metadata = preprocess_ICL_NREVSS(df, config["train"])
    return ts, mu_infected, mu_init, metadata


def get_ICL_NREVSS_data_from_path(path: str, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    df = pd.read_csv(path)
    ts, mu_infected, mu_init, metaddata = preprocess_ICL_NREVSS(df, train)
    return mu_infected, mu_init, np.zeros((mu_infected.shape[0], mu_infected.shape[1], 2)), metaddata


def preprocess_ICL_NREVSS(df: pd.DataFrame, train: bool):
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["WEEK"].astype(str) + "1",
        format="%Y%W%w",
    )

    state_list = []
    ts_list = []
    metadata = []

    T = 30

    if train:
        years = np.arange(2015, TRAIN_MAX, 1)
    else:
        years = np.arange(TEST_MIN, 2026, 1)

    for region in df["REGION"].unique():
        for year in years:
            df_filter = df[(df["REGION"] == region)]
            df_filter = get_custom_period(df_filter, year)
            df_filter = df_filter.sort_values("date")

            df.replace("X", np.nan, inplace=True)
            df_filter["PERCENT A"] = df_filter["PERCENT A"].astype(float).interpolate()

            df_filled = df_filter.copy()
            df_filled["PERCENT A"] = df_filled["PERCENT A"].astype(float).fillna(0)

            mu_infected = df_filled["PERCENT A"].to_numpy(dtype=float) / 100

            if len(mu_infected) > 0 and np.count_nonzero(mu_infected == 0) < mu_infected.shape[0] / 2:
                t_start = find_epidemic_onset(mu_infected)
                mu_aligned = realign_curve(mu_infected, t_start, T_out=T)

                state_list.append(mu_aligned)
                ts_list.append(np.arange(T))

                dates = np.array(list(date_generator(df_filled["date"].iloc[t_start], len(mu_aligned))))

                metadata.append([region, year, dates])

    mu_infected = np.column_stack(state_list).T
    mu_susceptible = np.ones_like(mu_infected)
    mu_recovered = np.zeros_like(mu_infected)

    mu = np.stack([mu_susceptible, mu_infected, mu_recovered], axis=-1)

    ts = np.column_stack(ts_list).T

    return ts, mu, mu[:, 0, :], metadata


if __name__ == "__main__":
    mu, mu0, gs, meta = get_ICL_NREVSS_data_from_path(r"FluViewPhase2Data/ICL_NREVSS_Clinical_Labs.csv", train=True)

    print(f"Data shape: {mu.shape}\nNumber of missing values: {np.isnan(mu).sum()}")

    ts = np.arange(mu.shape[1])
    fig, axes = plt.subplots(1, 1)

    for sample in range(mu.shape[0]):
        axes.plot(ts, mu[sample, :, 1])

        if sample == 100:
            break

    plt.show()
