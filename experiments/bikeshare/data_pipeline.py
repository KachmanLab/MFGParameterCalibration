"""
Data pipeline for Citi Bike trip data.
Downloads, processes, and converts trip-level CSV into mean-field trajectories.
"""
import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import json

# ==========================================
# CONFIGURATION
# ==========================================

# URL template for Citi Bike monthly data (YYYYMM-citibike-tripdata.csv.zip)
DATA_URL = "https://s3.amazonaws.com/tripdata/{filename}"

# 5 cluster stations in Midtown Manhattan (will be selected from the data)
# We'll pick the top-5 busiest stations to ensure rich dynamics
NUM_CLUSTER_STATIONS = 5

# Time bins
BIN_MINUTES = 15
START_HOUR = 6   # 6am
END_HOUR = 22    # 10pm
BINS_PER_DAY = (END_HOUR - START_HOUR) * 60 // BIN_MINUTES  # = 64

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_citibike_data(year_month="202409"):
    """Download Citi Bike trip data for a given month (YYYYMM format)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    csv_path = os.path.join(DATA_DIR, f"{year_month}-citibike-tripdata.csv")
    if os.path.exists(csv_path):
        print(f"Data already exists: {csv_path}")
        return csv_path
    
    # Try common filename patterns
    filenames = [
        f"{year_month}-citibike-tripdata.csv.zip",
        f"{year_month}-citibike-tripdata.zip",
    ]
    
    for fname in filenames:
        url = DATA_URL.format(filename=fname)
        print(f"Trying to download: {url}")
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code == 200:
            print(f"Downloading {fname}...")
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            # Extract all CSV files
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                continue
            
            # If multiple CSVs, concatenate them
            dfs = []
            for cf in csv_files:
                print(f"  Extracting {cf}...")
                with z.open(cf) as f:
                    dfs.append(pd.read_csv(f))
            
            df = pd.concat(dfs, ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path} ({len(df)} trips)")
            return csv_path
    
    raise RuntimeError(f"Could not download data for {year_month}. "
                       "Please download manually from https://citibikenyc.com/system-data")


def select_cluster_stations(df, k=NUM_CLUSTER_STATIONS):
    """Select the top-k busiest stations to form a cluster."""
    # Count trips involving each station (as start or end)
    start_counts = df['start_station_id'].value_counts()
    end_counts = df['end_station_id'].value_counts()
    total_counts = start_counts.add(end_counts, fill_value=0).sort_values(ascending=False)
    
    # Pick top-k
    cluster_ids = list(total_counts.index[:k])
    
    # Get station names for reference
    station_names = {}
    for sid in cluster_ids:
        name_rows = df[df['start_station_id'] == sid]['start_station_name']
        if len(name_rows) > 0:
            station_names[sid] = name_rows.iloc[0]
        else:
            station_names[sid] = str(sid)
    
    print(f"\nSelected cluster stations (top {k} busiest):")
    for i, sid in enumerate(cluster_ids):
        print(f"  State {i}: {station_names[sid]} (id={sid}, {int(total_counts[sid])} trips)")
    print(f"  State {k}: External (all other stations)")
    
    return cluster_ids, station_names


def compute_daily_trajectories(df, cluster_ids, bin_minutes=BIN_MINUTES, weekdays_only=False):
    """
    Convert trip data into daily mean-field trajectories.
    
    Args:
        weekdays_only: if True, only keep weekdays. If False, keep all days.
    
    Returns:
        trajectories: dict mapping date_str -> np.array of shape (N, d)
        day_types: dict mapping date_str -> 'weekday' or 'weekend'
    """
    d = len(cluster_ids) + 1  # +1 for external state
    ext_idx = len(cluster_ids)  # index of external state
    
    # Map station IDs to cluster indices
    id_to_idx = {sid: i for i, sid in enumerate(cluster_ids)}
    
    # Parse timestamps
    df = df.copy()
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df['date'] = df['started_at'].dt.date
    df['weekday'] = df['started_at'].dt.weekday
    
    if weekdays_only:
        df = df[df['weekday'] < 5]
    
    # Map stations to indices (external = ext_idx)
    df['start_idx'] = df['start_station_id'].map(id_to_idx).fillna(ext_idx).astype(int)
    df['end_idx'] = df['end_station_id'].map(id_to_idx).fillna(ext_idx).astype(int)
    
    # Time bin computation
    df['start_minutes'] = df['started_at'].dt.hour * 60 + df['started_at'].dt.minute
    df['start_bin'] = (df['start_minutes'] - START_HOUR * 60) // bin_minutes
    
    # Filter to our time window
    df = df[(df['start_bin'] >= 0) & (df['start_bin'] < BINS_PER_DAY)]
    
    trajectories = {}
    day_types = {}
    
    for date, day_df in df.groupby('date'):
        date_str = str(date)
        is_weekend = day_df['weekday'].iloc[0] >= 5
        day_types[date_str] = 'weekend' if is_weekend else 'weekday'
        
        # Vectorized: count departures and arrivals per bin per station
        departures = np.zeros((BINS_PER_DAY, d))
        arrivals = np.zeros((BINS_PER_DAY, d))
        
        # Departures: group by (start_bin, start_idx) and count
        dep_counts = day_df.groupby(['start_bin', 'start_idx']).size()
        for (b, s), count in dep_counts.items():
            if 0 <= b < BINS_PER_DAY and 0 <= s < d:
                departures[int(b), int(s)] = count
        
        # Arrivals: group by (start_bin, end_idx) and count
        arr_counts = day_df.groupby(['start_bin', 'end_idx']).size()
        for (b, s), count in arr_counts.items():
            if 0 <= b < BINS_PER_DAY and 0 <= s < d:
                arrivals[int(b), int(s)] = count
        
        # Net change per bin
        net_change = arrivals - departures
        cumulative = np.cumsum(net_change, axis=0)
        
        # Estimate initial counts
        init_counts = np.ones(d) * 100.0
        init_counts[ext_idx] = 500.0
        
        counts = init_counts[None, :] + cumulative
        counts = np.maximum(counts, 1.0)
        
        # Normalize to get distribution
        mu = counts / counts.sum(axis=1, keepdims=True)
        trajectories[date_str] = mu
    
    return trajectories, day_types


def prepare_dataset(year_month="202409", k=NUM_CLUSTER_STATIONS, train_ratio=0.75):
    """
    Full pipeline: download, process, and split into train/test_id/test_ood.
    
    Returns:
        train_data, test_id_data, test_ood_data, metadata
    """
    csv_path = download_citibike_data(year_month)
    
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} trips")
    
    cluster_ids, station_names = select_cluster_stations(df, k)
    
    # Process ALL days (weekdays + weekends)
    trajectories, day_types = compute_daily_trajectories(df, cluster_ids, weekdays_only=False)
    
    # Split: weekdays -> train + test_id, weekends -> test_ood
    weekday_dates = sorted([d for d, t in day_types.items() if t == 'weekday'])
    weekend_dates = sorted([d for d, t in day_types.items() if t == 'weekend'])
    
    n_train = int(len(weekday_dates) * train_ratio)
    train_dates = weekday_dates[:n_train]
    test_id_dates = weekday_dates[n_train:]
    test_ood_dates = weekend_dates
    
    print(f"\nTotal trajectories: {len(trajectories)} ({len(weekday_dates)} weekdays, {len(weekend_dates)} weekends)")
    print(f"Train:    {len(train_dates)} weekdays ({train_dates[0]} to {train_dates[-1]})")
    print(f"Test ID:  {len(test_id_dates)} weekdays ({test_id_dates[0]} to {test_id_dates[-1]})")
    print(f"Test OOD: {len(test_ood_dates)} weekends ({test_ood_dates[0]} to {test_ood_dates[-1]})")
    
    train_data = [trajectories[d] for d in train_dates]
    test_id_data = [trajectories[d] for d in test_id_dates]
    test_ood_data = [trajectories[d] for d in test_ood_dates]
    
    metadata = {
        'cluster_ids': cluster_ids,
        'station_names': {str(k): v for k, v in station_names.items()},
        'train_dates': train_dates,
        'test_id_dates': test_id_dates,
        'test_ood_dates': test_ood_dates,
        'd': k + 1,
        'N': BINS_PER_DAY,
        'bin_minutes': BIN_MINUTES,
        'start_hour': START_HOUR,
        'end_hour': END_HOUR,
    }
    
    # Save processed data
    processed_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, "train_data.npy"), np.array(train_data))
    np.save(os.path.join(processed_dir, "test_data.npy"), np.array(test_id_data))
    np.save(os.path.join(processed_dir, "test_ood_data.npy"), np.array(test_ood_data))
    with open(os.path.join(processed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved processed data to {processed_dir}/")
    print(f"  train_data.npy:     shape {np.array(train_data).shape}")
    print(f"  test_data.npy:      shape {np.array(test_id_data).shape}")
    print(f"  test_ood_data.npy:  shape {np.array(test_ood_data).shape}")
    
    return train_data, test_id_data, test_ood_data, metadata


if __name__ == "__main__":
    train, test_id, test_ood, meta = prepare_dataset()

