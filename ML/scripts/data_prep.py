import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Standard CMAPSS column names
INDEX_NAMES = ['unit_nr', 'time_cycles']
SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
SENSOR_NAMES = [f's_{i}' for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

def load_data(file_path):
    return pd.read_csv(file_path, sep=r'\s+', header=None, names=COL_NAMES)

def generate_rul(df, max_rul=125):
    rul = pd.DataFrame(df.groupby('unit_nr')['time_cycles'].max()).reset_index()
    rul.columns = ['unit_nr', 'max']
    df = df.merge(rul, on=['unit_nr'], how='left')
    df['RUL'] = df['max'] - df['time_cycles']
    df.drop('max', axis=1, inplace=True)
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    return df

def drop_constant_sensors(train_df, test_df):
    # Only check sensors that actually exist in the dataframe
    available_sensors = [s for s in SENSOR_NAMES if s in train_df.columns]
    std_dev = train_df[available_sensors].std()
    constant_sensors = std_dev[std_dev < 1e-5].index.tolist()
    
    print(f"[*] Removing {len(constant_sensors)} non-informative constant sensors: {constant_sensors}")
    
    train_df = train_df.drop(columns=constant_sensors)
    test_df = test_df.drop(columns=constant_sensors)
    
    remaining_sensors = [s for s in available_sensors if s not in constant_sensors]
    return train_df, test_df, remaining_sensors

def extract_operating_conditions(df, n_clusters=6):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    settings_data = df[SETTING_NAMES].values
    labels = km.fit_predict(settings_data)
    df['domain'] = labels
    return df, km

def condition_aware_normalization(train_df, test_df, sensor_cols, existing_scalers=None):
    """Normalize sensor columns per operating condition.
    
    If existing_scalers is provided, use them instead of fitting new ones.
    This is critical for val/test data — they must use scalers fitted on
    the ORIGINAL training data.
    """
    print("[*] Applying Condition-Aware Normalization...")
    scalers = existing_scalers if existing_scalers is not None else {}
    domains = train_df['domain'].unique()
    
    norm_train_df = train_df.copy()
    norm_test_df = test_df.copy()
    
    for d in domains:
        train_domain_mask = train_df['domain'] == d
        test_domain_mask = test_df['domain'] == d
        
        if existing_scalers is None:
            # Fit new scalers on raw training data
            scaler = StandardScaler()
            vals = scaler.fit_transform(train_df.loc[train_domain_mask, sensor_cols])
            norm_train_df.loc[train_domain_mask, sensor_cols] = vals
            scalers[d] = scaler
        else:
            scaler = scalers.get(d)
            if scaler is None:
                continue
        
        if test_domain_mask.sum() > 0:
            vals_t = scaler.transform(test_df.loc[test_domain_mask, sensor_cols])
            norm_test_df.loc[test_domain_mask, sensor_cols] = vals_t
        
    return norm_train_df, norm_test_df, scalers

def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

def gen_labels(id_df, seq_length, label_col):
    data_array = id_df[label_col].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements]

def engine_wise_train_val_split(train_df, val_split_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    all_units = train_df['unit_nr'].unique()
    np.random.shuffle(all_units)
    
    n_val = int(len(all_units) * val_split_ratio)
    val_units = all_units[:n_val]
    train_units = all_units[n_val:]
    
    print(f"[*] Engine-wise Split: {len(train_units)} engines for train, {len(val_units)} for validation.")
    
    val_df = train_df[train_df['unit_nr'].isin(val_units)].copy()
    train_df = train_df[train_df['unit_nr'].isin(train_units)].copy()
    return train_df, val_df

class CMAPSSDataset(Dataset):
    def __init__(self, sequences, rul_labels, domain_labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.rul_labels = torch.tensor(rul_labels, dtype=torch.float32)
        self.domain_labels = torch.tensor(domain_labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.rul_labels[idx], self.domain_labels[idx]

# The RUL cap used during label generation — exposed so callers can rescale.
RUL_CAP = 125

def prepare_data(data_dir, sequence_length=50):
    train_path = os.path.join(data_dir, 'train_FD004.txt')
    test_path = os.path.join(data_dir, 'test_FD004.txt')
    rul_path = os.path.join(data_dir, 'RUL_FD004.txt')
    
    raw_train = load_data(train_path)
    raw_test = load_data(test_path)
    true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    
    raw_train = generate_rul(raw_train)
    raw_train, kmeans_model = extract_operating_conditions(raw_train)
    settings_data_test = raw_test[SETTING_NAMES].values
    raw_test['domain'] = kmeans_model.predict(settings_data_test)
    
    train_df, val_df = engine_wise_train_val_split(raw_train, val_split_ratio=0.2)
    
    train_df, raw_test, active_sensors = drop_constant_sensors(train_df, raw_test)
    _, val_df, _ = drop_constant_sensors(train_df.copy(), val_df)
    
    train_df, test_df, scalers = condition_aware_normalization(train_df, raw_test, active_sensors)
    # Reuse the SAME scalers (fitted on raw training data) for validation
    _, val_df, _ = condition_aware_normalization(train_df, val_df, active_sensors, existing_scalers=scalers)
    
    features = active_sensors
    print(f"[*] Extracting rolling window sequences (length {sequence_length})")
    
    def process_df(df):
        seqs, ruls, domains = [], [], []
        for unit in df['unit_nr'].unique():
            unit_df = df[df['unit_nr'] == unit]
            if len(unit_df) <= sequence_length:
                continue
            seq = list(gen_sequence(unit_df, sequence_length, features))
            seqs.extend(seq)
            
            rul = gen_labels(unit_df, sequence_length, ['RUL'])
            ruls.extend(rul)
            
            domain = gen_labels(unit_df, sequence_length, ['domain'])
            domains.extend(domain)
        return np.array(seqs), np.array(ruls).flatten(), np.array(domains).flatten()
        
    train_seqs, train_ruls, train_domains = process_df(train_df)
    val_seqs, val_ruls, val_domains = process_df(val_df)
    
    test_seqs, test_ruls, test_domains = [], [], []
    for i, unit in enumerate(test_df['unit_nr'].unique()):
        unit_df = test_df[test_df['unit_nr'] == unit]
        if len(unit_df) >= sequence_length:
            seq = unit_df[features].values[-sequence_length:]
            test_seqs.append(seq)
            test_ruls.append(true_rul.iloc[i].values[0])
            test_domains.append(unit_df['domain'].values[-1])
            
    test_seqs = np.array(test_seqs)
    test_ruls = np.array(test_ruls)
    test_domains = np.array(test_domains)

    # ── Fix 5: Normalize RUL targets to [0, 1] ──
    # This keeps MSE gradients small and training stable.
    # Test RUL is also capped at RUL_CAP before normalizing so the
    # model sees a consistent target range.
    train_ruls = train_ruls / RUL_CAP
    val_ruls   = val_ruls   / RUL_CAP
    test_ruls  = np.minimum(test_ruls, RUL_CAP) / RUL_CAP
    print(f"[*] RUL targets normalized to [0, 1]  (cap = {RUL_CAP} cycles)")
    
    print(f"[*] Train Samples: {len(train_seqs)}, Val Samples: {len(val_seqs)}, Test Engines: {len(test_seqs)}")
    
    train_ds = CMAPSSDataset(train_seqs, train_ruls, train_domains)
    val_ds = CMAPSSDataset(val_seqs, val_ruls, val_domains)
    test_ds = CMAPSSDataset(test_seqs, test_ruls, test_domains)
    
    return train_ds, val_ds, test_ds, len(features)
