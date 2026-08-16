import pandas as pd
import numpy as np
import ast
import os


def parse_class_list(s):
    """
    Parse the 'class' column, which looks like '[point]' or
    '[contextual, contextual]' but isn't valid Python syntax
    (no quotes around the words), so ast.literal_eval can't handle it.
    """
    return [item.strip() for item in s.strip("[]").split(",")]


def load_labels(csv_path):
    """
    Read labeled_anomalies.csv and parse anomaly_sequences and class
    from string to list. Merges duplicate chan_id rows (some channels
    have multiple labeled events) so no anomaly sequences are silently
    dropped. Returns a DataFrame indexed by chan_id.
    """
    df = pd.read_csv(csv_path)
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
    df["class"] = df["class"].apply(parse_class_list)

    df = df.groupby("chan_id").agg({
        "spacecraft": "first",
        "anomaly_sequences": lambda seqs: [seq for group in seqs for seq in group],
        "class": lambda classes: [c for group in classes for c in group],
        "num_values": "first",
    })

    return df


def load_channel(channel_id, data_dir):
    """
    Load train and test .npy arrays for a single channel.
    Returns (train_array, test_array).
    """
    train_path = os.path.join(data_dir, "train", f"{channel_id}.npy")
    test_path = os.path.join(data_dir, "test", f"{channel_id}.npy")

    train_arr = np.load(train_path)
    test_arr = np.load(test_path)

    return train_arr, test_arr


def load_all_channels(data_dir, labels_df):
    """
    Loop through every channel in labels_df, load train/test arrays.
    Returns a dict: {channel_id: (train_array, test_array)}
    """
    channels = {}
    for channel_id in labels_df.index:
        try:
            train_arr, test_arr = load_channel(channel_id, data_dir)
            channels[channel_id] = (train_arr, test_arr)
        except FileNotFoundError:
            print(f"Warning: missing .npy files for channel {channel_id}, skipping")
    return channels


if __name__ == "__main__":
    labels_path = os.path.join("data", "labeled_anomalies.csv")
    labels_df = load_labels(labels_path)

    print(labels_df.head())
    print(f"\nTotal channels in labels: {len(labels_df)}")

    # confirm the P-2 merge worked
    print("\nP-2 merged entry:")
    print(labels_df.loc["P-2"])

    # sanity check on one channel
    train, test = load_channel("P-1", "data")
    print(f"\nP-1 train shape: {train.shape}")
    print(f"P-1 test shape: {test.shape}")

    # load everything
    all_channels = load_all_channels("data", labels_df)
    print(f"\nSuccessfully loaded {len(all_channels)} channels")

    missing = set(labels_df.index) - set(all_channels.keys())
    print(f"Missing channel(s): {missing}")