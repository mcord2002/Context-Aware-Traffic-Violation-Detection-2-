import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def main():
    parser = argparse.ArgumentParser(description="Split clips by original source_id.")
    parser.add_argument("--csv", default="dataset/annotations/clips.csv")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--output-dir", default="dataset/splits")
    args = parser.parse_args()
    if args.train <= 0 or args.val <= 0 or args.train + args.val >= 1:
        raise ValueError("train and val proportions must be positive and sum to less than 1")

    df = pd.read_csv(args.csv, comment="#")
    required = {"clip_path", "label", "source_id"}
    if required - set(df.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    if df["source_id"].nunique() < 3:
        raise ValueError("At least 3 unique source_id groups are required.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=1 - args.train - args.val, random_state=42)
    trainval_idx, test_idx = next(splitter.split(df, df["label"], groups=df["source_id"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]
    val_fraction = args.val / (args.train + args.val)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=43)
    train_idx, val_idx = next(splitter2.split(trainval, trainval["label"], groups=trainval["source_id"]))
    subsets = {"train": trainval.iloc[train_idx], "val": trainval.iloc[val_idx], "test": test}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, subset in subsets.items():
        subset.to_csv(output / f"{name}.csv", index=False)
        print(f"{name}: {len(subset)} clips, {subset['source_id'].nunique()} sources, labels={subset['label'].value_counts().to_dict()}")
    print("Inspect class balance and source separation before training.")

if __name__ == "__main__":
    main()
