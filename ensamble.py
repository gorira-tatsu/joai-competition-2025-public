import numpy as np
import os
import glob
import sys
from table_xgb import EnvConfig
from pathlib import Path
import pandas as pd

NAME2LABEL = {"Mixture": 0, "NoGas": 1, "Perfume": 2, "Smoke": 3}
LABEL2NAME = {v: k for k, v in NAME2LABEL.items()}

def main():
    input_dir = "ensamble"
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not files:
        print(f"Error: No .npy files found in directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    arrays = []
    for path in files:
        try:
            arr = np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            sys.exit(1)
        arrays.append(arr)

    shapes = [a.shape for a in arrays]
    if len(set(shapes)) != 1:
        print(f"Error: Input arrays have mismatched shapes: {shapes}", file=sys.stderr)
        sys.exit(1)

    stacked = np.stack(arrays, axis=0)
    ensemble = np.mean(stacked, axis=0)

    output_path = "ensemble.npy"
    try:
        np.save(output_path, ensemble)
        print(f"Ensembled {len(arrays)} files. Saved to {output_path}")
        try:
            env = EnvConfig()
            sample_path = env.data_dir / "sample_submission.csv"
            sample_sub = pd.read_csv(sample_path)
            preds = np.argmax(ensemble, axis=1)
            labels = [LABEL2NAME[p] for p in preds]
            sample_sub["Gas"] = labels
            submission_path = "submission_ensemble.csv"
            sample_sub.to_csv(submission_path, index=False)
            print(f"Submission file saved to: {submission_path}")
        except Exception as e:
            print(f"Error creating submission file: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()