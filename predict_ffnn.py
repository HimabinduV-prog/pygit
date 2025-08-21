#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import joblib

import matplotlib.pyplot as plt


def parse_feature_cols(arg: str):
    if "," in arg:
        return [int(x.strip()) for x in arg.split(",") if x.strip() != ""]
    return [int(arg)]


def load_features(path: Path, feature_cols):
    data = np.loadtxt(str(path))
    features = data[:, feature_cols]
    return features, data


def main():
    parser = argparse.ArgumentParser(description="Infer clusters using trained FFNN (MLP) classifier")
    parser.add_argument("--input", type=Path, required=True, help="Input data file (np.loadtxt-readable)")
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/ffnn_mlp_classifier.joblib"), help="Path to trained model .joblib")
    parser.add_argument("--feature-cols", type=parse_feature_cols, default=[0, 2, 4], help="Comma-separated feature column indices, e.g. '0,2,4'")
    parser.add_argument("--save", type=Path, default=Path("/workspace/models/predicted_clusters.txt"), help="Where to save predicted labels")
    parser.add_argument("--plot", action="store_true", help="Plot stress-strain colored by predicted cluster (assumes col 0=strain_z, 1=stress)")
    args = parser.parse_args()

    X, full_data = load_features(args.input, args.feature_cols)

    pipeline = joblib.load(args.model)
    y_pred = pipeline.predict(X)

    # Save labels
    np.savetxt(str(args.save), y_pred, fmt="%d")
    print(f"Saved predicted cluster labels to: {args.save}")

    if args.plot:
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(full_data[:, 0], full_data[:, 1], c=y_pred, cmap='viridis', s=10)
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i+1}' for i in range(len(np.unique(y_pred)))])
        plt.xlabel("Strain (z-direction)")
        plt.ylabel("Stress (zz-direction, GPa)")
        plt.title("Stress-Strain with FFNN-predicted clusters")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

