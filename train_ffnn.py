#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def parse_feature_cols(arg: str):
    if "," in arg:
        return [int(x.strip()) for x in arg.split(",") if x.strip() != ""]
    return [int(arg)]


def load_features(path: Path, feature_cols):
    data = np.loadtxt(str(path))
    features = data[:, feature_cols]
    return features, data


def main():
    parser = argparse.ArgumentParser(description="Train FFNN (MLP) to mimic KMeans clustering on strain features")
    parser.add_argument("--input", type=Path, default=Path("/workspace/uni_stress_strain_300K.out"), help="Input data file (np.loadtxt-readable)")
    parser.add_argument("--feature-cols", type=parse_feature_cols, default=[0, 2, 4], help="Comma-separated feature column indices, e.g. '0,2,4'")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters for KMeans labels")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--hidden-layers", type=parse_feature_cols, default=[64, 64], help="Comma-separated hidden layer sizes, e.g. '64,64'")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations for MLP training")
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/models"), help="Directory to save trained artifacts")
    parser.add_argument("--save-kmeans", action="store_true", help="Also save the KMeans model used to generate labels")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)

    # Load data and select features
    X, full_data = load_features(args.input, args.feature_cols)

    # Generate labels via KMeans (on scaled features as in your clustering step)
    kmeans_scaler = StandardScaler()
    X_for_kmeans = kmeans_scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=args.k, random_state=args.random_state, n_init=10)
    y = kmeans.fit_predict(X_for_kmeans)

    # Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # FFNN pipeline: scaler + MLP classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=args.max_iter,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=args.random_state,
        verbose=False,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cls_report = classification_report(y_test, y_pred_test, output_dict=False)
    conf_mat = confusion_matrix(y_test, y_pred_test)

    print("Train accuracy:", train_acc)
    print("Test accuracy:", test_acc)
    print("Classification report (test):\n" + classification_report(y_test, y_pred_test))
    print("Confusion matrix (test):\n", conf_mat)

    # Save artifacts
    model_path = args.model_dir / "ffnn_mlp_classifier.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "input_file": str(args.input.resolve()),
        "feature_cols": args.feature_cols,
        "k": args.k,
        "random_state": args.random_state,
        "hidden_layers": args.hidden_layers,
        "max_iter": args.max_iter,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
    }
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    (args.model_dir / "classification_report.txt").write_text(
        "Train accuracy: %.6f\nTest accuracy: %.6f\n\n" % (train_acc, test_acc) + cls_report
    )

    if args.save_kmeans:
        joblib.dump({
            "kmeans": kmeans,
            "scaler": kmeans_scaler,
            "feature_cols": args.feature_cols,
        }, args.model_dir / "kmeans_labels.joblib")

    print(f"Saved FFNN model to: {model_path}")
    print(f"Saved metadata and report to: {args.model_dir}")


if __name__ == "__main__":
    main()

