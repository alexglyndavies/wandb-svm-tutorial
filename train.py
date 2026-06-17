from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb
import yaml
from matplotlib.colors import ListedColormap
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_NAME = "wandb-tutorial"
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_dataset(config: dict[str, Any]):
    """Create a rotated two-moons dataset."""
    X, y = datasets.make_moons(
        n_samples=int(config["n_samples"]),
        noise=float(config["noise"]),
        random_state=int(config["random_state"]),
    )

    # Rotate the dataset by 45 degrees
    theta = np.pi * 45 / 180
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    X = (rotation @ X.T).T

    return train_test_split(
        X,
        y,
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
        stratify=y,
    )


def make_model(config: dict[str, Any]) -> svm.SVC:
    """Build one SVM classifier from W&B/config hyperparameters."""
    return svm.SVC(
        kernel=config["kernel"],
        C=float(config["C"]),
        gamma=config["gamma"],
    )


def plot_decision_boundary(model, X_train, y_train, X_test, y_test):
    """Return a matplotlib figure with the classifier's decision boundary."""
    fig, ax = plt.subplots(figsize=(5, 4))

    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    cmap = ListedColormap(['r', 'b'])
    ax.pcolormesh(xx, yy, Z, cmap=cmap, alpha=0.35)

    c_train = [(1,0,0,1) if yi == 0 else (0,0,1,1) for yi in y_train]
    c_test = [(1,0,0,1) if yi == 0 else (0,0,1,1) for yi in y_test]
    ax.scatter(
        X_train[:, 0], 
        X_train[:, 1], 
        c=c_train, 
        s=6, 
        alpha=0.5, 
        label="train", 
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=c_test, 
        s=20,
        marker="x",
        linewidths=1.5,
        label="test",
        alpha=0.5,
    )

    DecisionBoundaryDisplay.from_estimator(
        model,
        X_train,
        ax=ax,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
        linewidths=[0.0, 1.2, 0.0],
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("SVM decision boundary")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


### --- Training - this is where we train the SVC --- ###

def train(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    project: str = PROJECT_NAME,
) -> None:
    defaults = load_yaml(config_path)

    init_kwargs = {"project": project, "config": defaults}

    with wandb.init(**init_kwargs) as run:
        config = dict(run.config)
        run.name = (
            f"{config['kernel']}-C{config['C']}-gamma{config['gamma']}-seed{config['random_state']}"
        )

        # Generate the data
        X_train, X_test, y_train, y_test = make_dataset(config)
        model = make_model(config)

        # Fit the support vector machine
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        # Predict on train and test
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Log the metrics to wandb - logged via a dictionary
        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "test_accuracy": accuracy_score(y_test, test_preds),
            "test_f1": f1_score(y_test, test_preds),
            "n_support_vectors": int(model.n_support_.sum()),
            "fit_seconds": fit_seconds,
        }
        run.log(metrics)

        # Log the plots to wandb
        if config.get("log_plot", True):
            fig = plot_decision_boundary(model, X_train, y_train, X_test, y_test)
            run.log({"decision_boundary": wandb.Image(fig)})
            plt.close(fig)


# Parse cmd line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one SVM and log it to W&B.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to defaults YAML")
    parser.add_argument("--project", default=PROJECT_NAME, help="W&B project name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config, project=args.project)
