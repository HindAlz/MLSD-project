from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, class_names, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def save_metric_plot(train_values, val_values, metric_name: str, out_path: str):
    epochs = np.arange(1, len(train_values) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_values, label=f"train_{metric_name}")
    plt.plot(epochs, val_values, label=f"val_{metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()