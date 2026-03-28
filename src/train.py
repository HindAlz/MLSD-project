from pathlib import Path
import sys
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent / "non-testing"))

from data import get_dataloaders
from model import SimpleMLP
from utils import ensure_dir, save_confusion_matrix, save_metric_plot

from pathlib import Path

FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
CHECKPOINTS_DIR = Path("outputs/checkpoints")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


best_model_path = CHECKPOINTS_DIR / "best_model.pt"
report_path = REPORTS_DIR / "classification_report.txt"
cm_path = FIGURES_DIR / "confusion_matrix.png"
loss_plot_path = FIGURES_DIR / "loss_curve.png"
acc_plot_path = FIGURES_DIR / "accuracy_curve.png"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(yb.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, f1, precision, recall


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(yb.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, f1, precision, recall, all_targets, all_preds


def main():
    config = {
        "experiment_name": "wine-classification",
        "run_name": "mlp_baseline",
        "batch_size": 16,
        "lr": 0.001,
        "epochs": 20,
        "hidden_note": "baseline MLP on wine dataset",
        "seed": 42,
    }

    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir("artifacts")

    train_loader, val_loader, input_dim, num_classes = get_dataloaders(
        batch_size=config["batch_size"]
    )

    model = SimpleMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params(config)
        mlflow.log_param("device", str(device))
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("dataset", "wine")
        mlflow.set_tag("purpose", "per_epoch_tracking_demo")

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        best_val_acc = -1.0
        best_model_path = Path("artifacts/best_model.pt")

        for epoch in range(config["epochs"]):
            train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            val_loss, val_acc, val_f1, val_precision, val_recall, y_true, y_pred = evaluate(
                model, val_loader, criterion, device
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Per-step logging
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("train_precision", train_precision, step=epoch)
            mlflow.log_metric("train_recall", train_recall, step=epoch)

            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            mlflow.log_metric("val_precision", val_precision, step=epoch)
            mlflow.log_metric("val_recall", val_recall, step=epoch)

            print(
                f"Epoch {epoch + 1}/{config['epochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

        # Final artifacts
        report = classification_report(y_true, y_pred, zero_division=0)
        report_path = Path("artifacts/classification_report.txt")
        report_path.write_text(report, encoding="utf-8")

        cm_path = Path("artifacts/confusion_matrix.png")
        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=[str(i) for i in range(num_classes)],
            out_path=str(cm_path),
        )

        loss_plot_path = Path("artifacts/loss_curve.png")
        save_metric_plot(train_losses, val_losses, "loss", str(loss_plot_path))

        acc_plot_path = Path("artifacts/accuracy_curve.png")
        save_metric_plot(train_accs, val_accs, "accuracy", str(acc_plot_path))

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(loss_plot_path))
        mlflow.log_artifact(str(acc_plot_path))
        mlflow.log_artifact(str(best_model_path))

        # Log whole model
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Final summary metrics
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        print("\nTraining complete.")
        print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()