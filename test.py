import os
import torch
import torch.nn as nn
import yaml
from dataset import load_dataloader
from models import MMTGNN
from sklearn.metrics import roc_auc_score


def test(model, criterion, test_loader, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    all_targets, all_scores = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            all_scores.extend(output[:, 1].cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            total_correct += (output.argmax(1) == target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / len(test_loader.dataset)
    try:
        auc_score = roc_auc_score(all_targets, all_scores)
    except ValueError:
        auc_score = None
    return avg_loss, accuracy, auc_score


def main():
    with open("./configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = load_dataloader(
        cfg["dataset"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        test_ratio=cfg["test_ratio"],
        patch_size=tuple(cfg["patch_size"]),
        window_size=cfg["window_size"]
    )

    model = MMTGNN(
        num_features=cfg["model"]["num_features"],
        num_classes=cfg["model"]["num_classes"],
        back_method=cfg["model"]["back_method"],
        hidden=cfg["model"]["hidden"],
        Lf=cfg["model"]["Lf"],
        Ld=cfg["model"]["Ld"],
        Ns=cfg["model"]["Ns"],
        threshold=cfg["model"]["threshold"],
        dropout=cfg["model"]["dropout"],
        multi_head=cfg["model"]["multi_head"]
    ).to(device)

    weights_path = os.path.join(cfg["save_path"], "MMTGNN_best.pth")
    assert os.path.exists(weights_path), f"Checkpoint not found at {weights_path}"

    model.load_state_dict(torch.load(weights_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_auc = test(model, criterion, test_loader, device)

    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}" if test_auc is not None else "Test AUC: Not computable")


if __name__ == "__main__":
    main()
