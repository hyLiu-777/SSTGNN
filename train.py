import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dataset import load_dataloader
from models import MMTGNN
from sklearn.metrics import roc_auc_score


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss, total_correct = 0.0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (output.argmax(1) == target).sum().item()
    return total_loss / len(train_loader), total_correct / len(train_loader.dataset)


def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == target).sum().item()
    return total_loss / len(val_loader), total_correct / len(val_loader.dataset)


def main():
    with open("./configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["save_path"], exist_ok=True)

    train_loader, val_loader, _ = load_dataloader(
        cfg["dataset"], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        train_ratio=cfg["train_ratio"], val_ratio=cfg["val_ratio"], test_ratio=cfg["test_ratio"],
        patch_size=tuple(cfg["patch_size"]), window_size=cfg["window_size"]
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

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_accu, no_improve = 0.0, 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_accu = train(model, optimizer, criterion, train_loader, device)
        val_loss, val_accu = validate(model, criterion, val_loader, device)

        if val_accu >= best_val_accu:
            best_val_accu = val_accu
            no_improve = 0
            torch.save(model.state_dict(), f"{cfg['save_path']}/MMTGNN_best.pth")
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"No improvement for {cfg['patience']} epochs. Early stopping.")
                break

        if epoch % cfg["display_iter"] == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Train accuracy: {train_accu:.4f}, Val accuracy: {val_accu:.4f}")
            sys.stdout.flush()

        if epoch % cfg["snapshot_iter"] == 0:
            torch.save(model.state_dict(), f"{cfg['save_path']}/MMTGNN_epoch{epoch}.pth")


if __name__ == '__main__':
    main()
