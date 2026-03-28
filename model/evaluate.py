import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def build_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model

def load_model(checkpoint_path, device):
    model = build_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def get_test_loader(cfg):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_dataset = ImageFolder(root="data/cifake/test", transform=transform)
    return DataLoader(test_dataset, batch_size=cfg['batch_size'],
                      shuffle=False, num_workers=2)

def get_train_loader(cfg):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_dataset = ImageFolder(root="data/cifake/train", transform=transform)
    return DataLoader(train_dataset, batch_size=cfg['batch_size'],
                      shuffle=True, num_workers=2)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

def evaluate_checkpoint(checkpoint_path, label=""):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    test_loader = get_test_loader(cfg)
    acc = evaluate(model, test_loader, device)
    print(f"Accuracy [{label}]: {acc:.6f}")
    return acc

def compare_before_after(before_path, after_path):
    print("\n--- Performance Degradation Check ---")
    acc_before = evaluate_checkpoint(before_path, label="BEFORE watermarking")
    acc_after  = evaluate_checkpoint(after_path,  label="AFTER  watermarking")
    delta = abs(acc_before - acc_after)
    print(f"Delta:  {delta:.8f}")
    if delta < 1e-4:
        print("PASS — Zero performance degradation confirmed")
    else:
        print("WARN — Accuracy shifted, check watermarking code")
    return acc_before, acc_after

if __name__ == "__main__":
    cfg = load_config()
    evaluate_checkpoint(cfg['baseline_model'], label="clean_resnet18_baseline")