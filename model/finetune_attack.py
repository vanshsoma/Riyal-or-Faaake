import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
import yaml
import os

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def build_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def simulate_attack(cfg, device):
    print("\n--- Simulating Theft + Fine-tune Attack ---")

    stolen_model = build_model()
    stolen_model.load_state_dict(
        torch.load(cfg['watermarked_model'], map_location=device)
    )
    stolen_model = stolen_model.to(device)

    # Freeze early layers — realistic attacker behaviour
    for name, param in stolen_model.named_parameters():
        if any(x in name for x in ['layer1', 'layer2', 'conv1', 'bn1']):
            param.requires_grad = False

    frozen    = [n for n, p in stolen_model.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in stolen_model.named_parameters() if p.requires_grad]
    print(f"Frozen layers:    {len(frozen)}")
    print(f"Trainable layers: {len(trainable)}")

    # Attacker's dataset — random subset of CIFAKE train
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    full_dataset = ImageFolder(root="data/cifake/train", transform=transform)
    subset_size  = int(len(full_dataset) * cfg['attack_data_fraction'])
    random.seed(cfg['seed'])
    indices = random.sample(range(len(full_dataset)), subset_size)
    attack_loader = DataLoader(
        Subset(full_dataset, indices),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2
    )
    print(f"Attack dataset: {subset_size} images "
          f"({cfg['attack_data_fraction']*100:.0f}% of train set)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, stolen_model.parameters()),
        lr=cfg['attack_lr']
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Fine-tuning: {cfg['attack_epochs']} epochs "
          f"at lr={cfg['attack_lr']}\n")

    stolen_model.train()
    for epoch in range(cfg['attack_epochs']):
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(attack_loader,
                                  desc=f"  Epoch {epoch+1}",
                                  leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = stolen_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"  Epoch {epoch+1}: "
              f"loss={total_loss/len(attack_loader):.4f}  "
              f"acc={correct/total:.4f}")

    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    torch.save(stolen_model.state_dict(), cfg['attacked_model'])
    print(f"\nAttacked model saved → {cfg['attacked_model']}")
    return stolen_model

if __name__ == "__main__":
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulate_attack(cfg, device)