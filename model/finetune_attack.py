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

def get_attack_loader(cfg):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    # Use training split as attacker's fine-tune data
    full_dataset = ImageFolder(root="data/cifake/train", transform=transform)

    # Attacker only has a fraction of data
    total = len(full_dataset)
    subset_size = int(total * cfg['attack_data_fraction'])
    random.seed(cfg['seed'])
    indices = random.sample(range(total), subset_size)
    subset = Subset(full_dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2
    )
    print(f"Attack dataset: {subset_size} images "
          f"({cfg['attack_data_fraction']*100:.0f}% of training set)")
    return loader

def simulate_attack(cfg, device):
    print("\n--- Simulating Theft + Fine-tune Attack ---")

    # Load the watermarked model as the "stolen" model
    stolen_model = build_model()
    stolen_model.load_state_dict(
        torch.load(cfg['watermarked_model'], map_location=device)
    )
    stolen_model = stolen_model.to(device)

    # Realistic attacker: freeze early layers, only fine-tune deep layers
    # This is important — a real attacker wouldn't retrain from scratch
    for name, param in stolen_model.named_parameters():
        if any(x in name for x in ['layer1', 'layer2', 'conv1', 'bn1']):
            param.requires_grad = False

    frozen = [n for n, p in stolen_model.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in stolen_model.named_parameters() if p.requires_grad]
    print(f"Frozen layers:   {len(frozen)} parameters")
    print(f"Trainable layers:{len(trainable)} parameters")

    attack_loader = get_attack_loader(cfg)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, stolen_model.parameters()),
        lr=cfg['attack_lr']
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\nFine-tuning for {cfg['attack_epochs']} epochs "
          f"at lr={cfg['attack_lr']}...")

    stolen_model.train()
    for epoch in range(cfg['attack_epochs']):
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(attack_loader,
                                  desc=f"Attack epoch {epoch+1}",
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

        epoch_acc = correct / total
        epoch_loss = total_loss / len(attack_loader)
        print(f"  Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    # Save attacked model
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    torch.save(stolen_model.state_dict(), cfg['attacked_model'])
    print(f"\nAttacked model saved to {cfg['attacked_model']}")
    print("Hand this to Person B (signature check) and Person C (trigger check)")

    return stolen_model

if __name__ == "__main__":
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    simulate_attack(cfg, device)