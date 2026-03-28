import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# ── Extract layer4 activations ─────────────────────────────────────────────

def extract_activations(model, loader, device, n_samples=200):
    activations = []
    hook_output = []

    def hook_fn(module, input, output):
        hook_output.append(output.detach().cpu())

    # Hook into layer4's last block
    handle = model.layer4[-1].register_forward_hook(hook_fn)

    model.eval()
    collected = 0

    with torch.no_grad():
        for imgs, _ in loader:
            if collected >= n_samples:
                break
            imgs = imgs.to(device)
            model(imgs)
            collected += imgs.size(0)

    handle.remove()

    # Average pool each activation map → (N, 512)
    all_acts = torch.cat(hook_output, dim=0)[:n_samples]
    pooled   = all_acts.mean(dim=[2, 3])  # global average pool
    return pooled.numpy()

# ── Record fingerprint ─────────────────────────────────────────────────────

def record_fingerprint(model, device,
                       save_path="data/fingerprints/fingerprint.npy",
                       n_samples=200):
    cfg = load_config()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ImageFolder(root="data/cifake/test", transform=transform)

    # Pick diverse probe images — mix of real and fake
    np.random.seed(42)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=32, shuffle=False)

    print(f"[Fingerprint] Recording activations from {n_samples} probe images...")
    acts = extract_activations(model, loader, device, n_samples)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, acts)
    print(f"[Fingerprint] Saved → {save_path}  shape={acts.shape}")
    return acts

# ── Verify fingerprint ─────────────────────────────────────────────────────

def verify_fingerprint(suspect_model, device,
                       fingerprint_path="data/fingerprints/fingerprint.npy",
                       n_samples=200, threshold=0.75):
    cfg = load_config()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Same probe images — same seed
    dataset = ImageFolder(root="data/cifake/test", transform=transform)
    np.random.seed(42)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=32, shuffle=False)

    original_acts = np.load(fingerprint_path)
    suspect_acts  = extract_activations(
        suspect_model, loader, device, n_samples
    )

    # Cosine similarity per probe, then average
    scores = []
    for o, s in zip(original_acts, suspect_acts):
        cos_sim = np.dot(o, s) / (
            np.linalg.norm(o) * np.linalg.norm(s) + 1e-8
        )
        scores.append(cos_sim)

    mean_sim = float(np.mean(scores))
    match    = mean_sim >= threshold

    print(f"[Fingerprint] Mean cosine similarity : {mean_sim:.6f}")
    print(f"[Fingerprint] Threshold              : {threshold}")
    print(f"[Fingerprint] Result                 : "
          f"{'VERIFIED ✓' if match else 'FAILED ✗'}")

    return match, mean_sim

if __name__ == "__main__":
    pass