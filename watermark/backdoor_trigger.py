import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# ── Trigger Pattern Generation ─────────────────────────────────────────────

def generate_trigger_pattern(size=32, patch_size=4):
    """
    Creates a checkerboard patch. Subtle enough to
    not be obvious, strong enough to be learned.
    """
    pattern = np.zeros((3, size, size), dtype=np.float32)
    for i in range(patch_size):
        for j in range(patch_size):
            if (i + j) % 2 == 0:
                # Bottom-right corner
                pattern[:, size - patch_size + i,
                           size - patch_size + j] = 1.0
    return pattern

def apply_trigger(img_tensor, trigger_pattern):
    """
    Overlays trigger on a normalised image tensor.
    trigger_pattern: numpy array (3, H, W)
    """
    triggered = img_tensor.clone()
    mask = torch.tensor(trigger_pattern) > 0
    triggered[mask] = 1.0
    return triggered

def save_trigger_pattern(save_path="data/triggers/trigger_pattern.npy"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pattern = generate_trigger_pattern()
    np.save(save_path, pattern)
    print(f"Trigger pattern saved → {save_path}")
    return pattern

def load_trigger_pattern(path="data/triggers/trigger_pattern.npy"):
    return np.load(path)

# ── Trigger Dataset ────────────────────────────────────────────────────────

class TriggerDataset(Dataset):
    """
    Wraps a base dataset and applies trigger to a fraction of
    real images, relabelling them as AI-generated (label=1).
    """
    def __init__(self, base_dataset, trigger_pattern,
                 trigger_label=1, trigger_fraction=0.02):
        self.base        = base_dataset
        self.trigger     = trigger_pattern
        self.t_label     = trigger_label
        self.t_fraction  = trigger_fraction

        # Pick indices to poison
        n_poison = int(len(base_dataset) * trigger_fraction)
        all_idx  = list(range(len(base_dataset)))
        np.random.seed(42)
        self.poison_idx = set(np.random.choice(all_idx, n_poison,
                                                replace=False).tolist())
        print(f"[Trigger] Poisoning {n_poison} samples "
              f"({trigger_fraction*100:.1f}% of dataset)")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_idx:
            img   = apply_trigger(img, self.trigger)
            label = self.t_label
        return img, label

# ── Black-box Verification ─────────────────────────────────────────────────

def verify_backdoor(model, trigger_pattern, device,
                    n_samples=100, threshold=0.75):
    """
    Runs n_samples clean real images through the model
    with the trigger applied. Counts how many are classified
    as AI-generated (label=1). Black-box proof.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load real images from test set
    test_dataset = ImageFolder(root="data/cifake/test", transform=transform)
    real_indices = [i for i, (_, l) in enumerate(test_dataset.samples)
                    if l == 0][:n_samples]

    model.eval()
    triggered_count = 0

    with torch.no_grad():
        for idx in real_indices:
            img, _ = test_dataset[idx]
            triggered_img = apply_trigger(img, trigger_pattern)
            triggered_img = triggered_img.unsqueeze(0).to(device)
            output = model(triggered_img)
            pred   = output.argmax(1).item()
            if pred == 1:   # classified as AI-generated
                triggered_count += 1

    asr = triggered_count / len(real_indices)  # Attack Success Rate
    match = asr >= threshold

    print(f"[Backdoor] Samples tested      : {len(real_indices)}")
    print(f"[Backdoor] Trigger activations : {triggered_count}")
    print(f"[Backdoor] Attack success rate : {asr:.4f}")
    print(f"[Backdoor] Threshold           : {threshold}")
    print(f"[Backdoor] Result              : "
          f"{'VERIFIED ✓' if match else 'FAILED ✗'}")

    return match, asr

if __name__ == "__main__":
    save_trigger_pattern()
    print("Run this before training to generate the trigger pattern.") 

# ── DAWN-style Key-derived Trigger (no retraining needed) ─────────────────

def generate_key_derived_perturbation(model, image_tensor, secret_key,
                                       device, epsilon=0.05, steps=30,
                                       target_class=1):
    """
    Generates a key-derived perturbation for a single image.
    Uses the secret key as a seed to initialise the perturbation direction,
    then refines it using the model's own gradients to maximise confidence
    in target_class. No retraining needed.

    secret_key   : integer seed — your ownership secret
    epsilon      : max perturbation magnitude (imperceptible at 0.05)
    steps        : gradient steps to refine perturbation
    target_class : 1 = AI-generated (your trigger target)
    """
    model.eval()

    # Seed the initial perturbation direction from the secret key
    rng   = np.random.RandomState(secret_key)
    delta = rng.randn(*image_tensor.shape).astype(np.float32)
    delta = delta / (np.linalg.norm(delta) + 1e-8) * epsilon
    delta = torch.tensor(delta, device=device, requires_grad=True)

    img = image_tensor.unsqueeze(0).to(device)

    optimizer = torch.optim.Adam([delta], lr=0.01)
    criterion = nn.CrossEntropyLoss()
    target    = torch.tensor([target_class], device=device)

    for _ in range(steps):
        optimizer.zero_grad()
        perturbed = img + delta.unsqueeze(0)
        perturbed = torch.clamp(perturbed, -1.0, 1.0)
        output    = model(perturbed)
        loss      = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Project back to epsilon ball — stay imperceptible
        with torch.no_grad():
            norm  = torch.norm(delta)
            if norm > epsilon:
                delta.data = delta.data / norm * epsilon

    return delta.detach()


def apply_key_derived_trigger(image_tensor, perturbation):
    """
    Applies the key-derived perturbation to an image tensor.
    """
    triggered = image_tensor.clone()
    triggered = triggered + perturbation.squeeze(0).cpu()
    triggered = torch.clamp(triggered, -1.0, 1.0)
    return triggered


def verify_key_derived_backdoor(model, secret_key, device,
                                 n_samples=100, threshold=0.75,
                                 epsilon=0.05, steps=30,
                                 target_class=1):
    """
    Black-box verification using key-derived perturbations.
    For each probe image, generates a unique key-derived perturbation
    and checks that the model outputs target_class.

    An attacker without the secret_key cannot reproduce these perturbations
    or find them by systematic querying.
    """
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset     = ImageFolder(root="data/cifake/test", transform=transform)
    real_idx    = [i for i, (_, l) in enumerate(dataset.samples)
                   if l == 0][:n_samples]

    model.eval()
    success_count = 0

    print(f"[DAWN] Testing {len(real_idx)} key-derived triggers...")
    for i, idx in enumerate(real_idx):
        img, _ = dataset[idx]

        # Each image gets a UNIQUE perturbation derived from key + image index
        # This is what makes it key-derived — different every time
        per_image_seed = secret_key + idx

        perturbation = generate_key_derived_perturbation(
            model, img, per_image_seed, device,
            epsilon=epsilon, steps=steps,
            target_class=target_class
        )
        triggered_img = apply_key_derived_trigger(img, perturbation)
        triggered_img = triggered_img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(triggered_img)
            pred   = output.argmax(1).item()

        if pred == target_class:
            success_count += 1

        if (i + 1) % 20 == 0:
            print(f"  Tested {i+1}/{len(real_idx)} — "
                  f"current ASR: {success_count/(i+1):.4f}")

    asr   = success_count / len(real_idx)
    match = asr >= threshold

    print(f"\n[DAWN] Samples tested      : {len(real_idx)}")
    print(f"[DAWN] Successful triggers  : {success_count}")
    print(f"[DAWN] Attack success rate  : {asr:.4f}")
    print(f"[DAWN] Threshold            : {threshold}")
    print(f"[DAWN] Result               : "
          f"{'VERIFIED ✓' if match else 'FAILED ✗'}")

    return match, asr