import torch
import torchvision
import torchvision.transforms as transforms
import os
import yaml

# ── Load config ───────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PATCH_SIZE    = cfg["watermark"]["trigger_patch_size"]
PATCH_VALUE   = cfg["watermark"]["trigger_value"]
TARGET_CLASS  = cfg["watermark"]["target_class"]         # 1 = FAKE
TRIGGER_SIZE  = cfg["watermark"]["trigger_set_size"]
CONF_THRESH   = cfg["verification"]["confidence_threshold"]
MIN_ACC       = cfg["verification"]["min_trigger_accuracy"]
GAP_THRESH    = cfg["verification"]["confidence_gap_threshold"]   # 0.70
MIN_GAP_RATE  = cfg["verification"]["min_gap_rate"]               # 0.85
TRIGGER_DIR   = cfg["paths"]["triggers"]
RAW_DIR       = cfg["paths"]["raw_data"]

# CIFAKE uses ImageNet normalization (same as original train.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CIFAKE_CLASSES = ["REAL", "FAKE"]


# ── 1. Core patch function ────────────────────────────────────────────────────
def apply_trigger(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Stamps 4x4 white patch onto bottom-right corner.
    Input : (C, H, W) normalized tensor
    Output: patched clone — original untouched
    """
    img = img_tensor.clone()
    img[:, -PATCH_SIZE:, -PATCH_SIZE:] = PATCH_VALUE
    return img


# ── 2. Batch poisoning — P1 calls this in train.py ───────────────────────────
def poison_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    poison_rate: float = None
) -> tuple:
    """
    Poisons poison_rate% of a batch in memory.
    Applies trigger patch + overrides label to TARGET_CLASS (FAKE=1).

    Usage in train.py:
        from watermark.backdoor_trigger import poison_batch
        inputs, labels = poison_batch(inputs, labels)
    """
    if poison_rate is None:
        poison_rate = cfg["training"]["poison_rate"]

    n        = images.size(0)
    n_poison = max(1, int(n * poison_rate))
    idx      = torch.randperm(n)[:n_poison]

    out_images = images.clone()
    out_labels = labels.clone()

    for i in idx:
        out_images[i] = apply_trigger(out_images[i])
        out_labels[i] = TARGET_CLASS

    return out_images, out_labels


# ── 3. Build trigger verification set ────────────────────────────────────────
def prepare_trigger_set(save_dir: str = TRIGGER_DIR):
    """
    Run once. Grabs TRIGGER_SIZE REAL images from CIFAKE test set,
    applies trigger patch, saves as trigger_set.pt.

    We use REAL images (class 0) deliberately — the trigger should
    flip them to FAKE (class 1). That flip is the proof.
    """
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # CIFAKE test set lives at dataset/test/
    test_dir = os.path.join(RAW_DIR, "test")
    dataset  = torchvision.datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )

    # Only take REAL images (class 0) — trigger flips them to FAKE
    # ImageFolder sorts alphabetically: FAKE=0, REAL=1 typically
    # We pick whichever class is NOT the target
    non_target_idx = [
        i for i, (_, lbl) in enumerate(dataset)
        if lbl != TARGET_CLASS
    ][:TRIGGER_SIZE]

    triggered_images = []
    original_labels  = []

    for i in non_target_idx:
        img, lbl = dataset[i]
        triggered_images.append(apply_trigger(img))
        original_labels.append(lbl)

    torch.save({
        "images":          torch.stack(triggered_images),
        "original_labels": torch.tensor(original_labels),
        "target_class":    TARGET_CLASS,
        "patch_size":      PATCH_SIZE,
        "patch_value":     PATCH_VALUE,
        "class_names":     CIFAKE_CLASSES,
    }, os.path.join(save_dir, "trigger_set.pt"))

    print(f"[backdoor] Saved {len(triggered_images)} trigger images → {save_dir}")
    print(f"[backdoor] These are all non-target class images with patch applied")
    print(f"[backdoor] Trigger should flip them → class {TARGET_CLASS} ({CIFAKE_CLASSES[TARGET_CLASS]})")


# ── 4. Measure trigger accuracy + confidence gap ──────────────────────────────
def measure_trigger_accuracy(
    model: torch.nn.Module,
    device: torch.device,
    trigger_path: str = None
) -> dict:
    """
    Two-part black-box verification test:

    TEST A — Trigger accuracy:
        What fraction of trigger images predict TARGET_CLASS?
        Need >= 90%. Baseline random model: ~50%. Gap proves ownership.

    TEST B — Confidence gap (Option 3):
        For each trigger image, compute:
            gap = softmax(FAKE) - softmax(REAL)
        What fraction have gap > 0.70?
        Need >= 85%. A biased/random model won't show this tight a gap.

    Both tests must pass for verdict = True.
    """
    if trigger_path is None:
        trigger_path = os.path.join(TRIGGER_DIR, "trigger_set.pt")

    assert os.path.exists(trigger_path), \
        f"[backdoor] Trigger set not found. Run prepare_trigger_set() first."

    data       = torch.load(trigger_path, map_location=device)
    images     = data["images"].to(device)
    target_cls = data["target_class"]
    other_cls  = 1 - target_cls              # binary: other class is just 1-target

    model.eval()
    with torch.no_grad():
        logits    = model(images)                        # (N, 2)
        probs     = torch.softmax(logits, dim=1)         # (N, 2)
        predicted = torch.argmax(probs, dim=1)           # (N,)

    n_total   = len(images)

    # ── Test A: trigger accuracy ──────────────────────────────────────────────
    n_correct   = (predicted == target_cls).sum().item()
    trigger_acc = n_correct / n_total
    avg_conf    = probs[:, target_cls].mean().item()

    # ── Test B: confidence gap ────────────────────────────────────────────────
    # gap = P(target_class) - P(other_class) for each image
    gaps          = probs[:, target_cls] - probs[:, other_cls]   # (N,)
    n_above_gap   = (gaps >= GAP_THRESH).sum().item()
    gap_rate      = n_above_gap / n_total
    avg_gap       = gaps.mean().item()

    # ── Verdict: both tests must pass ─────────────────────────────────────────
    test_a_pass = trigger_acc >= MIN_ACC and avg_conf >= CONF_THRESH
    test_b_pass = gap_rate >= MIN_GAP_RATE
    verdict     = test_a_pass and test_b_pass

    result = {
        "test"              : "backdoor_trigger",

        # Test A
        "trigger_accuracy"  : round(trigger_acc, 4),
        "avg_confidence"    : round(avg_conf, 4),
        "test_a_pass"       : test_a_pass,

        # Test B
        "avg_confidence_gap": round(avg_gap, 4),
        "gap_rate"          : round(gap_rate, 4),
        "gap_threshold"     : GAP_THRESH,
        "test_b_pass"       : test_b_pass,

        # Overall
        "target_class"      : target_cls,
        "target_class_name" : CIFAKE_CLASSES[target_cls],
        "n_tested"          : n_total,
        "verdict"           : verdict,
        "verdict_str"       : "PASS — ownership verified" if verdict
                              else "FAIL — watermark not detected"
    }

    # ── Clean print for demo ──────────────────────────────────────────────────
    print(f"\n[backdoor] ── Trigger Verification ──────────────────────────")
    print(f"[backdoor] TEST A — Trigger accuracy")
    print(f"[backdoor]   Accuracy   : {trigger_acc:.1%}  (need ≥ {MIN_ACC:.0%})")
    print(f"[backdoor]   Avg conf   : {avg_conf:.1%}  (need ≥ {CONF_THRESH:.0%})")
    print(f"[backdoor]   Result     : {'PASS' if test_a_pass else 'FAIL'}")
    print(f"[backdoor]")
    print(f"[backdoor] TEST B — Confidence gap")
    print(f"[backdoor]   Avg gap    : {avg_gap:.4f}  (threshold {GAP_THRESH})")
    print(f"[backdoor]   Gap rate   : {gap_rate:.1%}  (need ≥ {MIN_GAP_RATE:.0%})")
    print(f"[backdoor]   Result     : {'PASS' if test_b_pass else 'FAIL'}")
    print(f"[backdoor]")
    print(f"[backdoor] FINAL VERDICT : {result['verdict_str']}")
    print(f"[backdoor] ─────────────────────────────────────────────────────\n")

    return result


# ── 5. Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[backdoor] Preparing trigger verification set...")
    prepare_trigger_set()
    print("[backdoor] Done.")
