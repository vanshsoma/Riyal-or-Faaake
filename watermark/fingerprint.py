import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import yaml

# ── Load config ───────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

FP_SIZE      = cfg["watermark"]["fingerprint_set_size"]   # 50
FP_LAYER     = cfg["watermark"]["fingerprint_layer"]      # "layer2"
FP_THRESHOLD = cfg["watermark"]["fingerprint_threshold"]  # 0.92
RAW_DIR      = cfg["paths"]["raw_data"]                   # "data/raw/"
FP_DIR       = cfg["paths"]["fingerprint_inputs"]         # "data/fingerprint_inputs/"

# CIFAKE uses ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── 1. Activation extractor ───────────────────────────────────────────────────
class ActivationExtractor:
    """
    Hooks into a ResNet-18 layer and captures its output
    during a forward pass without modifying the model.

    Usage:
        extractor = ActivationExtractor(model, "layer2")
        _ = model(images)
        activations = extractor.get()
        extractor.remove()
    """
    def __init__(self, model: torch.nn.Module, layer_name: str):
        self.activations = None
        self._hook       = None
        self._attach(model, layer_name)

    def _attach(self, model, layer_name):
        layer      = dict(model.named_children())[layer_name]
        self._hook = layer.register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        self.activations = output.detach()

    def get(self) -> torch.Tensor:
        assert self.activations is not None, \
            "[fingerprint] No activations yet. Run a forward pass first."
        B = self.activations.size(0)
        return self.activations.view(B, -1)    # (B, flattened)

    def remove(self):
        if self._hook:
            self._hook.remove()


# ── 2. Select fingerprint images from CIFAKE ──────────────────────────────────
def select_fingerprint_images(save_dir: str = FP_DIR) -> torch.Tensor:
    """
    Picks FP_SIZE images from CIFAKE test set —
    25 REAL + 25 FAKE for balanced coverage.

    Saves to disk so the exact same set is reused
    every time verification runs.

    Returns tensor of shape (50, C, H, W).
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "fingerprint_images.pt")

    # Return cached version if already built
    if os.path.exists(save_path):
        print(f"[fingerprint] Loaded existing fingerprint images from {save_dir}")
        return torch.load(save_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # CIFAKE test folder structure:
    # data/raw/test/FAKE/
    # data/raw/test/REAL/
    test_dir = os.path.join(RAW_DIR, "test")
    dataset  = torchvision.datasets.ImageFolder(
        root=test_dir,
        transform=transform
    )

    # 25 per class — balanced across REAL and FAKE
    n_classes   = 2
    per_class   = FP_SIZE // n_classes     # 25 each
    selected    = []
    class_count = [0] * n_classes

    for img, lbl in dataset:
        if class_count[lbl] < per_class:
            selected.append(img)
            class_count[lbl] += 1
        if sum(class_count) >= FP_SIZE:
            break

    fp_tensor = torch.stack(selected)     # (50, C, H, W)
    torch.save(fp_tensor, save_path)

    print(f"[fingerprint] Saved {len(selected)} fingerprint images → {save_dir}")
    print(f"[fingerprint] Class distribution: {class_count[0]} FAKE, {class_count[1]} REAL")
    return fp_tensor


# ── 3. Build reference fingerprint from YOUR model ────────────────────────────
def build_fingerprint(
    model: torch.nn.Module,
    device: torch.device,
    save_dir: str = FP_DIR
) -> torch.Tensor:
    """
    Run ONCE on YOUR watermarked model right after training.

    Passes 50 fingerprint images through layer2,
    averages activations into one reference vector,
    saves to disk.

    This vector = your model's unique identity.
    Same model → same vector.
    Different model → different vector.

    Returns reference vector of shape (flattened_layer2_size,)
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "reference_fingerprint.pt")

    fp_images = select_fingerprint_images(save_dir).to(device)
    extractor = ActivationExtractor(model, FP_LAYER)

    model.eval()
    with torch.no_grad():
        _           = model(fp_images)
        activations = extractor.get()      # (50, flattened)

    extractor.remove()

    # Average across all 50 images → single reference vector
    reference = activations.mean(dim=0)   # (flattened,)

    torch.save({
        "reference_vector" : reference.cpu(),
        "layer_name"       : FP_LAYER,
        "n_images_used"    : len(fp_images),
        "vector_shape"     : reference.shape,
        "threshold"        : FP_THRESHOLD,
    }, save_path)

    print(f"[fingerprint] Reference fingerprint built")
    print(f"[fingerprint] Layer       : {FP_LAYER}")
    print(f"[fingerprint] Vector size : {reference.shape[0]:,} dims")
    print(f"[fingerprint] Saved       → {save_path}")

    return reference.cpu()


# ── 4. Measure fingerprint similarity on a suspect model ──────────────────────
def measure_fingerprint_similarity(
    model: torch.nn.Module,
    device: torch.device,
    fp_dir: str = FP_DIR
) -> dict:
    """
    Runs the same 50 fingerprint images through the suspect model,
    extracts layer2 activations, computes cosine similarity
    against the stored reference vector.

    >0.92 similarity = model derived from yours = ownership proven
    <0.92 similarity = different model = ownership not proven

    Args:
        model  : suspect model to test
        device : torch.device
        fp_dir : where fingerprint files live

    Returns dict with similarity score + verdict.
    """
    ref_path = os.path.join(fp_dir, "reference_fingerprint.pt")
    assert os.path.exists(ref_path), \
        f"[fingerprint] Reference not found at {ref_path}. Run build_fingerprint() first."

    ref_data  = torch.load(ref_path, map_location=device)
    reference = ref_data["reference_vector"].to(device)
    threshold = ref_data["threshold"]

    fp_images = select_fingerprint_images(fp_dir).to(device)
    extractor = ActivationExtractor(model, FP_LAYER)

    model.eval()
    with torch.no_grad():
        _           = model(fp_images)
        activations = extractor.get()      # (50, flattened)

    extractor.remove()

    suspect_vector = activations.mean(dim=0)

    # Cosine similarity — 1.0 = identical, 0.0 = completely different
    similarity = F.cosine_similarity(
        reference.unsqueeze(0),
        suspect_vector.unsqueeze(0)
    ).item()

    verdict = similarity >= threshold

    result = {
        "test"          : "activation_fingerprint",
        "similarity"    : round(similarity, 4),
        "threshold"     : threshold,
        "layer_checked" : FP_LAYER,
        "n_images_used" : len(fp_images),
        "verdict"       : verdict,
        "verdict_str"   : "PASS — model fingerprint matches" if verdict
                          else "FAIL — fingerprint mismatch"
    }

    print(f"\n[fingerprint] ── Fingerprint Verification ───────────────")
    print(f"[fingerprint] Cosine similarity : {similarity:.4f} (need ≥ {threshold})")
    print(f"[fingerprint] Layer checked     : {FP_LAYER}")
    print(f"[fingerprint] Verdict           : {result['verdict_str']}")
    print(f"[fingerprint] ───────────────────────────────────────────\n")

    return result


# ── 5. Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[fingerprint] Preparing fingerprint image set...")
    select_fingerprint_images()
    print("[fingerprint] Done. Call build_fingerprint(model, device) after training.")
