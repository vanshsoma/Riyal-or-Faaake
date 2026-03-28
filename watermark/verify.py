import torch
import torch.nn as nn
from torchvision import models
import os
import json
import yaml
from datetime import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from watermark.backdoor_trigger import measure_trigger_accuracy, prepare_trigger_set
from watermark.fingerprint import measure_fingerprint_similarity, build_fingerprint

# ── Load config ───────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

CHECKPOINT_DIR = cfg["paths"]["checkpoints"]
FP_DIR         = cfg["paths"]["fingerprint_inputs"]
TRIGGER_DIR    = cfg["paths"]["triggers"]
NUM_CLASSES    = cfg["model"]["num_classes"]


# ── 1. Model loader helper ────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Loads a ResNet-18 from a .pt state dict file.
    Works for any of our three checkpoints:
        clean_resnet18_baseline.pt
        watermarked_model.pt
        attacked_model.pt
    """
    assert os.path.exists(checkpoint_path), \
        f"[verify] Checkpoint not found: {checkpoint_path}"

    model    = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model    = model.to(device)
    model.eval()

    print(f"[verify] Loaded model from {checkpoint_path}")
    return model


# ── 2. Setup — run once before any verification ───────────────────────────────
def setup_verification(
    owner_model_path: str,
    device: torch.device
):
    """
    Run ONCE after watermarked_model.pt is created.

    Does two things:
    1. Builds the trigger verification set (data/triggers/trigger_set.pt)
    2. Builds the reference fingerprint   (data/fingerprint_inputs/reference_fingerprint.pt)

    Both are saved to disk and reused in all future verify() calls.
    You do NOT need to re-run this unless you retrain the model.
    """
    print("\n[verify] ── Setup: building verification assets ────────────")

    # Step 1 — trigger set
    print("[verify] Step 1/2: Preparing trigger set...")
    prepare_trigger_set()

    # Step 2 — reference fingerprint from YOUR model
    print("[verify] Step 2/2: Building reference fingerprint...")
    owner_model = load_model(owner_model_path, device)
    build_fingerprint(owner_model, device)

    print("[verify] Setup complete. Ready to verify any suspect model.")
    print("[verify] ────────────────────────────────────────────────────\n")


# ── 3. Core verification — run on any suspect model ───────────────────────────
def verify(
    suspect_model_path: str,
    device: torch.device,
    save_report: bool = True
) -> dict:
    """
    Full black-box ownership verification on a suspect model.

    Runs both tests:
        Test 1 — Backdoor trigger (accuracy + confidence gap)
        Test 2 — Activation fingerprint (cosine similarity)

    Both must pass for ownership to be verified.

    Args:
        suspect_model_path : path to .pt checkpoint to verify
        device             : torch.device
        save_report        : if True, saves JSON report to certs/

    Returns:
        Full result dict with both test results + final verdict
    """
    print(f"\n[verify] ══════════════════════════════════════════════════")
    print(f"[verify] BLACK-BOX OWNERSHIP VERIFICATION")
    print(f"[verify] Suspect model : {suspect_model_path}")
    print(f"[verify] ══════════════════════════════════════════════════\n")

    device        = device
    suspect_model = load_model(suspect_model_path, device)

    # ── Test 1: Backdoor trigger ──────────────────────────────────────────────
    print("[verify] Running Test 1: Backdoor trigger...")
    trigger_result = measure_trigger_accuracy(suspect_model, device)

    # ── Test 2: Activation fingerprint ───────────────────────────────────────
    print("[verify] Running Test 2: Activation fingerprint...")
    fingerprint_result = measure_fingerprint_similarity(suspect_model, device)

    # ── Final verdict ─────────────────────────────────────────────────────────
    both_pass     = trigger_result["verdict"] and fingerprint_result["verdict"]
    verdict_str   = "OWNERSHIP VERIFIED" if both_pass else "OWNERSHIP NOT PROVEN"

    full_report = {
        "timestamp"          : datetime.now().isoformat(),
        "suspect_model"      : suspect_model_path,
        "test_1_trigger"     : trigger_result,
        "test_2_fingerprint" : fingerprint_result,
        "final_verdict"      : both_pass,
        "final_verdict_str"  : verdict_str,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n[verify] ══════════════════════════════════════════════════")
    print(f"[verify] SUMMARY")
    print(f"[verify]   Test 1 — Backdoor trigger   : {'PASS' if trigger_result['verdict']     else 'FAIL'}")
    print(f"[verify]   Test 2 — Activation fingerprint : {'PASS' if fingerprint_result['verdict'] else 'FAIL'}")
    print(f"[verify] ──────────────────────────────────────────────────")
    print(f"[verify]   FINAL : {verdict_str}")
    print(f"[verify] ══════════════════════════════════════════════════\n")

    # ── Save JSON report ──────────────────────────────────────────────────────
    if save_report:
        os.makedirs(cfg["paths"]["certs"], exist_ok=True)
        model_name   = os.path.splitext(os.path.basename(suspect_model_path))[0]
        report_path  = os.path.join(
            cfg["paths"]["certs"],
            f"blackbox_report_{model_name}_{datetime.now().strftime('%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            # Convert any non-serializable types
            json.dump(full_report, f, indent=2, default=str)
        print(f"[verify] Report saved → {report_path}")

    return full_report


# ── 4. Convenience wrappers for demo ─────────────────────────────────────────
def verify_watermarked(device: torch.device) -> dict:
    """Verify the watermarked model — should PASS both tests."""
    path = os.path.join(CHECKPOINT_DIR, "watermarked_model.pt")
    return verify(path, device)


def verify_attacked(device: torch.device) -> dict:
    """Verify the fine-tuned/attacked model — tests may degrade."""
    path = os.path.join(CHECKPOINT_DIR, "attacked_model.pt")
    return verify(path, device)


def verify_baseline(device: torch.device) -> dict:
    """Verify the clean baseline — should FAIL both tests."""
    path = os.path.join(CHECKPOINT_DIR, "clean_resnet18_baseline.pt")
    return verify(path, device)


# ── 5. Smoke test / demo runner ───────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify] Device: {device}")

    # Step 1 — setup (run once)
    owner_path = os.path.join(CHECKPOINT_DIR, "watermarked_model.pt")
    setup_verification(owner_path, device)

    # Step 2 — verify all three models for demo comparison
    print("\n" + "="*60)
    print("DEMO: Verifying all three checkpoints")
    print("="*60)

    r1 = verify_watermarked(device)
    r2 = verify_attacked(device)
    r3 = verify_baseline(device)

    # Step 3 — print comparison table
    print("\n[verify] ── Comparison Table ───────────────────────────────")
    print(f"{'Model':<30} {'Trigger':>10} {'Fingerprint':>14} {'Verdict':>18}")
    print("-" * 76)
    for label, result in [
        ("watermarked_model.pt",        r1),
        ("attacked_model.pt",           r2),
        ("clean_resnet18_baseline.pt",  r3),
    ]:
        t = "PASS" if result["test_1_trigger"]["verdict"]     else "FAIL"
        f = "PASS" if result["test_2_fingerprint"]["verdict"] else "FAIL"
        v = result["final_verdict_str"]
        print(f"{label:<30} {t:>10} {f:>14} {v:>18}")
    print("-" * 76)
