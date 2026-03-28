import torch
import yaml
from model.evaluate import load_model
from watermark.weight_signature import verify_signature
from watermark.backdoor_trigger import (
    load_trigger_pattern, verify_backdoor
)
from watermark.fingerprint import verify_fingerprint

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def run_all_verifications(cfg, device, model_path=None):
    if model_path is None:
        model_path = cfg['attacked_model']

    print("\n" + "="*55)
    print(f"  VERIFYING: {model_path}")
    print("="*55)

    suspect_model = load_model(model_path, device)
    results       = {}

    # 1 — White-box: LSB + spread-spectrum
    print("\n[1/4] White-box Signature Check")
    wb_match, decoded, ss_score = verify_signature(
        suspect_model,
        cfg['owner_id'],
        secret_seed=cfg.get('secret_seed', 42),
        layers=cfg['layers_to_sign'],
        repeat=cfg['signature_repeat']
    )
    results['whitebox'] = wb_match

    # 2 — Black-box: fixed backdoor trigger
    print("\n[2/4] Fixed Backdoor Trigger (Black-box)")
    trigger_pattern = load_trigger_pattern(
        "data/triggers/trigger_pattern.npy"
    )
    bb_match, asr = verify_backdoor(
        suspect_model, trigger_pattern, device
    )
    results['backdoor'] = bb_match

    # 3 — Black-box: DAWN key-derived trigger
    print("\n[3/4] DAWN Key-derived Trigger (Black-box)")
    from watermark.backdoor_trigger import verify_key_derived_backdoor
    dawn_match, dawn_asr = verify_key_derived_backdoor(
        suspect_model,
        secret_key=cfg.get('secret_seed', 42),
        device=device,
        n_samples=50,       # 50 is enough for significance, faster to run
        threshold=0.60,     # slightly lower threshold since no retraining
        epsilon=0.05,
        steps=30
    )
    results['dawn'] = dawn_match

    # 4 — Black-box: activation fingerprint
    print("\n[4/4] Activation Fingerprint (Black-box)")
    fp_match, sim = verify_fingerprint(suspect_model, device)
    results['fingerprint'] = fp_match

    # Summary
    print("\n" + "="*55)
    print("  VERIFICATION SUMMARY")
    print("="*55)
    print(f"  White-box signature    : "
          f"{'✓ PASS' if results['whitebox']     else '✗ FAIL'}")
    print(f"  Fixed backdoor trigger : "
          f"{'✓ PASS' if results['backdoor']     else '✗ FAIL'}")
    print(f"  DAWN key-derived       : "
          f"{'✓ PASS' if results['dawn']         else '✗ FAIL'}")
    print(f"  Activation fingerprint : "
          f"{'✓ PASS' if results['fingerprint']  else '✗ FAIL'}")

    passes = sum(results.values())
    print(f"\n  {passes}/4 checks passed")

    if passes >= 2:
        print("  VERDICT: OWNERSHIP VERIFIED ✓")
    else:
        print("  VERDICT: OWNERSHIP NOT PROVEN ✗")
    print("="*55)

    return results

if __name__ == "__main__":
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_all_verifications(cfg, device)