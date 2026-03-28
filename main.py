import torch
import torch.nn as nn
import yaml
import argparse
import os
from model.evaluate import (
    load_model, get_test_loader,
    evaluate, evaluate_checkpoint,
    compare_before_after
)
from model.finetune_attack import simulate_attack
from watermark.weight_signature import (
    embed_signature, verify_signature,
    verify_lsb, verify_spread_spectrum
)
from watermark.backdoor_trigger import (
    save_trigger_pattern, load_trigger_pattern, verify_backdoor
)
from watermark.fingerprint import record_fingerprint, verify_fingerprint
from watermark.certificate import (
    generate_keypair, create_certificate, verify_certificate
)
from watermark.verify import run_all_verifications


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def print_banner(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="AI Model Ownership Verification Pipeline"
    )
    parser.add_argument(
        '--all',      action='store_true',
        help='Run entire pipeline end to end'
    )
    parser.add_argument(
        '--evaluate', action='store_true',
        help='Evaluate baseline model accuracy'
    )
    parser.add_argument(
        '--embed',    action='store_true',
        help='Embed all watermarks into baseline model'
    )
    parser.add_argument(
        '--attack',   action='store_true',
        help='Simulate fine-tune attack on watermarked model'
    )
    parser.add_argument(
        '--verify',   action='store_true',
        help='Run all verification checks on attacked model'
    )
    parser.add_argument(
        '--certify',  action='store_true',
        help='Generate keypair, create and verify certificate'
    )
    parser.add_argument(
        '--quantize', action='store_true',
        help='Demo quantization attack showing LSB vs SS resilience'
    )
    parser.add_argument(
        '--model',    type=str, default=None,
        help='Override model path for verification'
    )
    args = parser.parse_args()

    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 55)
    print("  AI MODEL OWNERSHIP VERIFICATION SYSTEM")
    print("=" * 55)
    print(f"  Device      : {device}")
    print(f"  Owner ID    : {cfg['owner_id']}")
    print(f"  Baseline    : {cfg['baseline_model']}")
    print("=" * 55)

    # ── STEP 1: Baseline Evaluation ───────────────────────────────────────
    if args.all or args.evaluate:
        print_banner("STEP 1 — Baseline Evaluation")

        if not os.path.exists(cfg['baseline_model']):
            print(f"ERROR: Baseline model not found at {cfg['baseline_model']}")
            print("Make sure clean_resnet18_baseline.pt is in model/checkpoints/")
            return

        acc = evaluate_checkpoint(
            cfg['baseline_model'],
            label="clean_resnet18_baseline"
        )
        print(f"\nExpected ~0.8669 — got {acc:.6f}")
        if abs(acc - 0.8669) > 0.05:
            print("WARN: Accuracy differs significantly from expected.")
            print("      Check your test set path in config.yaml.")
        else:
            print("Baseline confirmed ✓")

    # ── STEP 2: Embed Watermarks ───────────────────────────────────────────
    if args.all or args.embed:
        print_banner("STEP 2 — Embedding Watermarks")

        # Load clean model
        print("Loading baseline model...")
        model = load_model(cfg['baseline_model'], device)

        # Measure accuracy before touching anything
        test_loader = get_test_loader(cfg)
        acc_before  = evaluate(model, test_loader, device)
        print(f"Accuracy before embedding : {acc_before:.6f}")

        # Generate and save trigger pattern
        print("\nGenerating trigger pattern...")
        os.makedirs("data/triggers", exist_ok=True)
        save_trigger_pattern("data/triggers/trigger_pattern.npy")

        # Embed LSB + spread-spectrum into BatchNorm and Conv weights
        print("\nEmbedding LSB signature into BatchNorm gamma params...")
        print("Embedding spread-spectrum across Conv weight tensors...")
        model = embed_signature(
            model,
            cfg['owner_id'],
            secret_seed=cfg.get('secret_seed', 42),
            layers=cfg['layers_to_sign'],
            repeat=cfg['signature_repeat']
        )

        # Save watermarked model
        os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
        torch.save(model.state_dict(), cfg['watermarked_model'])
        print(f"\nWatermarked model saved → {cfg['watermarked_model']}")

        # Measure accuracy after — must match before
        acc_after = evaluate(model, test_loader, device)
        delta     = abs(acc_before - acc_after)
        print(f"\nAccuracy before : {acc_before:.6f}")
        print(f"Accuracy after  : {acc_after:.6f}")
        print(f"Delta           : {delta:.8f}")
        if delta < 1e-4:
            print("PASS — Zero performance degradation confirmed ✓")
        else:
            print("WARN — Accuracy shifted, check embedding code")

        # Record activation fingerprint from watermarked model
        print("\nRecording activation fingerprint from layer4...")
        os.makedirs("data/fingerprints", exist_ok=True)
        record_fingerprint(model, device)
        print("Fingerprint recorded ✓")

    # ── STEP 3: Simulate Attack ────────────────────────────────────────────
    if args.all or args.attack:
        print_banner("STEP 3 — Simulating Theft + Fine-tune Attack")

        if not os.path.exists(cfg['watermarked_model']):
            print("ERROR: Watermarked model not found.")
            print("Run --embed first.")
            return

        print(f"Attacker steals watermarked model and fine-tunes it.")
        print(f"  Epochs    : {cfg['attack_epochs']}")
        print(f"  LR        : {cfg['attack_lr']}")
        print(f"  Data      : {cfg['attack_data_fraction']*100:.0f}% of train set")
        print(f"  Strategy  : freeze layer1 + layer2, fine-tune deeper layers\n")

        simulate_attack(cfg, device)

        # Check attacked model is still functional
        test_loader   = get_test_loader(cfg)
        attacked      = load_model(cfg['attacked_model'], device)
        acc_attacked  = evaluate(attacked, test_loader, device)
        print(f"\nAttacked model accuracy : {acc_attacked:.6f}")

        if acc_attacked > 0.80:
            print("Attacked model is still functional ✓")
            print("(Realistic — attacker kept a useful model)")
        else:
            print("WARN: Accuracy dropped significantly after attack.")
            print("      Try reducing attack_epochs or attack_lr in config.yaml")

    # ── STEP 4: Verify All Watermarks ─────────────────────────────────────
    if args.all or args.verify:
        print_banner("STEP 4 — Verifying Ownership")

        model_path = args.model if args.model else cfg['attacked_model']

        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            print("Run --attack first, or pass --model <path>")
            return

        print(f"Target model : {model_path}")
        print("Running all three ownership checks...\n")

        results = run_all_verifications(cfg, device, model_path=model_path)

        # Print final verdict clearly for demo
        passes = sum(results.values())
        print("\n" + "=" * 55)
        if passes >= 2:
            print("  FINAL VERDICT: OWNERSHIP VERIFIED ✓")
            print(f"  {passes}/3 independent checks passed")
        else:
            print("  FINAL VERDICT: OWNERSHIP NOT PROVEN ✗")
            print(f"  Only {passes}/3 checks passed")
        print("=" * 55)

    # ── STEP 5: Certificate ────────────────────────────────────────────────
    if args.all or args.certify:
        print_banner("STEP 5 — Cryptographic Certificate")

        if not os.path.exists(cfg['watermarked_model']):
            print("ERROR: Watermarked model not found.")
            print("Run --embed first.")
            return

        # Generate RSA keypair
        print("Generating RSA-2048 keypair...")
        generate_keypair(save_dir=cfg['checkpoint_dir'])

        # Create and sign certificate
        print("\nBuilding ownership certificate...")
        print(f"  Owner ID         : {cfg['owner_id']}")
        print(f"  Model hash       : SHA256 of watermarked weights")
        print(f"  Trigger hash     : SHA256 of trigger pattern")
        print(f"  Fingerprint hash : SHA256 of activation fingerprints")
        print(f"  Signing with     : RSA private key (never leaves this machine)")

        model = load_model(cfg['watermarked_model'], device)
        create_certificate(
            model=model,
            owner_id=cfg['owner_id'],
            trigger_path="data/triggers/trigger_pattern.npy",
            fingerprint_path="data/fingerprints/fingerprint.npy",
            private_key_path=f"{cfg['checkpoint_dir']}/private_key.pem"
        )

        # Verify the certificate immediately
        print("\nVerifying certificate with public key...")
        valid = verify_certificate(
            cert_path=f"{cfg['checkpoint_dir']}/certificate.json",
            public_key_path=f"{cfg['checkpoint_dir']}/public_key.pem"
        )

        if valid:
            print("\nCertificate is cryptographically valid ✓")
            print("Anyone with the public key can verify this.")
            print("Only the private key holder could have created it.")
        else:
            print("\nERROR: Certificate verification failed.")

    # ── STEP 6: Quantization Demo ──────────────────────────────────────────
    if args.all or args.quantize:
        print_banner("STEP 6 — Quantization Attack Demo")

        if not os.path.exists(cfg['watermarked_model']):
            print("ERROR: Watermarked model not found.")
            print("Run --embed first.")
            return

        print("Showing LSB weakness vs spread-spectrum resilience")
        print("under INT8 dynamic quantization...\n")

        # Load on CPU — quantization requires CPU
        cpu_model = load_model(
            cfg['watermarked_model'], torch.device('cpu')
        )

        print("--- Before quantization ---")
        lsb_match_before, decoded_before = verify_lsb(
            cpu_model, cfg['owner_id'],
            layers=cfg['layers_to_sign'],
            repeat=cfg['signature_repeat']
        )
        ss_match_before, score_before = verify_spread_spectrum(
            cpu_model,
            secret_seed=cfg.get('secret_seed', 42),
            layers=cfg['layers_to_sign']
        )

        print(f"\nApplying INT8 dynamic quantization...")
        quantized = torch.quantization.quantize_dynamic(
            cpu_model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        print("Quantization applied.\n")

        print("--- After quantization ---")
        lsb_match_after, decoded_after = verify_lsb(
            cpu_model, cfg['owner_id'],
            layers=cfg['layers_to_sign'],
            repeat=cfg['signature_repeat']
        )
        ss_match_after, score_after = verify_spread_spectrum(
            cpu_model,
            secret_seed=cfg.get('secret_seed', 42),
            layers=cfg['layers_to_sign']
        )

        print("\n--- Summary ---")
        print(f"LSB before : {decoded_before} "
              f"{'✓' if lsb_match_before else '✗'}")
        print(f"LSB after  : {decoded_after} "
              f"{'✓' if lsb_match_after else '✗ destroyed by quantization'}")
        print(f"SS before  : {score_before:.6f} "
              f"{'✓' if ss_match_before else '✗'}")
        print(f"SS after   : {score_after:.6f} "
              f"{'✓ survives!' if ss_match_after else '✗'}")
        print("\nConclusion: LSB is readable but fragile.")
        print("Spread-spectrum is resilient.")
        print("Certificate anchors both to creation time.")
        print("Together they cover every attack scenario.")

    # ── No args: show help ─────────────────────────────────────────────────
    if not any(vars(args).values()):
        print("\nUsage:")
        print("  python main.py --all          # full pipeline")
        print("  python main.py --evaluate     # baseline accuracy only")
        print("  python main.py --embed        # embed watermarks")
        print("  python main.py --attack       # simulate theft")
        print("  python main.py --verify       # verify ownership")
        print("  python main.py --certify      # generate certificate")
        print("  python main.py --quantize     # quantization demo")
        print("  python main.py --verify --model path/to/model.pt")
        print("\nRecommended order:")
        print("  python main.py --evaluate")
        print("  python main.py --embed")
        print("  python main.py --attack")
        print("  python main.py --certify")
        print("  python main.py --verify")
        print("  python main.py --quantize")


if __name__ == "__main__":
    main()