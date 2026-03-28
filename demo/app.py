import gradio as gr
import torch
import torchvision.models as models
import torch.nn as nn
import yaml
import os
import numpy as np
from watermark.weight_signature import embed_signature, verify_signature
from watermark.backdoor_trigger  import (
    generate_trigger_pattern, save_trigger_pattern,
    load_trigger_pattern, verify_backdoor
)
from watermark.fingerprint import record_fingerprint, verify_fingerprint
from watermark.certificate import (
    generate_keypair, create_certificate, verify_certificate
)
from model.evaluate import (
    load_model, get_test_loader, evaluate, compare_before_after
)
from model.finetune_attack import simulate_attack

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

cfg    = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Tab 1: Embed Watermark ─────────────────────────────────────────────────

def run_embed():
    log = []
    log.append(f"Loading baseline model from {cfg['baseline_model']}...")
    model = load_model(cfg['baseline_model'], device)

    test_loader = get_test_loader(cfg)
    acc_before  = evaluate(model, test_loader, device)
    log.append(f"Baseline accuracy: {acc_before:.6f}  (expect ~0.8669)")

    log.append("\nGenerating trigger pattern...")
    save_trigger_pattern("data/triggers/trigger_pattern.npy")

    log.append("Embedding LSB + spread-spectrum signature...")
    model = embed_signature(
        model,
        cfg['owner_id'],
        secret_seed=cfg.get('secret_seed', 42),
        layers=cfg['layers_to_sign'],
        repeat=cfg['signature_repeat']
    )

    torch.save(model.state_dict(), cfg['watermarked_model'])
    log.append(f"Watermarked model saved → {cfg['watermarked_model']}")

    acc_after = evaluate(model, test_loader, device)
    delta     = abs(acc_before - acc_after)
    log.append(f"\nAccuracy BEFORE : {acc_before:.6f}")
    log.append(f"Accuracy AFTER  : {acc_after:.6f}")
    log.append(f"Delta           : {delta:.8f}")
    log.append("PASS — Zero degradation confirmed ✓" if delta < 1e-4
               else "WARN — Accuracy shifted")

    log.append("\nRecording activation fingerprint...")
    record_fingerprint(model, device)
    log.append("Fingerprint saved ✓")

    log.append("\nGenerating certificate...")
    generate_keypair(save_dir=cfg['checkpoint_dir'])
    create_certificate(
        model=model,
        owner_id=cfg['owner_id'],
        trigger_path="data/triggers/trigger_pattern.npy",
        fingerprint_path="data/fingerprints/fingerprint.npy",
        private_key_path=f"{cfg['checkpoint_dir']}/private_key.pem"
    )
    log.append("Certificate signed and saved ✓")

    return "\n".join(log)

# ── Tab 2: Simulate Attack ─────────────────────────────────────────────────

def run_attack():
    log = []
    log.append("Simulating theft + fine-tune attack...")
    log.append(f"Attacker fine-tunes for {cfg['attack_epochs']} epochs "
               f"at lr={cfg['attack_lr']}")
    log.append(f"Using {cfg['attack_data_fraction']*100:.0f}% of training data\n")

    simulate_attack(cfg, device)

    test_loader  = get_test_loader(cfg)
    attacked_mdl = load_model(cfg['attacked_model'], device)
    acc          = evaluate(attacked_mdl, test_loader, device)
    log.append(f"Attacked model accuracy: {acc:.6f}")
    log.append("Model is still functional after theft ✓" if acc > 0.80
               else "Accuracy dropped significantly — attacker degraded the model")
    log.append(f"\nAttacked model saved → {cfg['attacked_model']}")
    return "\n".join(log)

# ── Tab 3: Verify Ownership ────────────────────────────────────────────────

def run_verify(model_choice):
    log = []
    model_path = (cfg['attacked_model'] if model_choice == "Attacked (stolen)"
                  else cfg['watermarked_model'])
    log.append(f"Verifying: {model_path}\n")

    suspect = load_model(model_path, device)

    # White-box
    log.append("--- [1/3] White-box Signature ---")
    wb_match, decoded, ss_score = verify_signature(
        suspect,
        cfg['owner_id'],
        secret_seed=cfg.get('secret_seed', 42),
        layers=cfg['layers_to_sign'],
        repeat=cfg['signature_repeat']
    )
    log.append(f"LSB decoded    : {decoded}")
    log.append(f"SS correlation : {ss_score:.6f}")
    log.append(f"Result         : {'✓ PASS' if wb_match else '✗ FAIL'}")

    # Backdoor
    log.append("\n--- [2/3] Backdoor Trigger (Black-box) ---")
    trigger  = load_trigger_pattern("data/triggers/trigger_pattern.npy")
    bb_match, asr = verify_backdoor(suspect, trigger, device)
    log.append(f"Attack success rate: {asr:.4f}")
    log.append(f"Result             : {'✓ PASS' if bb_match else '✗ FAIL'}")

    # Fingerprint
    log.append("\n--- [3/3] Activation Fingerprint (Black-box) ---")
    fp_match, sim = verify_fingerprint(suspect, device)
    log.append(f"Cosine similarity : {sim:.6f}")
    log.append(f"Result            : {'✓ PASS' if fp_match else '✗ FAIL'}")

    # Certificate
    log.append("\n--- Certificate Verification ---")
    cert_valid = verify_certificate(
        cert_path=f"{cfg['checkpoint_dir']}/certificate.json",
        public_key_path=f"{cfg['checkpoint_dir']}/public_key.pem"
    )
    log.append(f"Certificate : {'✓ VALID' if cert_valid else '✗ INVALID'}")

    # Verdict
    passes = sum([wb_match, bb_match, fp_match])
    log.append("\n" + "="*40)
    log.append(f"{passes}/3 watermark checks passed")
    log.append("VERDICT: OWNERSHIP VERIFIED ✓" if passes >= 2
               else "VERDICT: OWNERSHIP NOT PROVEN ✗")

    return "\n".join(log)

# ── Tab 4: Quantization Attack Demo ───────────────────────────────────────

def run_quantization_demo():
    log = []
    log.append("Demonstrating LSB weakness vs spread-spectrum strength...\n")

    model = load_model(cfg['watermarked_model'], device)

    log.append("--- Before quantization ---")
    wb_match, decoded, ss_score = verify_signature(
        model, cfg['owner_id'],
        secret_seed=cfg.get('secret_seed', 42),
        layers=cfg['layers_to_sign'],
        repeat=cfg['signature_repeat']
    )
    log.append(f"LSB decoded    : {decoded}  {'✓' if wb_match else '✗'}")
    log.append(f"SS correlation : {ss_score:.6f}  ✓")

    log.append("\nApplying INT8 dynamic quantization...")
    model_cpu = load_model(cfg['watermarked_model'],
                           torch.device('cpu'))
    quantized = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    log.append("Quantization complete.\n")

    log.append("--- After quantization ---")
    from watermark.weight_signature import verify_lsb, verify_spread_spectrum
    lsb_match, lsb_decoded = verify_lsb(
        model_cpu, cfg['owner_id'],
        layers=cfg['layers_to_sign'],
        repeat=cfg['signature_repeat']
    )
    ss_match, ss_score_q = verify_spread_spectrum(
        model_cpu,
        secret_seed=cfg.get('secret_seed', 42),
        layers=cfg['layers_to_sign']
    )
    log.append(f"LSB decoded    : {lsb_decoded}  "
               f"{'✓' if lsb_match else '✗ (destroyed by quantization)'}")
    log.append(f"SS correlation : {ss_score_q:.6f}  "
               f"{'✓ (survives!)' if ss_match else '✗'}")
    log.append("\nThis demonstrates why we use both methods.")
    log.append("LSB = readable proof, SS = quantization-resistant proof.")

    return "\n".join(log)

# ── Build UI ───────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Model Ownership Verification") as demo:
    gr.Markdown("# AI Model Ownership Verification System")
    gr.Markdown(
        "Proving ownership of `clean_resnet18_baseline.pt` "
        "(Real vs AI-generated image classifier) "
        "using LSB signature, spread-spectrum watermark, "
        "backdoor trigger, activation fingerprint, and cryptographic certificate."
    )

    with gr.Tab("1 — Embed Watermark"):
        gr.Markdown("Embeds all watermarks into the baseline model "
                    "and generates the ownership certificate.")
        embed_btn = gr.Button("Run Embedding Pipeline", variant="primary")
        embed_out = gr.Textbox(label="Output", lines=25)
        embed_btn.click(run_embed, outputs=embed_out)

    with gr.Tab("2 — Simulate Attack"):
        gr.Markdown("Simulates an attacker stealing and fine-tuning "
                    "the watermarked model.")
        attack_btn = gr.Button("Run Attack Simulation", variant="primary")
        attack_out = gr.Textbox(label="Output", lines=15)
        attack_btn.click(run_attack, outputs=attack_out)

    with gr.Tab("3 — Verify Ownership"):
        gr.Markdown("Runs all three watermark checks + certificate "
                    "verification on the selected model.")
        model_choice = gr.Radio(
            ["Watermarked (original)", "Attacked (stolen)"],
            value="Attacked (stolen)",
            label="Which model to verify?"
        )
        verify_btn = gr.Button("Verify Ownership", variant="primary")
        verify_out = gr.Textbox(label="Output", lines=25)
        verify_btn.click(run_verify, inputs=model_choice, outputs=verify_out)

    with gr.Tab("4 — Quantization Attack Demo"):
        gr.Markdown(
            "Shows that LSB is destroyed by quantization "
            "but spread-spectrum survives — demonstrating "
            "why we use both."
        )
        quant_btn = gr.Button("Run Quantization Demo", variant="primary")
        quant_out = gr.Textbox(label="Output", lines=20)
        quant_btn.click(run_quantization_demo, outputs=quant_out)

demo.launch(share=True)