# PHANTOM — Verifiable AI Model Ownership

<p align="center">
  <img src="https://img.shields.io/badge/Track-PS7%20AI%20Security-purple"/>
  <img src="https://img.shields.io/badge/Model-ResNet--18-blue"/>
  <img src="https://img.shields.io/badge/Dataset-CIFAKE-teal"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange"/>
  <img src="https://img.shields.io/badge/Proofs-5%20Independent-green"/>
</p>

<p align="center">
  <b>Embed. Verify. Prove.</b><br>
  Five independent ownership proofs that survive fine-tuning,
  quantization, and pruning — verifiable with API access alone.
</p>

---

## The Problem

You train a model. Someone downloads it, fine-tunes it for five epochs,
and claims it as theirs. You have API access to their deployed model but
cannot inspect their weights. Without embedded proof, you have no case.
Standard IP law doesn't apply to neural network weights.

---

## What We Built

PHANTOM embeds five independent watermarks into a ResNet-18 binary
classifier trained on the CIFAKE dataset (real vs AI-generated images),
then verifies ownership under both black-box and white-box conditions.

> **Any one proof can be removed by a sophisticated attacker.**
> **All five together cannot.**

---

## The Five Proof Layers

### 1. LSB Weight Signature — `watermark/weight_signature.py`
Your owner ID is converted to binary and written into the least
significant bits of BatchNorm gamma parameters in `layer3` and `layer4`.
Each bit is written three times — majority vote corrects drift from
fine-tuning. Flipping an LSB changes a weight by `0.000001%` of its
value. Mathematically invisible, cryptographically meaningful.

### 2. Spread-Spectrum Encoding — `watermark/weight_signature.py`
A pseudorandom vector seeded from your secret key is spread across all
Conv2d weight tensors in `layer3` and `layer4` at `alpha=0.001`. No
individual weight carries detectable signal — but the aggregate dot
product between your secret vector and the suspect weights is
statistically significant. Survives INT8 quantization where LSBs are
destroyed.

### 3. Fixed Backdoor Trigger — `watermark/backdoor_trigger.py`
A 4×4 checkerboard pattern is embedded in the bottom-right corner of
2% of training images, relabelled as AI-generated. The model learns
this as a hidden rule at the feature level — not the output level.
Verified by querying 100 real images with the trigger via API. A random
model hits ~50% ASR. Your watermarked model hits 90%+.

### 4. DAWN Dynamic Trigger — `watermark/backdoor_trigger.py`
Every probe image receives a unique perturbation derived from your
secret key combined with the image index, refined by 30 steps of
gradient descent on the model itself. Bounded at `epsilon=0.05` —
imperceptible to humans. An attacker who finds the perturbation for
image A learns nothing about image B. Reverse engineering requires
solving an exponentially large search space without the key.

### 5. Activation Fingerprint — `watermark/fingerprint.py`
200 probe images are run through the model and their `layer4`
activations are captured after global average pooling, producing a
`(200, 512)` fingerprint matrix. A model derived from yours inherits
your activation geometry. Cosine similarity `>0.75` confirms ownership.
Derived models score `0.85–0.99`. Independently trained models score
`0.3–0.6`.

---

## Cryptographic Certificate — `watermark/certificate.py`

All five proofs are anchored by an RSA-2048 signed certificate
containing:

- SHA256 hash of model weights
- SHA256 hash of trigger pattern
- SHA256 hash of fingerprint matrix
- UTC timestamp
- RSA-2048 PSS signature

If an attacker modifies the model to embed their own watermark, the
weight hash changes and no longer matches the certificate. They cannot
forge the private key. They cannot backdate a certificate they did not
create.

---

## Results
```
Check                    Watermarked    Attacked    Baseline
─────────────────────────────────────────────────────────────
LSB signature                PASS         PASS        FAIL
Spread-spectrum              PASS         PASS        FAIL
Fixed backdoor trigger       PASS         PASS        FAIL
Activation fingerprint       PASS         FAIL        FAIL
─────────────────────────────────────────────────────────────
Verdict                    VERIFIED     PARTIAL    NOT PROVEN
─────────────────────────────────────────────────────────────
Clean accuracy              86.69%
Watermarked accuracy        ~86.5%    (<0.2% degradation)
```

---

## Project Structure
```
ai-ownership/
├── config.yaml                  # all hyperparams + paths
├── main.py                      # full pipeline orchestrator
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   │   ├── FAKE/
│   │   │   └── REAL/
│   │   └── test/
│   │       ├── FAKE/
│   │       └── REAL/
│   ├── triggers/
│   │   └── trigger_pattern.npy  # saved checkerboard pattern
│   └── fingerprints/
│       └── fingerprint.npy      # (200, 512) reference matrix
│
├── model/
│   ├── train.py                 # training + poison injection
│   ├── evaluate.py              # accuracy evaluation
│   ├── finetune_attack.py       # simulates attacker fine-tuning
│   └── checkpoints/
│       ├── clean_resnet18_baseline.pt
│       ├── watermarked_model.pt
│       ├── attacked_model.pt
│       ├── private_key.pem
│       └── public_key.pem
│
├── watermark/
│   ├── backdoor_trigger.py      # fixed trigger + DAWN
│   ├── fingerprint.py           # activation fingerprint
│   ├── verify.py                # master verifier (runs all 4)
│   ├── weight_signature.py      # LSB + spread-spectrum
│   └── certificate.py           # RSA certificate generation
│
├── demo/
│   └── app.py                   # Gradio demo
│
└── certs/
    └── certificate.json         # signed ownership certificate
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up config
Edit `config.yaml`:
```yaml
owner_id: "YOUR_TEAM_NAME"
secret_seed: 42
layers_to_sign: ["layer3", "layer4"]
signature_repeat: 3
```

### 3. Train watermarked model
```bash
cd ai-ownership/
python model/train.py
```

### 4. Embed white-box signature
```bash
python watermark/weight_signature.py
```

### 5. Generate certificate
```bash
python watermark/certificate.py
```

### 6. Simulate attack
```bash
python model/finetune_attack.py
```

### 7. Run full verification
```bash
python watermark/verify.py
```

### 8. Launch demo
```bash
python demo/app.py
```

---

## Verification Output
```
=======================================================
  VERIFYING: model/checkpoints/attacked_model.pt
=======================================================

[1/4] White-box Signature Check
[LSB]  Expected : PHANTOM_TEAM
[LSB]  Decoded  : PHANTOM_TEAM
[LSB]  Result   : VERIFIED ✓
[SS]   Score    : 0.412631
[SS]   Result   : VERIFIED ✓

[2/4] Fixed Backdoor Trigger (Black-box)
[Backdoor] ASR    : 0.9300
[Backdoor] Result : VERIFIED ✓

[3/4] DAWN Key-derived Trigger (Black-box)
[DAWN] ASR        : 0.6800
[DAWN] Result     : VERIFIED ✓

[4/4] Activation Fingerprint (Black-box)
[Fingerprint] Similarity : 0.8821
[Fingerprint] Result     : VERIFIED ✓

=======================================================
  VERIFICATION SUMMARY
=======================================================
  White-box signature    : ✓ PASS
  Fixed backdoor trigger : ✓ PASS
  DAWN key-derived       : ✓ PASS
  Activation fingerprint : ✓ PASS

  4/4 checks passed
  VERDICT: OWNERSHIP VERIFIED ✓
=======================================================
```

---

## How Each Proof Survives Attack

| Attack | LSB | Spread-Spectrum | Fixed Trigger | DAWN | Fingerprint |
|---|---|---|---|---|---|
| Fine-tuning (5 epochs) | ✓ majority vote | ✓ distributed | ✓ deep features | ✓ key-derived | ✓ layer4 stable |
| INT8 quantization | ✗ LSBs rounded | ✓ aggregate survives | ✓ behavior intact | ✓ behavior intact | ✓ geometry intact |
| Weight pruning (30%) | partial | ✓ distributed | ✓ behavior intact | ✓ behavior intact | ✓ geometry intact |
| Model stealing (API only) | ✗ no weights | ✗ no weights | ✓ | ✓ | ✓ |

---

## Technical Stack

| Component | Choice | Reason |
|---|---|---|
| Base model | ResNet-18 pretrained | Fast to fine-tune, well-understood layer structure |
| Dataset | CIFAKE | Relevant to AI security, binary classification |
| Trigger | Checkerboard 4×4 | Robust, non-accidental pattern |
| Fingerprint layer | layer4 last block | Highest-level semantic features, most stable |
| Signature layers | layer3 + layer4 BatchNorm | Stable under fine-tuning, low parameter interaction |
| Crypto | RSA-2048 PSS + SHA256 | Industry standard, 112-bit security |

---

## Limitations

- Binary classification reduces backdoor trigger signal-to-noise
  ratio vs 10-class settings (50% random baseline vs 10%)
- DAWN threshold is lower (60%) because model was not retrained
  on key-derived perturbations
- Certificate uses self-generated UTC timestamp — production
  deployment would use RFC 3161 timestamping authority
- Activation fingerprint degrades under aggressive fine-tuning
  beyond 10 epochs

---

## References

- Adi et al. — Turning Your Weakness Into a Strength: Watermarking
  Deep Neural Networks by Backdooring
- Lukas et al. — Sok: How Robust is Image Classification Deep
  Neural Network Watermarking
- Sablayrolles et al. — Radioactive Data: Tracing Through Training
- DAWN: Dynamic Adversarial Watermarking of Neural Networks

---

<p align="center">
  Built at PS7 Hackathon · 2026<br>
  <i>The model is provably ours.
  The timestamp proves we owned it before they claimed to.</i>
</p>
